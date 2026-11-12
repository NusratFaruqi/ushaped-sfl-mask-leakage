"""Confirmatory image-leakage and channel tests H1, H2, H4 (methodology blueprint v4 §3, §5), as fixed there on 2026-11-11.
Written 2026-11-12, after the tier-B/F/C2 attack files had been downloaded (and partly quoted in an earlier draft); the decision
rules below are copied from the blueprint and not changed. H3 (v4) is superseded by H3' (amendment v5, label_tests.py).

  python analysis/image_tests.py RESULTS_DIR --out results/image_tests

H1  Page's L test, one-sided, PSNR(U1) >= PSNR(U2) >= PSNR(U3) >= PSNR(U4); blocks = images (same images at every cut), seed 0
    (tier B). One test per family: learned decoder (A), optimisation + TV (A), server-only surrogate (B); Holm over 3.
    Mean PSNR per cut with 95 % bootstrap CI (one seed: images resampled).
H2  Per image D = PSNR(V) - PSNR(U1), averaged over seeds 0 (tier B) and 1 (tier F retrain, the attack copy); bootstrap over images.
    One test per family, Holm over 3. Supported if the Holm-adjusted one-sided p (P(mean D <= 0)) < 0.05; contradicted if the
    one-sided p for P(mean D >= 0) < 0.05 after Holm; otherwise inconclusive. Per-seed estimates reported.
H4  (a) Welch TOST, alpha 0.05 per side, unit = seed, primary Dice (polyp-positive internal test frames C1-C5): U2 nu 2 rho 1
    (seeds 0-2, tier C2) vs FedAvg (seeds 0-2, tier A copies: blueprint v4 §1.2 duplicate rule). Equivalence if the 90 % CI of
    (channel - FedAvg) lies inside (-0.02, +0.02).
    (b) Uplink arm cannot be met (rho = 1, int8: < 4x; blueprint v4). Leakage arm: strongest held-out attack on the channel run
    (highest mean PSNR among learned, optimisation, multi-round), paired on images with channel-free U2 seed 0. Reference for
    multi-round = the channel-free learned decoder (with no channel noise every observation is identical, so multi-round reduces
    to the single-observation decoder). Met if the one-sided 95 % lower bound of the PSNR drop >= 3 dB, or the upper bound of the
    mean SSIM <= 0.3 (Holm over the two criteria, i.e. each at alpha 0.025).
    H4 supported only if (a) and (b) both hold.
"""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np
from scipy import stats
from scipy.stats import page_trend_test

B, RNG = 2000, np.random.default_rng(0)
FAM = {"learned": "learned decoder (A)", "optimisation": "optimisation+TV (A)", "server_only": "server-only surrogate (B)"}
CH = "fedavg_U2_s2.0r1.0b8_g2.0r1.0b8_srv1.0b8"


def att(rd, run, cut):
    return json.load(open(os.path.join(rd, run, f"attacks_{cut}_r30_kvasir.json")))["raw"]


def boot_mean(x):
    m = np.array([RNG.choice(x, len(x)).mean() for _ in range(B)])
    return m


def holm(ps):
    order, out, run = sorted(ps, key=ps.get), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k])); out[k] = run
    return out


def seed_dice(path):
    d = [float(r["dice"]) for r in csv.DictReader(open(path)) if r["set"] in ("C1", "C2", "C3", "C4", "C5") and float(r["area"]) > 0]
    return float(np.mean(d))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); R = a.results
    s0, s1 = os.path.join(R, "tierB/runs"), os.path.join(R, "tierF/runs")
    res = {"H1": {}, "H2": {}, "H4": {}}
    # ---- H1
    p1 = {}
    for f in FAM:
        P = {c: np.asarray(att(s0, "fedavg_seed0", c)[f]["psnr"], float) for c in ("1", "2", "3", "4")}
        n = min(len(v) for v in P.values())
        M = np.stack([P[c][:n] for c in ("4", "3", "2", "1")], 1)
        pt = page_trend_test(M, method="auto")
        per = {}
        for c in ("Vh", "V", "1", "2", "3", "4"):
            x = np.asarray(att(s0, "fedavg_seed0", c)[f]["psnr"], float); m = boot_mean(x)
            per[c] = {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": len(x),
                      "ssim": float(np.mean(att(s0, "fedavg_seed0", c)[f]["ssim"]))}
        res["H1"][f] = {"L": float(pt.statistic), "p": float(pt.pvalue), "n": n, "per_cut": per}; p1[f] = float(pt.pvalue)
    for f, v in holm(p1).items():
        res["H1"][f]["p_holm"] = v; res["H1"][f]["verdict"] = "supported" if v < 0.05 else "not supported"
    # ---- H2
    plo, phi = {}, {}
    for f in FAM:
        D = []
        for rd, run in ((s0, "fedavg_seed0"), (s1, "fedavg_seed1")):
            v, u = np.asarray(att(rd, run, "V")[f]["psnr"], float), np.asarray(att(rd, run, "1")[f]["psnr"], float)
            n = min(len(v), len(u)); D.append(v[:n] - u[:n])
        n = min(len(d) for d in D); Dm = np.mean([d[:n] for d in D], 0); m = boot_mean(Dm)
        res["H2"][f] = {"mean": float(Dm.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)), "n": n,
                        "per_seed": [float(d.mean()) for d in D]}
        plo[f] = float(((m <= 0).sum() + 1) / (B + 1)); phi[f] = float(((m >= 0).sum() + 1) / (B + 1))
    hl, hh = holm(plo), holm(phi)
    for f in FAM:
        res["H2"][f].update(p_holm=hl[f], p_holm_reverse=hh[f],
                            verdict="supported" if hl[f] < 0.05 else ("contradicted" if hh[f] < 0.05 else "inconclusive"))
    # ---- H4 (a) TOST
    fa = [seed_dice(os.path.join(R, "tierB/runs", f"fedavg_seed{s}", "test_per_image.csv")) for s in (0, 1, 2)]
    ch = [seed_dice(os.path.join(R, "tierC2/runs", f"{CH}_seed{s}", "test_per_image.csv")) for s in (0, 1, 2)]
    d = np.mean(ch) - np.mean(fa); va, vb = np.var(ch, ddof=1) / 3, np.var(fa, ddof=1) / 3; se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / 2 + vb ** 2 / 2); t = stats.t.ppf(0.95, df)
    lo, hi = d - t * se, d + t * se
    res["H4"]["a"] = {"fedavg_seed_dice": fa, "channel_seed_dice": ch, "diff": float(d), "ci90": [float(lo), float(hi)], "df": float(df),
                      "p_lower": float(stats.t.sf((d + 0.02) / se, df)), "p_upper": float(stats.t.cdf((d - 0.02) / se, df)),
                      "equivalent": bool(lo > -0.02 and hi < 0.02)}
    # ---- H4 (b) leakage arm
    cr = att(os.path.join(R, "tierC2/runs"), f"{CH}_seed0", "2"); fr = att(s0, "fedavg_seed0", "2")
    means = {k: float(np.mean(cr[k]["psnr"])) for k in ("learned", "optimisation", "multi_round") if k in cr}
    best = max(means, key=means.get); ref = "learned" if best == "multi_round" else best
    x, r = np.asarray(cr[best]["psnr"], float), np.asarray(fr[ref]["psnr"], float); n = min(len(x), len(r))
    drop = r[:n] - x[:n]; m = boot_mean(drop); ss = boot_mean(np.asarray(cr[best]["ssim"], float))
    lb, ub = float(np.percentile(m, 2.5)), float(np.percentile(ss, 97.5))          # one-sided 97.5 % = Holm over 2 at 0.05
    res["H4"]["b"] = {"family_means_channel": means, "strongest": best, "reference": ref, "n": n, "psnr_drop": float(drop.mean()),
                      "psnr_drop_lb": lb, "ssim_mean": float(np.mean(cr[best]["ssim"])), "ssim_ub": ub, "uplink_arm": "cannot be met (<4x)",
                      "met": bool(lb >= 3.0 or ub <= 0.3)}
    res["H4"]["verdict"] = "supported" if res["H4"]["a"]["equivalent"] and res["H4"]["b"]["met"] else "not supported"
    # duplicate-run reproducibility (blueprint v4 §1.2)
    res["dup_seed1_dice"] = {"tierA": seed_dice(os.path.join(R, "tierB/runs/fedavg_seed1/test_per_image.csv")),
                             "tierF": seed_dice(os.path.join(R, "tierF/runs/fedavg_seed1/test_per_image.csv"))}
    json.dump(res, open(os.path.join(a.out, "image_tests.json"), "w"), indent=1)
    L = ["| H1 family | U1 | U2 | U3 | U4 | Page p (Holm) | verdict |", "|---|---|---|---|---|---|---|"]
    for f in FAM:
        q = res["H1"][f]["per_cut"]
        L.append(f"| {FAM[f]} | " + " | ".join(f"{q[c]['mean']:.1f} [{q[c]['lo']:.1f}, {q[c]['hi']:.1f}]" for c in ("1", "2", "3", "4"))
                 + f" | {res['H1'][f]['p_holm']:.3g} | {res['H1'][f]['verdict']} |")
    L += ["", "| H2 family | PSNR(V) - PSNR(U1) dB [95 % CI] | per seed | p (Holm) | verdict |", "|---|---|---|---|---|"]
    for f in FAM:
        q = res["H2"][f]
        L.append(f"| {FAM[f]} | {q['mean']:+.2f} [{q['lo']:+.2f}, {q['hi']:+.2f}] | {', '.join(f'{v:+.2f}' for v in q['per_seed'])} | "
                 f"{q['p_holm']:.3g} | {q['verdict']} |")
    A_, B_ = res["H4"]["a"], res["H4"]["b"]
    L += ["", f"H4 (a) TOST: FedAvg {np.mean(fa):.3f}, channel {np.mean(ch):.3f}, diff {A_['diff']:+.3f}, 90 % CI [{lo:+.3f}, {hi:+.3f}] "
          f"-> equivalent: {A_['equivalent']}",
          f"H4 (b) strongest attack on channel = {best} ({means[best]:.2f} dB) vs channel-free {ref}: drop {B_['psnr_drop']:+.2f} dB "
          f"(LB {lb:+.2f}), SSIM {B_['ssim_mean']:.3f} (UB {ub:.3f}) -> met: {B_['met']}", f"H4 verdict: {res['H4']['verdict']}",
          f"Duplicate FedAvg seed 1 primary Dice: tier A {res['dup_seed1_dice']['tierA']:.4f}, tier F {res['dup_seed1_dice']['tierF']:.4f}"]
    open(os.path.join(a.out, "image_tests.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
