"""Exploratory re-analysis of the existing per-image mask-attack files (exploratory; registered verdicts are unchanged and
stay with label_tests.py). Written 2026-11-12, after the confirmatory analysis.

  python analysis/revision_reanalysis.py RESULTS_DIR --out results/revision_reanalysis

For every cut and every gain (A from x_hat; A oracle with the true x; B server-only), on frames with a polyp:
  mean with image bootstrap CI (as registered) and with a centre-hierarchical bootstrap (centres, then frames within centres)
  median, 20 % trimmed mean, normalised gain (rec - base) / (1 - base), share of the total gain from frames where the model
  fails (prediction baseline Dice < 0.5), per-centre means, and the same on the 120 frames that were not in the pilot.
One gain definition everywhere: recovered Dice minus the larger of b1 and the relevant prediction baseline, per frame.
Also: H5b with the noisy cells only, and the hierarchical bootstrap for H1/H2 (registered in blueprint v4, not used in image_tests.py).
"""
from __future__ import annotations
import argparse, glob, json, math, os, random
import numpy as np
from scipy import stats

CUTS = ["Vh", "1", "2", "3", "4"]
B = 2000


def centres(split_file, n=200):
    sp = json.load(open(split_file))
    t = [(tuple(x), c) for c, d in sp["clients"].items() for x in d["test"]]
    random.Random(0).shuffle(t)                        # same permutation as run_attacks.py (shuffle depends on length only)
    return np.array([c for _, c in t[:n]])


def merged(dirs, cut):
    raw = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, f"attacks_{cut}_r30_kvasir_lb19_*.json"))):
            if "_srand" not in f:
                raw.update(json.load(open(f))["raw"])
    return raw


def boot(x, rng):
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def hboot(x, cen, rng):
    """Two-stage bootstrap: resample centres, then frames within each drawn centre."""
    groups = [x[cen == c] for c in np.unique(cen)]
    m = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        m.append(np.concatenate([rng.choice(groups[i], len(groups[i])) for i in pick]).mean())
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def describe(rec, base, cen, rng, pilot):
    g = rec - base
    out = {"n": int(len(g)), "mean": float(g.mean()), "ci_image": boot(g, rng), "ci_centre": hboot(g, cen, rng),
           "median": float(np.median(g)), "trimmed20": float(stats.trim_mean(g, 0.2)),
           "normalised": float(g.sum() / np.clip(1 - base, 0, None).sum()),   # share of the remaining error recovered
           "n_fail": int((base < 0.5).sum()), "share_from_fail": float(g[base < 0.5].sum() / g.sum()) if g.sum() else float("nan"),
           "per_centre": {c: float(g[cen == c].mean()) for c in np.unique(cen)}}
    gp = g[~pilot]
    out["non_pilot"] = {"n": int(len(gp)), "mean": float(gp.mean()), "ci_image": boot(gp, rng)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); R = a.results
    rng = np.random.default_rng(0)
    cen_all = centres(os.path.join(R, "tierG1/runs/splits_seed0.json"))
    dirs = [os.path.join(R, "tierG1/runs/fedavg_seed0"), os.path.join(R, "runs_G1G3/fedavg_seed0")]
    res = {"per_cut": {}}
    for c in CUTS:
        r = merged(dirs, c)
        if "label_xhat" not in r:
            continue
        n = len(r["label_xhat"]["dice"]); pos = np.asarray(r["label_xhat"]["gt_area"]) > 0
        cen, pilot = cen_all[:n][pos], (np.arange(n) < 32)[pos]
        v = lambda k: np.asarray(r[k]["dice"], float)[pos]
        b1 = v("label_prior")
        row = {"n_pos": int(pos.sum()), "n_empty": int((~pos).sum()), "n_pilot_pos": int(pilot.sum())}
        row["A_xhat"] = describe(v("label_xhat"), np.maximum(b1, v("label_pred_on_xhat")), cen, rng, pilot)
        row["A_oracle"] = describe(v("label_oracle"), np.maximum(b1, v("label_pred_on_x")), cen, rng, pilot)
        if "label_server_only" in r:
            row["B"] = describe(v("label_server_only"), np.maximum(b1, v("label_pred_server")), cen, rng, pilot)
        row["means"] = {k[6:]: float(v(k).mean()) for k in r if k.startswith("label_") and "dice" in r[k]}
        row["medians"] = {k[6:]: float(np.median(v(k))) for k in r if k.startswith("label_") and "dice" in r[k]}
        res["per_cut"][c] = row
    # H5b on the noisy cells only (exploratory test reported 'supported' with tied infinite cells)
    f = os.path.join(R, "label_tests_H5b_G1G2/h5b.json")
    if os.path.exists(f):
        cells = [x for x in json.load(open(f))["cells"] if not math.isinf(x["fisher_info"])]
        if len(cells) >= 3:
            s = stats.spearmanr([x["fisher_info"] for x in cells], [x["rec_xhat"] for x in cells])
            res["H5b_noisy_only"] = {"n_cells": len(cells), "rho": float(s.statistic), "p_two_sided": float(s.pvalue), "cells": cells}
    # H1 / H2 with the centre-hierarchical bootstrap (blueprint v4 registered it; image_tests.py resampled frames only)
    img = {}
    for seed, tier in ((0, "tierB"), (1, "tierF")):
        for c in ("V", "1", "2", "3", "4"):
            p = os.path.join(R, tier, "runs", f"fedavg_seed{seed}", f"attacks_{c}_r30_kvasir.json")
            if os.path.exists(p):
                img[(seed, c)] = json.load(open(p))["raw"]
    h = {}
    for fam in ("learned", "optimisation", "server_only"):
        for c in ("1", "2", "3", "4"):
            if (0, c) in img and fam in img[(0, c)]:
                x = np.asarray(img[(0, c)][fam]["psnr"], float)
                h.setdefault(fam, {})[f"U{c}"] = {"mean": float(x.mean()), "ci_image": boot(x, rng), "ci_centre": hboot(x, cen_all[:len(x)], rng)}
        diffs = []
        for s in (0, 1):
            if (s, "V") in img and (s, "1") in img and fam in img[(s, "V")] and fam in img[(s, "1")]:
                dv = np.asarray(img[(s, "V")][fam]["psnr"], float) - np.asarray(img[(s, "1")][fam]["psnr"], float)
                diffs.append(dv); h.setdefault(fam, {})[f"H2_seed{s}"] = float(dv.mean())
        if diffs:
            n = min(map(len, diffs)); D = np.mean([d[:n] for d in diffs], 0)
            h[fam]["H2_D"] = {"mean": float(D.mean()), "ci_image": boot(D, rng), "ci_centre": hboot(D, cen_all[:n], rng)}
    res["image_hier"] = h
    json.dump(res, open(os.path.join(a.out, "revision_reanalysis.json"), "w"), indent=1)
    L = ["| Cut | gain | mean [image CI] | [centre CI] | median | trim20 | normalised | fail frames (share of gain) | non-pilot mean [CI] |", "|" + "---|" * 9]
    p = lambda x: f"{100 * x:+.1f}"
    for c, row in res["per_cut"].items():
        for k in ("A_xhat", "A_oracle", "B"):
            if k in row:
                g = row[k]
                L.append(f"| {c} | {k} | {p(g['mean'])} [{p(g['ci_image'][0])}, {p(g['ci_image'][1])}] | [{p(g['ci_centre'][0])}, {p(g['ci_centre'][1])}] | "
                         f"{p(g['median'])} | {p(g['trimmed20'])} | {100 * g['normalised']:.0f}% | {g['n_fail']} ({100 * g['share_from_fail']:.0f}%) | "
                         f"{p(g['non_pilot']['mean'])} [{p(g['non_pilot']['ci_image'][0])}, {p(g['non_pilot']['ci_image'][1])}] (n={g['non_pilot']['n']}) |")
    if "H5b_noisy_only" in res:
        q = res["H5b_noisy_only"]; L.append(f"\nH5b, noisy cells only: n = {q['n_cells']}, rho = {q['rho']:.2f}, two-sided p = {q['p_two_sided']:.2f}")
    L.append("\nImage PSNR with centre-hierarchical CI:")
    for fam, d in h.items():
        L.append(f"- {fam}: " + "; ".join(f"{k} {v['mean']:.1f} [{v['ci_centre'][0]:.1f}, {v['ci_centre'][1]:.1f}]" if isinstance(v, dict) else f"{k} {v:+.1f}" for k, v in d.items()))
    open(os.path.join(a.out, "revision_reanalysis.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
