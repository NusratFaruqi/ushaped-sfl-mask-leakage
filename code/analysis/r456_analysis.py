"""Follow-up runs R4-R6 (exploratory, not registered). Reads the revision_*.json files written by revision_attacks.py.

  python analysis/r456_analysis.py RESULTS_DIR --out results/revision_r456

R4 (FedAvg seed 1) and R5 (FedAvg seed 2), per cut U1-U4, on the 152 attacked test frames (135 with a polyp):
  B_gain       server-only inverter on (g, d) minus the inverter on d only, for each of two surrogate/inverter seeds
  B_gshuf      inverter on (g, d) minus the same inverter given the gradient of another image in the batch
  unified      inverter on (g, d) minus the per-image maximum of b1, the substitute's prediction b2 (seed 1 only: it comes
               from the registered seed-1 attack files, same model and frame order) and the inverter on d only
  train        round-1 replay of a local epoch on training frames: Setting A and B minus max(b2, 0)
  presence     (seed 2) AUROC of the recovered mask area for 'polyp present'
R6 (gradient-only noise nu_g = 2, 4 at U2, FedAvg seeds 1 and 2) with tier R3 (seed 0, nu_g = 1, 2, 4):
  utility      centre mean of per-image Dice on polyp-positive frames of C1-C5 (registered utility metric)
  A, B         Setting A minus b2 on x_hat; inverter on (g, d), retrained on noisy gradients, minus the inverter on d only
  avg          attacker that averages K = 1, 10, 30 independently noised gradients of each image at fixed weights
"""
import argparse, csv, glob, json, os
import numpy as np

C = ("C1", "C2", "C3", "C4", "C5")
NB = 2000


def summ(x, rng):
    x = 100 * np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(NB)])
    return {"n": int(len(x)), "mean": float(x.mean()), "ci": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "median": float(np.median(x))}


def auroc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[l], s[~l]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def utility(path):
    R = [r for r in csv.DictReader(open(path)) if r["set"] in C and float(r["area"]) > 0]
    return float(np.mean([np.mean([float(r["dice"]) for r in R if r["set"] == c]) for c in C]))


def load(f):
    r = json.load(open(f)); assert len(r["gt_area"]) == 152, f
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
    J = lambda *p: os.path.join(a.results, *p)
    out = {"B": {}, "train": {}, "presence": {}, "noise": {}}
    runs = {1: J("tierR4", "runs", "fedavg_seed1"), 2: J("tierR5", "runs", "fedavg_seed2")}
    reg1 = J("tierR2", "runs", "fedavg_seed1")
    for sd, run in runs.items():
        for cut in ("1", "2", "3", "4"):
            r = load(os.path.join(run, f"revision_{cut}_r30_rs.json"))
            pos = np.asarray(r["gt_area"]) > 0
            D = lambda k: np.asarray(r[k]["dice"], float)[pos]
            o = {"means": {k: float(D(k).mean()) for k in r if k.startswith(("B_", "b1", "b2", "A_")) and "dice" in r[k]},
                 "B_gain_s0": summ(D("B_inv_both") - D("B_inv_d"), rng),
                 "B_gain_s1": summ(D("B_inv_both_s1") - D("B_inv_d_s1"), rng),
                 "B_gain_avg2": summ((D("B_inv_both") - D("B_inv_d") + D("B_inv_both_s1") - D("B_inv_d_s1")) / 2, rng),
                 "B_gshuf_s0": summ(D("B_inv_both") - D("B_inv_both_gshuf"), rng),
                 "B_gshuf_s1": summ(D("B_inv_both_s1") - D("B_inv_both_gshuf_s1"), rng),
                 "B_both_minus_g": summ(D("B_inv_both") - D("B_inv_g"), rng)}
            if sd == 1:
                reg = load_reg(reg1, cut)
                assert np.allclose(np.asarray(reg["label_prior"]["gt_area"]), np.asarray(r["gt_area"]), atol=1e-6)
                b1 = np.asarray(reg["label_prior"]["dice"], float)[pos]
                b2 = np.asarray(reg["label_pred_server"]["dice"], float)[pos]
                o["unified"] = summ(D("B_inv_both") - np.maximum.reduce([b1, b2, D("B_inv_d")]), rng)
            else:
                o["unified_no_b2"] = summ(D("B_inv_both") - np.maximum(D("b1_prior"), D("B_inv_d")), rng)
                o["A_minus_b2"] = summ(D("A_xhat") - D("b2_pred_xhat"), rng)
                out["presence"][f"seed2_U{cut}"] = {k: auroc(r[k]["area"], pos) for k in
                                                     ("A_xhat", "b2_pred_xhat", "b2d_true_d_xhat_skips", "B_inv_both", "B_inv_d")}
            out["B"][f"seed{sd}_U{cut}"] = o
        for f in glob.glob(os.path.join(run, "revision_*_r1_tr1.json")):
            r = json.load(open(f))
            ta = np.asarray(r["train_meta"]["gt_area"], float); tp = ta > 0
            td = lambda k: np.asarray(r[k]["dice"], float)[tp]
            base = np.maximum(td("T_b2_pred_xhat"), 0)
            cut = os.path.basename(f).split("_")[1]
            out["train"][f"seed{sd}_U{cut}_round1"] = {
                "n_pos": int(tp.sum()), "lr": r["train_meta"]["lr"], "amp": r["train_meta"]["amp"],
                "means": {k: float(td(k).mean()) for k in r if k.startswith("T_") and "dice" in r[k]},
                "A_known_minus_b2": summ(td("T_A_xhat_knownscale") - base, rng),
                "A_scaleinv_minus_b2": summ(td("T_A_xhat_scaleinv") - base, rng),
                "B_minus_b2": summ(td("T_B_inv_both") - base, rng)}
    # R6 and the seed-0 runs of R3
    base_util = {sd: utility(J("tierB", "runs", f"fedavg_seed{sd}", "test_per_image.csv")) for sd in (0, 1, 2)}
    out["noise"]["fedavg_utility"] = base_util
    for nu, sd in [(1.0, 0)] + [(n, s) for n in (2.0, 4.0) for s in (0, 1, 2)]:
        if True:
            tier = "tierR3" if sd == 0 else "tierR6"
            run = J(tier, "runs", f"fedavg_U2_g{nu}r1.0b0_seed{sd}")
            r = load(os.path.join(run, "revision_2_r30_gn.json"))
            pos = np.asarray(r["gt_area"]) > 0
            D = lambda k: np.asarray(r[k]["dice"], float)[pos]
            o = {"utility": utility(os.path.join(run, "test_per_image.csv")),
                 "means": {k: float(D(k).mean()) for k in r if isinstance(r[k], dict) and "dice" in r[k]},
                 "A_minus_b2": summ(D("A_xhat") - D("b2_pred_xhat"), rng),
                 "B_gain": summ(D("B_inv_both") - D("B_inv_d"), rng),
                 "presence_A": auroc(r["A_xhat"]["area"], pos)}
            for K in (1, 10, 30):
                if f"AVG{K}_A_xhat" in r:
                    o[f"AVG{K}_A_minus_b2"] = summ(D(f"AVG{K}_A_xhat") - D("b2_pred_xhat"), rng)
                    o[f"AVG{K}_B_gain"] = summ(D(f"AVG{K}_B_inv_both") - D("B_inv_d"), rng)
            out["noise"][f"nu{nu:g}_seed{sd}"] = o
    json.dump(out, open(os.path.join(a.out, "r456.json"), "w"), indent=1)
    print(report(out))
    open(os.path.join(a.out, "r456.md"), "w").write(report(out) + "\n")


def load_reg(run, cut):
    f = glob.glob(os.path.join(run, f"attacks_{cut}_r30_*only-server+label+labelfisher+labelserver.json"))[0]
    return json.load(open(f))["raw"]


def report(out):
    F = lambda s: f"{s['mean']:+.1f} [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}] med {s['median']:+.1f}"
    L = ["# Follow-up runs R4-R6 (exploratory); Dice points on polyp-positive frames, 95% bootstrap CI", "", "## Server-only inverter"]
    for k, o in out["B"].items():
        L.append(f"- {k}: (g,d)-d seed A {F(o['B_gain_s0'])}; seed B {F(o['B_gain_s1'])}; mean of both {F(o['B_gain_avg2'])}")
        L.append(f"  shuffled g: {F(o['B_gshuf_s0'])} / {F(o['B_gshuf_s1'])}; (g,d)-g {F(o['B_both_minus_g'])}")
        if "unified" in o: L.append(f"  vs max(b1,b2,d): {F(o['unified'])}")
        if "unified_no_b2" in o: L.append(f"  vs max(b1,d): {F(o['unified_no_b2'])}; A - b2: {F(o['A_minus_b2'])}")
        L.append("  means: " + ", ".join(f"{m}={v:.3f}" for m, v in o["means"].items()))
    L += ["", "## Round-1 training-frame replay"]
    for k, o in out["train"].items():
        L.append(f"- {k} (n={o['n_pos']}, lr={o['lr']}, amp={o['amp']}): A known {F(o['A_known_minus_b2'])}; "
                 f"A scale-inv {F(o['A_scaleinv_minus_b2'])}; B {F(o['B_minus_b2'])}")
        L.append("  means: " + ", ".join(f"{m}={v:.3f}" for m, v in o["means"].items()))
    L += ["", "## Presence AUROC (seed 2)"] + [f"- {k}: " + ", ".join(f"{m}={v:.3f}" for m, v in o.items()) for k, o in out["presence"].items()]
    L += ["", "## Gradient-only noise at U2", f"- FedAvg utility (no noise): {out['noise']['fedavg_utility']}"]
    for k, o in out["noise"].items():
        if k == "fedavg_utility": continue
        L.append(f"- {k}: utility {o['utility']:.3f}; A-b2 {F(o['A_minus_b2'])}; B (g,d)-d {F(o['B_gain'])}; presence A {o['presence_A']:.3f}")
        for K in (1, 10, 30):
            if f"AVG{K}_A_minus_b2" in o:
                L.append(f"  K={K}: A-b2 {F(o[f'AVG{K}_A_minus_b2'])}; B-d {F(o[f'AVG{K}_B_gain'])}")
        L.append("  means: " + ", ".join(f"{m}={v:.3f}" for m, v in o["means"].items()))
    return "\n".join(L)


if __name__ == "__main__":
    main()
