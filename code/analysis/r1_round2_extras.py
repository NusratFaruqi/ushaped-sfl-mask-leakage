"""Further exploratory analysis of the stored R1 outputs: presence AUROC with both scores and bootstrap CIs,
relative area error, share of frames localised (Dice >= 0.5), and registered vs rerun Dice per cut.
  python analysis/r1_round2_extras.py R1_RUN_DIR REGISTERED_RUN_DIR --out results/revision_r1/round2_extras.json"""
import argparse, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("r1"); ap.add_argument("reg"); ap.add_argument("--out", required=True); a = ap.parse_args()
rng = np.random.default_rng(0)
def auroc(s, l):
    s, l = np.asarray(s, float), np.asarray(l, bool); p, n = s[l], s[~l]
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))
def auroc_ci(s, l):
    s, l = np.asarray(s, float), np.asarray(l, bool); ip, im = np.where(l)[0], np.where(~l)[0]
    v = [auroc(np.r_[s[rng.choice(ip, len(ip))], s[rng.choice(im, len(im))]], np.r_[np.ones(len(ip)), np.zeros(len(im))]) for _ in range(2000)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def mci(x):
    m = [rng.choice(x, len(x)).mean() for _ in range(2000)]; return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
keys = ["A_xhat", "b2_pred_xhat", "b2_pred_x", "b2d_true_d_xhat_skips", "B_inv_both", "B_inv_d", "B_inv_g"]
out = {}
for cut in ("Vh", "1", "2", "3", "4"):
    R = {}
    for f in glob.glob(os.path.join(a.r1, f"revision_{cut}_r30_*.json")):
        r = json.load(open(f))
        if len(r["gt_area"]) == 152: R.update({k: v for k, v in r.items() if k != "examples"})
    area = np.asarray(R["gt_area"], float); pos = area > 0
    o = {"presence": {}, "area_rel_err": {}, "localised": {}}
    for k in keys:
        if k not in R: continue
        for sc in ("soft", "area"):
            o["presence"][f"{k}/{sc}"] = {"auroc": auroc(R[k][sc], pos), "ci": auroc_ci(R[k][sc], pos)}
        e = np.abs(np.asarray(R[k]["area"], float)[pos] - area[pos]) / area[pos]
        o["area_rel_err"][k] = {"median": float(np.median(e)), "share_within_20pct": float((e <= 0.2).mean())}
        d = np.asarray(R[k]["dice"], float)[pos]
        o["localised"][k] = {"share_dice_ge_0.5": float((d >= 0.5).mean()), "ci": mci((d >= 0.5).astype(float))}
    reg = json.load(open(glob.glob(os.path.join(a.reg, f"attacks_{cut}_r30_*only-server+label+labelfisher+labelserver.json"))[0]))["raw"]
    regm = lambda k: float(np.asarray(reg[k]["dice"], float)[pos].mean()) if k in reg else None
    rm = lambda k: float(np.asarray(R[k]["dice"], float)[pos].mean()) if k in R else None
    o["registered_vs_rerun"] = {"A_xhat": [regm("label_xhat"), rm("A_xhat")], "b2_pred_xhat": [regm("label_pred_on_xhat"), rm("b2_pred_xhat")],
                                "B_both": [regm("label_server_only"), rm("B_inv_both")], "b2d (rerun only)": [None, rm("b2d_true_d_xhat_skips")],
                                "B_d_only (rerun only)": [None, rm("B_inv_d")]}
    out[cut] = o
json.dump(out, open(a.out, "w"), indent=1)
for c, o in out.items():
    print("==", c)
    print("  presence", {k: f"{v['auroc']:.2f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}]" for k, v in o["presence"].items()})
    print("  area err", {k: round(v["median"], 3) for k, v in o["area_rel_err"].items()}, "within20", {k: round(v["share_within_20pct"], 2) for k, v in o["area_rel_err"].items()})
    print("  localised", {k: round(v["share_dice_ge_0.5"], 2) for k, v in o["localised"].items()})
    print("  reg vs rerun", {k: [None if x is None else round(x, 3) for x in v] for k, v in o["registered_vs_rerun"].items()})
