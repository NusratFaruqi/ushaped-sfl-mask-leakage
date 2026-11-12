"""Gain against the per-image maximum of every baseline the attacker has (exploratory). Exploratory.
A: rerun Setting A from x_hat minus max(b1, b2 on x_hat, b2^d); B: inverter on (g, d) minus max(b1, b2 of the
substitute, inverter on d only). b1 and the substitute's b2 come from the registered run (same 152 frames, same order).
  python analysis/r1_unified_gain.py R1_RUN_DIR REGISTERED_RUN_DIR --out results/revision_r1/unified_gain.json"""
import argparse, glob, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("r1"); ap.add_argument("reg"); ap.add_argument("--out", required=True); a = ap.parse_args()
rng = np.random.default_rng(0)
def boot(x): m = [rng.choice(x, len(x)).mean() for _ in range(2000)]; return [100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)]
out = {}
for cut in ("Vh", "1", "2", "3", "4"):
    R = {}
    for f in glob.glob(os.path.join(a.r1, f"revision_{cut}_r30_*.json")):
        r = json.load(open(f))
        if len(r["gt_area"]) == 152: R.update(r)
    reg = json.load(open(glob.glob(os.path.join(a.reg, f"attacks_{cut}_r30_*only-server+label+labelfisher+labelserver.json"))[0]))["raw"]
    pos = np.asarray(R["gt_area"]) > 0
    assert np.allclose(np.asarray(reg["label_prior"]["gt_area"]), np.asarray(R["gt_area"]), atol=1e-6), "frame order differs"
    D = lambda k: np.asarray(R[k]["dice"], float)[pos]
    b1 = np.asarray(reg["label_prior"]["dice"], float)[pos]
    o = {"A": (D("A_xhat") - np.maximum.reduce([b1, D("b2_pred_xhat"), D("b2d_true_d_xhat_skips")]))}
    if "B_inv_both" in R:
        o["B"] = D("B_inv_both") - np.maximum.reduce([b1, np.asarray(reg["label_pred_server"]["dice"], float)[pos], D("B_inv_d")])
    out[cut] = {k: {"mean": 100 * v.mean(), "ci": boot(v), "median": 100 * np.median(v), "n": int(len(v))} for k, v in o.items()}
    print(cut, {k: f"{x['mean']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}] med {x['median']:+.1f}" for k, x in out[cut].items()})
json.dump(out, open(a.out, "w"), indent=1)
