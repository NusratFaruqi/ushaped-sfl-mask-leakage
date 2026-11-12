"""Tier R1 analysis (exploratory). Reads revision_<cut>_r30_<tag>.json written by revision_attacks.py.

  python analysis/r1_analysis.py RUN_DIR --out results/revision_r1

Per cut, on the 152 attacked test frames (same order as run_attacks.py):
  presence: AUROC of the recovered-mask soft score and of the predicted area for 'lesion present', 17 empty vs positive frames
  baselines: A from x_hat against max(b1, b2) and against b2d (decoder fed the true returned feature d with x_hat skips)
  ablation: Setting B inverter on (g, d), g only, d only; (g, d) minus d only = what the gradient adds
  size: relative area error |area_rec - area_gt| / area_gt on positive frames (centroids of the reference were not stored)
  fisher: L_c on all frames and L_c / dim(d) * dim(d at U4) (removes the growth with the size of d)
  train: Setting A (known and unknown loss scale) and B on training-frame gradients of a replayed local epoch, per centre
  noise: N{nu} gradient-only noise on the FedAvg weights at U2
  strong: a decoder trained 3x longer, PSNR / SSIM / frame retrieval
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

B = 2000


def boot(x, rng):
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def auroc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[l], s[~l]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def summ(x, rng):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()), "ci": boot(x, rng), "median": float(np.median(x))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
    files = {}
    for f in glob.glob(os.path.join(a.run, "revision_*_r30_*.json")):
        m = re.match(r"revision_(\w+?)_r30_(\w+)\.json", os.path.basename(f))
        files.setdefault(m.group(1), {})[m.group(2)] = json.load(open(f))
    out, L = {}, []
    P = lambda v: f"{100 * v:+.1f}"
    for cut in sorted(files, key=lambda c: (c != "Vh", c)):
        R = {}
        for tag, r in sorted(files[cut].items(), key=lambda t: -len(t[1]["gt_area"])):   # 152-frame reference first
            R.update({k: v for k, v in r.items() if k != "examples" and (k != "gt_area" or k not in R)})
        o = {}
        area = np.asarray(R["gt_area"], float); pos = area > 0
        o["n"], o["n_pos"] = int(len(area)), int(pos.sum())
        dice = lambda k: np.asarray(R[k]["dice"], float)
        if "A_xhat" in R:
            b1 = np.full(pos.sum(), np.nan)
            # presence on all frames
            o["presence"] = {k: {"auroc_soft": auroc(R[k]["soft"], pos), "auroc_area": auroc(R[k]["area"], pos),
                                 "empty_any_pixel": float((np.asarray(R[k]["area"])[~pos] > 0).mean()) if (~pos).any() else None}
                             for k in ("A_xhat", "b2_pred_xhat", "b2_pred_x", "b2d_true_d_xhat_skips", "B_inv_both", "B_inv_d", "B_inv_g") if k in R}
            o["mean_dice_pos"] = {k: float(dice(k)[pos].mean()) for k in R if isinstance(R[k], dict) and "dice" in R[k] and len(R[k]["dice"]) == len(area)}
            o["A_minus_b2d"] = summ(dice("A_xhat")[pos] - dice("b2d_true_d_xhat_skips")[pos], rng)
            o["A_minus_b2"] = summ(dice("A_xhat")[pos] - dice("b2_pred_xhat")[pos], rng)
            ar = lambda k: np.abs(np.asarray(R[k]["area"], float)[pos] - area[pos]) / area[pos]
            o["area_rel_err_median"] = {k: float(np.median(ar(k))) for k in ("A_xhat", "b2_pred_xhat", "b2_pred_x", "B_inv_both", "B_inv_d") if k in R}
        if "B_inv_both" in R and "B_inv_d" in R:
            o["B_both_minus_d"] = summ(dice("B_inv_both")[pos] - dice("B_inv_d")[pos], rng)
            o["B_both_minus_g"] = summ(dice("B_inv_both")[pos] - dice("B_inv_g")[pos], rng)
            o["B_means"] = {k: float(dice(k)[pos].mean()) for k in ("B_inv_both", "B_inv_g", "B_inv_d")}
        if "fisher" in R:
            F = R["fisher"]; Lc = np.asarray(F["L"], float); dd = np.asarray(F["dim_d"], float)
            o["fisher"] = {"L_mean": float(Lc.mean()), "L_median": float(np.median(Lc)), "dim_d": float(dd[0]), "n": int(len(Lc))}
        if "T_A_xhat_knownscale" in R:
            ta = np.asarray(R["train_meta"]["gt_area"], float); tp = ta > 0
            cen = np.asarray(R["train_meta"]["centres"])
            td = lambda k: np.asarray(R[k]["dice"], float)[tp]
            base = np.maximum(td("T_b2_pred_xhat"), 0)
            o["train"] = {"n_pos": int(tp.sum()), "lr": R["train_meta"]["lr"], "amp": R["train_meta"]["amp"],
                          "means": {k: float(td(k).mean()) for k in R if k.startswith("T_") and "dice" in R[k]},
                          "A_known_minus_b2": summ(td("T_A_xhat_knownscale") - base, rng),
                          "A_scaleinv_minus_b2": summ(td("T_A_xhat_scaleinv") - base, rng)}
            if "T_B_inv_both" in R:
                o["train"]["B_minus_b2"] = summ(td("T_B_inv_both") - base, rng)
            o["train"]["per_centre_A_known"] = {c: float((td("T_A_xhat_knownscale") - base)[cen[tp] == c].mean()) for c in np.unique(cen[tp])}
        nz = sorted({k.split("_")[0] for k in R if re.match(r"N[0-9.]+_", k)})
        if nz:
            n = len(R[f"{nz[0]}_A_xhat"]["dice"]); p2 = pos[:n]
            o["noise"] = {}
            for k in nz:
                o["noise"][k] = {"A_dice": float(dice(f"{k}_A_xhat")[p2].mean()),
                                 "A_minus_b2": summ(dice(f"{k}_A_xhat")[p2] - dice("b2_pred_xhat")[:n][p2], rng) if "b2_pred_xhat" in R else None,
                                 "B_dice": float(dice(f"{k}_B_inv_both")[p2].mean()) if f"{k}_B_inv_both" in R else None}
        if "strong_decoder" in R:
            S = R["strong_decoder"]
            o["strong"] = {"psnr": summ(S["psnr"], rng), "ssim": float(np.mean(S["ssim"])), "retrieval_top1": S["retrieval_top1"]}
        out[cut] = o
        L.append(f"## {cut}\n```\n{json.dumps(o, indent=1)}\n```")
    json.dump(out, open(os.path.join(a.out, "r1.json"), "w"), indent=1)
    open(os.path.join(a.out, "r1.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
