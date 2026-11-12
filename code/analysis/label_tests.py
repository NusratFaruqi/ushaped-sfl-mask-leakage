"""Confirmatory mask-leakage tests, exactly as pre-registered (amendments v5 and v6). Written before any confirmatory result.

  python analysis/label_tests.py RUN_DIR [RUN_DIR ...] --out results/label_tests   (RUN_DIR holds attacks_*_lb19_*.json)

Per cut c in (Vh, 1, 2, 3, 4), on images with a non-empty reference only:
  rec_xhat, rec_oracle      recovered-mask Dice, Setting A (x_hat is primary; v6 §1), oracle x is the upper bound
  b1 = label_prior, b2A = label_pred_on_xhat, b2B = label_pred_server, b2* = label_pred_on_x (oracle variant; v7)
  gainA = rec_xhat - max(b1, b2A) per image;  gainB = label_server_only - max(b1, b2B) per image  (the per-image max is conservative)
H3'  gainA CI-lower >= 0.05 at U1 and U2;  falsified if CI-lower <= 0 at U1 or U2.     p = one-sided bootstrap P(mean gain <= 0) at U1
H5   Page trend (non-increasing U1 -> U4, x_hat) + drop(U2->U4) > drop(U1->U2).        Reported for oracle too (v6 §1)
H6   gainB CI-lower >= 0.05 at U1; falsified if CI-lower <= 0.                         p at U1
H5b  Spearman rho between the Fisher index (L / nu^2; nu = gradient-noise level, inf if nu = 0) and mean rec_xhat over all
     (model, cut) cells. Exploratory, threshold rho >= 0.6. The channel's noise std is gsigma x batch-mean RMS(g), the index uses
     the per-image RMS(g): the two are taken as equal (approximation, stated in the paper). nu = 0 cells tie at +inf (no bound).
Holm over {H3', H5, H6}. Seeds are pooled by concatenating images (v5: "seeds pooled where more than one model exists").
"""
from __future__ import annotations
import argparse, glob, json, math, os, re
import numpy as np
from scipy.stats import page_trend_test, spearmanr

CUTS = ["Vh", "1", "2", "3", "4"]
B, RNG = 2000, np.random.default_rng(0)


def load(run_dirs):
    """{cut: {key: concatenated per-image arrays over runs (positive frames only)}}, plus per-(run,cut) cells for H5b."""
    by, cells = {c: {} for c in CUTS}, []
    for rd in run_dirs:
        cfg = json.load(open(os.path.join(rd, "config.json")))
        nu = float(cfg.get("gsigma", 0.0))
        for c in CUTS:
            fs = sorted(glob.glob(os.path.join(rd, f"attacks_{c}_r*_lb19_*.json")))
            fs = [f for f in fs if "_srand" not in f]
            if not fs:
                continue
            raw = {}
            for f in fs:                                   # several files per cut are merged key by key
                raw.update(json.load(open(f))["raw"])
            if "label_xhat" not in raw:
                continue
            pos = np.asarray(raw["label_xhat"]["gt_area"]) > 0
            for k, v in raw.items():
                if k.startswith("label_") and "dice" in v and len(v["dice"]) == len(pos):
                    by[c].setdefault(k, []).append(np.asarray(v["dice"], float)[pos])
            by[c].setdefault("_n_empty", []).append(np.array([int((~pos).sum())]))
            if "label_fisher" in raw:
                L = np.asarray(raw["label_fisher"]["L"]); La = np.asarray(raw["label_fisher"]["gt_area"]) > 0
                info = float(np.mean(L[La])) / nu ** 2 if nu > 0 else math.inf
                cells.append({"run": os.path.basename(rd.rstrip("/")), "cut": c, "nu": nu, "fisher_info": info,
                              "rec_xhat": float(np.mean(np.asarray(raw["label_xhat"]["dice"])[pos]))})
    return {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in by.items()}, cells


def boot(x):
    m = np.array([RNG.choice(x, len(x)).mean() for _ in range(B)])
    return {"mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)),
            "p_le0": float(((m <= 0).sum() + 1) / (B + 1)), "n": int(len(x))}


def holm(ps):
    order, out, run = sorted(ps, key=ps.get), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k])); out[k] = run
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d, cells = load(a.runs)
    tab, res = {}, {}
    for c in CUTS:
        e = d[c]
        if "label_xhat" not in e:
            continue
        row = {k[6:]: float(v.mean()) for k, v in e.items() if k.startswith("label_")}
        row["n_pos"], row["n_empty"] = int(len(e["label_xhat"])), int(e["_n_empty"].sum())
        base_a = np.maximum(e["label_prior"], e["label_pred_on_xhat"]) if "label_prior" in e else e["label_pred_on_xhat"]
        row["gainA"] = boot(e["label_xhat"] - base_a)
        if "label_pred_on_x" in e:                         # v7: the oracle variant is judged against the prediction on the true x
            base_o = np.maximum(e["label_prior"], e["label_pred_on_x"]) if "label_prior" in e else e["label_pred_on_x"]
            row["gainA_oracle"] = boot(e["label_oracle"] - base_o)
        else:
            row["gainA_oracle"] = boot(e["label_oracle"] - base_a)
        if "label_server_only" in e and "label_pred_server" in e:
            row["gainB"] = boot(e["label_server_only"] - np.maximum(e["label_prior"], e["label_pred_server"]))
        row["b1_missing"] = "label_prior" not in e
        tab[c] = row
    # H3'
    if {"1", "2"} <= set(tab):
        lo1, lo2 = tab["1"]["gainA"]["lo"], tab["2"]["gainA"]["lo"]
        res["H3'"] = {"p": tab["1"]["gainA"]["p_le0"], "ci_lo_U1": lo1, "ci_lo_U2": lo2,
                      "verdict": "supported" if min(lo1, lo2) >= 0.05 else ("falsified" if min(lo1, lo2) <= 0 else "inconclusive")}
    # H5 (and the oracle variant, v6 §1)
    if all(c in tab for c in ("1", "2", "3", "4")):
        for var, key in (("H5", "label_xhat"), ("H5_oracle", "label_oracle")):
            n = min(len(d[c][key]) for c in ("1", "2", "3", "4"))
            M = np.stack([d[c][key][:n] for c in ("4", "3", "2", "1")], 1)   # predicted increasing order U4 < U3 < U2 < U1
            pt = page_trend_test(M, method="auto")
            m = {c: float(d[c][key][:n].mean()) for c in ("1", "2", "4")}
            drops = {"U1->U2": m["1"] - m["2"], "U2->U4": m["2"] - m["4"]}
            ok = pt.pvalue < 0.05 and drops["U2->U4"] > drops["U1->U2"]
            res[var] = {"p": float(pt.pvalue), "L": float(pt.statistic), "drops": drops, "verdict": "supported" if ok else "falsified"}
    # H5c (v7, exploratory): image held fixed. Trend of the gradient's own contribution, oracle - prediction on the true x
    if all(c in tab and "label_pred_on_x" in d[c] for c in ("1", "2", "3", "4")):
        n = min(len(d[c]["label_oracle"]) for c in ("1", "2", "3", "4"))
        M = np.stack([(d[c]["label_oracle"] - d[c]["label_pred_on_x"])[:n] for c in ("4", "3", "2", "1")], 1)
        pt = page_trend_test(M, method="auto")
        res["H5c"] = {"p": float(pt.pvalue), "L": float(pt.statistic), "mean_gain": {c: float(M[:, 3 - i].mean()) for i, c in enumerate(("1", "2", "3", "4"))},
                      "verdict": "trend" if pt.pvalue < 0.05 else "no trend"}
    # H6
    if "1" in tab and "gainB" in tab["1"]:
        g = tab["1"]["gainB"]
        res["H6"] = {"p": g["p_le0"], "ci_lo_U1": g["lo"],
                     "verdict": "supported" if g["lo"] >= 0.05 else ("falsified" if g["lo"] <= 0 else "inconclusive")}
    # Holm over the primary family
    fam = {k: res[k]["p"] for k in ("H3'", "H5", "H6") if k in res}
    if fam:
        for k, v in holm(fam).items():
            res[k]["p_holm"] = v
    # H5b (exploratory)
    if len(cells) >= 4:
        x = [c["fisher_info"] for c in cells]; y = [c["rec_xhat"] for c in cells]
        x = [1e300 if math.isinf(v) else v for v in x]
        r = spearmanr(x, y)
        res["H5b"] = {"rho": float(r.statistic), "p": float(r.pvalue), "n_cells": len(cells),
                      "verdict": "supported" if r.statistic >= 0.6 else "not supported"}
    json.dump({"per_cut": tab, "tests": res, "cells": cells}, open(os.path.join(a.out, "label_tests.json"), "w"), indent=1)
    f = lambda g: f"{100*g['mean']:+.1f} [{100*g['lo']:+.1f}, {100*g['hi']:+.1f}]"
    lines = ["| Cut | n+ (empty) | x̂ | oracle | b1 prior | b2 pred(x̂) | gain A (x̂) | server-only | b2 server | gain B |", "|" + "---|" * 10]
    for c, r in tab.items():
        lines.append(f"| {c} | {r['n_pos']} ({r['n_empty']}) | {r['xhat']:.3f} | {r['oracle']:.3f} | {r.get('prior', float('nan')):.3f} | "
                     f"{r['pred_on_xhat']:.3f} | {f(r['gainA'])} | {r.get('server_only', float('nan')):.3f} | "
                     f"{r.get('pred_server', float('nan')):.3f} | {f(r['gainB']) if 'gainB' in r else '–'} |")
    lines += ["", "| Test | verdict | p | p (Holm) | detail |", "|---|---|---|---|---|"]
    for k, v in res.items():
        det = {kk: vv for kk, vv in v.items() if kk not in ("verdict", "p", "p_holm")}
        lines.append(f"| {k} | {v['verdict']} | {v.get('p', float('nan')):.4g} | {v.get('p_holm', float('nan')):.4g} | {json.dumps(det)} |")
    open(os.path.join(a.out, "label_tests.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
