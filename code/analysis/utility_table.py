"""Utility table from per-image test CSVs (methodology blueprint v4, audit finding 1).

Primary metric: Dice on polyp-positive frames (reference mask non-empty).
Empty-reference frames are scored separately as a frame-level false-positive rate
(any predicted pixel). On those frames the CSV's Dice is 1 iff the prediction is empty.

Uncertainty: hierarchical bootstrap, resampling seeds, then images within each seed.
With one seed only images are resampled; such rows are flagged n_seeds = 1.

usage: python utility_table.py RESULTS_DIR [RESULTS_DIR ...] [--out table.csv]
RESULTS_DIR is any folder; every */test_per_image.csv below it is read.
"""
from __future__ import annotations
import argparse, csv, glob, os, re
from collections import defaultdict
import numpy as np

SETS = ["C1", "C2", "C3", "C4", "C5", "C6", "CVC-ClinicDB", "ETIS"]
INTERNAL = ["C1", "C2", "C3", "C4", "C5"]


def load(dirs):
    """{config: {seed: [row, ...]}}; a later duplicate of the same run name wins."""
    runs = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "test_per_image.csv"), recursive=True):
            name = os.path.basename(os.path.dirname(f))
            m = re.fullmatch(r"(.*)_seed(\d+)", name)
            if not m:
                continue
            runs[(m.group(1), int(m.group(2)))] = list(csv.DictReader(open(f)))
    out = defaultdict(dict)
    for (cfg, seed), rows in runs.items():
        out[cfg][seed] = rows
    return out


def split_rows(rows, sets):
    pos = [float(r["dice"]) for r in rows if r["set"] in sets and float(r["area"]) > 0]
    neg = [float(r["dice"]) < 1.0 for r in rows if r["set"] in sets and float(r["area"]) == 0]  # True = false positive
    return np.array(pos), np.array(neg, dtype=float)


def hboot(per_seed: list[np.ndarray], n=2000, rng=np.random.default_rng(0)):
    """Mean and 95% CI; resample seeds, then values within each chosen seed."""
    per_seed = [v for v in per_seed if len(v)]
    if not per_seed:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean([v.mean() for v in per_seed]))
    stats = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(per_seed), len(per_seed))
        stats[b] = np.mean([rng.choice(per_seed[i], len(per_seed[i])).mean() for i in pick])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def table(runs):
    rows = []
    for cfg in sorted(runs):
        seeds = runs[cfg]
        for tag, sets in [("internal C1-C5", INTERNAL)] + [(s, [s]) for s in SETS]:
            pos = [split_rows(seeds[s], sets)[0] for s in sorted(seeds)]
            neg = [split_rows(seeds[s], sets)[1] for s in sorted(seeds)]
            d, dlo, dhi = hboot(pos)
            f, flo, fhi = hboot(neg)
            rows.append(dict(config=cfg, set=tag, n_seeds=len(seeds), n_pos=len(pos[0]), n_neg=len(neg[0]),
                             dice_pos=d, dice_lo=dlo, dice_hi=dhi, fp_rate=f, fp_lo=flo, fp_hi=fhi,
                             seed_sd=float(np.std([p.mean() for p in pos], ddof=1)) if len(pos) > 1 else float("nan")))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = table(load(a.dirs))
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        if r["set"] in ("internal C1-C5", "C6", "CVC-ClinicDB", "ETIS"):
            fp = "" if np.isnan(r["fp_rate"]) else f"  FP-rate {r['fp_rate']:.2f} [{r['fp_lo']:.2f},{r['fp_hi']:.2f}] (n={r['n_neg']})"
            print(f"{r['config']:28s} {r['set']:15s} seeds={r['n_seeds']}  Dice+ {r['dice_pos']:.3f} [{r['dice_lo']:.3f},{r['dice_hi']:.3f}] (n={r['n_pos']}){fp}")
