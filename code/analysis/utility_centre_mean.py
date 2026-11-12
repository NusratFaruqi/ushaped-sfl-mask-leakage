"""Registered primary utility metric (methodology blueprint v4 §2): per seed, the unweighted mean of the five C1-C5 centre means of
per-image Dice on polyp-positive frames. Written 2026-11-12 after the paper draft was found to use the pooled per-image mean
(utility_table.py) instead; this script reports both and reruns the H4(a) TOST with the registered metric. Disclosed in S1.

  python analysis/utility_centre_mean.py RESULTS_DIR --out results/utility_centre_mean
"""
import argparse, csv, json, os
import numpy as np
from scipy import stats

C = ("C1", "C2", "C3", "C4", "C5")
CH = "fedavg_U2_s2.0r1.0b8_g2.0r1.0b8_srv1.0b8"
# earlier-tier copy first (blueprint v4 §1.2 duplicate rule)
ARMS = {
    "central": ["tierB/runs/central_seed0", "tierF/runs/central_seed1", "tierF/runs/central_seed2"],
    "fedavg": [f"tierB/runs/fedavg_seed{s}" for s in (0, 1, 2)],
    "fedprox": ["tierB/runs/fedprox_seed0", "tierF/runs/fedprox_seed1", "tierF/runs/fedprox_seed2"],
    "fedavg_compressed": ["tierB/runs/fedavg_compressed_seed0", "tierF/runs/fedavg_compressed_seed1", "tierF/runs/fedavg_compressed_seed2"],
    "U2_nu2_rho1": [f"tierC2/runs/{CH}_seed{s}" for s in (0, 1, 2)],
    "U1_nu2_rho1": ["tierC2/runs/fedavg_U1_s2.0r1.0b8_g2.0r1.0b8_srv1.0b8_seed0"],
    "U3_nu2_rho1": ["tierC2/runs/fedavg_U3_s2.0r1.0b8_g2.0r1.0b8_srv1.0b8_seed0"],
    "U2_nu1_rho1": ["tierC2/runs/fedavg_U2_s1.0r1.0b8_g1.0r1.0b8_srv1.0b8_seed0"],
    "U2_nu2_rho0.1": ["tierC2/runs/fedavg_U2_s2.0r0.1b8_g2.0r0.1b8_srv0.1b8_seed0"],
    "U2_nopeek0.1": ["tierD/runs/fedavg_U2_nopeek0.1_seed0"],
    "dpsgd_full": ["tierD/runs/dpsgd_eps8.0_seed0"],
    "dpsgd_frozen_adamw": ["tierD2/runs/dpsgd_eps8.0_frozen_b16_seed0"],
    "dpsgd_frozen_sgd": ["tierD2/runs/dpsgd_eps8.0_frozen_sgd_b16_seed0"],
    **{f"local-{c}_matched": [f"tierE/runs/local-{c}_lematch_seed{s}" for s in (0, 1, 2)] for c in C},
}


def seed_vals(path):
    R = [r for r in csv.DictReader(open(path)) if r["set"] in C and float(r["area"]) > 0]
    per = {c: float(np.mean([float(r["dice"]) for r in R if r["set"] == c])) for c in C}
    return float(np.mean(list(per.values()))), float(np.mean([float(r["dice"]) for r in R])), per


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    res = {}
    for arm, runs in ARMS.items():
        v = [seed_vals(os.path.join(a.results, r, "test_per_image.csv")) for r in runs]
        res[arm] = {"n_seeds": len(v), "centre_mean": float(np.mean([x[0] for x in v])), "pooled": float(np.mean([x[1] for x in v])),
                    "per_seed_centre_mean": [x[0] for x in v],
                    "per_centre": {c: float(np.mean([x[2][c] for x in v])) for c in C}}
    fa, ch = res["fedavg"]["per_seed_centre_mean"], res["U2_nu2_rho1"]["per_seed_centre_mean"]
    d = np.mean(ch) - np.mean(fa); va, vb = np.var(ch, ddof=1) / 3, np.var(fa, ddof=1) / 3; se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / 2 + vb ** 2 / 2); t = stats.t.ppf(0.95, df)
    res["H4a_registered_metric"] = {"diff": float(d), "ci90": [float(d - t * se), float(d + t * se)],
                                    "equivalent": bool(d - t * se > -0.02 and d + t * se < 0.02)}
    json.dump(res, open(os.path.join(a.out, "utility_centre_mean.json"), "w"), indent=1)
    L = ["| Arm | seeds | centre-mean Dice (registered) | pooled per-image Dice | C1 | C2 | C3 | C4 | C5 |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, q in res.items():
        if arm.startswith("H4"): continue
        L.append(f"| {arm} | {q['n_seeds']} | {q['centre_mean']:.3f} | {q['pooled']:.3f} | " + " | ".join(f"{q['per_centre'][c]:.3f}" for c in C) + " |")
    h = res["H4a_registered_metric"]
    L.append(f"\nH4(a) with the registered metric: diff {h['diff']:+.3f}, 90 % CI [{h['ci90'][0]:+.3f}, {h['ci90'][1]:+.3f}], equivalent: {h['equivalent']}")
    open(os.path.join(a.out, "utility_centre_mean.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
