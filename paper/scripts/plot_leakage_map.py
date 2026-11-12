"""Figure 2 of the paper: image and mask leakage against the cut (polyp, FedAvg seed 0, round 30), from the confirmatory test outputs.
  python paper/scripts/plot_leakage_map.py   (run from the project root)
Reads results/image_tests/image_tests.json and results/label_tests_polyp_final/label_tests.json; writes paper/figs/fig_leakage.pdf.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
img = json.load(open(ROOT / "results/image_tests/image_tests.json"))["H1"]
lab = json.load(open(ROOT / "results/label_tests_polyp_final/label_tests.json"))["per_cut"]
# Okabe-Ito, colour-blind safe
C = {"learned": "#0072B2", "optimisation": "#E69F00", "server_only": "#009E73", "xhat": "#0072B2", "oracle": "#CC79A7",
     "server": "#009E73", "b2": "#7f7f7f"}
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.35))

cuts = ["V", "Vh", "1", "2", "3", "4"]; lbl = ["$V$", "$V_h$", "$U_1$", "$U_2$", "$U_3$", "$U_4$"]
for f, name in (("learned", "learned decoder (A)"), ("optimisation", "optimisation + TV (A)"), ("server_only", "server-only surrogate (B)")):
    q = img[f]["per_cut"]; m = [q[c]["mean"] for c in cuts]
    e = [[q[c]["mean"] - q[c]["lo"] for c in cuts], [q[c]["hi"] - q[c]["mean"] for c in cuts]]
    ax[0].errorbar(range(6), m, yerr=e, marker="o", ms=3, lw=1.2, capsize=1.5, color=C[f], label=name)
ax[0].set_xticks(range(6), lbl); ax[0].set_ylabel("image attack PSNR (dB)"); ax[0].set_title("(a) Image leakage", loc="left")
ax[0].axvspan(-0.4, 1.4, color="0.93", zorder=0); ax[0].legend(frameon=False, fontsize=6.5)

mc = ["Vh", "1", "2", "3", "4"]; ml = ["$V_h$", "$U_1$", "$U_2$", "$U_3$", "$U_4$"]
def ci(key, c):
    return lab[c][key]
series = [("xhat", "recovered from $\\hat{x}$ (A)", C["xhat"], "-"), ("oracle", "recovered with true $x$ (A, bound)", C["oracle"], "-"),
          ("server_only", "server-only inverter (B)", C["server"], "-"), ("pred_on_x", "model prediction on true $x$ ($b_2^*$)", C["b2"], "--"),
          ("pred_on_xhat", "model prediction on $\\hat{x}$ ($b_2$)", C["b2"], ":")]
for key, name, col, ls in series:
    xs = [i for i, c in enumerate(mc) if key in lab[c]]; ys = [lab[mc[i]][key] for i in xs]
    ax[1].plot(xs, ys, ls, marker="o" if ls == "-" else None, ms=3, lw=1.2, color=col, label=name)
ax[1].axhline(lab["1"]["prior"], color="0.6", lw=0.8, ls="-.")
ax[1].text(4.0, lab["1"]["prior"] + 0.015, "prior $b_1$", ha="right", va="bottom", fontsize=6, color="0.4")
ax[1].set_xticks(range(5), ml); ax[1].set_ylabel("Dice on polyp frames"); ax[1].set_ylim(0.2, 1.0)
ax[1].set_title("(b) Mask leakage", loc="left")
ax[1].legend(frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 0.11))
fig.tight_layout(); out = ROOT / "paper/figs/fig_leakage.pdf"; fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200)
print("wrote", out)
