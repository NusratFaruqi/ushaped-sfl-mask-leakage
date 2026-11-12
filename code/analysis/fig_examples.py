"""Example figure from the 88x88 thumbnails stored by revision_attacks.py --only examples (tag pf).

  python analysis/fig_examples.py RUN_DIR --out paper/figs/fig_examples.pdf --cuts 1 4 --frames 0 1 2

Rows: frames; column groups: cuts. Per cut: reconstructed image x_hat, model prediction on x_hat (b2), Setting A recovered
mask; the true image and reference mask are shown once per frame. Thresholded at 0.5 for masks, Dice in the corner.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt


def dice(a, b):
    a, b = a > 127, b > 127
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--out", required=True)
    ap.add_argument("--cuts", nargs="+", default=["1", "4"]); ap.add_argument("--frames", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    E = {c: json.load(open(os.path.join(a.run, f"revision_{c}_r30_pf.json")))["examples"] for c in a.cuts}
    e0 = E[a.cuts[0]]
    ncol = 2 + 3 * len(a.cuts)
    fig, ax = plt.subplots(len(a.frames), ncol, figsize=(7.16, 1.0 * len(a.frames) + 0.3))
    ax = np.atleast_2d(ax)
    img = lambda t, i: np.transpose(np.asarray(t[i], np.uint8), (1, 2, 0))
    msk = lambda t, i: np.asarray(t[i], np.uint8)[0]
    titles = ["image", "reference"] + sum([[f"$\\hat x$, $U_{c}$", f"$b_2$, $U_{c}$", f"A, $U_{c}$"] for c in a.cuts], [])
    for r, i in enumerate(a.frames):
        y = msk(e0["y"], i)
        panels = [(img(e0["x"], i), None), (y, None)]
        for c in a.cuts:
            e = E[c]
            panels += [(img(e["xhat"], i), None), (msk(e["b2_pred_xhat"], i), dice(msk(e["b2_pred_xhat"], i), y)),
                       (msk(e["A_xhat"], i), dice(msk(e["A_xhat"], i), y))]
        for k, (p, dv) in enumerate(panels):
            A = ax[r, k]
            A.imshow(p if p.ndim == 3 else (p > 127), cmap=None if p.ndim == 3 else "gray", vmin=0, vmax=1 if p.ndim == 2 else None)
            A.set_xticks([]); A.set_yticks([])
            if dv is not None and dv == dv:
                A.text(2, 84, f"{dv:.2f}", color="yellow", fontsize=6, va="bottom")
            if r == 0:
                A.set_title(titles[k], fontsize=6.5)
    plt.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99, top=0.9, bottom=0.01)
    fig.savefig(a.out); print("wrote", a.out)


if __name__ == "__main__":
    main()
