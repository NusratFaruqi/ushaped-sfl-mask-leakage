"""Per-image segmentation metrics (Dice, IoU, precision, recall, HD95)."""
from __future__ import annotations
import numpy as np
import torch
from scipy import ndimage


def _nsd(p: np.ndarray, g: np.ndarray, tol: float) -> float:
    """Normalized surface distance: share of boundary points within `tol` pixels of the other boundary."""
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    pb = p ^ ndimage.binary_erosion(p); gb = g ^ ndimage.binary_erosion(g)
    dg = ndimage.distance_transform_edt(~gb); dp = ndimage.distance_transform_edt(~pb)
    return float(((dg[pb] <= tol).sum() + (dp[gb] <= tol).sum()) / (pb.sum() + gb.sum()))


def _hd95(p: np.ndarray, g: np.ndarray) -> float:
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float(np.hypot(*p.shape))  # worst case: image diagonal
    pb = p ^ ndimage.binary_erosion(p)
    gb = g ^ ndimage.binary_erosion(g)
    dg = ndimage.distance_transform_edt(~gb)
    dp = ndimage.distance_transform_edt(~pb)
    d = np.concatenate([dg[pb], dp[gb]])
    return float(np.percentile(d, 95))


@torch.no_grad()
def batch_metrics(logits: torch.Tensor, y: torch.Tensor, hd: bool = True, masks_orig=None) -> list[dict]:
    """masks_orig: optional list of original-resolution boolean masks; logits are then resized to
    each mask's size before scoring (metrics at native resolution, NSD tolerance = 1% of width)."""
    import torch.nn.functional as F
    if masks_orig is not None:
        p = [(F.interpolate(l[None], size=mo.shape, mode="bilinear", align_corners=False)[0, 0] > 0).cpu().numpy()
             for l, mo in zip(logits, masks_orig)]
        g = list(masks_orig)
    else:
        p = list((logits.sigmoid() > 0.5).cpu().numpy().astype(bool)[:, 0])
        g = list((y > 0.5).cpu().numpy().astype(bool)[:, 0])
    out = []
    for pi, gi in zip(p, g):
        tp = (pi & gi).sum(); fp = (pi & ~gi).sum(); fn = (~pi & gi).sum()
        eps = 1e-7
        empty = tp + fp + fn == 0
        out.append({
            "dice": 1.0 if empty else 2 * tp / (2 * tp + fp + fn + eps),
            "iou": 1.0 if empty else tp / (tp + fp + fn + eps),
            "precision": 1.0 if tp + fp == 0 and fn == 0 else tp / (tp + fp + eps),
            "recall": 1.0 if tp + fn == 0 else tp / (tp + fn + eps),
            "hd95": _hd95(pi, gi) if hd else float("nan"),
            "nsd": _nsd(pi, gi, 0.01 * gi.shape[1]) if hd else float("nan"),
            "area": float(gi.mean()),
        })
    return out


def summarize(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
