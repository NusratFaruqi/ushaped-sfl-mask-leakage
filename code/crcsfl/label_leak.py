"""Mask leakage in U-shaped SFL (reframe v5, pre-registration amendment v5: H5b, H6).

A U-shaped client at cut c returns g = dL/dd_{c+1} = J_c^T dL/dz for every image (per-sample: the loss is
BCE + per-image soft Dice, so g_i depends on y_i only). For BCE, dL/dz is affine in the mask y, so the server
observes a linear measurement of y through the client decoder's Jacobian.

1. label_fisher: attacker-independent bound. G = dg/dy (exact, including the Dice term, by double backprop).
   Scale-free index  L_c = tr(G^T G) / (HW * rms(g)^2).  If the client adds Gaussian noise of std nu*rms(g)
   to g (our gradient channel), the per-pixel Fisher information about y is L_c / nu^2 and, by Cramer-Rao,
   any unbiased attacker has per-pixel MSE >= nu^2 / L_c.  Analogue of dFIL (Maeng et al.) for labels.
2. Server-only attack (Setting B): the server knows its own (averaged) server part, builds a surrogate client
   on public auxiliary data, simulates (g, d, y) through surrogate client + real server, trains a
   gradient-to-mask inverter, and applies it to the real clients' returned gradients.
"""
from __future__ import annotations
import copy, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attacks import norm, InvDecoder, returned_gradient
from .losses import seg_loss


# ------------------------------------------------------------------ 1. Fisher bound
def label_fisher(net, x01, y, probes: int = 16):
    """Returns (L_c per image (B,), rms(g) per image (B,)). net: SplitNet in eval mode."""
    x = norm(x01)
    with torch.no_grad():
        skips = [s.detach() for s in net.encode(x)]
        d = net.serve(skips).detach()
    HW = y[0].numel()
    tr = torch.zeros(x.shape[0], device=x.device)
    g = None
    for _ in range(probes):
        yy = y.detach().float().clone().requires_grad_(True)
        dd = d.clone().requires_grad_(True)
        g, = torch.autograd.grad(seg_loss(net.decode(dd, skips), yy), dd, create_graph=True)
        w = torch.randint_like(g, 2) * 2 - 1                      # Rademacher probe in gradient space
        gy, = torch.autograd.grad((g * w).sum(), yy)              # G^T w
        tr += gy.flatten(1).pow(2).sum(1)                         # E ||G^T w||^2 = tr(G G^T)
    rms = g.detach().flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-30)
    return tr / probes / (HW * rms ** 2), rms


# ------------------------------------------------------------------ 2. server-only attack
class MaskInverter(nn.Module):
    """(g, d) at the cut -> mask probability at full resolution. g is RMS-normalised per image."""

    def __init__(self, ch: int, spatial: int, size: int, w: int = 64, use: str = "both"):
        super().__init__()
        self.use = use                       # "both" (paper), "g" (gradient only) or "d" (server feature only): revision ablation
        self.dec = InvDecoder([(2 * ch, spatial)], size, w)
        self.dec.out = nn.Conv2d(w, 1, 1)

    def forward(self, g, d):
        gn = g / g.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-30).view(-1, 1, 1, 1)
        dn = d / d.flatten(1).pow(2).mean(1).sqrt().clamp_min(1e-30).view(-1, 1, 1, 1)
        if self.use == "g":
            dn = torch.zeros_like(dn)
        elif self.use == "d":
            gn = torch.zeros_like(gn)
        return self.dec([torch.cat([gn, dn], 1)])


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def surrogate_client(net, aux_loader, steps: int, dev, random_init: bool = False, lr: float = 1e-4):
    """Copy of the split net whose CLIENT part is re-initialised (public ImageNet weights, or random) and
    trained on auxiliary data with the REAL server part frozen: all a server-only adversary can do."""
    from .split import SplitNet
    if random_init:
        os.environ["CRCSFL_NO_PRETRAIN"] = "1"
    sur = SplitNet(net.cut, up=copy.deepcopy(net.up), down_grad=copy.deepcopy(net.gch), server=copy.deepcopy(net.srv)) \
        if hasattr(net, "cut") else None
    os.environ.pop("CRCSFL_NO_PRETRAIN", None)
    sur.server.load_state_dict(net.server.state_dict())
    sur = sur.to(dev)
    for p_ in sur.server.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW([p_ for p_ in sur.client.parameters()], lr=lr, weight_decay=1e-4)
    it = _cycle(aux_loader)
    sur.train()
    for _ in range(steps):
        x, y, _ = next(it)
        loss = seg_loss(sur(x.to(dev)), y.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sur.eval()
    return sur


def train_mask_inverter(sur, aux_loader, size: int, steps: int, dev, lr: float = 1e-3, use: str = "both"):
    """Simulate returned gradients on auxiliary data through the surrogate and learn g, d -> y."""
    from .attacks import denorm
    it = _cycle(aux_loader)
    x, y, _ = next(it)
    g, d = returned_gradient(sur, denorm(x.to(dev)), y.to(dev))
    inv = MaskInverter(g.shape[1], g.shape[-1], size, use=use).to(dev)
    opt = torch.optim.Adam(inv.parameters(), lr=lr)
    if sur.gch.active or sur.up.active:                        # same relative noise as the real protocol
        sur.calibrate(x.to(dev), y.to(dev)); sur.eval()
    for _ in range(steps):
        x, y, _ = next(it)
        x01, y = denorm(x.to(dev)), y.to(dev)
        g, d = returned_gradient(sur, x01, y)
        p = inv(g, d).clamp(1e-6, 1 - 1e-6)
        pf, yf = p.flatten(1), y.flatten(1)
        loss = F.binary_cross_entropy(p, y) + (1 - (2 * (pf * yf).sum(1) + 1) / (pf.sum(1) + yf.sum(1) + 1)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    inv.eval()
    return inv


def prior_mask(aux_masks: torch.Tensor) -> torch.Tensor:
    """Baseline b1: mean auxiliary mask, thresholded so its area equals the mean auxiliary mask area."""
    m = aux_masks.float().mean(0, keepdim=True)                   # (1,1,H,W)
    area = aux_masks.float().mean()
    thr = torch.quantile(m.flatten(), (1 - area).clamp(0, 1).item())
    return (m >= thr).float()


def dice_per_image(yh, y):
    """Dice of binary masks; returns (dice (B,), reference area (B,)). Empty-reference images must be excluded."""
    yh, y = (yh > 0.5).float().flatten(1), (y > 0.5).float().flatten(1)
    return 2 * (yh * y).sum(1) / (yh.sum(1) + y.sum(1)).clamp_min(1), y.mean(1)
