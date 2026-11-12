"""Attacks and leakage measures (paper Sections 4 and 5).

Images enter the attacks in [0,1]; `norm` maps them to the network's input space,
so every bound and metric is in pixel units.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import seg_loss

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def norm(x01):
    return (x01 - MEAN.to(x01.device)) / STD.to(x01.device)


def denorm(x):
    return (x * STD.to(x.device) + MEAN.to(x.device)).clamp(0, 1)


# ------------------------------------------------------------------ feature maps seen by the server
class Leak:
    """Wraps a trained model and exposes what crosses the network for a given cut.
    cut 'V' -> all encoder skips z1..z5 (encoder cut); int c -> z_c (U-shaped cut).
    `noisy(x)` adds the channel as configured; `clean(x)` returns noise-free features."""

    def __init__(self, splitnet=None, full=None, cut="V", levels=None):
        self.net, self.full, self.cut = splitnet, full, cut
        self.levels = levels  # for V: subset of levels to expose (None = all)

    def clean(self, x01):
        x = norm(x01)
        if self.cut == "V":
            z = self.full.encoder(x)
            return [z[l - 1] for l in (self.levels or range(1, 6))]
        return [self.net.client.encode(x)[-1]]

    def noisy(self, x01):
        z = self.clean(x01)
        if self.cut == "V" or not self.net.up.active:
            return z
        return [self.net.up(z[0])]


# ------------------------------------------------------------------ learned inversion
class InvDecoder(nn.Module):
    """Decoder from one or more feature maps (any resolutions) back to a [0,1] image."""

    def __init__(self, shapes: list[tuple[int, int]], size: int, w: int = 64):
        super().__init__()
        self.shapes = shapes  # (channels, spatial) per input, any order
        self.size = size
        self.proj = nn.ModuleList(nn.Conv2d(c, w, 1) for c, _ in shapes)
        n_up = int(round(math.log2(size / min(s for _, s in shapes))))
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.ReLU(True),
            nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.ReLU(True)) for _ in range(n_up + 1))
        self.out = nn.Conv2d(w, 3, 1)

    def forward(self, feats):
        res = min(s for _, s in self.shapes)
        h = None
        for blk in self.blocks:
            add = sum(p(f) for p, f, (_, s) in zip(self.proj, feats, self.shapes) if s == res)
            h = add if h is None else h + add if not isinstance(add, int) else h
            h = blk(h)
            if res < self.size:
                h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False); res *= 2
        return torch.sigmoid(self.out(h))


def ssim(a, b, win=11, sigma=1.5, mask=None):
    g = torch.exp(-(torch.arange(win) - win // 2) ** 2 / (2 * sigma ** 2)).to(a.device)
    g = (g[:, None] * g[None]) / g.sum() ** 2
    k = g.expand(a.shape[1], 1, win, win)
    f = lambda t: F.conv2d(t, k, padding=win // 2, groups=a.shape[1])
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    s = ((2 * ma * mb + 1e-4) * (2 * cab + 9e-4)) / ((ma ** 2 + mb ** 2 + 1e-4) * (va + vb + 9e-4))
    s = s.mean(1, keepdim=True)
    if mask is None:
        return s.flatten(1).mean(1)
    return (s * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)


def fov_mask(x01, thr=10 / 255):
    """1 inside the endoscope's field of view (excludes the black border)."""
    return (x01.max(1, keepdim=True).values > thr).float()


def psnr(a, b, mask=None):
    m = mask if mask is not None else torch.ones_like(a[:, :1])
    mse = (((a - b) ** 2) * m).flatten(1).sum(1) / (m.flatten(1).sum(1) * a.shape[1]).clamp_min(1)
    return 10 * torch.log10(1 / mse.clamp_min(1e-10))


def train_inverter(leak: Leak, aux_loader, size, epochs=60, lr=1e-3, dev="cuda", feat_fn=None):
    """feat_fn(x01) -> list of features (defaults to leak.noisy). Returns a trained InvDecoder."""
    feat_fn = feat_fn or leak.noisy
    x0 = next(iter(aux_loader))[0][:1].to(dev)
    with torch.no_grad():
        f0 = feat_fn(denorm(x0))
    G = InvDecoder([(f.shape[1], f.shape[-1]) for f in f0], size).to(dev)
    opt = torch.optim.Adam(G.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    amp = str(dev).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for _ in range(epochs):
        for xb, _, _ in aux_loader:
            x01 = denorm(xb.to(dev))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                f = [t.float() for t in feat_fn(x01)]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                r = G(f)
            r = r.float()
            loss = F.l1_loss(r, x01) + (1 - ssim(r, x01)).mean()
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
    return G.eval()


def optimise_inversion(leak: Leak, obs, x_init, steps=500, lr=0.05, tv=1e-4):
    """Optimisation attack (collusion): match clean features to observed ones, starting from x_init."""
    u = torch.logit(x_init.clamp(1e-3, 1 - 1e-3)).detach().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    masks = [(o != 0).float() for o in obs]               # top-k: compare kept positions only
    for _ in range(steps):
        x = torch.sigmoid(u)
        f = leak.clean(x)
        loss = sum((((fi - oi) * mi) ** 2).sum() / mi.sum().clamp_min(1) for fi, oi, mi in zip(f, obs, masks))
        loss = loss + tv * ((x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())
        opt.zero_grad(); loss.backward(); opt.step()
    return torch.sigmoid(u).detach()


@torch.no_grad()
def multi_round_obs(leak: Leak, x01, R=100):
    """Average of R independently noised observations (worst-case linking attacker)."""
    acc = None
    for _ in range(R):
        f = leak.noisy(x01)
        acc = f if acc is None else [a + b for a, b in zip(acc, f)]
    return [a / R for a in acc]


# ------------------------------------------------------------------ server-only substitute (FORA-style)
def mmd(a, b, scales=(1.0, 4.0, 16.0)):
    a = F.adaptive_avg_pool2d(a, 8).flatten(1); b = F.adaptive_avg_pool2d(b, 8).flatten(1)
    x = torch.cat([a, b]); d = torch.cdist(x, x) ** 2
    k = sum(torch.exp(-d / (s * d.detach().mean())) for s in scales)
    n = a.shape[0]
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


def train_substitute(sub_encoder, observed_feats, aux_loader, steps=3000, lr=1e-3, dev="cuda"):
    """Fit a substitute client encoder whose features match the distribution of the observed
    features (list of tensors recorded by the server). Returns the trained encoder."""
    opt = torch.optim.Adam(sub_encoder.parameters(), lr=lr)
    obs = torch.cat(observed_feats).to(dev)
    it = iter(aux_loader)
    for s in range(steps):
        try:
            xb = next(it)[0]
        except StopIteration:
            it = iter(aux_loader); xb = next(it)[0]
        f = sub_encoder(xb.to(dev))
        idx = torch.randint(0, obs.shape[0], (f.shape[0],), device=dev)
        loss = mmd(f, obs[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return sub_encoder.eval()


# ------------------------------------------------------------------ mask recovery (Section 5.2)
def returned_gradient(net, x01, y, d=None):
    """What a U-shaped client sends back: g = dL/dd_{c+1} for a batch (loss = training loss),
    with the client's gradient channel applied. Also returns the d_{c+1} the server sent."""
    x = norm(x01)
    skips = net.encode(x)
    if d is None:
        with torch.no_grad():
            d = net.serve(skips)
    d = d.detach().requires_grad_(True)
    loss = seg_loss(net.decode(d, skips), y)
    g, = torch.autograd.grad(loss, d)
    if net.gch.active:
        g = net.gch.apply(g)
    return g.detach(), d.detach()


def recover_masks(net, x01_hat, d, g, steps=300, lr=0.1, tv=1e-3):
    """Gradient matching (paper Eq. 2): find y in [0,1]^P whose gradient at d matches the observed g."""
    x = norm(x01_hat)
    with torch.no_grad():
        skips = [s.detach() for s in net.encode(x)]
        o0 = net.decode(d, skips)
    v = o0.detach().clone().requires_grad_(True)          # y = sigmoid(v), init = model prediction
    opt = torch.optim.Adam([v], lr=lr)
    dd = d.detach().requires_grad_(True)
    for _ in range(steps):
        y = torch.sigmoid(v)
        loss = seg_loss(net.decode(dd, skips), y)
        gy, = torch.autograd.grad(loss, dd, create_graph=True)
        obj = ((gy - g) ** 2).sum() / (g ** 2).sum().clamp_min(1e-12)
        obj = obj + tv * ((y[..., 1:, :] - y[..., :-1, :]).abs().mean() + (y[..., :, 1:] - y[..., :, :-1]).abs().mean())
        opt.zero_grad(); obj.backward(); opt.step()
    return torch.sigmoid(v).detach()


def jacobian_rank(net, x01, d, n_probe=None, tol=1e-6):
    """Numerical rank of J_h = d logits / d d_{c+1} for ONE image (small inputs only; uses the full Jacobian
    through random sketching when P is large)."""
    x = norm(x01[:1])
    skips = [s.detach() for s in net.encode(x)]
    f = lambda dd: net.decode(dd, skips).flatten()
    P = f(d[:1]).numel()
    k = n_probe or min(P, 2048)
    Q = torch.randn(P, k, device=d.device) / math.sqrt(k)
    # sketch J^T Q via VJPs; rank(J) >= rank(J^T Q) with high probability when k <= rank
    cols = []
    for i in range(k):
        _, vjp = torch.func.vjp(f, d[:1])
        cols.append(vjp(Q[:, i])[0].flatten())
    S = torch.linalg.svdvals(torch.stack(cols, 1))
    return int((S > tol * S[0]).sum()), k


# ------------------------------------------------------------------ dFIL (Supplement S4)
def dfil(net, x01, y, s_z, s_g=None, probes=64):
    """Per-image dFIL = [tr(Jz^T Jz)/s_z^2 + tr(Jg^T Jg | z)/s_g^2] / d_x with x in [0,1] and the
    trace restricted to field-of-view pixels. The g term is I(g~ | z~): d_{c+1} is computed from
    the (detached) feature, so g depends on x only through the client's skips.
    Hutchinson with Rademacher probes in the output space; returns a tensor (B,)."""
    B = x01.shape[0]
    fov = fov_mask(x01)                                   # (B,1,H,W)
    d_x = (fov.flatten(1).sum(1) * x01.shape[1]).clamp_min(1)
    out = torch.zeros(B, device=x01.device)
    for _ in range(probes):
        x = x01.detach().requires_grad_(True)
        skips = net.client.encode(norm(x))
        z = skips[-1]
        v = torch.randint_like(z, 2) * 2 - 1
        gx, = torch.autograd.grad((z * v).sum(), x, retain_graph=s_g is not None)
        tz = (gx * fov).flatten(1).pow(2).sum(1) / s_z ** 2
        out += tz
        if s_g is not None:
            with torch.no_grad():
                dsv = net.server(z.detach())
            d = dsv.detach().requires_grad_(True)
            loss = seg_loss(net.client.decode(d, skips), y)
            g, = torch.autograd.grad(loss, d, create_graph=True)
            w = torch.randint_like(g, 2) * 2 - 1
            gx2, = torch.autograd.grad((g * w).sum(), x)
            out += (gx2 * fov).flatten(1).pow(2).sum(1) / s_g ** 2
    return out / probes / d_x


# ------------------------------------------------------------------ FedAvg update inversion (Geiping et al.)
def invert_fedavg_gradient(model, x_shape, y, true_grads, steps=500, lr=0.1, tv=1e-4, x_init=None):
    """Reconstruct a batch from one local gradient (labels y assumed known: worst case)."""
    dev = y.device
    u = (torch.logit(x_init.clamp(1e-3, 1 - 1e-3)) if x_init is not None
         else torch.zeros(x_shape, device=dev)).detach().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    params = [p for p in model.parameters() if p.requires_grad]
    for _ in range(steps):
        x = torch.sigmoid(u)
        loss = seg_loss(model(norm(x)), y)
        gs = torch.autograd.grad(loss, params, create_graph=True)
        num = sum((a * b).sum() for a, b in zip(gs, true_grads))
        den = torch.sqrt(sum((a ** 2).sum() for a in gs)) * torch.sqrt(sum((b ** 2).sum() for b in true_grads))
        obj = 1 - num / den + tv * ((x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())
        opt.zero_grad(); obj.backward(); opt.step()
    return torch.sigmoid(u).detach()


def invert_fedavg_update(model, delta, x_shape, y, steps_local, batch, lr_local, betas=(0.9, 0.999),
                         wd=1e-4, iters=300, lr=0.05, tv=1e-4):
    """FedAvg-aware attack (Dimitrov et al., TMLR 2022) adapted to AdamW with state reset each round:
    simulate `steps_local` AdamW steps on unknown images (labels y known, worst case) and match the
    observed weight delta by cosine similarity. x_shape = (steps_local * batch, 3, H, W)."""
    from torch.func import functional_call
    names = [n for n, p in model.named_parameters()]
    p0 = {n: p.detach().clone().requires_grad_(True) for n, p in model.named_parameters()}
    bufs = {n: b for n, b in model.named_buffers()}
    u = torch.zeros(x_shape, device=y.device, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    target = torch.cat([delta[n].flatten() for n in names])
    for _ in range(iters):
        x = torch.sigmoid(u)
        p = dict(p0); m = {n: torch.zeros_like(v) for n, v in p0.items()}; v2 = {n: torch.zeros_like(v) for n, v in p0.items()}
        for t in range(1, steps_local + 1):
            sl = slice((t - 1) * batch, t * batch)
            loss = seg_loss(functional_call(model, ({**p, **bufs},), (norm(x[sl]),)), y[sl])
            g = torch.autograd.grad(loss, [p[n] for n in names], create_graph=True)
            newp = {}
            for n, gi in zip(names, g):
                m[n] = betas[0] * m[n] + (1 - betas[0]) * gi
                v2[n] = betas[1] * v2[n] + (1 - betas[1]) * gi * gi
                mh, vh = m[n] / (1 - betas[0] ** t), v2[n] / (1 - betas[1] ** t)
                newp[n] = p[n] * (1 - lr_local * wd) - lr_local * mh / (vh.sqrt() + 1e-8)
            p = newp
        est = torch.cat([(p[n] - p0[n].detach()).flatten() for n in names])
        obj = 1 - F.cosine_similarity(est, target, dim=0)
        obj = obj + tv * ((x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())
        opt.zero_grad(); obj.backward(); opt.step()
    return torch.sigmoid(u).detach()
