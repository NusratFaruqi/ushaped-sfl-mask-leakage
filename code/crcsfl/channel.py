"""Channel applied to tensors that cross the network (paper Section 7.5, Supplement S4).

Per sample:  1) add N(0, s^2), s = sigma * scale  (scale = mean RMS of the tensor on
                 the client's validation images, refreshed once per round -> fixed
                 noise std, so the dFIL bound (Supplement S4) applies)
             2) keep the fraction rho of largest-magnitude NOISY entries (post-processing)
             3) symmetric b-bit quantisation with a per-sample scale (post-processing)
Gradients pass through 2)-3) with a straight-through estimator.

`GradChannel` applies the same steps to the gradient flowing *backwards* through a
tensor (used on dL/dd_{c+1}, which the U-cut client sends to the server).

No formal differential privacy is claimed.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch


def payload_bits(d: int, rho: float, bits: int) -> int:
    """Bits for one sample of a d-dim tensor: values + positions + 32-bit scale."""
    k = d if rho >= 1 else max(1, math.ceil(rho * d))
    val = k * (bits if bits > 0 else 32)
    pos = 0 if k == d else min(d, k * math.ceil(math.log2(d)))  # bitmap or indices
    return val + pos + (32 if bits > 0 else 0)


@dataclass
class Channel:
    sigma: float = 0.0
    rho: float = 1.0
    bits: int = 0            # 0 = fp32 (no quantisation)
    scale: float = 1.0       # set by calibrate()
    bytes_sent: float = 0.0

    @property
    def active(self):
        return self.sigma > 0 or self.rho < 1 or self.bits > 0

    @torch.no_grad()
    def calibrate(self, t: torch.Tensor):
        """t: a batch of the clean tensor; sets scale = mean per-sample RMS."""
        self.scale = float(t.flatten(1).pow(2).mean(1).sqrt().mean())

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        flat = t.flatten(1)
        d = flat.shape[1]
        self.bytes_sent += B * payload_bits(d, self.rho, self.bits) / 8
        if not self.active:
            return t
        if self.sigma > 0:
            s = self.sigma * self.scale
            flat = flat + s * torch.randn_like(flat)
        if self.rho < 1:
            k = max(1, math.ceil(self.rho * d))
            idx = flat.detach().abs().topk(k, dim=1).indices
            mask = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True)
            flat = flat * mask
        if self.bits > 0:
            q = flat.detach().abs().amax(1, keepdim=True).clamp_min(1e-8)
            lv = 2 ** (self.bits - 1) - 1
            quant = torch.round(flat / q * lv) / lv * q
            flat = flat + (quant - flat).detach()
        return flat.view_as(t)

    __call__ = apply


class _GradChan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ch, ids):
        ctx.ch, ctx.ids = ch, ids
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        with torch.no_grad():
            return ctx.ch.apply(g), None, None


def grad_channel(x: torch.Tensor, ch: Channel, ids=None) -> torch.Tensor:
    """Identity forward; applies `ch` to the gradient flowing back through x."""
    return _GradChan.apply(x, ch, ids)
