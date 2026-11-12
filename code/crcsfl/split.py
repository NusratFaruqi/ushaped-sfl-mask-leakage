"""Single-process simulation of SFL-V1.

By Proposition 1 (paper), SFL-V1 without a channel equals FedAvg. With a channel,
SFL-V1 equals FedAvg on the *composite* network client -> channel -> server ->
grad-channel -> client, because each client has its own server copy and both
parts are averaged with the same weights. So we simulate a U-cut as one module
and train it with the FedAvg loop in train.py. Tensors that would cross the
network are exposed for attacks via `forward(..., keep=True)`.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .model import build
from .channel import Channel, grad_channel


class SplitNet(nn.Module):
    def __init__(self, cut: int, width=0.5, up: Channel | None = None, down_grad: Channel | None = None,
                 server: Channel | None = None):
        super().__init__()
        self.cut = cut
        self.client, self.server = build("ucut", width, c=cut)
        self.up = up or Channel()            # on z_c (client -> server)
        self.gch = down_grad or Channel()    # on dL/dd_{c+1} (client -> server)
        self.srv = server or Channel()       # compression only, on d_{c+1} and dL/dz_c (server -> client)
        self.srv_g = Channel(0.0, self.srv.rho, self.srv.bits)
        self.last = {}

    def client_params(self):
        return list(self.client.parameters())

    def forward(self, x, ids=None, keep=False):
        skips = self.client.encode(x)
        zc = self.up(skips[-1])
        if self.training and self.srv_g.active:
            zc = grad_channel(zc, self.srv_g)          # compress dL/dz_c sent back to the client
        d = self.server(zc)
        if self.srv.active:
            d = self.srv(d)                             # compress d_{c+1} sent to the client
        if self.training and self.gch.active:
            d = grad_channel(d, self.gch, ids)
        if keep:
            if d.requires_grad:
                d.retain_grad()
            self.last = {"z": zc, "d": d, "skips": skips}
        return self.client.decode(d, skips)

    @torch.no_grad()
    def calibrate(self, x, y):
        """Set channel noise scales from a clean validation batch (once per round)."""
        was = self.training
        self.eval()
        skips = self.client.encode(x)
        self.up.calibrate(skips[-1])
        self.train(was)
        if self.gch.active:
            with torch.enable_grad():
                skips = self.client.encode(x)
                d = self.server(skips[-1]).detach().requires_grad_(True)
                out = self.client.decode(d, skips)
                from .losses import seg_loss
                g, = torch.autograd.grad(seg_loss(out, y), d)
            self.gch.calibrate(g)   # same batch size as training -> same gradient scale


def from_full(full_state: dict, cut: int) -> dict:
    """Map a FullUNet state dict onto SplitNet(cut) parameter names (client/server parts)."""
    out = {}
    for k, v in full_state.items():
        if k.startswith("encoder.levels."):
            i, rest = k[len("encoder.levels."):].split(".", 1); i = int(i)
            out[f"client.enc.{i}.{rest}" if i < cut else f"server.enc.{i - cut}.{rest}"] = v
        elif k.startswith("decoder.bottleneck"):
            out["server." + k[len("decoder."):]] = v
        elif k.startswith("decoder.blocks."):
            j, rest = k[len("decoder.blocks."):].split(".", 1); j = int(j)   # j=0 is level 4
            lvl = 4 - j
            out[f"server.ups.{j}.{rest}" if lvl > cut else f"client.ups.{cut - lvl}.{rest}"] = v
        elif k.startswith("decoder.head"):
            out["client.head" + k[len("decoder.head"):]] = v
    return out


class VhNet(nn.Module):
    """Encoder cut with the head at the client (V_h): client = encoder + D1 + head,
    sends z2..z5, receives d2, returns dL/dd2. Same interface as SplitNet for attacks."""

    def __init__(self, width=0.5):
        super().__init__()
        from .model import build as _b
        self.client, self.server_dec = _b("skip", width)
        self.up = Channel(); self.gch = Channel(); self.srv = Channel()

    @staticmethod
    def from_full_state(full_state, width=0.5):
        net = VhNet(width)
        st = {}
        for k, v in full_state.items():
            if k.startswith("encoder."):
                st["client." + k] = v
            elif k.startswith("decoder.blocks.3."):
                st["client.tail.block." + k[len("decoder.blocks.3."):]] = v
            elif k.startswith("decoder.head"):
                st["client.tail.head" + k[len("decoder.head"):]] = v
            elif k.startswith("decoder."):
                st["server_dec." + k[len("decoder."):]] = v
        net.load_state_dict(st)
        return net

    def encode(self, x):
        return self.client.encoder(x)

    def serve(self, skips):
        return self.server_dec({i + 1: t for i, t in enumerate(skips)})

    def decode(self, d, skips):
        return self.client.tail(d, skips[0])

    def forward(self, x):
        z = self.encode(x)
        return self.decode(self.serve(z), z)


def _serve_ucut(self, skips):
    return self.server(self.up(skips[-1]))


SplitNet.serve = _serve_ucut
SplitNet.encode = lambda self, x: self.client.encode(x)
SplitNet.decode = lambda self, d, skips: self.client.decode(d, skips)
