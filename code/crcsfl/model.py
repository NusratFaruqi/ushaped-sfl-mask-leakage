"""Attention U-Net and its two split variants.

full    : the whole network (centralized / FedAvg-family baselines)
vanilla : client = encoder E1..E5, sends z1..z5 (+ masks); server = rest + loss
          (SplitFed with a U-Net, as in most prior SFL segmentation work)
skip    : client = encoder + D1 + head; z2..z5 go up, d2 comes down
ucut    : horizontal U-cut at depth c (RoS-FL/DC-SFL topology generalised):
          client = E1..Ec + Dc..D1 + head; only z_c goes up, d_{c+1} comes down.

GroupNorm is used by default so that each sample's representation does not
depend on the other samples of the batch (needed for per-sample noise and
for small, non-IID client batches).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTHS = (64, 128, 256, 512, 1024)
RESNET_WIDTHS = (64, 64, 128, 256, 512)       # ResNet-34 stages: /2, /4, /8, /16, /32
ARCH = {"name": "resnet34"}                    # set_arch("unet") for the original from-scratch network


def set_arch(name: str):
    assert name in ("resnet34", "unet"), name
    ARCH["name"] = name


def resnet_levels():
    """ImageNet-pretrained ResNet-34 as five levels. BatchNorm is frozen (pretrained statistics, no
    batch dependence), so each sample is processed independently, as the per-sample channel needs."""
    import os, torchvision
    from torchvision.ops.misc import FrozenBatchNorm2d
    w = None if os.environ.get("CRCSFL_NO_PRETRAIN") else torchvision.models.ResNet34_Weights.IMAGENET1K_V1
    r = torchvision.models.resnet34(weights=w, norm_layer=FrozenBatchNorm2d)
    return [nn.Sequential(r.conv1, r.bn1, r.relu), nn.Sequential(r.maxpool, r.layer1), r.layer2, r.layer3, r.layer4]


def make_head(c: int) -> nn.Module:
    if ARCH["name"] == "resnet34":   # level 1 is at half resolution
        return nn.Sequential(nn.Conv2d(c, 1, 1), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
    return nn.Conv2d(c, 1, 1)


def norm_layer(c: int, kind: str) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(c)
    return nn.GroupNorm(min(8, c), c)


class DoubleConv(nn.Module):
    def __init__(self, cin, cout, norm="gn"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), norm_layer(cout, norm), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), norm_layer(cout, norm), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Schlemper et al., MedIA 2019)."""

    def __init__(self, c, norm="gn"):
        super().__init__()
        ci = max(c // 2, 8)
        self.wg = nn.Sequential(nn.Conv2d(c, ci, 1), norm_layer(ci, norm))
        self.wx = nn.Sequential(nn.Conv2d(c, ci, 1), norm_layer(ci, norm))
        self.psi = nn.Conv2d(ci, 1, 1)

    def forward(self, g, x):
        return x * torch.sigmoid(self.psi(F.relu(self.wg(g) + self.wx(x))))


class Encoder(nn.Module):
    def __init__(self, widths, norm="gn"):
        super().__init__()
        if ARCH["name"] == "resnet34":
            self.levels = nn.ModuleList(resnet_levels())
            return
        self.levels = nn.ModuleList()
        c = 3
        for i, w in enumerate(widths):
            blk = DoubleConv(c, w, norm)
            self.levels.append(blk if i == 0 else nn.Sequential(nn.MaxPool2d(2), blk))
            c = w

    def forward(self, x):
        out = []
        for lvl in self.levels:
            x = lvl(x)
            out.append(x)
        return out  # z1..z5


class UpBlock(nn.Module):
    def __init__(self, cin, cout, attention=True, norm="gn"):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.gate = AttentionGate(cout, norm) if attention else None
        self.conv = DoubleConv(2 * cout, cout, norm)

    def forward(self, x, skip):
        g = self.up(x)
        s = self.gate(g, skip) if self.gate is not None else skip
        return self.conv(torch.cat([s, g], 1))


class Decoder(nn.Module):
    """Bottleneck + up-blocks for levels 4..`stop`; optional output head."""

    def __init__(self, widths, attention=True, norm="gn", stop=1, head=True):
        super().__init__()
        self.bottleneck = DoubleConv(widths[-1], widths[-1], norm)
        self.blocks = nn.ModuleList()
        self.stop = stop
        for lvl in range(4, stop - 1, -1):  # level index 4..stop (1-based widths)
            self.blocks.append(UpBlock(widths[lvl], widths[lvl - 1], attention, norm))
        self.head = make_head(widths[0]) if head else None

    def forward(self, feats):
        # feats: dict level->tensor, must contain 5 and the skips needed
        x = self.bottleneck(feats[5])
        for blk, lvl in zip(self.blocks, range(4, self.stop - 1, -1)):
            x = blk(x, feats[lvl])
        return self.head(x) if self.head is not None else x


class ClientTail(nn.Module):
    """Last decoder level (uses z1 locally) + output head, kept on the client."""

    def __init__(self, widths, attention=True, norm="gn"):
        super().__init__()
        self.block = UpBlock(widths[1], widths[0], attention, norm)
        self.head = make_head(widths[0])

    def forward(self, d2, z1):
        return self.head(self.block(d2, z1))


class ClientModel(nn.Module):
    def __init__(self, widths, split, attention=True, norm="gn"):
        super().__init__()
        self.split = split
        self.encoder = Encoder(widths, norm)
        self.tail = ClientTail(widths, attention, norm) if split == "skip" else None


class UClient(nn.Module):
    """Horizontal U-cut at depth c: encoder levels 1..c, decoder levels c..1, head.
    Sends z_c up; receives the server feature d_{c+1} (at level-(c+1) resolution)."""

    def __init__(self, widths, c, attention=True, norm="gn"):
        super().__init__()
        self.c = c
        full = Encoder(widths, norm)
        self.enc = nn.ModuleList(full.levels[:c])
        self.ups = nn.ModuleList(UpBlock(widths[l], widths[l - 1], attention, norm) for l in range(c, 0, -1))
        self.head = make_head(widths[0])

    def encode(self, x):
        skips = []
        for lvl in self.enc:
            x = lvl(x)
            skips.append(x)
        return skips  # z1..zc ; zc is transmitted

    def decode(self, d, skips):
        for blk, z in zip(self.ups, reversed(skips)):
            d = blk(d, z)
        return self.head(d)


class UServer(nn.Module):
    """Server part of the U-cut: encoder levels c+1..5, bottleneck, decoder levels 4..c+1."""

    def __init__(self, widths, c, attention=True, norm="gn"):
        super().__init__()
        self.c = c
        full = Encoder(widths, norm)
        self.enc = nn.ModuleList(full.levels[c:])
        self.bottleneck = DoubleConv(widths[-1], widths[-1], norm)
        self.ups = nn.ModuleList(UpBlock(widths[l], widths[l - 1], attention, norm) for l in range(4, c, -1))

    def forward(self, zc):
        skips, x = [], zc
        for lvl in self.enc:
            x = lvl(x)
            skips.append(x)
        x = self.bottleneck(skips[-1])
        for blk, z in zip(self.ups, reversed(skips[:-1])):
            x = blk(x, z)
        return x  # d_{c+1}


def build(split: str, width: float = 0.5, attention=True, norm="gn", c: int = 2):
    """split: full | vanilla (encoder cut, all skips cross) | ucut (horizontal cut at depth c)."""
    widths = RESNET_WIDTHS if ARCH["name"] == "resnet34" else tuple(max(8, int(w * width)) for w in WIDTHS)
    if split == "full":
        return FullUNet(widths, attention, norm), None
    if split == "ucut":
        return UClient(widths, c, attention, norm), UServer(widths, c, attention, norm)
    client = ClientModel(widths, split, attention, norm)
    server = Decoder(widths, attention, norm, stop=1 if split == "vanilla" else 2, head=(split == "vanilla"))
    return client, server


class FullUNet(nn.Module):
    def __init__(self, widths, attention=True, norm="gn"):
        super().__init__()
        self.encoder = Encoder(widths, norm)
        self.decoder = Decoder(widths, attention, norm, stop=1, head=True)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder({i + 1: t for i, t in enumerate(z)})


def n_params(m):
    return 0 if m is None else sum(p.numel() for p in m.parameters())


@torch.no_grad()
def count_macs(fn, modules, x):
    """MACs of conv layers executed inside `modules` while running fn(x)."""
    total = [0]

    def hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        total[0] += out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * k

    hs = [m.register_forward_hook(hook) for mod in modules for m in mod.modules()
          if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    fn(x)
    for h in hs:
        h.remove()
    return total[0]
