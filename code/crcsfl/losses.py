import torch
import torch.nn.functional as F


def seg_loss(logits, y, eps=1.0):
    """BCE + soft Dice (per image, then mean)."""
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = logits.sigmoid().flatten(1); t = y.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return bce + dice.mean()


def dcor(x, z):
    """Sample distance correlation (NoPeek). x, z: (B, ...)."""
    def cent(a):
        d = torch.cdist(a, a)
        return d - d.mean(0, keepdim=True) - d.mean(1, keepdim=True) + d.mean()
    A, B = cent(x.flatten(1)), cent(z.flatten(1))
    dxy = (A * B).mean(); dxx = (A * A).mean(); dyy = (B * B).mean()
    return dxy.clamp_min(0).sqrt() / (dxx.sqrt() * dyy.sqrt()).clamp_min(1e-9).sqrt()
