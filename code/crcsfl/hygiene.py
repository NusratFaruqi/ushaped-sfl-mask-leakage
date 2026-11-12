"""Split hygiene: perceptual-hash clustering of near-duplicate frames and
cross-dataset duplicate checks (paper Section 6.1)."""
from __future__ import annotations
import random
import numpy as np
from PIL import Image
from scipy.fft import dctn


def phash(path: str) -> int:
    g = np.asarray(Image.open(path).convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)
    c = dctn(g, norm="ortho")[:8, :8].flatten()
    bits = c > np.median(c[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def cluster(paths: list[str], thr: int = 8) -> list[int]:
    """Union-find on pairs with Hamming distance <= thr. Returns a cluster id per path."""
    h = [phash(p) for p in paths]
    parent = list(range(len(h)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            if hamming(h[i], h[j]) <= thr:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(h))]


def group_split(items: list, groups: list, seed: int, fracs=(0.7, 0.1, 0.2)) -> dict:
    """Assign whole groups to train/val/test so that group sizes approximate fracs."""
    rng = random.Random(seed)
    by = {}
    for it, g in zip(items, groups):
        by.setdefault(g, []).append(it)
    keys = sorted(by, key=str)
    rng.shuffle(keys)
    n, out, acc = len(items), {"train": [], "val": [], "test": []}, 0
    cut1, cut2 = fracs[0] * n, (fracs[0] + fracs[1]) * n
    for k in keys:
        part = "train" if acc < cut1 else "val" if acc < cut2 else "test"
        out[part] += by[k]
        acc += len(by[k])
    return out


def cross_duplicates(a_paths: list[str], b_paths: list[str], thr: int = 8) -> list[tuple[str, str]]:
    ha = [(p, phash(p)) for p in a_paths]
    hb = [(p, phash(p)) for p in b_paths]
    return [(pa, pb) for pa, x in ha for pb, y in hb if hamming(x, y) <= thr]
