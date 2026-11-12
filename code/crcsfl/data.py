"""Dataset discovery, leakage-free splits and on-the-fly augmentation.

Splits are made BEFORE any augmentation. Frames of CVC-ClinicDB / CVC-ColonDB
come from video sequences and are split by sequence (group) so that frames of
one polyp never appear in both train and test. If a sequence map CSV
(columns: file,group) is not provided, contiguous blocks of frame indices are
used as groups (frames are stored in sequence order in both datasets).
"""
from __future__ import annotations
import csv, json, math, os, random, re
from glob import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
IMG_DIRS = ("images", "image", "original", "img")
MASK_DIRS = ("masks", "mask", "label", "labels", "ground truth", "groundtruth", "gt")
N_SEQ = {"cvc-clinicdb": 31, "cvc-colondb": 15}


def _stem(p):
    return re.sub(r"(_mask|_gt|-mask)$", "", Path(p).stem.lower())


def discover(root: str) -> list[tuple[str, str, str]]:
    """Returns (image, mask, class_or_group) triplets found below `root`."""
    pairs = []
    for d, subdirs, _ in os.walk(root):
        subdirs[:] = [s for s in subdirs if s.lower() != "tif"]     # CVC-ClinicDB ships PNG and TIF copies
        low = {s.lower(): s for s in subdirs}
        for s in subdirs:                                          # BKAI layout: train/train + train_gt/train_gt
            if s.lower() + "_gt" in low:
                im_root, gt_root = os.path.join(d, s), os.path.join(d, low[s.lower() + "_gt"])
                masks = {_stem(m): m for m in glob(os.path.join(gt_root, "**", "*"), recursive=True) if m.lower().endswith(IMG_EXT)}
                for im in sorted(glob(os.path.join(im_root, "**", "*"), recursive=True)):
                    if im.lower().endswith(IMG_EXT) and _stem(im) in masks:
                        pairs.append((im, masks[_stem(im)], s))
        idir = next((low[k] for k in IMG_DIRS if k in low), None)
        mdir = next((low[k] for k in MASK_DIRS if k in low), None)
        if not (idir and mdir):
            continue
        masks = {_stem(m): m for m in glob(os.path.join(d, mdir, "*")) if m.lower().endswith(IMG_EXT)}
        for im in sorted(glob(os.path.join(d, idir, "*"))):
            if im.lower().endswith(IMG_EXT) and _stem(im) in masks:
                pairs.append((im, masks[_stem(im)], Path(d).name))
    return pairs


def polypgen(root: str) -> dict[str, list[tuple[str, str, str]]]:
    """PolypGen single frames grouped by centre (C1..C6). Works with the official
    layout data_C{k}/images_C{k}, data_C{k}/masks_C{k} and close variants."""
    out = {}
    for d, subdirs, _ in os.walk(root):
        for s in subdirs:
            m = re.fullmatch(r"images?_?(C\d)", s, re.I)
            if not m or "seq" in d.lower():
                continue
            cen = m.group(1).upper()
            mdir = next((t for t in subdirs if re.fullmatch(rf"masks?_?{cen}", t, re.I)), None)
            if mdir is None:
                continue
            masks = {_stem(p): p for p in glob(os.path.join(d, mdir, "*")) if p.lower().endswith(IMG_EXT)}
            for im in sorted(glob(os.path.join(d, s, "*"))):
                if im.lower().endswith(IMG_EXT) and _stem(im) in masks:
                    out.setdefault(cen, []).append((im, masks[_stem(im)], cen))
    return dict(sorted(out.items()))


def _frame_index(path):
    m = re.findall(r"\d+", Path(path).stem)
    return int(m[-1]) if m else 0


def make_split(name: str, pairs, seed: int, test_frac=0.2, val_frac=0.1, seq_csv: str | None = None):
    rng = random.Random(seed)
    key = name.lower()
    if key in N_SEQ:  # group split by sequence
        if seq_csv and os.path.exists(seq_csv):
            gmap = {r["file"]: r["group"] for r in csv.DictReader(open(seq_csv))}
            groups = [gmap.get(Path(p[0]).name, Path(p[0]).name) for p in pairs]
        else:
            order = sorted(range(len(pairs)), key=lambda i: _frame_index(pairs[i][0]))
            per = math.ceil(len(pairs) / N_SEQ[key])
            groups = [None] * len(pairs)
            for rank, i in enumerate(order):
                groups[i] = rank // per
        uniq = sorted(set(groups), key=str)
        rng.shuffle(uniq)
        n_test = max(1, round(test_frac * len(uniq)))
        n_val = max(1, round(val_frac * len(uniq)))
        test_g, val_g = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
        split = {"train": [], "val": [], "test": []}
        for p, g in zip(pairs, groups):
            split["test" if g in test_g else "val" if g in val_g else "train"].append(p)
        return split
    # stratified by class folder (EBHI-Seg) or random (Kvasir-SEG)
    by_cls = {}
    for p in pairs:
        by_cls.setdefault(p[2], []).append(p)
    split = {"train": [], "val": [], "test": []}
    for items in by_cls.values():
        items = sorted(items)
        rng.shuffle(items)
        nt, nv = round(test_frac * len(items)), round(val_frac * len(items))
        split["test"] += items[:nt]
        split["val"] += items[nt:nt + nv]
        split["train"] += items[nt + nv:]
    return split


_CACHE: dict = {}          # path -> (uint8 image CxHxW, uint8 mask 1xHxW); shared by all datasets in a process


def _read(im, mk, size):
    x = Image.open(im).convert("RGB")
    x.draft("RGB", (size * 2, size * 2))               # fast JPEG downscale on decode
    x = np.asarray(x.resize((size, size), Image.BILINEAR)).copy()
    # max over RGB so colour-coded masks (e.g. BKAI-IGH red/green classes) become binary
    y = Image.fromarray(np.asarray(Image.open(mk).convert("RGB")).max(2)).resize((size, size), Image.NEAREST)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous(), (torch.from_numpy(np.asarray(y).copy()) > 127).to(torch.uint8)[None]


def preload(pairs, size, threads=8):
    """Decodes and resizes every image once (threaded) and keeps it in RAM as uint8."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [(im, mk) for im, mk, _ in pairs if (im, size) not in _CACHE]
    with ThreadPoolExecutor(threads) as ex:
        for (im, mk), v in zip(todo, ex.map(lambda p: _read(p[0], p[1], size), todo)):
            _CACHE[(im, size)] = v


class SegDataset(Dataset):
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, pairs, size=224, augment=False, cache=True):
        self.pairs, self.size, self.augment = pairs, size, augment

    def __len__(self):
        return len(self.pairs)

    def _load(self, i):
        im, mk, _ = self.pairs[i]
        if (im, self.size) not in _CACHE:
            _CACHE[(im, self.size)] = _read(im, mk, self.size)
        x, y = _CACHE[(im, self.size)]
        return x.float() / 255.0, y.float()

    def __getitem__(self, i):
        x, y = self._load(i)
        if self.augment:                 # CPU path (only used when augment_batch is not applied on GPU)
            x, y = augment(x, y)
        return (x - self.MEAN) / self.STD, y, i


def augment_batch(x, y):
    """GPU augmentation of a normalised batch: same random ops as `augment`, per sample."""
    m, s = SegDataset.MEAN.to(x.device), SegDataset.STD.to(x.device)
    xs, ys = [], []
    for xi, yi in zip(x * s + m, y):
        xi, yi = augment(xi, yi)
        xs.append((xi - m) / s); ys.append(yi)
    return torch.stack(xs), torch.stack(ys)


def augment(x, y):
    if random.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if random.random() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    if random.random() < 0.5:
        k = random.randint(1, 3)
        x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
    if random.random() < 0.5:  # brightness / contrast
        x = ((x - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1)).clamp(0, 1)
    if random.random() < 0.3:  # gaussian blur
        k = torch.tensor([1., 4., 6., 4., 1.], device=x.device); k = (k[:, None] * k[None]) / 256
        x = F.conv2d(x[None], k.expand(3, 1, 5, 5), padding=2, groups=3)[0]
    if random.random() < 0.3:  # additive noise
        x = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
    if random.random() < 0.3:  # grid distortion (smooth random displacement field)
        H = x.shape[-1]
        d = F.interpolate(torch.randn(1, 2, 5, 5, device=x.device) * 0.03, size=(H, H), mode="bicubic", align_corners=True)
        base = torch.stack(torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, H, device=x.device), indexing="xy"), -1)[None]
        grid = base + d.permute(0, 2, 3, 1)
        x = F.grid_sample(x[None], grid, align_corners=True, padding_mode="reflection")[0]
        y = (F.grid_sample(y[None], grid, align_corners=True, mode="nearest")[0] > 0.5).float()
    return x, y


def synthetic_pairs(root: str, n: int = 40, size: int = 96, seed: int = 0):
    """Creates a tiny synthetic dataset (ellipses on noise) for smoke tests."""
    rng = np.random.default_rng(seed)
    os.makedirs(f"{root}/images", exist_ok=True); os.makedirs(f"{root}/masks", exist_ok=True)
    for i in range(n):
        yy, xx = np.mgrid[:size, :size]
        cx, cy, r = rng.integers(20, size - 20, 2).tolist() + [rng.integers(8, 20)]
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
        img = (rng.normal(0.4, 0.1, (size, size, 3)) + 0.4 * m[..., None]).clip(0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(f"{root}/images/{i:04d}.png")
        Image.fromarray((m * 255).astype(np.uint8)).save(f"{root}/masks/{i:04d}.png")
    return discover(root)
