"""HAM10000 with the owner-released lesion masks (second modality; research/dataset_decision.md).

Official source: Harvard Dataverse doi:10.7910/DVN/DBW86T (CC BY-NC 4.0), fetched with the Dataverse API inside the
Kaggle kernel. Images: Tschandl, Rosendahl & Kittler, Sci Data 2018. Masks ("curated by a single dermatologist"):
Tschandl et al., Nat Med 2020.

Clients are the three Vienna acquisition sources; the fourth source (rosendahl, Queensland: another institution)
is the attacker's public auxiliary data, as Kvasir-SEG is for the polyp federation. Splits are made by lesion_id.
"""
from __future__ import annotations
import csv, os, subprocess, zipfile
from glob import glob

DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/"
FILES = {"img1.zip": 3172585, "img2.zip": 3172584, "masks.zip": 3838943, "meta.csv": "4338392?format=original"}
CLIENTS = {"vidir_molemax": "H1", "vidir_modern": "H2", "vienna_dias": "H3"}
AUX_SOURCE = "rosendahl"


def fetch(root: str = os.environ.get("CRCSFL_HAM_ROOT", "/tmp/ham10000")) -> str:
    """Download and unpack once (idempotent). Kept outside /kaggle/working so it is not saved as output."""
    os.makedirs(root, exist_ok=True)
    for name, fid in FILES.items():
        dst = os.path.join(root, name)
        if not os.path.exists(dst):
            subprocess.run(["curl", "-sSL", "--retry", "5", "-o", dst + ".part", f"{DATAVERSE}{fid}"], check=True)
            os.replace(dst + ".part", dst)
        if name.endswith(".zip") and not os.path.exists(dst + ".done"):
            zipfile.ZipFile(dst).extractall(os.path.join(root, "images" if name.startswith("img") else "masks"))
            open(dst + ".done", "w").close()
    return root


def pairs(root: str):
    """Returns ({client: [(image, mask, lesion_id)]}, aux [(image, mask, lesion_id)])."""
    img = {os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(root, "images", "**", "*.jpg"), recursive=True)}
    msk = {os.path.basename(p).replace("_segmentation.png", ""): p
           for p in glob(os.path.join(root, "masks", "**", "*_segmentation.png"), recursive=True)}
    cl, aux, missing = {}, [], 0
    for r in csv.DictReader(open(os.path.join(root, "meta.csv"))):
        i = r["image_id"]
        if i not in img or i not in msk:
            missing += 1
            continue
        t = (img[i], msk[i], r["lesion_id"])
        if r["dataset"] == AUX_SOURCE:
            aux.append(t)
        elif r["dataset"] in CLIENTS:
            cl.setdefault(CLIENTS[r["dataset"]], []).append(t)
    if missing:
        print(f"HAM10000: {missing} metadata rows without image or mask", flush=True)
    return dict(sorted(cl.items())), aux
