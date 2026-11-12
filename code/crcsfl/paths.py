"""Finds a Kaggle dataset folder whatever the mount layout
(/kaggle/input/<slug>, /kaggle/input/datasets/<owner>/<slug>, ...)."""
import os


def find_input(p: str, base: str = "/kaggle/input") -> str:
    if os.path.isdir(p):
        return p
    slug = os.path.basename(p.rstrip("/"))
    if os.path.isdir(base):
        for d, subdirs, _ in os.walk(base):
            if d.count(os.sep) - base.count(os.sep) > 3:
                subdirs[:] = []
                continue
            if slug in subdirs:
                return os.path.join(d, slug)
    raise FileNotFoundError(f"dataset folder '{slug}' not found under {base}; attached: "
                            f"{sorted(os.listdir(base)) if os.path.isdir(base) else 'none'}")
