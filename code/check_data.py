"""First Kaggle step: confirm every dataset is found and the splits look sane."""
import argparse, json
from crcsfl.data import polypgen, discover
from crcsfl.paths import find_input
p = argparse.ArgumentParser()
p.add_argument("--polypgen", default="/kaggle/input/polypgen20021-video"); p.add_argument("--cvc", default="/kaggle/input/cvcclinicdb")
p.add_argument("--etis", default="/kaggle/input/etis-laribpolypdb"); p.add_argument("--kvasir", default="/kaggle/input/kvasirseg")
p.add_argument("--bkai", default="/kaggle/input/bkai-igh-neopolyp")
a = p.parse_args()
for n in ('polypgen', 'cvc', 'etis', 'kvasir', 'bkai'): setattr(a, n, find_input(getattr(a, n))); print(n, '->', getattr(a, n))
pg = polypgen(a.polypgen)
out = {"polypgen": {k: len(v) for k, v in pg.items()}}
for n in ("cvc", "etis", "kvasir", "bkai"):
    out[n] = len(discover(getattr(a, n)))
print(json.dumps(out, indent=1))
assert set(pg) >= {"C1", "C2", "C3", "C4", "C5", "C6"}, "PolypGen centres not found: check folder names"
expect = {"cvc": 612, "etis": 196, "kvasir": 1000, "bkai": 1000}
assert all(out[n] == v for n, v in expect.items()), f"unexpected counts {out}, expected {expect}"
assert all(len(pg[c]) > 50 for c in ("C1", "C2", "C3", "C4", "C5", "C6")), "a PolypGen centre is nearly empty"
from crcsfl.data import SegDataset
for n, pairs in [("polypgen", pg["C1"]), ("cvc", discover(a.cvc)), ("bkai", discover(a.bkai))]:
    x, y, _ = SegDataset(pairs[:1], 352)[0]
    assert 0.001 < y.mean() < 0.9, f"{n}: mask looks empty/full ({float(y.mean()):.3f})"
from crcsfl.model import build
build("full")          # downloads the pretrained encoder once, before parallel jobs start
print("DATA OK")
