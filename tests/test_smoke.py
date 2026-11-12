"""Quick checks that need no data and no GPU (about one minute on a CPU).

    python -m pytest tests

1. The two confirmatory analysis scripts are byte-identical to the versions hashed in the registration
   documents (registration/label_tests.sha256 and image_tests.sha256 in the data record).
2. Training and the attacks run end to end on synthetic data (--smoke). This checks the code paths only;
   the numbers are meaningless.
"""
import glob
import hashlib
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")

REGISTERED = {
    "analysis/label_tests.py": "60fc61cb66992b2085b37d01d0c7a87ed93d157179a33dacb9bb565d376af11f",
    "analysis/image_tests.py": "0452e3a89b996b485e45bd5e3927c0ea8191aaed4e38df14ad170ab401dcff56",
}


@pytest.mark.parametrize("rel,sha", sorted(REGISTERED.items()))
def test_registered_scripts_unchanged(rel, sha):
    with open(os.path.join(CODE, rel), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == sha


def run(args, cwd=CODE):
    r = subprocess.run([sys.executable, "-W", "ignore"] + args, cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


@pytest.fixture(scope="module")
def fedavg_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    run(["train.py", "--smoke", "--rounds", "1", "--width", "0.25", "--save-rounds", "1", "--out", str(out)])
    return out / "fedavg_seed0"


def test_train_fedavg(fedavg_run):
    assert fedavg_run.is_dir() and any(fedavg_run.iterdir())


def test_train_ushaped_cut_with_channel(tmp_path):
    run(["train.py", "--smoke", "--rounds", "1", "--cut", "2", "--sigma", "1", "--rho", "0.1", "--bits", "8",
         "--gsigma", "1", "--width", "0.25", "--save-rounds", "1", "--out", str(tmp_path)])
    assert glob.glob(str(tmp_path / "fedavg_U2_*_seed0"))


def test_attacks_at_u2(fedavg_run):
    run(["run_attacks.py", "--smoke", "--run", str(fedavg_run), "--cut", "2", "--round", "1"])
    assert glob.glob(str(fedavg_run / "attacks_2_r1_*.json"))
