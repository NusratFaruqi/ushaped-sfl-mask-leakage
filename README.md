# Mask leakage in U-shaped split federated learning

Code for the paper

> N. Faruqi, M. S. R. Joy, et al. *Labels Do Not Stay Local: Mask Leakage in U-Shaped Split Federated Learning for Medical Image Segmentation.* Manuscript submitted for publication.

In U-shaped split federated learning a hospital keeps the first and last layers of a segmentation network, so its images and masks never leave the hospital. The server only sees an intermediate feature, and the hospital returns the gradient of the loss with respect to the server's output. We show that for the cross-entropy loss this returned gradient is an affine function of a smoothed copy of each image's mask, and we measure how much an honest-but-curious server can recover from it. The experiments use PolypGen (five colonoscopy centres) and HAM10000 (dermoscopy) with an ImageNet-pretrained ResNet-34 U-Net, cut at six points.

The result files, data splits and pre-registration documents are archived separately as a data record on Zenodo (doi: [DATA DOI]). Every number in the paper's tables and figures can be traced to a file there and to the script in this repository that produced it (see [From results to the paper](#from-results-to-the-paper)).

## Contents

```
code/
  crcsfl/              library: model and cuts, split training, channel, attacks, data, metrics
    model.py           ResNet-34 U-Net with attention-gated skips; full network and cuts V, V_h, U_1..U_4
    split.py           split network (U-shaped cut with uplink/downlink channels), V_h network, weight mapping
    channel.py         noise, top-k sparsification and int8 quantisation of tensors that cross the network; byte counts
    attacks.py         image reconstruction (learned decoder, optimisation), multi-round attacker, server-only surrogate,
                       mask recovery from the returned gradient, dFIL, FedAvg gradient/update inversion
    label_leak.py      mask-recovery attacks of Settings A and B and their baselines
    hygiene.py         perceptual-hash grouping of near-duplicate frames, group-wise splits, cross-dataset checks
    data.py, ham.py    PolypGen/CVC/ETIS/Kvasir/BKAI loaders; HAM10000 download and splits
    metrics.py, losses.py
  train.py             centralised, local, FedAvg, FedProx, FedGN and DP-SGD training, with any cut and channel
  run_attacks.py       all registered leakage measurements on a saved checkpoint
  revision_attacks.py  follow-up attack experiments (exploratory, not registered)
  check_data.py        checks that every dataset is found and has the expected size
  analysis/            scripts that turn result files into the statistics reported in the paper
  kaggle/              plan.py (every run, grouped in tiers) and make_kernel.py (builds a Kaggle notebook per tier)
paper/
  scripts/             cost_table.py (communication and compute cost), plot_leakage_map.py (Figure 2)
  tables/              output folder of cost_table.py
  figs/                output folder of the figure scripts
tests/                 checks that run without data or GPU (registered-script hashes, smoke runs)
SHA256SUMS             checksum of every file in this repository
```

## Installation

Python 3.10 or later. A CUDA GPU is needed for the full experiments; the smoke tests below run on a CPU in seconds.

```bash
git clone https://github.com/NusratFaruqi/ushaped-sfl-mask-leakage.git
cd ushaped-sfl-mask-leakage
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The experiments in the paper ran on Kaggle GPU notebooks (2 x NVIDIA T4) with Kaggle's preinstalled PyTorch, plus `opacus` and `lpips` installed by each notebook. The smoke tests were also checked locally with PyTorch 2.15, torchvision 0.29, NumPy 2.5 and SciPy 1.18. The ImageNet weights of ResNet-34 are downloaded by torchvision on first use.

## Data

No images are included; obtain them from their owners under their licences.

| Dataset | Role | Source | Kaggle mirror used |
|---|---|---|---|
| PolypGen (centres C1–C6) | clients C1–C5; C6 held-out test | Ali et al., *Sci. Data* 2023 | `gokulraj324/polypgen20021-video` |
| CVC-ClinicDB | external test | Bernal et al., *CMIG* 2015 | `balraj98/cvcclinicdb` |
| ETIS-Larib | external test | Silva et al., *IJCARS* 2014 | `nguyenvoquocduong/etis-laribpolypdb` |
| Kvasir-SEG | attacker's auxiliary data | Jha et al., *MMM* 2020 | `debeshjha1/kvasirseg` |
| BKAI-IGH NeoPolyp | auxiliary-data ablation | Lan et al., *ISVC* 2021 | `dtruon46/bkai-igh-neopolyp` |
| HAM10000 + lesion masks | second domain (clients H1–H3, auxiliary = fourth source) | Tschandl et al., *Sci. Data* 2018; masks: Tschandl et al., *Nat. Med.* 2020 | downloaded by `crcsfl/ham.py` from Harvard Dataverse, doi:10.7910/DVN/DBW86T |

The loaders look for each dataset by folder name under `/kaggle/input`; elsewhere, pass the folders explicitly (`--polypgen`, `--cvc`, `--etis`, `--kvasir`, `--bkai`; for HAM10000 set `CRCSFL_HAM_ROOT`). `python code/check_data.py` reports what was found.

**Splits.** The data record contains `splits/polypgen_splits_seed0.json` and `splits/ham10000_splits_seed0.json`, which list, for every client, the training, validation and test files (and the auxiliary set) used in all runs. They store the Kaggle paths of the mirrors above; the file names are those of the original releases. `train.py --prepare-splits` rebuilds them from the data (perceptual-hash grouping, split seed 0).

## Quick check (CPU, synthetic data)

The test suite checks that the two confirmatory analysis scripts match their registered SHA-256 hashes and that training and the attacks run end to end on synthetic data. It needs no data or GPU and takes about a minute on a CPU:

```bash
pip install pytest
python -m pytest tests
```

The same smoke runs by hand (they test the code paths, not a result):

```bash
cd code
python train.py --smoke --rounds 1 --out /tmp/smoke --width 0.25 --save-rounds 1
python run_attacks.py --smoke --run /tmp/smoke/fedavg_seed0 --cut 2 --round 1
python train.py --smoke --rounds 1 --cut 2 --sigma 1 --rho 0.1 --bits 8 --gsigma 1 --out /tmp/smoke --width 0.25 --save-rounds 1
```

## Reproducing the experiments

All runs are listed in `code/kaggle/plan.py`, grouped in tiers: 0 and A–H hold the main experiments, R1–R6 the exploratory follow-ups. Which analyses are confirmatory is stated in the registration record. `python code/kaggle/plan.py <tier>` prints the commands of a tier. Typical commands:

```bash
cd code
python train.py --prepare-splits                                   # splits and perceptual-hash cache
python train.py --method fedavg --seed 0 --rounds 30 --save-rounds 1,30
python run_attacks.py --run runs/fedavg_seed0 --cut 1 --round 30   # U_1: image and mask leakage
python run_attacks.py --run runs/fedavg_seed0 --cut Vh --round 30  # V_h
python train.py --method fedavg --cut 2 --sigma 1 --rho 1 --bits 8 --gsigma 1 --grho 1 --gbits 8 --sbits 8 --rounds 30
```

To run a tier on Kaggle as we did:

```bash
python code/kaggle/make_kernel.py <tier> <kaggle-username>        # writes code/kaggle/build/tier<tier>/
kaggle kernels push -p code/kaggle/build/tier<tier>
```

`make_kernel.py` embeds the code in the notebook and attaches the datasets listed above.

## From results to the paper

| Paper item | Script | Result files |
|---|---|---|
| Table 1 (registered hypotheses): H1, H2, H4 | `code/analysis/image_tests.py` | `results/image_tests/` |
| Table 1: H3′, H5, H6 (polyps) | `code/analysis/label_tests.py` | `results/label_tests_polyp_final/`, `results/label_tests_seed1/`, `results/label_tests_seeds01/` |
| Table 1: H7a–c (HAM10000) | `code/analysis/label_tests.py` | `results/label_tests_H/` |
| Table 2 (utility) | `code/analysis/utility_table.py`, `code/analysis/utility_centre_mean.py` | `results/utility_table.csv`, `results/utility_centre_mean/` |
| Table 3 (mask recovery per cut) | `label_tests.py`, `code/analysis/revision_reanalysis.py`, `code/analysis/r1_unified_gain.py` | `results/label_tests_polyp_final/`, `results/revision_reanalysis/`, `results/revision_r1/unified_gain.json` |
| Table 4 (leakage per cut) | as Tables 1 and 3, `code/analysis/r1_round2_extras.py` (presence AUROC), `paper/scripts/cost_table.py` | as above, `results/revision_r1/round2_extras.json`, `paper/tables/cost.json` |
| Figure 1 (cuts) | drawn in TikZ | — |
| Figure 2 (leakage against the cut) | `paper/scripts/plot_leakage_map.py` | `results/image_tests/`, `results/label_tests_polyp_final/` |
| Figure 3 (example masks) | `code/analysis/fig_examples.py` | `results/tierR1/` |
| Supplement: cost (Table S6) | `paper/scripts/cost_table.py` | `paper/tables/cost.tex` |
| Supplement: channel runs and leakage (S4, Table S7) | `label_tests.py`, `image_tests.py` | `results/label_tests_G2/`, `results/tierC*`, `results/tierE/` |
| Supplement: exploratory controls (Table S8) | `code/analysis/r1_analysis.py` | `results/revision_r1/` |
| Section 7.4 (further training seeds), Section 7.6 (gradient-only noise), Supplement Tables S9 and S10 | `code/analysis/r456_analysis.py` | `results/revision_r456/`, `results/tierR3/`–`results/tierR6/` |

To use the stored results, download the data record (doi: [DATA DOI]) and extract it into the root of this repository; it adds `results/`, `splits/`, `registration/` and `paper/tables/`, which the paths below refer to. Each `results/tier*/` folder holds the small output files of one Kaggle notebook: per-image test metrics (`test_per_image.csv`), attack outputs (`attacks_*.json`, `revision_*.json`), per-command logs and the notebook log. Model checkpoints are not included. Kaggle account names in the logs are replaced by `<kaggle-user>`.

The analysis scripts take the result folders as arguments, for example:

```bash
python code/analysis/image_tests.py results --out /tmp/image_tests
python paper/scripts/plot_leakage_map.py
```

## Pre-registration

The data record's `registration/` folder holds the registered documents in the order they were written: the design (v3), the methodology blueprint (v4), amendments v5–v7 and the pilot decision, with the SHA-256 hashes recorded when each file was written (`*.sha256`). `code/analysis/label_tests.py` and `code/analysis/image_tests.py` in this repository are byte-identical to the hashed versions. `registration/registration_record.pdf` lists every document, its timing relative to the results it governs, and every deviation of the executed protocol from the design. Confirmatory and exploratory analyses are labelled as such in the code, the result files and the paper.

Check the hashes (after extracting the data record) with

```bash
shasum -a 256 code/analysis/label_tests.py code/analysis/image_tests.py registration/preregistration_amendment_v*.md
shasum -a 256 -c SHA256SUMS
```

## Citation

If you use this code, please cite the paper (reference above; it will be updated on publication) and this software release (see `CITATION.cff`). If you use the results or splits, please also cite the data record (doi: [DATA DOI]).

## Licence

Code: MIT (`LICENSE`). The data record (results, splits, registration documents): CC BY 4.0. The datasets are not covered by these licences; they remain under their owners' terms (HAM10000: CC BY-NC 4.0).
