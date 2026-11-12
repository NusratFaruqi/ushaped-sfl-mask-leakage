"""Tiered run plan. `python kaggle/plan.py TIER` prints the commands of one tier; the Kaggle
script runs them in order and copies results to /kaggle/working. Budget per tier on a T4 (estimate)."""
import sys
SEEDS5, SEEDS3 = range(5), range(3)
GRID = [(nu, rho) for nu in (0.5, 1, 2, 4) for rho in (1.0, 0.1)]
ROUNDS = 30                        # bench2: FedAvg val Dice plateaus by round ~20 (0.73)
RR = f"--rounds {ROUNDS} --save-rounds 1,{ROUNDS}"
R = "/kaggle/working/runs_dry"
Q = f"--rounds 2 --save-rounds 1,2 --out {R}"
U2 = f"$(ls -d {R}/fedavg_U2_s1.0r0.1b8*_seed0 | head -1)"
T = {
 # dry run: every code path on the real data, 2 rounds, tiny attack budgets (~30-45 min)
 "dry": ["python check_data.py", "python train.py --equiv-check"]
      + [f"python train.py --method {m} {Q}" for m in ("fedavg", "central", "fedprox", "fedgn")]
      + [f"python train.py --method local --client C1 {Q}", f"python train.py --method dpsgd --eps 8 {Q}",
         f"python train.py --method fedavg --compress {Q}",
         f"python train.py --method fedavg --cut 2 --sigma 1 --rho 0.1 --bits 8 --gsigma 1 --grho 0.1 --gbits 8 --srho 0.1 --sbits 8 {Q}",
         f"python train.py --method fedavg --cut 2 --nopeek 0.1 {Q}"]
      + [f"python run_attacks.py --smoke --run {R}/fedavg_seed0 --cut {c} --round 2" for c in ("V", "Vh", 1, 2, 3, 4)]
      + [f"python run_attacks.py --smoke --run {R}/fedavg_seed0 --fedavg --round 2",
         f"python run_attacks.py --smoke --run {R}/fedavg_seed0 --cut 2 --round 2 --aux bkai",
         f"python run_attacks.py --smoke --run {U2} --cut 2 --round 2"],
 "0": ["python check_data.py", "python train.py --equiv-check", "python train.py --prepare-splits"]
      + [f"python train.py --method fedavg --seed {s} --rounds 100 --no-amp" for s in (0, 1)],      # fp32 pilot (ran 2026-11-11)
 # speed benchmark: AMP vs fp32 per round, and one full-budget attack (~1 h)
 "bench": ["python check_data.py", "python train.py --prepare-splits"]
      + ["python train.py --method fedavg --rounds 3 --save-rounds 3 --out /kaggle/working/runs_bench",
         "python train.py --method fedavg --rounds 3 --save-rounds 3 --no-amp --out /kaggle/working/runs_fp32",
         "python run_attacks.py --run /kaggle/working/runs_bench/fedavg_seed0 --cut 2 --round 3"],
 # pretrained ResNet-34 U-Net: convergence + speed (FedAvg and central, 30 rounds) and one full attack
 "bench2": ["python check_data.py", "python train.py --prepare-splits",
            "python train.py --method fedavg --rounds 30 --save-rounds 1,30",
            "python train.py --method central --rounds 30 --save-rounds 30",
            "while [ ! -f /kaggle/working/runs/fedavg_seed0/test_per_image.csv ]; do sleep 30; done; "
            "python run_attacks.py --run /kaggle/working/runs/fedavg_seed0 --cut 2 --round 30"],
 # ---- lean main plan; ROUNDS chosen from the tier-0 validation curve ----
 # A: utility. Every channel-free cut equals FedAvg (Remark 1), so FedAvg seeds cover V, V_h, U1-U4.
 "A": ["python check_data.py", "python train.py --prepare-splits"]
      + [f"python train.py --method fedavg --seed {s} {RR}" for s in SEEDS3]
      + [f"python train.py --method central --seed 0 {RR}", f"python train.py --method fedprox --seed 0 {RR}",
         f"python train.py --method fedavg --compress --seed 0 {RR}"]
      + [f"python train.py --method local --client C{k} --seed 0 {RR}" for k in range(1, 6)],
 # B: leakage of every cut on the FedAvg checkpoint (needs A)
 "B": ["python check_data.py"]
      + [f"python run_attacks.py --run /kaggle/working/runs/fedavg_seed0 --cut {c} --round {ROUNDS}" for c in ("V", "Vh", 1, 2, 3, 4)]
      + [f"python run_attacks.py --run /kaggle/working/runs/fedavg_seed0 --cut {c} --round 1" for c in ("V", 2)]
      + [f"python run_attacks.py --run /kaggle/working/runs/fedavg_seed0 --fedavg --round {ROUNDS}",
         f"python run_attacks.py --run /kaggle/working/runs/fedavg_seed0 --cut 2 --round {ROUNDS} --aux bkai"],
 # C1: channel calibration grid at U2 only, one seed (independent of A)
 "C1": ["python check_data.py", "python train.py --prepare-splits"]
      + [f"python train.py --method fedavg --cut 2 --sigma {nu} --rho {rho} --bits 8 --gsigma {nu} --grho {rho} --gbits 8 --srho {rho} --sbits 8 --seed 0 {RR}"
         for nu, rho in GRID],
 # D2: DP-SGD on all 29M weights did not learn (val Dice 0.07); standard DP transfer recipe:
 # public pretrained encoder frozen, decoder trained privately, batch 16; optimizer chosen on validation
 "D2": ["python check_data.py", "python train.py --prepare-splits",
        f"python train.py --method dpsgd --eps 8 --dp-freeze-encoder --dp-opt adamw --batch 16 --lr 3e-4 --seed 0 {RR}",
        f"python train.py --method dpsgd --eps 8 --dp-freeze-encoder --dp-opt sgd --batch 16 --lr 0.05 --seed 0 {RR}"],
 # C1d: dFIL (selection rule) at the round-1 checkpoint of every grid point; round 1 has the same lr in both schedules
 "C1d": None,
 # C1 split over two accounts
 "C1a": None, "C1b": None,
 # C2 (written after C1): chosen setting at U1 and U3 (seed 0) and U2 (seeds 1, 2), plus attacks on U1-U3
 # D: defences at U2 and DP-SGD (independent of A)
 "D": ["python check_data.py", "python train.py --prepare-splits",
       f"python train.py --method fedavg --cut 2 --nopeek 0.1 --seed 0 {RR}",
       f"python train.py --method dpsgd --eps 8 --seed 0 {RR}"],
}


def est_h(c):
    """GPU hours per command from the dry run on a T4 (100 rounds ~ 80 s/round for 5 clients)."""
    if "run_attacks" in c:
        return 0.3 if "--fedavg" in c else 1.5
    if "--method local" in c:
        return 0.7
    if "--method dpsgd" in c:
        return 2.8
    return 2.3 if "--cut" in c else 2.2


def _dfil_cmd(nu, rho):
    ch = f"--sigma {nu} --rho {rho} --bits 8 --gsigma {nu} --grho {rho} --gbits 8 --srho {rho} --sbits 8"
    name = f"fedavg_U2_s{float(nu)}r{float(rho)}b8_g{float(nu)}r{float(rho)}b8_srv{float(rho)}b8_seed0"
    return (f"python train.py --method fedavg --cut 2 {ch} --seed 0 --rounds 1 --save-rounds 1 --out /kaggle/working/runs && "
            f"python run_attacks.py --run /kaggle/working/runs/{name} --cut 2 --round 1 --only dfil")
_pre = ["python check_data.py", "python train.py --prepare-splits"]
T["C1da"] = _pre + [_dfil_cmd(nu, rho) for nu, rho in GRID[:4]]
T["C1db"] = _pre + [_dfil_cmd(nu, rho) for nu, rho in GRID[4:]]
_grid = T["C1"][2:]
T["C1a"] = T["C1"][:2] + _grid[:4]
T["C1b"] = T["C1"][:2] + _grid[4:]
# which account runs what (B must share A's account: it reads A's checkpoints)
ACCOUNTS = {"A": "kaggle1", "B": "kaggle1", "C1a": "kaggle2", "C1b": "kaggle3", "D": "kaggle4"}

def c2(nu, rho, nu_use=1.0):
    """Chosen channel (dFIL rule) at U1-U3, its sparsified/unsparsified twin at U2, a usable-accuracy point
    (nu_use, rho=1) at U2, and NoPeek at U2, each attacked; dFIL over rounds 1/10/20/30 for the chosen U2 run."""
    W = "/kaggle/working/runs"
    ch = lambda n, r: f"--sigma {n} --rho {r} --bits 8 --gsigma {n} --grho {r} --gbits 8 --srho {r} --sbits 8"
    name = lambda c, n, r, s: f"fedavg_U{c}_s{float(n)}r{float(r)}b8_g{float(n)}r{float(r)}b8_srv{float(r)}b8_seed{s}"
    keep = f"--rounds {ROUNDS} --save-rounds 1,10,20,{ROUNDS}"
    alt = 0.1 if rho >= 1 else 1.0
    runs = [f"python train.py --method fedavg --cut 2 {ch(nu, rho)} --seed {s} {keep}" for s in (0, 1, 2)]
    runs += [f"python train.py --method fedavg --cut {c} {ch(nu, rho)} --seed 0 {RR}" for c in (1, 3)]
    runs += [f"python train.py --method fedavg --cut 2 {ch(nu, alt)} --seed 0 {RR}",
             f"python train.py --method fedavg --cut 2 {ch(nu_use, 1.0)} --seed 0 {RR}",
             f"python train.py --method fedavg --cut 2 --nopeek 0.1 --seed 0 {RR}"]
    wait = lambda d: f"while [ ! -f {W}/{d}/test_per_image.csv ]; do sleep 30; done; "
    att = [wait(name(2, nu, rho, 0)) + f"python run_attacks.py --run {W}/{name(2, nu, rho, 0)} --cut 2 --round {r}" + ("" if r == ROUNDS else " --only dfil")
           for r in (ROUNDS, 1, 10, 20)]
    att += [wait(name(c, nu, rho, 0)) + f"python run_attacks.py --run {W}/{name(c, nu, rho, 0)} --cut {c} --round {ROUNDS}" for c in (1, 3)]
    att += [wait(name(2, nu, alt, 0)) + f"python run_attacks.py --run {W}/{name(2, nu, alt, 0)} --cut 2 --round {ROUNDS}",
            wait(name(2, nu_use, 1.0, 0)) + f"python run_attacks.py --run {W}/{name(2, nu_use, 1.0, 0)} --cut 2 --round {ROUNDS}",
            wait("fedavg_U2_nopeek0.1_seed0") + f"python run_attacks.py --run {W}/fedavg_U2_nopeek0.1_seed0 --cut 2 --round {ROUNDS}"]
    return ["python check_data.py", "python train.py --prepare-splits"] + runs + att


T["C2"] = c2(2, 1.0)        # dFIL rule (beta = 100): nu = 2, rho = 1 (val Dice 0.392, dFIL p95 52.5)
# A2: centralized and local training overflowed in fp16 (no averaging every ~30 steps); rerun in fp32
T["A2"] = ["python check_data.py", "python train.py --prepare-splits",
           f"python train.py --method central --seed 0 --no-amp {RR}"] + [f"python train.py --method local --client C{k} --seed 0 --no-amp {RR}" for k in range(1, 6)]
_pre = ["python check_data.py", "python train.py --prepare-splits"]
T["C1da"] = _pre + [_dfil_cmd(nu, rho) for nu, rho in GRID[:4]]
T["C1db"] = _pre + [_dfil_cmd(nu, rho) for nu, rho in GRID[4:]]
_grid = T["C1"][2:]
T["C1a"] = T["C1"][:2] + _grid[:4]
T["C1b"] = T["C1"][:2] + _grid[4:]
# which account runs what (B must share A's account: it reads A's checkpoints)
ACCOUNTS = {"A": "kaggle1", "B": "kaggle1", "C1a": "kaggle2", "C1b": "kaggle3", "D": "kaggle4"}

def c2(nu, rho):
    """Chosen channel (from C1's dFIL rule) plus the U2 defence rows of the channel table (Supplement Table S7), with their attacks."""
    W = "/kaggle/working/runs"
    ch = lambda r: f"--sigma {nu} --rho {r} --bits 8 --gsigma {nu} --grho {r} --gbits 8 --srho {r} --sbits 8"
    name = lambda c, r, s: f"fedavg_U{c}_s{float(nu)}r{float(r)}b8_g{float(nu)}r{float(r)}b8" + (f"_srv{float(r)}b8" if r < 1 else "_srv1.0b8") + f"_seed{s}"
    keep = f"--rounds {ROUNDS} --save-rounds 1,10,20,{ROUNDS}"
    runs = [f"python train.py --method fedavg --cut 2 {ch(rho)} --seed {s} {keep}" for s in (0, 1, 2)]
    runs += [f"python train.py --method fedavg --cut {c} {ch(rho)} --seed 0 {RR}" for c in (1, 3)]
    runs += [f"python train.py --method fedavg --cut 2 {ch(1.0)} --seed 0 {RR}",          # noise only (rho = 1)
             f"python train.py --method fedavg --cut 2 --nopeek 0.1 --seed 0 {RR}"]
    wait = lambda d: f"while [ ! -f {W}/{d}/test_per_image.csv ]; do sleep 30; done; "
    att = [wait(name(2, rho, 0)) + f"python run_attacks.py --run {W}/{name(2, rho, 0)} --cut 2 --round {r}" + ("" if r == ROUNDS else " --only dfil")
           for r in (ROUNDS, 1, 10, 20)]
    att += [wait(name(c, rho, 0)) + f"python run_attacks.py --run {W}/{name(c, rho, 0)} --cut {c} --round {ROUNDS}" for c in (1, 3)]
    att += [wait(name(2, 1.0, 0)) + f"python run_attacks.py --run {W}/{name(2, 1.0, 0)} --cut 2 --round {ROUNDS}",
            wait("fedavg_U2_nopeek0.1_seed0") + f"python run_attacks.py --run {W}/fedavg_U2_nopeek0.1_seed0 --cut 2 --round {ROUNDS}"]
    return ["python check_data.py", "python train.py --prepare-splits"] + runs + att


# ---- audit fixes (research/audit_2026-11-11.md) ----
# E: local baselines with step-matched epochs (local-C4 collapsed at 1 epoch/round), 3 seeds, fp32 like A2
T["E"] = _pre + [f"python train.py --method local --client C{k} --local-epochs match --seed {s} --no-amp {RR}"
                 for s in SEEDS3 for k in range(1, 6)]
# F: extra seeds for single-seed utility rows, and a second FedAvg seed for the headline cut attacks
_W = "/kaggle/working/runs"
T["F"] = _pre + [f"python train.py --method fedavg --seed 1 {RR}"] \
    + [f"python train.py --method central --seed {s} --no-amp {RR}" for s in (1, 2)] \
    + [f"python train.py --method fedprox --seed {s} {RR}" for s in (1, 2)] \
    + [f"python train.py --method fedavg --compress --seed {s} {RR}" for s in (1, 2)] \
    + [f"while [ ! -f {_W}/fedavg_seed1/test_per_image.csv ]; do sleep 30; done; "
       f"python run_attacks.py --run {_W}/fedavg_seed1 --cut {c} --round {ROUNDS}" for c in ("V", 1, 2)]

# G1 (account of A/B; kernel source = tier A): audit reruns on the FedAvg seed-0 model at round 30
_A = "/kaggle/working/runs/fedavg_seed0"
T["G1"] = _pre + [f"python run_attacks.py --run {_A} --cut 2 --round {ROUNDS} --aux bkai"] \
    + [f"python run_attacks.py --run {_A} --cut {c} --round {ROUNDS} --only server --surrogate-init random" for c in ("V", 1, 2)] \
    + [f"python run_attacks.py --run {_A} --cut {c} --round {ROUNDS} --only server,label,labelfisher,labelserver --label-batches 19" for c in ("Vh", 1, 2, 3, 4)] \
# G3 (account1 after G1; source = tier A): server-only mask attack with a randomly initialised surrogate client
# longest jobs first (G1 log: label jobs take 107-173 min and grow with depth), so the two GPUs finish together
T["G3"] = _pre + [f"python run_attacks.py --run {_A} --cut {c} --round {ROUNDS} --only label --label-batches 19" for c in (4, 3, 2, 1, "Vh")] \
    + [f"python run_attacks.py --run {_A} --cut {c} --round {ROUNDS} --only server,labelserver --surrogate-init random --label-batches 19" for c in (1, 2)]   # label: adds b2* = prediction on x (v7)
# G2 (account of C2; kernel source = tier C2): multi-round attack with R = rounds actually run, larger label attack under the channel
_ch = lambda c, n, r: f"/kaggle/working/runs/fedavg_U{c}_s{float(n)}r{float(r)}b8_g{float(n)}r{float(r)}b8_srv{float(r)}b8_seed0"
_g2 = [(2, 2, 1.0), (1, 2, 1.0), (3, 2, 1.0), (2, 2, 0.1), (2, 1.0, 1.0)]
T["G2"] = _pre + [f"python run_attacks.py --run {_ch(c, n, r)} --cut {c} --round {ROUNDS} --only multi" for c, n, r in _g2] \
    + [f"python run_attacks.py --run {_ch(2, n, r)} --cut 2 --round {ROUNDS} --only server,label,labelfisher,labelserver --label-batches 19" for n, r in ((2, 1.0), (1.0, 1.0))] \
    + [f"python run_attacks.py --run {_ch(c, 2, 1.0)} --cut {c} --round {ROUNDS} --only server,label,labelfisher,labelserver --label-batches 19" for c in (1, 3)]

# H (second modality, HAM10000; research/dataset_decision.md): clients = 3 Vienna sources, aux = rosendahl, lesion-level splits.
# 6 rounds of ~5.4k training images = about the same number of optimisation steps as 30 rounds of PolypGen (~1.0k images).
HR, _H = 6, "/kaggle/working/runs_ham"
_hall = "learned,opt,multi,server,label,dfil,labelfisher,labelserver"
T["H"] = ["python check_data.py", f"python train.py --prepare-splits --dataset ham --out {_H}"] \
    + [f"python train.py --dataset ham --method fedavg --seed {s} --rounds {HR} --save-rounds 1,{HR} --out {_H}" for s in SEEDS3] \
    + [f"while [ ! -f {_H}/fedavg_seed0/test_per_image.csv ]; do sleep 30; done; "
       f"python run_attacks.py --run {_H}/fedavg_seed0 --cut {c} --round {HR} --label-batches 19 --only " + ("learned,opt,multi,server,dfil" if c == "V" else _hall)
       for c in ("V", "Vh", 1, 2, 3, 4)]

# ---- follow-up experiments R1-R3 (2026-11-12); exploratory, not registered ----
# R1 (account4; source = tier A): controls on the FedAvg seed-0 model: presence, d/g ablation, Fisher on all frames,
# training-frame gradients in a local epoch, post-hoc gradient noise, longer-trained image decoder
_rv = lambda c, o, t, x="": f"python revision_attacks.py --run {_A} --cut {c} --round {ROUNDS} --only {o} --tag {t} {x}".strip()
T["R1"] = _pre + [_rv(2, "noise", "nz", "--noise-batches 4")] \
    + [_rv(c, "train", "tr") for c in (4, 1)] \
    + [_rv(c, "presence,fisher,examples", "pf") for c in (4, 3, 2, 1)] + [_rv("Vh", "presence,examples", "pf")] \
    + [_rv(c, "ablation", "ab") for c in (4, 3, 2, 1)] \
    + [_rv(c, "strong", "st", "--label-batches 1") for c in (4, 3)]
# R2 (account3; source = tier F): the registered seed-1 mask attacks (amendment v5 "where available"), frozen code
T["R2"] = _pre + [f"python run_attacks.py --run /kaggle/working/runs/fedavg_seed1 --cut {c} --round {ROUNDS} --only server,label,labelfisher,labelserver --label-batches 19"
                  for c in (4, 3, 2, 1, "Vh")]
# R3 (account2; independent): defence aimed at the mechanism: noise on the returned gradient only, then its attacks
_gn = lambda n: f"/kaggle/working/runs/fedavg_U2_g{float(n)}r1.0b0_seed0"
T["R3"] = _pre + [f"python train.py --method fedavg --cut 2 --sigma 0 --rho 1 --bits 0 --gsigma {n} --grho 1 --gbits 0 --srho 1 --sbits 0 --seed 0 {RR}" for n in (1, 2, 4)] \
    + [f"while [ ! -f {_gn(n)}/test_per_image.csv ]; do sleep 30; done; python revision_attacks.py --run {_gn(n)} --cut 2 --round {ROUNDS} --only presence,ablation --tag gn" for n in (1, 2, 4)]

# R1b / R1c (account1, owner of tier A; 2026-11-12 18:50): backup of the slowest R1 commands in case R1 hits the 12 h cap.
# Same model, code and tags as R1; outputs go to results/tierR1b, results/tierR1c and are used only for files R1 lacks.
T["R1b"] = _pre + [_rv(c, "ablation", "ab") for c in (1, 3)]
T["R1c"] = _pre + [_rv(4, "ablation", "ab")] + [_rv(c, "strong", "st", "--label-batches 1") for c in (4, 3)]
# ---- follow-up experiments R4-R5 (2026-11-12); exploratory, not registered ----
# R4 (account3; source = tier F, FedAvg seed 1): server-only inverter repeated over 2 surrogate/inverter seeds with a
# shuffled-gradient control at every cut; replayed training-frame gradients at round 1 (early training, high lr).
_S1 = "/kaggle/working/runs/fedavg_seed1"
_rvr = lambda run, c, o, t, x="", r=ROUNDS: f"python revision_attacks.py --run {run} --cut {c} --round {r} --only {o} --tag {t} {x}".strip()
T["R4"] = _pre + [_rvr(_S1, c, "ablation", "rs", "--inv-seeds 2") for c in (1, 2, 3, 4)] + [_rvr(_S1, 1, "train", "tr1", r=1)]
# R5 (account2; independent): a third FedAvg seed trained here, then presence (with b1) and the 2-seed ablation per cut,
# and the round-1 training-frame replay at U4.
_S2 = "/kaggle/working/runs/fedavg_seed2"
_w2 = f"while [ ! -f {_S2}/test_per_image.csv ]; do sleep 30; done; "
T["R5"] = _pre + [f"python train.py --method fedavg --seed 2 {RR}"] \
    + [_w2 + _rvr(_S2, c, "presence,ablation", "rs", "--inv-seeds 2") for c in (1, 2, 3, 4)] + [_w2 + _rvr(_S2, 4, "train", "tr1", r=1)]

# ---- follow-up experiment R6 (2026-11-12); exploratory, not registered ----
# R6 (account4; independent): the gradient-noise defence over two more training seeds (nu_g = 2, 4 at U2), with
# presence, the server-only ablation retrained on noisy gradients, and an adaptive attacker that averages K = 1, 10, 30
# independently noised gradients of each image at fixed weights (favours the attacker).
_gns = lambda n, sd: f"/kaggle/working/runs/fedavg_U2_g{float(n)}r1.0b0_seed{sd}"
_R6 = [(n, sd) for sd in (1, 2) for n in (2, 4)]
T["R6"] = _pre + [f"python train.py --method fedavg --cut 2 --sigma 0 --rho 1 --bits 0 --gsigma {n} --grho 1 --gbits 0 --srho 1 --sbits 0 --seed {sd} {RR}" for n, sd in _R6] \
    + [f"while [ ! -f {_gns(n, sd)}/test_per_image.csv ]; do sleep 30; done; python revision_attacks.py --run {_gns(n, sd)} --cut 2 --round {ROUNDS} --only presence,ablation,avg --tag gn" for n, sd in _R6]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("\n".join(T[sys.argv[1]]))
    else:
        for k, v in T.items():
            print(f"{k:4s} {len(v):3d} cmds  ~{sum(map(est_h, v)):5.1f} GPU-h  ~{sum(map(est_h, v)) / 2:4.1f} h wall")
