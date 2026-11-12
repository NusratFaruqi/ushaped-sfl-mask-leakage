"""Training entry point (run on Kaggle; see kaggle/README.md).

Examples
  python train.py --method fedavg --seed 0
  python train.py --method fedavg --cut 2 --sigma 0.5 --rho 0.1 --bits 8 --gsigma 0.5 --seed 0
  python train.py --method fedavg --cut 2 --nopeek 0.5 --seed 0
  python train.py --method central --seed 0
  python train.py --method local --client C3 --seed 0
  python train.py --method dpsgd --eps 8 --seed 0
  python train.py --equiv-check            # Proposition 1, 10 rounds
"""
from __future__ import annotations
import argparse, copy, csv, json, math, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from crcsfl.data import SegDataset, polypgen, discover, synthetic_pairs, preload, augment_batch
from crcsfl.paths import find_input
from crcsfl.hygiene import cluster, group_split, cross_duplicates
from crcsfl.model import build
from crcsfl.split import SplitNet
from crcsfl.channel import Channel, payload_bits
from crcsfl.losses import seg_loss, dcor
from crcsfl.metrics import batch_metrics

p = argparse.ArgumentParser()
p.add_argument("--method", default="fedavg", choices=["central", "local", "fedavg", "fedprox", "fedgn", "dpsgd"])
p.add_argument("--cut", type=int, default=0, help="0 = full network; 1..4 = U-cut depth")
p.add_argument("--sigma", type=float, default=0.0); p.add_argument("--rho", type=float, default=1.0)
p.add_argument("--bits", type=int, default=0)
p.add_argument("--gsigma", type=float, default=0.0); p.add_argument("--grho", type=float, default=1.0)
p.add_argument("--gbits", type=int, default=0)
p.add_argument("--srho", type=float, default=1.0); p.add_argument("--sbits", type=int, default=0, help="server-side compression")
p.add_argument("--nopeek", type=float, default=0.0)
p.add_argument("--compress", action="store_true", help="FedAvg: top-10%% + int8 updates with error feedback")
p.add_argument("--mu", type=float, default=0.01)
p.add_argument("--eps", type=float, default=8.0); p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--client", default=None)
p.add_argument("--dp-freeze-encoder", action="store_true", help="DP-SGD: keep the public pretrained encoder fixed, train the rest privately")
p.add_argument("--dp-opt", default="adamw", choices=["adamw", "sgd"])
p.add_argument("--rounds", type=int, default=100); p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--arch", default="resnet34", choices=["resnet34", "unet"]); p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--width", type=float, default=0.5); p.add_argument("--size", type=int, default=352)
p.add_argument("--seed", type=int, default=0); p.add_argument("--split-seed", type=int, default=0)
p.add_argument("--polypgen", default="/kaggle/input/polypgen20021-video")
p.add_argument("--cvc", default="/kaggle/input/cvcclinicdb"); p.add_argument("--etis", default="/kaggle/input/etis-laribpolypdb")
p.add_argument("--kvasir", default="/kaggle/input/kvasirseg")
p.add_argument("--dataset", default="polyp", choices=["polyp", "ham"], help="ham = HAM10000 second modality (crcsfl/ham.py)")
p.add_argument("--out", default="runs"); p.add_argument("--workers", type=int, default=2)
p.add_argument("--smoke", action="store_true", help="tiny synthetic run for code checks")
p.add_argument("--equiv-check", action="store_true")
p.add_argument("--no-amp", action="store_true", help="train in fp32 (DP-SGD always does)")
p.add_argument("--prepare-splits", action="store_true", help="build the split/pHash cache and exit")
p.add_argument("--save-rounds", default="1,50,100")
p.add_argument("--local-epochs", default="1", help="epochs per round; 'match' = enough epochs that a local client takes as many steps per round as central")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
from crcsfl.model import set_arch
set_arch(a.arch)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ data
def federation():
    """Returns {client: {train,val,test}}, {name: test pairs} for held-out sets, aux pairs."""
    if a.smoke:
        root = os.path.join(a.out, "_synthetic")
        cl = {f"C{k}": synthetic_pairs(f"{root}/C{k}", n=24, size=64, seed=k) for k in range(1, 4)}
        splits = {k: group_split(v, list(range(len(v))), a.split_seed) for k, v in cl.items()}
        aux = synthetic_pairs(f"{root}/aux", n=16, size=64, seed=7)
        json.dump({"clients": splits, "aux": aux}, open(os.path.join(a.out, f"splits_seed{a.split_seed}.json"), "w"))
        return splits, {"C6": synthetic_pairs(f"{root}/C6", n=8, size=64, seed=9)}, aux
    cache = os.path.join(a.out, f"splits_seed{a.split_seed}.json")
    if a.dataset == "ham":
        return federation_ham(cache)
    for n in ("polypgen", "cvc", "etis", "kvasir"):
        setattr(a, n, find_input(getattr(a, n)))
    pg = polypgen(a.polypgen)
    if not {"C1", "C2", "C3", "C4", "C5", "C6"} <= set(pg):
        raise SystemExit(f"PolypGen centres missing under {a.polypgen}: found {sorted(pg)}")
    held = {"C6": pg.pop("C6", [])}
    held["CVC-ClinicDB"] = discover(a.cvc)
    held["ETIS"] = discover(a.etis)
    aux = discover(a.kvasir)
    if os.path.exists(cache) and json.load(open(cache)).get("clients"):
        js = json.load(open(cache))
        return {k: {s: [tuple(t) for t in v] for s, v in d.items()} for k, d in js["clients"].items()}, held, [tuple(t) for t in js["aux"]]
    splits = {}
    for cen, pairs in pg.items():
        groups = cluster([q[0] for q in pairs], thr=8)
        splits[cen] = group_split(pairs, groups, a.split_seed)
    # remove auxiliary images that duplicate any client or test image
    others = [q[0] for d in splits.values() for v in d.values() for q in v] + [q[0] for v in held.values() for q in v]
    dup = {pa for pa, _ in cross_duplicates([q[0] for q in aux], others)}
    report = {"aux_removed": len(dup), "sizes": {k: {s: len(v) for s, v in d.items()} for k, d in splits.items()}}
    if len(dup) > 0.2 * len(aux):
        raise SystemExit(f"pHash removed {len(dup)}/{len(aux)} auxiliary images: threshold too loose, check hygiene")
    aux = [q for q in aux if q[0] not in dup]
    os.makedirs(a.out, exist_ok=True)
    json.dump({"clients": splits, "aux": aux, "report": report}, open(cache, "w"))
    print("split report:", report)
    return splits, held, aux


def federation_ham(cache):
    """HAM10000: three Vienna sources as clients, rosendahl as auxiliary data; splits by lesion_id; no external test sets."""
    from crcsfl import ham
    if os.path.exists(cache) and json.load(open(cache)).get("clients"):
        js = json.load(open(cache))
        return {k: {s: [tuple(t) for t in v] for s, v in d.items()} for k, d in js["clients"].items()}, {}, [tuple(t) for t in js["aux"]]
    cl, aux = ham.pairs(ham.fetch())
    splits = {k: group_split(v, [t[2] for t in v], a.split_seed) for k, v in cl.items()}
    others = [q[0] for d in splits.values() for v in d.values() for q in v]
    dup = {pa for pa, _ in cross_duplicates([q[0] for q in aux], others)}
    if len(dup) > 0.2 * len(aux):
        raise SystemExit(f"pHash removed {len(dup)}/{len(aux)} auxiliary images: threshold too loose, check hygiene")
    aux = [q for q in aux if q[0] not in dup]
    report = {"dataset": "ham10000", "aux_removed": len(dup), "sizes": {k: {s: len(v) for s, v in d.items()} for k, d in splits.items()},
              "lesions": {k: len({t[2] for t in v}) for k, v in cl.items()}}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"clients": splits, "aux": aux, "report": report}, open(cache, "w"))
    print("split report:", report)
    return splits, {}, aux


def loader(pairs, train):
    # images are decoded once into RAM (preload); augmentation runs on the GPU in local_epoch
    ds = SegDataset(pairs, size=64 if a.smoke else a.size, augment=False)
    return DataLoader(ds, batch_size=a.batch, shuffle=train, drop_last=train and len(ds) > a.batch,
                      num_workers=0, pin_memory=dev == "cuda")


# ------------------------------------------------------------------ model
def make_model():
    if a.cut == 0:
        return build("full", a.width)[0].to(dev)
    up = Channel(a.sigma, a.rho, a.bits)
    gch = Channel(a.gsigma, a.grho, a.gbits)
    srv = Channel(0.0, a.srho, a.sbits)
    return SplitNet(a.cut, a.width, up, gch, srv).to(dev)


def fwd(model, x, ids):
    if isinstance(model, SplitNet):
        return model(x, ids=ids, keep=a.nopeek > 0)
    return model(x)


def is_gn(name):
    return ".norm" in name or "GroupNorm" in name


# ------------------------------------------------------------------ local training
def local_epoch(model, dl, global_state=None):
    model.train()
    if a.method == "dpsgd" and a.dp_freeze_encoder:
        for n, p_ in model.named_parameters():
            p_.requires_grad_(not n.startswith("encoder."))
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    if a.method == "dpsgd" and a.dp_opt == "sgd":
        opt = torch.optim.SGD(trainable, lr=a.lr_now, momentum=0.9)
    else:
        opt = torch.optim.AdamW(trainable, lr=a.lr_now, weight_decay=a.wd)  # state reset each round
    gparams = None
    if a.method == "fedprox" and global_state is not None:
        gparams = {k: v.detach().clone() for k, v in global_state.items()}
    amp = dev == "cuda" and not a.no_amp and a.method != "dpsgd"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for x, y, idx in (b for _ in range(getattr(a, "le_now", 1)) for b in dl):
        x, y = augment_batch(x.to(dev, non_blocking=True), y.to(dev, non_blocking=True))
        ids = [dl.dataset.pairs[i][0] for i in idx.tolist()]
        if a.method == "dpsgd":
            dp_step(model, opt, x, y)
            continue
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = fwd(model, x, ids)
        loss = seg_loss(out.float(), y)
        if a.nopeek > 0:
            xs = torch.nn.functional.interpolate(x, size=32, mode="area")
            loss = loss + a.nopeek * dcor(xs.float(), model.last["z"].float())
        if gparams is not None:
            loss = loss + a.mu / 2 * sum(((p_ - gparams[n]) ** 2).sum() for n, p_ in model.named_parameters())
        if not torch.isfinite(loss):
            a.skipped = getattr(a, "skipped", 0) + 1
            continue
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()


def dp_step(model, opt, x, y):
    """DP-SGD with per-sample clipping (torch.func) and Gaussian noise."""
    from torch.func import functional_call, vmap, grad
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    fixed = {k: v.detach() for k, v in model.named_parameters() if not v.requires_grad}
    bufs = {k: v.detach() for k, v in model.named_buffers()}

    def loss_one(p_, xi, yi):
        return seg_loss(functional_call(model, ({**p_, **fixed}, bufs), (xi[None],)), yi[None])
    g = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)
    norms = torch.sqrt(sum(v.flatten(1).pow(2).sum(1) for v in g.values()))
    f = (a.clip / (norms + 1e-6)).clamp(max=1.0)
    for n, p_ in model.named_parameters():
        if n not in g:
            continue
        gs = (g[n] * f.view(-1, *[1] * (g[n].dim() - 1))).sum(0)
        p_.grad = (gs + torch.randn_like(gs) * a.noise_mult * a.clip) / x.shape[0]
    opt.step()


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, pairs, tag, rows=None, hd=False):
    model.eval()
    res = []
    for x, y, idx in loader(pairs, False):
        ids = [pairs[i][0] for i in idx.tolist()]
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda" and not a.no_amp):
            out = fwd(model, x.to(dev), ids).float()
        orig = None
        if hd:   # final test: score at native resolution
            from PIL import Image
            orig = [np.asarray(Image.open(pairs[i][1]).convert("RGB")).max(2) > 127 for i in idx.tolist()]
        m = batch_metrics(out, y.to(dev), hd=hd, masks_orig=orig)
        res += m
        if rows is not None:
            rows += [{"set": tag, "image": os.path.basename(i), **r} for i, r in zip(ids, m)]
    return float(np.mean([r["dice"] for r in res])) if res else float("nan")


def average(states, weights, keep_local=()):
    tot = sum(weights)
    avg = {}
    for k in states[0]:
        if states[0][k].dtype.is_floating_point and k not in keep_local:
            avg[k] = sum(s[k] * (w / tot) for s, w in zip(states, weights))
        else:
            avg[k] = states[0][k]
    return avg


# ------------------------------------------------------------------ main
def tag():
    t = f"{a.method}"
    if a.method == "local": t += f"-{a.client}"
    if a.cut: t += f"_U{a.cut}"
    if a.sigma or a.rho < 1 or a.bits: t += f"_s{a.sigma}r{a.rho}b{a.bits}"
    if a.gsigma or a.grho < 1 or a.gbits: t += f"_g{a.gsigma}r{a.grho}b{a.gbits}"
    if a.srho < 1 or a.sbits: t += f"_srv{a.srho}b{a.sbits}"
    if a.nopeek: t += f"_nopeek{a.nopeek}"
    if a.method == "dpsgd": t += f"_eps{a.eps}" + ("_frozen" if a.dp_freeze_encoder else "") + ("_sgd" if a.dp_opt == "sgd" else "") + (f"_b{a.batch}" if a.batch != 8 else "")
    if a.compress: t += "_compressed"
    if a.local_epochs != "1": t += f"_le{a.local_epochs}"
    return t + f"_seed{a.seed}"


def main():
    seed_all(a.seed)
    splits, held, aux = federation()
    if a.prepare_splits:
        return
    t_pre = time.time()
    preload([q for d in splits.values() for v in d.values() for q in v] + [q for v in held.values() for q in v], 64 if a.smoke else a.size)
    print(f"preloaded images in {time.time() - t_pre:.0f}s", flush=True)
    names = sorted(splits)
    if a.method == "local":
        names = [a.client]
    if a.method == "central":  # pooled data, one "client"
        pooled = {s: sum((splits[k][s] for k in names), []) for s in ("train", "val", "test")}
        train_sets = {"pooled": pooled["train"]}
    else:
        train_sets = {k: splits[k]["train"] for k in names}
    weights = [len(v) for v in train_sets.values()]
    n_pool = sum(len(splits[k]["train"]) for k in splits)   # central training-set size, for --local-epochs match
    if a.method == "dpsgd":
        from opacus.accountants.utils import get_noise_multiplier
        n_min = min(weights)
        a.noise_mult = get_noise_multiplier(target_epsilon=a.eps, target_delta=1e-5,
                                            sample_rate=a.batch / n_min, epochs=a.rounds, accountant="rdp")
        print("DP-SGD noise multiplier", a.noise_mult)
    out = os.path.join(a.out, tag()); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(f"{out}/config.json", "w"), indent=1)
    model = make_model()
    gstate = copy.deepcopy(model.state_dict())
    gn_keys = [k for k in gstate if is_gn(k)] if a.method == "fedgn" else []
    local_gn = {k: {n: gstate[n].clone() for n in gn_keys} for k in train_sets}
    save_at = {int(r) for r in a.save_rounds.split(",")}
    log = open(f"{out}/rounds.csv", "w", newline=""); lw = csv.writer(log)
    lw.writerow(["round", "val_dice", "bytes_up_cum", "seconds"])
    best, t0 = -1, time.time()
    residual, comp_bytes = {}, [0.0]
    val_pairs = {k: splits[k]["val"] for k in sorted(splits)}
    for r in range(1, a.rounds + 1):
        a.lr_now = a.lr * 0.5 * (1 + math.cos(math.pi * (r - 1) / a.rounds))
        states = []
        for k, tr in train_sets.items():
            model.load_state_dict(gstate)
            if gn_keys:
                model.load_state_dict(local_gn[k], strict=False)
            if isinstance(model, SplitNet) and model.up.active | model.gch.active:
                vk = val_pairs.get(k) or next(iter(val_pairs.values()))
                xb, yb, _ = next(iter(loader(vk, False)))
                model.calibrate(xb.to(dev), yb.to(dev))
            a.le_now = (max(1, round(n_pool / len(tr))) if a.local_epochs == "match" else int(a.local_epochs))
            local_epoch(model, loader(tr, True), gstate)
            st = {n: v.detach().clone() for n, v in model.state_dict().items()}
            if a.compress:                                    # compress the update, keep the residual locally
                err = residual.setdefault(k, {n: torch.zeros_like(v) for n, v in st.items() if v.dtype.is_floating_point})
                for n in err:
                    dlt = st[n] - gstate[n] + err[n]
                    q = Channel(0.0, 0.1, 8).apply(dlt.flatten()[None]).view_as(dlt)
                    err[n] = dlt - q
                    st[n] = gstate[n] + q
                    comp_bytes[0] += payload_bits(dlt.numel(), 0.1, 8) / 8
            states.append(st)
            if gn_keys:
                local_gn[k] = {n: st[n] for n in gn_keys}
        gstate = average(states, weights)
        model.load_state_dict(gstate)
        sent = sum(c.bytes_sent for c in (model.up, model.gch, model.srv, model.srv_g)) if isinstance(model, SplitNet) else comp_bytes[0]
        comp_bytes[0] = 0.0
        cum_sent = getattr(a, "cum_sent", 0) + sent; a.cum_sent = cum_sent
        vd = np.mean([evaluate(model, v, "val") for k, v in val_pairs.items() if k in names or a.method != "local"])
        if isinstance(model, SplitNet):
            for c in (model.up, model.gch, model.srv, model.srv_g):
                c.bytes_sent = 0.0                              # eval traffic is not counted
        lw.writerow([r, f"{vd:.4f}", f"{cum_sent:.0f}", f"{time.time() - t0:.0f}"]); log.flush()
        if vd > best:
            best = vd; torch.save(gstate, f"{out}/best.pt")
        if r in save_at:
            torch.save(gstate, f"{out}/round{r}.pt")
        print(f"round {r} val dice {vd:.4f} ({time.time() - t0:.0f}s)" + (f" skipped non-finite steps: {a.skipped}" if getattr(a, "skipped", 0) else ""), flush=True)
        a.skipped = 0
    # final test with the best checkpoint
    model.load_state_dict(torch.load(f"{out}/best.pt", map_location=dev))
    rows = []
    for k in sorted(splits):
        if gn_keys:
            model.load_state_dict(torch.load(f"{out}/best.pt", map_location=dev)); model.load_state_dict(local_gn[k], strict=False)
        evaluate(model, splits[k]["test"], k, rows, hd=True)
    model.load_state_dict(torch.load(f"{out}/best.pt", map_location=dev))
    for name, pairs in held.items():
        if pairs:
            evaluate(model, pairs, name, rows, hd=True)
    with open(f"{out}/test_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("done", out)


def equiv_check():
    """Proposition 1: SFL-V1 at every cut (no channel) == FedAvg, given identical batches."""
    seed_all(0)
    x = torch.randn(16, 3, 64, 64); y = (torch.rand(16, 1, 64, 64) > 0.7).float()
    full = build("full", 0.25)[0]
    for c in (1, 2, 3, 4):
        sn = SplitNet(c, 0.25)
        # copy weights: full encoder levels / decoder blocks -> split parts
        fs = full.state_dict(); ss = sn.state_dict()
        mp = {}
        for k in ss:
            if k.startswith("client.enc."): mp[k] = "encoder.levels." + k.split("client.enc.")[1]
            elif k.startswith("server.enc."):
                i, rest = k.split("server.enc.")[1].split(".", 1); mp[k] = f"encoder.levels.{int(i) + c}.{rest}"
            elif k.startswith("server.bottleneck"): mp[k] = "decoder." + k.split("server.")[1]
            elif k.startswith("server.ups."):
                i, rest = k.split("server.ups.")[1].split(".", 1); mp[k] = f"decoder.blocks.{i}.{rest}"
            elif k.startswith("client.ups."):
                i, rest = k.split("client.ups.")[1].split(".", 1); mp[k] = f"decoder.blocks.{int(i) + 4 - c}.{rest}"
            elif k.startswith("client.head"): mp[k] = "decoder.head" + k.split("client.head")[1]
        sn.load_state_dict({k: fs[mp[k]] for k in ss})
        diffs = []
        for m in (full, sn):
            mm = copy.deepcopy(m); opt = torch.optim.AdamW(mm.parameters(), lr=1e-3)
            for i in range(0, 16, 8):
                loss = seg_loss(mm(x[i:i + 8]), y[i:i + 8]); opt.zero_grad(); loss.backward(); opt.step()
            diffs.append(mm)
        a_, b_ = diffs[0].state_dict(), diffs[1].state_dict()
        md = max((a_[mp[k]] - b_[k]).abs().max().item() for k in b_)
        print(f"U{c}: max |theta_FedAvg - theta_SFL| after 2 steps = {md:.2e}")


if __name__ == "__main__":
    equiv_check() if a.equiv_check else main()
