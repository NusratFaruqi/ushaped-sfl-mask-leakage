"""Leakage measurements on a saved checkpoint (paper Sections 5 and 7).

  python run_attacks.py --run runs/fedavg_seed0 --cut 1 --round 100          # channel-free U_1 (from FedAvg)
  python run_attacks.py --run runs/fedavg_seed0 --cut V --round 100          # encoder cut
  python run_attacks.py --run runs/fedavg_seed0 --cut Vh --round 100
  python run_attacks.py --run runs/fedavg_U2_s1.0r0.1b8_..._seed0 --cut 2    # trained with a channel
  python run_attacks.py --run runs/fedavg_seed0 --fedavg --round 100         # FedAvg update inversion
Writes <run>/attacks_<cut>_r<round>[_<aux>].json and a sample grid (.npz).
"""
from __future__ import annotations
import argparse, json, os, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from crcsfl.data import SegDataset, discover
from crcsfl.model import build, Encoder
from crcsfl.split import SplitNet, VhNet, from_full
from crcsfl.channel import Channel
from crcsfl.losses import seg_loss
from crcsfl import attacks as A
from crcsfl.paths import find_input

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--cut", default="2")
p.add_argument("--round", type=int, default=100); p.add_argument("--aux", default="kvasir", choices=["kvasir", "bkai"])
p.add_argument("--bkai", default="/kaggle/input/bkai-igh-neopolyp")
p.add_argument("--n-test", type=int, default=200); p.add_argument("--epochs", type=int, default=30)
p.add_argument("--only", default="learned,opt,multi,server,label,dfil", help="comma list")
p.add_argument("--fedavg", action="store_true"); p.add_argument("--size", type=int, default=352)
p.add_argument("--smoke", action="store_true")
p.add_argument("--surrogate-init", default="imagenet", choices=["imagenet", "random"], help="server-only attack: start the substitute encoder from public ImageNet weights or from random init")
p.add_argument("--sur-steps", type=int, default=3000); p.add_argument("--inv-steps", type=int, default=4000)
p.add_argument("--label-batches", type=int, default=4, help="test batches (of 8) used by the label attack")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0); np.random.seed(0)
cfg = json.load(open(f"{a.run}/config.json"))
from crcsfl.model import set_arch
set_arch(cfg.get("arch", "unet"))
W, S = cfg["width"], (64 if a.smoke else a.size)
split = json.load(open(os.path.join(os.path.dirname(a.run.rstrip("/")), f"splits_seed{cfg['split_seed']}.json")))
tests = [tuple(t) for d in split["clients"].values() for t in d["test"]]
vals = [tuple(t) for d in split["clients"].values() for t in d["val"]]
aux = [tuple(t) for t in split["aux"]] if a.aux == "kvasir" else discover(find_input(a.bkai))
if a.aux != "kvasir":   # Kvasir aux is de-duplicated in train.py; do the same for BKAI against all client images
    from crcsfl.hygiene import cross_duplicates
    _cl = [t[0] for d in split["clients"].values() for v in d.values() for t in v]
    _dup = {pa for pa, _ in cross_duplicates([q[0] for q in aux], _cl)}
    print(f"aux {a.aux}: removed {len(_dup)}/{len(aux)} pHash duplicates of client images", flush=True)
    aux = [q for q in aux if q[0] not in _dup]
random.Random(0).shuffle(tests); tests = tests[: a.n_test]
state = torch.load(f"{a.run}/round{a.round}.pt", map_location=dev)
dl = lambda pairs, shuf=False, bs=8: DataLoader(SegDataset(pairs, S, augment=False), batch_size=bs, shuffle=shuf, drop_last=shuf)


def load_net():
    if cfg["cut"]:                                           # trained as a split with a channel
        net = SplitNet(cfg["cut"], W, Channel(cfg["sigma"], cfg["rho"], cfg["bits"]),
                       Channel(cfg["gsigma"], cfg["grho"], cfg["gbits"]), Channel(0, cfg["srho"], cfg["sbits"]))
        net.load_state_dict(state); return net.to(dev).eval()
    full = build("full", W)[0]; full.load_state_dict(state)
    if a.cut == "V":
        return full.to(dev).eval()
    if a.cut == "Vh":
        return VhNet.from_full_state(state, W).to(dev).eval()
    net = SplitNet(int(a.cut), W); net.load_state_dict(from_full(state, int(a.cut)))
    return net.to(dev).eval()


net = load_net()
if isinstance(net, SplitNet) and (net.up.active or net.gch.active):
    xb, yb, _ = next(iter(dl(vals)))                          # colluding attacker knows the noise scale
    net.calibrate(xb.to(dev), yb.to(dev)); net.eval()
leak = A.Leak(splitnet=net if isinstance(net, SplitNet) else None,
              full=net if a.cut == "V" else None, cut="V" if a.cut == "V" else int(a.cut) if a.cut != "Vh" else "V")
if a.cut == "Vh":                                             # V_h sends z2..z5
    leak = A.Leak(full=type("E", (), {"encoder": net.client.encoder})(), cut="V", levels=[2, 3, 4, 5])
res, grid = {"cut": a.cut, "round": a.round, "aux": a.aux, "run": a.run}, {}


def score(xh, x01, tag):
    m = A.fov_mask(x01)
    out = {"psnr": A.psnr(xh, x01, m).tolist(), "ssim": A.ssim(xh, x01, mask=m).tolist()}
    mse = (((xh - x01) ** 2) * m).sum() / (m.sum() * 3)
    out["psnr_of_mean_mse"] = float(10 * torch.log10(1 / mse))
    try:
        import lpips
        fn = score.lp = getattr(score, "lp", None) or lpips.LPIPS(net="alex").to(dev)
        out["lpips"] = fn(xh * 2 - 1, x01 * 2 - 1).flatten().tolist()
    except Exception:
        pass
    res.setdefault(tag, {k: [] for k in out})
    for k, v in out.items():
        res[tag][k] = res[tag].get(k, []) + (v if isinstance(v, list) else [v])


def reid(xhs, xs):
    """Top-1 retrieval of the true frame among all test frames (ImageNet ResNet-50 features)."""
    import torchvision
    m = torchvision.models.resnet50(weights="IMAGENET1K_V2").to(dev).eval(); m.fc = torch.nn.Identity()
    with torch.no_grad():
        f = lambda t: torch.nn.functional.normalize(m(A.norm(t)), dim=1)
        q, g = torch.cat([f(x) for x in xhs]), torch.cat([f(x) for x in xs])
    return float(((q @ g.T).argmax(1) == torch.arange(len(q), device=dev)).float().mean())


only = set(a.only.split(","))
import time as _t
_T = [_t.time()]
def lap(name):
    print(f"[timing] {name}: {(_t.time() - _T[0]) / 60:.1f} min", flush=True); _T[0] = _t.time()
if a.fedavg:
    # single-step gradient (upper bound) and the real one-epoch update of a 16-image client
    model = build("full", W)[0].to(dev); model.load_state_dict(state)
    x, y, _ = next(iter(dl(tests[:16], bs=16))); x01, y = A.denorm(x.to(dev)), y.to(dev)
    loss = seg_loss(model(A.norm(x01[:8])), y[:8])
    tg = [g.detach() for g in torch.autograd.grad(loss, list(model.parameters()))]
    score(A.invert_fedavg_gradient(model, x01[:8].shape, y[:8], tg, steps=5 if a.smoke else 500), x01[:8], "fedavg_single_step")
    m2 = build("full", W)[0].to(dev); m2.load_state_dict(state)
    opt = torch.optim.AdamW(m2.parameters(), lr=cfg["lr"] * 0.5, weight_decay=cfg["wd"])
    for t in range(2):
        l = seg_loss(m2(A.norm(x01[8 * t:8 * t + 8])), y[8 * t:8 * t + 8]); opt.zero_grad(); l.backward(); opt.step()
    delta = {n: (q2.detach() - q1.detach()) for (n, q1), (_, q2) in zip(model.named_parameters(), m2.named_parameters())}
    xh = A.invert_fedavg_update(model, delta, x01.shape, y, steps_local=2, batch=8, lr_local=cfg["lr"] * 0.5, wd=cfg["wd"], iters=5 if a.smoke else 300)
    score(xh, x01, "fedavg_update")
else:
    xs, ys, xhs = [], [], []
    if only & {"learned", "opt", "multi", "label"}:
        G = A.train_inverter(leak, dl(aux, True), S, epochs=2 if a.smoke else a.epochs, dev=dev)
        for x, y, _ in dl(tests):
            x01 = A.denorm(x.to(dev))
            with torch.no_grad():
                xh = G(leak.noisy(x01))
            xs.append(x01); ys.append(y.to(dev)); xhs.append(xh); score(xh, x01, "learned")
        res["learned"]["reid_top1"] = reid(xhs, xs)
        lap("learned")
        grid["orig"], grid["learned"] = xs[0][:4].cpu().numpy(), xhs[0][:4].cpu().numpy()
    if "opt" in only:
        for x01, xh in list(zip(xs, xhs))[:4]:
            with torch.no_grad():
                obs = leak.noisy(x01)
            score(A.optimise_inversion(leak, obs, xh, steps=20 if a.smoke else 500), x01, "optimisation")
    lap("opt")
    if "multi" in only and isinstance(net, SplitNet) and net.up.sigma > 0:
        for x01 in xs[:4]:
            with torch.no_grad():
                score(G(A.multi_round_obs(leak, x01, R=5 if a.smoke else min(a.round, cfg["rounds"]))), x01, "multi_round")
    lap("multi")
    if "server" in only:
        c = 5 if a.cut in ("V", "Vh") else int(a.cut)
        widths = tuple(max(8, int(w * W)) for w in (64, 128, 256, 512, 1024))
        if a.surrogate_init == "random": os.environ["CRCSFL_NO_PRETRAIN"] = "1"
        sub = Encoder(widths); os.environ.pop("CRCSFL_NO_PRETRAIN", None); sub.levels = sub.levels[:c]; sub = sub.to(dev)
        lv = [2, 3, 4, 5] if a.cut == "Vh" else (list(range(1, 6)) if a.cut == "V" else [c])
        subf = lambda x01: [t for i, t in enumerate(sub(A.norm(x01)), 1) if i in lv]
        with torch.no_grad():                                   # features the server recorded from client data
            rec = [leak.noisy(A.denorm(x.to(dev)))[-1] for x, _, _ in dl([tuple(t) for d in split["clients"].values() for t in d["train"]][:400])]
        opt_s = torch.optim.Adam(sub.parameters(), 1e-3); obs = torch.cat(rec); it = iter(dl(aux, True))
        for s in range(20 if a.smoke else 3000):
            try: xa = next(it)[0]
            except StopIteration: it = iter(dl(aux, True)); xa = next(it)[0]
            f = sub(xa.to(dev))[-1]; l = A.mmd(f, obs[torch.randint(0, len(obs), (len(f),), device=dev)])
            opt_s.zero_grad(); l.backward(); opt_s.step()
        sub.eval()
        Gs = A.train_inverter(leak, dl(aux, True), S, epochs=2 if a.smoke else a.epochs, dev=dev, feat_fn=lambda x: [t.detach() for t in subf(x)])
        for x, _, _ in list(dl(tests))[:8]:
            x01 = A.denorm(x.to(dev))
            with torch.no_grad():
                score(Gs(leak.noisy(x01)), x01, "server_only")
    lap("server")
    if "label" in only and a.cut != "V":
        for x01, y, xh in list(zip(xs, ys, xhs))[:a.label_batches]:
            g, d = A.returned_gradient(net, x01, y)
            for tag, xin in (("label_oracle", x01), ("label_xhat", xh)):
                yh = (A.recover_masks(net, xin, d, g, steps=20 if a.smoke else 300) > 0.5).float()
                dice = (2 * (yh * y).flatten(1).sum(1) / (yh + y).flatten(1).sum(1).clamp_min(1)).tolist()
                res.setdefault(tag, {"dice": []})["dice"] += dice
                res[tag].setdefault("gt_area", []).extend(y.flatten(1).mean(1).tolist())   # 0 = empty mask: exclude when scoring
            for tag, xin in (("label_pred_on_xhat", xh), ("label_pred_on_x", x01)):   # b2 for x_hat, and for the oracle (v7)
                with torch.no_grad():
                    pred = net(A.norm(xin)).sigmoid().gt(0.5).float()
                dice = (2 * (pred * y).flatten(1).sum(1) / (pred + y).flatten(1).sum(1).clamp_min(1)).tolist()
                res.setdefault(tag, {"dice": []})["dice"] += dice
                res[tag].setdefault("gt_area", []).extend(y.flatten(1).mean(1).tolist())
    lap("label")
    # ---- mask leakage (pre-registration amendment v5): Fisher bound (H5b), prior baseline b1, server-only attack (H6)
    if only & {"labelfisher", "labelserver"} and a.cut != "V":
        from crcsfl import label_leak as LL
        tb = list(dl(tests))[: a.label_batches]
        am = torch.cat([y for _, y, _ in list(dl(aux))[: (2 if a.smoke else 50)]])
        prior = LL.prior_mask(am).to(dev)
        for _, y, _ in tb:
            dz, ar = LL.dice_per_image(prior.expand_as(y.to(dev)), y.to(dev))
            res.setdefault("label_prior", {"dice": [], "gt_area": []}); res["label_prior"]["dice"] += dz.tolist(); res["label_prior"]["gt_area"] += ar.tolist()
        if "labelfisher" in only:
            for x, y, _ in tb[: (1 if a.smoke else 4)]:
                L, rms = LL.label_fisher(net, A.denorm(x.to(dev)), y.to(dev), probes=2 if a.smoke else 16)
                res.setdefault("label_fisher", {"L": [], "gt_area": []}); res["label_fisher"]["L"] += L.tolist()
                res["label_fisher"]["gt_area"] += y.flatten(1).mean(1).tolist()
        if "labelserver" in only and isinstance(net, SplitNet):
            sur = LL.surrogate_client(net, dl(aux, True), steps=a.sur_steps, dev=dev, random_init=a.surrogate_init == "random")
            inv = LL.train_mask_inverter(sur, dl(aux, True), S, steps=a.inv_steps, dev=dev)
            for x, y, _ in tb:
                x01, yy = A.denorm(x.to(dev)), y.to(dev)
                g, d = A.returned_gradient(net, x01, yy)          # what the REAL client returns
                with torch.no_grad():
                    dz, ar = LL.dice_per_image(inv(g, d), yy)
                res.setdefault("label_server_only", {"dice": [], "gt_area": []})
                res["label_server_only"]["dice"] += dz.tolist(); res["label_server_only"]["gt_area"] += ar.tolist()
                if "server" in only:                                  # baseline b2 for Setting B (amendment v5): surrogate's
                    with torch.no_grad():                             # prediction on the server-only reconstruction x_hat
                        xs_hat = Gs(leak.noisy(x01)).clamp(0, 1)
                        dz, ar = LL.dice_per_image(sur(A.norm(xs_hat)).sigmoid(), yy)
                    res.setdefault("label_pred_server", {"dice": [], "gt_area": []})
                    res["label_pred_server"]["dice"] += dz.tolist(); res["label_pred_server"]["gt_area"] += ar.tolist()
    lap("labelleak")
    if "dfil" in only and isinstance(net, SplitNet) and net.up.sigma > 0:
        vals_b = []
        for x, y, _ in list(dl(vals))[: (1 if a.smoke else 25)]:
            vals_b += A.dfil(net, A.denorm(x.to(dev)), y.to(dev), net.up.sigma * net.up.scale,
                             net.gch.sigma * net.gch.scale if net.gch.sigma > 0 else None, probes=4 if a.smoke else 64).tolist()
        res["dfil"] = {"per_image": vals_b, "p95": float(np.percentile(vals_b, 95))}
lap("dfil/fedavg")
summary = {k: {m: float(np.nanmean(v)) if isinstance(v, list) else v for m, v in d.items()} if isinstance(d, dict) else d for k, d in res.items()}
tag = f"attacks_{'fedavg' if a.fedavg else a.cut}_r{a.round}_{a.aux}" + ("_srand" if a.surrogate_init == "random" else "") + (f"_lb{a.label_batches}" if a.label_batches != 4 else "") + ("" if a.only == "learned,opt,multi,server,label,dfil" else "_only-" + a.only.replace(",", "+"))
json.dump({"summary": summary, "raw": res}, open(f"{a.run}/{tag}.json", "w"))
np.savez_compressed(f"{a.run}/{tag}_grid.npz", **grid)
print(json.dumps(summary, indent=1)[:2000])
