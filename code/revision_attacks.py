"""Follow-up experiments (2026-11-12; exploratory, not registered). Attack-only, on saved checkpoints.

  --only presence   Setting A (x_hat) recovery and baselines on ALL attacked frames, empty ones included: per-image soft
                    and binary area, centroid, Dice; baseline b2 (prediction on x_hat) and b2d (prediction from the true
                    server feature d with x_hat's skips).                                      [review C1, C2, C4, S8]
  --only ablation   Setting B inverter trained on (g, d), g only, d only.                       [review C2]
  --only fisher     Fisher index L_c on all attacked frames, with rms(g), P and dim(d) to rescale offline.  [C5]
  --only train      Gradients a client really returns in a local epoch: training frames, augmentation, AMP (scaled
                    gradients), weights drifting step by step; attacked with the round-start weights.        [C3]
  --only noise      Post-hoc Gaussian noise on the returned gradient only (nu x mean RMS), on THIS checkpoint, so noise
                    is not confounded with a different trained model. Setting A and B.          [S4, S5]
  --only strong     Learned image decoder trained 3x longer (90 epochs), PSNR / SSIM / retrieval.  [S3]
  --only examples   88x88 thumbnails of 4 frames (image, x_hat, mask, recovered masks) for a figure.   [S15]

Writes <run>/revision_<cut>_r<round>[_<tag>].json. Every list is per frame, in the fixed shuffled test order used by
run_attacks.py (frames 0..31 are the 32 pilot frames).
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from crcsfl.data import SegDataset, augment_batch
from crcsfl.model import build, set_arch
from crcsfl.split import SplitNet, VhNet, from_full
from crcsfl.channel import Channel
from crcsfl.losses import seg_loss
from crcsfl import attacks as A
from crcsfl import label_leak as LL

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--cut", default="2"); p.add_argument("--round", type=int, default=30)
p.add_argument("--only", default="presence,ablation,fisher,examples")
p.add_argument("--label-batches", type=int, default=19); p.add_argument("--n-test", type=int, default=200)
p.add_argument("--epochs", type=int, default=30); p.add_argument("--size", type=int, default=352)
p.add_argument("--sur-steps", type=int, default=3000); p.add_argument("--inv-steps", type=int, default=4000)
p.add_argument("--train-batches", type=int, default=2, help="local-epoch batches attacked per centre")
p.add_argument("--nus", default="0.5,1,2,4"); p.add_argument("--noise-batches", type=int, default=6)
p.add_argument("--tag", default=""); p.add_argument("--smoke", action="store_true")
p.add_argument("--inv-seeds", type=int, default=1, help="repeat surrogate + inverter training (follow-up analysis)")
p.add_argument("--avg-k", default="1,10,30", help="adaptive attacker: average K independently noised gradients of an image")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0); np.random.seed(0)
cfg = json.load(open(f"{a.run}/config.json")); set_arch(cfg.get("arch", "unet"))
W, S = cfg["width"], (64 if a.smoke else a.size)
split = json.load(open(os.path.join(os.path.dirname(a.run.rstrip("/")), f"splits_seed{cfg['split_seed']}.json")))
tests = [tuple(t) for d in split["clients"].values() for t in d["test"]]
vals = [tuple(t) for d in split["clients"].values() for t in d["val"]]
aux = [tuple(t) for t in split["aux"]]
random.Random(0).shuffle(tests); tests = tests[: a.n_test]           # identical order to run_attacks.py
state = torch.load(f"{a.run}/round{a.round}.pt", map_location=dev)
dl = lambda pairs, shuf=False, bs=8, aug=False: DataLoader(SegDataset(pairs, S, augment=aug), batch_size=bs, shuffle=shuf, drop_last=shuf)
only = set(a.only.split(","))
T0 = [time.time()]
def lap(n): print(f"[timing] {n}: {(time.time() - T0[0]) / 60:.1f} min", flush=True); T0[0] = time.time()


def load_net(gch=None):
    if cfg["cut"]:
        net = SplitNet(cfg["cut"], W, Channel(cfg["sigma"], cfg["rho"], cfg["bits"]),
                       gch or Channel(cfg["gsigma"], cfg["grho"], cfg["gbits"]), Channel(0, cfg["srho"], cfg["sbits"]))
        net.load_state_dict(state)
    elif a.cut == "Vh":
        net = VhNet.from_full_state(state, W)
    else:
        net = SplitNet(int(a.cut), W, down_grad=gch); net.load_state_dict(from_full(state, int(a.cut)))
    net = net.to(dev).eval()
    if isinstance(net, SplitNet) and (net.up.active or net.gch.active):
        xb, yb, _ = next(iter(dl(vals)))
        net.calibrate(xb.to(dev), yb.to(dev)); net.eval()
    return net


net = load_net()
is_split = isinstance(net, SplitNet)
leak = A.Leak(splitnet=net, cut=int(a.cut)) if is_split else \
    A.Leak(full=type("E", (), {"encoder": net.client.encoder})(), cut="V", levels=[2, 3, 4, 5])
res = {"cut": a.cut, "round": a.round, "run": a.run, "only": sorted(only), "note": "revision, exploratory"}


def stats(m):
    """Per image: soft mean, binary area, centroid (row, col in [0,1]); m: (B,1,H,W) probabilities."""
    b = (m > 0.5).float()
    H, Wd = m.shape[-2:]
    r = torch.linspace(0, 1, H, device=m.device).view(1, 1, H, 1); c = torch.linspace(0, 1, Wd, device=m.device).view(1, 1, 1, Wd)
    ar = b.flatten(1).sum(1)
    cr = (b * r).flatten(1).sum(1) / ar.clamp_min(1); cc = (b * c).flatten(1).sum(1) / ar.clamp_min(1)
    return {"soft": m.flatten(1).mean(1).tolist(), "area": (ar / (H * Wd)).tolist(), "cy": cr.tolist(), "cx": cc.tolist()}


def put(tag, m, y):
    d, _ = LL.dice_per_image(m, y)
    out = {"dice": d.tolist(), **stats(m)}
    r = res.setdefault(tag, {k: [] for k in out})
    for k, v in out.items():
        r[k] += v


def recover_scaleinv(net_, x01_hat, d, g, steps=300, lr=0.1, tv=1e-3):
    """Gradient matching that does not need the gradient's scale (AMP loss scaling): per-image unit-norm match."""
    x = A.norm(x01_hat)
    with torch.no_grad():
        skips = [s.detach() for s in net_.encode(x)]
        o0 = net_.decode(d, skips)
    v = o0.detach().clone().requires_grad_(True); opt = torch.optim.Adam([v], lr=lr)
    dd = d.detach().requires_grad_(True)
    un = lambda t: t.flatten(1) / t.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-20)
    for _ in range(steps):
        y = torch.sigmoid(v)
        gy, = torch.autograd.grad(seg_loss(net_.decode(dd, skips), y), dd, create_graph=True)
        obj = ((un(gy) - un(g)) ** 2).sum(1).mean()
        obj = obj + tv * ((y[..., 1:, :] - y[..., :-1, :]).abs().mean() + (y[..., :, 1:] - y[..., :, :-1]).abs().mean())
        opt.zero_grad(); obj.backward(); opt.step()
    return torch.sigmoid(v).detach()


steps_rec = 20 if a.smoke else 300
# ---------------------------------------------------------------- learned decoder (x_hat) and test batches
G = A.train_inverter(leak, dl(aux, True), S, epochs=2 if a.smoke else a.epochs, dev=dev)
batches = []
for i, (x, y, _) in enumerate(dl(tests)):
    if i >= a.label_batches: break
    x01 = A.denorm(x.to(dev))
    with torch.no_grad():
        xh = G(leak.noisy(x01)).clamp(0, 1)
    batches.append((x01, y.to(dev), xh))
res["gt_area"] = [v for _, y, _ in batches for v in y.flatten(1).mean(1).tolist()]
lap("decoder")
_am = torch.cat([y for _, y, _ in list(dl(aux))[: (2 if a.smoke else 50)]])
prior = LL.prior_mask(_am).to(dev)                     # b1, as in run_attacks.py

if "presence" in only:
    for x01, y, xh in batches:
        g, d = A.returned_gradient(net, x01, y)
        put("A_xhat", A.recover_masks(net, xh, d, g, steps=steps_rec), y)
        with torch.no_grad():
            put("b2_pred_xhat", net(A.norm(xh)).sigmoid(), y)
            put("b2_pred_x", net(A.norm(x01)).sigmoid(), y)
            put("b2d_true_d_xhat_skips", net.decode(d, net.encode(A.norm(xh))).sigmoid(), y)
            put("b1_prior", prior.expand_as(y).float(), y)
    lap("presence")

sur = None
if "ablation" in only and is_split:
    for s_ in range(a.inv_seeds):
        sfx = "" if s_ == 0 else f"_s{s_}"
        if s_:
            torch.manual_seed(1000 * s_); random.seed(1000 * s_); np.random.seed(1000 * s_)
        sur_s = LL.surrogate_client(net, dl(aux, True), steps=20 if a.smoke else a.sur_steps, dev=dev)
        invs_s = {}
        for use in (("both", "g", "d") if s_ == 0 else ("both", "d")):
            invs_s[use] = LL.train_mask_inverter(sur_s, dl(aux, True), S, steps=20 if a.smoke else a.inv_steps, dev=dev, use=use)
            for x01, y, _ in batches:
                g, d = A.returned_gradient(net, x01, y)
                with torch.no_grad():
                    put(f"B_inv_{use}{sfx}", invs_s[use](g, d), y)
                    if use == "both":                      # control: the gradient of another image in the batch
                        put(f"B_inv_both_gshuf{sfx}", invs_s[use](torch.roll(g, 1, 0), d), y)
        if s_ == 0:
            sur, invs = sur_s, invs_s
        lap(f"ablation seed {s_}")

if "avg" in only and is_split and net.gch.active:
    # Adaptive attacker against gradient noise (exploratory): the same image's gradient observed K times
    # with fresh noise and the weights held at round R, then averaged. Holding the weights fixed favours the attacker.
    for K in [int(k) for k in a.avg_k.split(",")]:
        for x01, y, xh in batches:
            gs = [A.returned_gradient(net, x01, y) for _ in range(K)]
            g = torch.stack([t[0] for t in gs]).mean(0); d = gs[0][1]
            put(f"AVG{K}_A_xhat", A.recover_masks(net, xh, d, g, steps=steps_rec), y)
            if sur is not None:
                with torch.no_grad():
                    put(f"AVG{K}_B_inv_both", invs["both"](g, d), y)
    lap("avg")

if "fisher" in only and is_split:
    r = res.setdefault("fisher", {"L": [], "rms": [], "P": [], "dim_d": []})
    for x01, y, _ in batches:
        L, rms = LL.label_fisher(net, x01, y, probes=2 if a.smoke else 16)
        with torch.no_grad():
            dim_d = net.serve(net.encode(A.norm(x01[:1])))[0].numel()
        r["L"] += L.tolist(); r["rms"] += rms.tolist(); r["P"] += [y[0].numel()] * len(L); r["dim_d"] += [dim_d] * len(L)
    lap("fisher")

if "train" in only and is_split:
    lr_now = cfg["lr"] * 0.5 * (1 + math.cos(math.pi * (a.round - 1) / cfg["rounds"]))   # lr used in this round
    inv = invs["both"] if sur is not None else LL.train_mask_inverter(
        LL.surrogate_client(net, dl(aux, True), steps=20 if a.smoke else a.sur_steps, dev=dev), dl(aux, True), S,
        steps=20 if a.smoke else a.inv_steps, dev=dev)
    amp = dev == "cuda"
    res["train_meta"] = {"lr": lr_now, "amp": amp, "centres": [], "loss_scale": []}
    ygt = []
    for cen, dsp in split["clients"].items():
        m = copy.deepcopy(net).train()                                  # this client's copy for one local epoch
        opt = torch.optim.AdamW(m.parameters(), lr=lr_now, weight_decay=cfg["wd"])
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        for bi, (x, y, _) in enumerate(dl([tuple(t) for t in dsp["train"]], shuf=True)):
            if bi >= a.train_batches: break
            x, y = augment_batch(x.to(dev), y.to(dev))
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = m(x, keep=True)
            loss = seg_loss(out.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            g_sent, d_sent, z_sent = m.last["d"].grad.detach().float(), m.last["d"].detach().float(), m.last["z"].detach().float()
            sc = float(scaler.get_scale()) if amp else 1.0
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
            x01 = A.denorm(x)
            with torch.no_grad():
                xh = G([z_sent]).clamp(0, 1)
            put("T_A_xhat_knownscale", A.recover_masks(net, xh, d_sent, g_sent / sc, steps=steps_rec), y)
            put("T_A_xhat_scaleinv", recover_scaleinv(net, xh, d_sent, g_sent, steps=steps_rec), y)
            with torch.no_grad():
                put("T_B_inv_both", inv(g_sent, d_sent), y)
                put("T_b2_pred_xhat", net(A.norm(xh)).sigmoid(), y)
                put("T_b2_pred_x", net(A.norm(x01)).sigmoid(), y)
            res["train_meta"]["centres"] += [cen] * len(y); res["train_meta"]["loss_scale"].append(sc)
            ygt += y.flatten(1).mean(1).tolist()
            m.last = {}
        m.train(False)
    res["train_meta"]["gt_area"] = ygt
    lap("train")

if "noise" in only and is_split and not cfg["cut"]:
    for nu in [float(v) for v in a.nus.split(",")]:
        nn_ = load_net(gch=Channel(nu, 1.0, 0))                        # noise only, on g only, same weights
        sur_n = LL.surrogate_client(nn_, dl(aux, True), steps=20 if a.smoke else a.sur_steps, dev=dev)
        inv_n = LL.train_mask_inverter(sur_n, dl(aux, True), S, steps=20 if a.smoke else a.inv_steps, dev=dev)
        for x01, y, xh in batches[: a.noise_batches]:
            g, d = A.returned_gradient(nn_, x01, y)
            put(f"N{nu}_A_xhat", A.recover_masks(net, xh, d, g, steps=steps_rec), y)
            with torch.no_grad():
                put(f"N{nu}_B_inv_both", inv_n(g, d), y)
        lap(f"noise {nu}")

if "strong" in only:
    G3 = A.train_inverter(leak, dl(aux, True), S, epochs=2 if a.smoke else 3 * a.epochs, dev=dev)
    r = res.setdefault("strong_decoder", {"psnr": [], "ssim": []}); xs, xhs = [], []
    for x, _, _ in dl(tests):
        x01 = A.denorm(x.to(dev))
        with torch.no_grad():
            xh = G3(leak.noisy(x01)).clamp(0, 1)
        mk = A.fov_mask(x01)
        r["psnr"] += A.psnr(xh, x01, mk).tolist(); r["ssim"] += A.ssim(xh, x01, mask=mk).tolist(); xs.append(x01); xhs.append(xh)
    import torchvision
    fm = torchvision.models.resnet50(weights=None if a.smoke else "IMAGENET1K_V2").to(dev).eval(); fm.fc = torch.nn.Identity()
    with torch.no_grad():
        f = lambda t: F.normalize(fm(A.norm(t)), dim=1)
        q, gal = torch.cat([f(x) for x in xhs]), torch.cat([f(x) for x in xs])
    r["retrieval_top1"] = float(((q @ gal.T).argmax(1) == torch.arange(len(q), device=dev)).float().mean())
    lap("strong")

if "examples" in only:
    x01, y, xh = batches[0]
    th = lambda t: (F.interpolate(t[:4].float(), size=88, mode="area") * 255).round().byte().cpu().numpy().tolist()
    ex = {"x": th(x01), "xhat": th(xh), "y": th(y)}
    if "presence" in only:
        g, d = A.returned_gradient(net, x01[:4], y[:4])
        ex["A_xhat"] = th(A.recover_masks(net, xh[:4], d, g, steps=steps_rec))
        with torch.no_grad():
            ex["b2_pred_xhat"] = th(net(A.norm(xh[:4])).sigmoid())
    if sur is not None:
        g, d = A.returned_gradient(net, x01[:4], y[:4])
        with torch.no_grad():
            ex["B_inv_both"] = th(invs["both"](g, d))
    res["examples"] = ex

name = f"revision_{a.cut}_r{a.round}" + (f"_{a.tag}" if a.tag else "") + ".json"
json.dump(res, open(os.path.join(a.run, name), "w"))
print("wrote", os.path.join(a.run, name), flush=True)
