"""Exact cost of each cut (no training): client parameters, forward MACs, traffic per
image in each direction, total traffic of one training run, and time per round.
Units are decimal (MB = 1e6 bytes, GB = 1e9 bytes). Writes tables/cost.tex and cost.json."""
import json, math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "code"))
import torch
from crcsfl.model import build, n_params, count_macs

W, S, R, N, B = 0.5, 352, 30, 200, 8      # 200 = 25 full batches = mean over C1-C5 train splits (drop_last)      # width, input size, rounds, images/client, batch
BW, RTT = 100e6, 0.02                      # 100 Mb/s link, 20 ms round-trip time
x = torch.zeros(1, 3, S, S)
full, _ = build("full", W)
P, F = n_params(full), count_macs(full, [full], x)
steps = N // B                   # mini-batches per round
rows = []


@torch.no_grad()
def act_numel(fn, modules):
    """Values stored for the backward pass: outputs of every leaf module of `modules` (per image)."""
    tot = [0]
    leaves = [m for mod in modules for m in mod.modules() if len(list(m.children())) == 0]
    hs = [m.register_forward_hook(lambda m_, i, o: tot.__setitem__(0, tot[0] + o.numel())) for m in leaves]
    fn(x)
    for h in hs:
        h.remove()
    return tot[0]


def train_mem(params, acts):
    """GB of training memory at batch B: weights, gradients, two AdamW moments (4 x fp32) + activations."""
    return (16 * params + 4 * B * acts) / 1e9


def add(name, params, macs, up, down, exchanges, mem=0.0):
    """up/down: bytes per image per local step; exchanges: sequential round trips per mini-batch."""
    run = (up + down) * N * R + 2 * params * 4 * R               # + client-part averaging
    t = (up + down) * N / BW * 8 + 2 * params * 4 * 8 / BW + steps * exchanges * RTT
    rows.append(dict(name=name, params=params, macs=macs, up=up / 1e6, down=down / 1e6, run=run / 1e9, round_min=t / 60, mem=mem))


rows.append(dict(name="FedAvg (whole network)", params=P, macs=F, up=0, down=0,
                 run=2 * P * 4 * R / 1e9, round_min=(2 * P * 4 * 8 / BW + 2 * RTT) / 60,
                 mem=train_mem(P, act_numel(full, [full]))))
cv, _ = build("vanilla", W)
v = sum(t.numel() for t in cv.encoder(x))
add("$V$: encoder cut, loss at server", n_params(cv), count_macs(lambda t: cv.encoder(t), [cv], x),
    v * 4 + S * S, v * 4, 1, train_mem(n_params(cv), act_numel(lambda t: cv.encoder(t), [cv])))
ch, sh = build("skip", W)                                       # encoder + D1 + head at client
z = ch.encoder(x); vh = sum(t.numel() for t in z[1:])
d2 = sh({i + 1: t for i, t in enumerate(z)}).numel()
enc_macs = count_macs(lambda t: ch.encoder(t), [ch], x)
both = count_macs(lambda t: ch.tail(sh({i + 1: u for i, u in enumerate(ch.encoder(t))}), ch.encoder(t)[0]), [ch], x)
add("$V_h$: encoder cut, head at client", n_params(ch), both - enc_macs,  # lambda runs the encoder twice
    (vh + d2) * 4, (vh + d2) * 4, 2,
    train_mem(n_params(ch), act_numel(lambda t: ch.tail(sh({i + 1: u for i, u in enumerate(ch.encoder(t))}), ch.encoder(t)[0]), [ch.tail])
              + act_numel(lambda t: ch.encoder(t), [ch])))
for c in (1, 2, 3, 4):
    cl, sv = build("ucut", W, c=c)
    sk = cl.encode(x); d = sv(sk[-1]); zc, dd = sk[-1].numel(), d.numel()
    add(f"$U_{c}$", n_params(cl),
        count_macs(lambda t: (lambda s_: cl.decode(sv(s_[-1]), s_))(cl.encode(t)), [cl], x),
        (zc + dd) * 4, (zc + dd) * 4, 2,
        train_mem(n_params(cl), act_numel(lambda t: (lambda s_: cl.decode(sv(s_[-1]), s_))(cl.encode(t)), [cl])))
out = pathlib.Path(__file__).resolve().parents[1] / "tables"
json.dump(dict(width=W, size=S, rounds=R, images=N, batch=B, rows=rows, full_params=P, full_macs=F),
          open(out / "cost.json", "w"), indent=1)
L = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
     r"Client part & \makecell{Params\\(M)} & \makecell{GMAC\\per image} & \makecell{Train\\mem.\ (GB)} & \makecell{Up\\(MB/img)} & \makecell{Down\\(MB/img)} & \makecell{Run\\(GB)} & \makecell{Round\\(min)} \\", r"\midrule"]
for r in rows:
    pct = f" ({r['macs'] / F * 100:.0f}\\%)"
    up = "--" if r["up"] == 0 else f"{r['up']:.1f}"
    dn = "--" if r["down"] == 0 else f"{r['down']:.1f}"
    L.append(f"{r['name']} & {r['params'] / 1e6:.2f} & {r['macs'] / 1e9:.1f}{pct} & {r['mem']:.1f} & {up} & {dn} & {r['run']:.0f} & {r['round_min']:.1f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
(out / "cost.tex").write_text("\n".join(L) + "\n")
print("\n".join(L))
