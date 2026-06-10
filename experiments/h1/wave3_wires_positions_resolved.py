"""H1 wave 3: D4-lite at the LEPTON position (parallel-readers test, causal both ways)
+ A3-lite resolved-stratum battery (thesis model).

Prereg: (i) swapping the b1h3+b2h2 scalars at lep flips LEP ~99% while nu holds (and
swapping at both positions flips both together, restoring the lockstep); (ii) resolved
events: the verdict heads' attention from lep/nu spreads over the W-sjet PAIR but less
cleanly than the boosted W-ljet; the wires' qqbb-side value in resolved events is
carried by the W-sjets.
"""
import os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
torch.set_num_threads(4)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
PAD, NU, LJ, SJ = 5, 2, 3, 4
W = 2
stds = np.ones(7); stds[:4] = 1e5
WIRES = [(1, 3), (2, 2)]

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in range(510115, 510125)},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)

model, _ = load_model("thesis-ent1-bn1-d152", checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=1)]
attn_modules = [blk["self_attention"] for blk in model.attention_blocks]
D = attn_modules[0].embed_dim; NH = attn_modules[0].num_heads; DH = D // NH

OV = {"active": False, "cfg": {}}     # cfg: (blk,h) -> list of (pos_tensor, target_tensor)
def make_override(blk):
    def hook(module, args, kwargs, output):
        if not OV["active"]:
            return output
        items = [(k, v) for k, v in OV["cfg"].items() if k[0] == blk]
        if not items:
            return output
        out, w = output
        out = out.clone()
        bidx = torch.arange(out.shape[0])
        for (b, h), entries in items:
            w_up = model.attention_blocks[blk]["bottleneck_up"][h].weight[:, 0]
            for pos, tgt in entries:
                s_cur = cache[f"block_{blk}_attention"]["bottleneck_activation"][bidx, h, pos, 0]
                out[bidx, pos] = out[bidx, pos] + (tgt - s_cur).unsqueeze(-1) * w_up
        return (out, w)
    return hook
for bi, m in enumerate(attn_modules):
    handles.append(m.register_forward_hook(make_override(bi), with_kwargs=True))

Xs, Ts = [], []
for _ in range(6):
    b = next(dl)
    types = b["types"]
    ok = ((types == NU).sum(1) == 1) & (((types == 0) | (types == 1)).sum(1) == 1)
    Xs.append(b["x"][ok].clone()); Ts.append(types[ok].clone())
X = torch.cat(Xs); T = torch.cat(Ts); N = len(X)
TRU = torch.round(X[..., -1]).long()
ar = torch.arange(N)
npos = (T == NU).float().argmax(1); lpos = ((T == 0) | (T == 1)).float().argmax(1)
lvbb = TRU[ar, lpos] == 3
resolved = (~lvbb) & (((T == SJ) & (TRU == 2)).sum(1) == 2)
print(f"events={N} (lvbb {lvbb.sum()}, resolved {resolved.sum()})")

# ---- pass 1: baseline preds + scalars at lep/nu + resolved attention/decomposition
msgs = {("nu",) + k: [] for k in WIRES} | {("lep",) + k: [] for k in WIRES}
pred_nu, pred_lep = [], []
# resolved: attention of nu to {W-sjet pair (sum), H-ljet, other}, per verdict head
res_attn = {k: [] for k in WIRES + [(2, 3), (2, 0)]}
res_contrib = {k: {c: [] for c in ("W-sjets", "H-ljet", "other-jets", "lepton", "nu-self")} for k in WIRES}
def sval(blk, h):
    x_in = cache[f"block_{blk}_attention"]["input"][0]
    attn = attn_modules[blk]
    Wv = attn.in_proj_weight[2 * D:3 * D]; bv = attn.in_proj_bias[2 * D:3 * D]
    v = x_in @ Wv.t() + bv
    v_h = v[..., h * DH:(h + 1) * DH]
    Wout_h = attn.out_proj.weight[:, h * DH:(h + 1) * DH]
    dw = model.attention_blocks[blk]["bottleneck_down"][h].weight[0]
    return (v_h @ Wout_h.t()) @ dw          # [B,N]

for c0 in range(0, N, 2048):
    sl = slice(c0, min(c0 + 2048, N))
    xx, tt = X[sl], T[sl]
    with torch.no_grad():
        out = model(xx[..., :5], tt)
    B = len(xx); bidx = torch.arange(B)
    np_, lp_ = npos[sl], lpos[sl]
    pr = out.argmax(-1)
    pred_nu.append(pr[bidx, np_]); pred_lep.append(pr[bidx, lp_])
    tru_ = TRU[sl]
    wsj = (tt == SJ) & (tru_ == 2)
    hlj = (tt == LJ) & (tru_ == 1)
    oth = ((tt == LJ) | (tt == SJ)) & ~wsj & ~hlj
    lepm = (tt == 0) | (tt == 1); num = tt == NU
    for k in WIRES:
        bna = cache[f"block_{k[0]}_attention"]["bottleneck_activation"]
        msgs[("nu",) + k].append(bna[bidx, k[1], np_, 0])
        msgs[("lep",) + k].append(bna[bidx, k[1], lp_, 0])
        sv = sval(*k)
        aw = cache[f"block_{k[0]}_attention"]["attn_weights_per_head"][:, k[1]]
        per_key = aw[bidx, np_] * sv
        for cname, mm in (("W-sjets", wsj), ("H-ljet", hlj), ("other-jets", oth),
                          ("lepton", lepm), ("nu-self", num)):
            res_contrib[k][cname].append((per_key * mm).sum(1))
    for k in WIRES + [(2, 3), (2, 0)]:
        aw = cache[f"block_{k[0]}_attention"]["attn_weights_per_head"][:, k[1]]
        a_nu = aw[bidx, np_]
        res_attn[k].append(torch.stack([(a_nu * wsj).sum(1), (a_nu * hlj).sum(1),
                                        (a_nu * oth).sum(1)], -1))
pred_nu = torch.cat(pred_nu); pred_lep = torch.cat(pred_lep)
M = {k: torch.cat(v) for k, v in msgs.items()}
pn = (pred_nu == W); pl = (pred_lep == W)

# ---- D4-lite at the lepton (and both) positions ----
mean_W = {}; mean_N = {}
for k in WIRES:
    mean_W[("nu",) + k] = M[("nu",) + k][pn].mean(); mean_N[("nu",) + k] = M[("nu",) + k][~pn].mean()
    mean_W[("lep",) + k] = M[("lep",) + k][pl].mean(); mean_N[("lep",) + k] = M[("lep",) + k][~pl].mean()

def swap_targets(tok, sl):
    base = pn if tok == "nu" else pl
    pos = npos if tok == "nu" else lpos
    return [(pos[sl], torch.where(base[sl], mean_N[(tok,) + k], mean_W[(tok,) + k]).float())
            for k in WIRES]   # aligned with WIRES order

CONFIGS = [("swap wires @ nu", ("nu",)), ("swap wires @ lep", ("lep",)),
           ("swap wires @ both", ("nu", "lep"))]
print("\n=== D4-lite by position [prereg: each position flips ~99% independently; both => lockstep restored] ===")
print(f"  {'config':20s} {'nu flip':>8s} {'lep flip':>9s} {'P(nu=lep)':>10s}")
for name, toks in CONFIGS:
    f_n, f_l, ag = [], [], []
    for c0 in range(0, N, 2048):
        sl = slice(c0, min(c0 + 2048, N))
        OV["cfg"] = {}
        for tok in toks:
            tg = swap_targets(tok, sl)
            for k, entry in zip(WIRES, tg):
                OV["cfg"].setdefault(k, []).append(entry)
        OV["active"] = True
        with torch.no_grad():
            out = model(X[sl][..., :5], T[sl])
        OV["active"] = False
        bidx = torch.arange(len(out))
        pr_n = out[bidx, npos[sl]].argmax(-1); pr_l = out[bidx, lpos[sl]].argmax(-1)
        f_n.append(pr_n != pred_nu[sl]); f_l.append(pr_l != pred_lep[sl]); ag.append(pr_n == pr_l)
    print(f"  {name:20s} {torch.cat(f_n).float().mean():8.4f} {torch.cat(f_l).float().mean():9.4f} "
          f"{torch.cat(ag).float().mean():10.4f}")

# ---- A3-lite: resolved stratum ----
res = resolved
print(f"\n=== A3-lite: resolved stratum (n={res.sum()}) ===")
print("  nu attention mass on {W-sjet pair | H-ljet | other jets} per verdict head:")
for k in WIRES + [(2, 3), (2, 0)]:
    a = torch.cat(res_attn[k])[res]
    print(f"    b{k[0]}h{k[1]}: {a[:,0].mean():.3f} | {a[:,1].mean():.3f} | {a[:,2].mean():.3f}")
print("  per-sender contribution to nu's wire scalars (resolved vs lvbb):")
for k in WIRES:
    print(f"    b{k[0]}h{k[1]}:")
    for cname in ("W-sjets", "H-ljet", "other-jets", "lepton", "nu-self"):
        v = torch.cat(res_contrib[k][cname])
        print(f"      {cname:>10s}  resolved {v[res].mean():+8.3f}   lvbb {v[lvbb].mean():+8.3f}")

for hh in handles: hh.remove()
print("\ndone.")
