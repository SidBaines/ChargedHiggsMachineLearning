"""RT2: sufficiency direction of the wire claim (thesis model, unseen slice C).

The waves showed: overwriting nu's two scalars flips its verdict (necessity of the
wires for the BASELINE verdict). Untested direction: when the verdict is flipped from
the INPUT side, does it flow THROUGH those two scalars? Clamp nu's b1h3+b2h2 to their
clean-run per-event values while applying input interventions that flip the verdict:

  arm 1: boosted qqbb, surgical mask of the truth-W ljet      (round-2c: ~77% flips)
  arm 2: arm 1 + clamp nu wires to clean                       (account: nu flips ~<5%)
  arm 3: arm 1 + clamp LEP wires to clean (nu free)            (symmetric control)
  arm 4: lvbb, insert real W-ljet copy                         (A4: lep 0.97->0.65)
  arm 5: arm 4 + clamp nu wires to clean

Restoration rate = among events the intervention alone flips, fraction un-flipped by
the clamp. Their circuit account predicts high restoration; low restoration would mean
major verdict paths into nu BYPASS the two wires.
"""
import os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
torch.set_num_threads(6)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
PAD, NU, LJ, SJ = 5, 2, 3, 4
W = 2
stds = np.ones(7); stds[:4] = 1e5
WIRES = [(1, 3), (2, 2)]
torch.manual_seed(0)

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in range(510115, 510125)},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)
for _ in range(12):
    next(dl)

model, _ = load_model("thesis-ent1-bn1-d152", checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=1)]
attn_modules = [blk["self_attention"] for blk in model.attention_blocks]

OV = {"active": False, "cfg": {}, "pos": None}   # cfg: (blk,h) -> target [B]
def make_override(blk):
    def hook(module, args, kwargs, output):
        if not OV["active"]:
            return output
        heads = [(b, h) for (b, h) in OV["cfg"] if b == blk]
        if not heads:
            return output
        out, w = output
        out = out.clone()
        bidx = torch.arange(out.shape[0])
        pos = OV["pos"]
        for (b, h) in heads:
            tgt = OV["cfg"][(b, h)]
            s_cur = cache[f"block_{blk}_attention"]["bottleneck_activation"][bidx, h, pos, 0]
            w_up = model.attention_blocks[blk]["bottleneck_up"][h].weight[:, 0]
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
boosted = (~lvbb) & (((T == LJ) & (TRU == 2)).sum(1) == 1)
print(f"slice C: events={N} (lvbb {lvbb.sum()}, boosted {boosted.sum()})")

def run(X_, T_, pos_n, pos_l, clamp=None):
    """clamp: None | ('nu', dict k->target[B]) | ('lep', ...). Returns preds at nu, lep."""
    pn_, pl_ = [], []
    for c0 in range(0, len(X_), 2048):
        sl = slice(c0, min(c0 + 2048, len(X_)))
        if clamp is not None:
            tok, tgts = clamp
            OV["active"] = True
            OV["pos"] = (pos_n if tok == "nu" else pos_l)[sl]
            OV["cfg"] = {k: t[sl] for k, t in tgts.items()}
        with torch.no_grad():
            out = model(X_[sl][..., :5], T_[sl])
        OV["active"] = False
        bidx = torch.arange(out.shape[0])
        pn_.append(out[bidx, pos_n[sl]].argmax(-1)); pl_.append(out[bidx, pos_l[sl]].argmax(-1))
    return torch.cat(pn_), torch.cat(pl_)

# clean pass: store nu & lep wire scalars + preds
clean = {("nu",) + k: [] for k in WIRES} | {("lep",) + k: [] for k in WIRES}
pred_nu, pred_lep = [], []
for c0 in range(0, N, 2048):
    sl = slice(c0, min(c0 + 2048, N))
    with torch.no_grad():
        out = model(X[sl][..., :5], T[sl])
    bidx = torch.arange(out.shape[0])
    pred_nu.append(out[bidx, npos[sl]].argmax(-1)); pred_lep.append(out[bidx, lpos[sl]].argmax(-1))
    for k in WIRES:
        bna = cache[f"block_{k[0]}_attention"]["bottleneck_activation"]
        clean[("nu",) + k].append(bna[bidx, k[1], npos[sl], 0])
        clean[("lep",) + k].append(bna[bidx, k[1], lpos[sl], 0])
pred_nu = torch.cat(pred_nu); pred_lep = torch.cat(pred_lep)
CL = {k: torch.cat(v) for k, v in clean.items()}

def report(name, base_n, base_l, new_n, new_l, ref_flip_n=None):
    f_n = (new_n != base_n); f_l = (new_l != base_l)
    line = f"  {name:46s} nu flip={f_n.float().mean():.4f}  lep flip={f_l.float().mean():.4f}"
    if ref_flip_n is not None:
        restored = (~f_n[ref_flip_n]).float().mean()
        line += f"  restoration(nu)={restored:.4f} (of n={ref_flip_n.sum().item()})"
    print(line)
    return f_n, f_l

# ---------------- arms 1-3: surgical truth-W mask in boosted ----------------
bsel = torch.where(boosted)[0]
wp = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
Xm = X[bsel].clone(); Tm = T[bsel].clone()
bar = torch.arange(len(bsel))
Xm[bar, wp] = 0.0; Tm[bar, wp] = PAD
bn, bl = pred_nu[bsel], pred_lep[bsel]
pn_, pl_ = run(Xm, Tm, npos[bsel], lpos[bsel])
print("\n=== boosted qqbb: surgical truth-W-ljet mask (n=%d) ===" % len(bsel))
f1n, f1l = report("arm1: mask only", bn, bl, pn_, pl_)
nu_clamp = {k: CL[("nu",) + k][bsel] for k in WIRES}
pn_, pl_ = run(Xm, Tm, npos[bsel], lpos[bsel], clamp=("nu", nu_clamp))
report("arm2: mask + clamp NU wires clean", bn, bl, pn_, pl_, ref_flip_n=f1n)
lep_clamp = {k: CL[("lep",) + k][bsel] for k in WIRES}
pn_, pl_ = run(Xm, Tm, npos[bsel], lpos[bsel], clamp=("lep", lep_clamp))
f3n, f3l = report("arm3: mask + clamp LEP wires clean (nu free)", bn, bl, pn_, pl_)
print(f"        arm3 lep restoration: {(~(f3l[f1l])).float().mean():.4f} (of n={f1l.sum().item()})")

# ---------------- arms 4-5: real-W insertion into lvbb ----------------
rng = np.random.default_rng(0)
lsel = torch.where(lvbb & ((T == PAD).sum(1) >= 1))[0][:4000]
bo_all = torch.where(boosted)[0]
wrow_of = ((T[bo_all] == LJ) & (TRU[bo_all] == 2)).float().argmax(1)
j = rng.integers(len(bo_all), size=len(lsel))
rows = X[bo_all[j], wrow_of[j]].clone(); rows[:, 5] = 0; rows[:, -1] = 0
Xi = X[lsel].clone(); Ti = T[lsel].clone()
slot = (Ti == PAD).float().argmax(1)
lar = torch.arange(len(lsel))
Xi[lar, slot] = rows; Ti[lar, slot] = LJ
bn, bl = pred_nu[lsel], pred_lep[lsel]
pn_, pl_ = run(Xi, Ti, npos[lsel], lpos[lsel])
print("\n=== lvbb: real W-ljet inserted (n=%d) ===" % len(lsel))
f4n, f4l = report("arm4: insert only", bn, bl, pn_, pl_)
nu_clamp = {k: CL[("nu",) + k][lsel] for k in WIRES}
pn_, pl_ = run(Xi, Ti, npos[lsel], lpos[lsel], clamp=("nu", nu_clamp))
report("arm5: insert + clamp NU wires clean", bn, bl, pn_, pl_, ref_flip_n=f4n)

for h in handles: h.remove()
print("\ndone.")
