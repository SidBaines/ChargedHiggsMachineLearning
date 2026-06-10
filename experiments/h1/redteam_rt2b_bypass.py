"""RT2b: identify the bypass path left after clamping the two wires (thesis model).

RT2 arm 2: surgical truth-W mask + clamp nu's b1h3+b2h2 -> 14.3% of nu verdicts still
flip. Which scalars carry that? Clamp increasing sets of nu's scalars to clean values
under the same mask:
   wires | wires+b2h0+b2h3 | wires+all blk2 | wires+all blk0 | ALL 12 (machinery
   sanity: must be ~0 because nu's stream is then fully reconstructed clean).
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
ALL_HEADS = [(b, h) for b in range(3) for h in range(4)]
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

OV = {"active": False, "cfg": {}, "pos": None}
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

# clean pass: all 12 scalars at nu + baseline preds
msgs = {k: [] for k in ALL_HEADS}
pred_nu = []
for c0 in range(0, N, 2048):
    sl = slice(c0, min(c0 + 2048, N))
    with torch.no_grad():
        out = model(X[sl][..., :5], T[sl])
    bidx = torch.arange(out.shape[0])
    pred_nu.append(out[bidx, npos[sl]].argmax(-1))
    for k in ALL_HEADS:
        bna = cache[f"block_{k[0]}_attention"]["bottleneck_activation"]
        msgs[k].append(bna[bidx, k[1], npos[sl], 0])
pred_nu = torch.cat(pred_nu)
M = {k: torch.cat(v) for k, v in msgs.items()}

bsel = torch.where(boosted)[0]
wp = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
bar = torch.arange(len(bsel))
Xm = X[bsel].clone(); Tm = T[bsel].clone()
Xm[bar, wp] = 0.0; Tm[bar, wp] = PAD
base_n = pred_nu[bsel]

SETS = [
    ("no clamp", []),
    ("wires", WIRES),
    ("wires + b2h0 + b2h3", WIRES + [(2, 0), (2, 3)]),
    ("wires + all blk2", WIRES + [(2, 0), (2, 1), (2, 3)]),
    ("wires + all blk0", WIRES + [(0, h) for h in range(4)]),
    ("ALL 12 (sanity: ~0)", ALL_HEADS),
]
print(f"boosted n={len(bsel)}; surgical truth-W mask; clamp sets at NU (clean per-event values)")
print(f"  {'clamp set':28s} {'nu flip':>8s}")
for name, ks in SETS:
    flips = []
    for c0 in range(0, len(bsel), 2048):
        sl = slice(c0, min(c0 + 2048, len(bsel)))
        gsl = bsel[sl]
        if ks:
            OV["active"] = True
            OV["pos"] = npos[gsl]
            OV["cfg"] = {k: M[k][gsl].float() for k in ks}
        with torch.no_grad():
            out = model(Xm[sl][..., :5], Tm[sl])
        OV["active"] = False
        bidx = torch.arange(out.shape[0])
        flips.append(out[bidx, npos[gsl]].argmax(-1) != base_n[sl])
    print(f"  {name:28s} {torch.cat(flips).float().mean():8.4f}")

for h in handles: h.remove()
print("\ndone.")
