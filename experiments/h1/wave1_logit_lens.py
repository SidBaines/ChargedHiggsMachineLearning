"""H1 wave 1, D2: logit-lens timing — who knows the channel, when?

Captures the per-token residual stream at each depth (post-embedding, after each
block) and decodes it with the final classifier (no final LayerNorm -> exact lens).

Preregistration (H-A "ask the jets"): the truth-W jet's claim crystallizes EARLY
(possibly already post-embedding, since boosted candidacy ~ jet mass, a single-object
feature); the lep/nu verdict develops LATER (needs attention). H-B: simultaneous.
H-C: lep/nu first. Caveat: lens is miscalibrated at early depths; relative timing
across token types at the SAME depth is the readout, not absolute values.

Usage: .venv/bin/python tmp_h1_wave1_lens.py [--model ent1-d20-2blk] [--batches 6]
       [--ckpt-override /path/to/chkpt.pth]
"""
import os, sys, argparse
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
torch.set_num_threads(4)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser()
p.add_argument("--model", default="ent1-d20-2blk")
p.add_argument("--batches", type=int, default=6)
p.add_argument("--ckpt-override", default=None)
ARGS = p.parse_args()

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))
PAD, NU, LJ, SJ = 5, 2, 3, 4
W = 2
stds = np.ones(7); stds[:4] = 1e5
torch.manual_seed(0)

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)

BN = 1 if "bn1" in ARGS.model else None
model, _ = load_model(ARGS.model, checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False, checkpoint_override=ARGS.ckpt_override)
model.eval()
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=BN)]

# ---- residual-stream capture ----
# stream BEFORE block b = input (query arg) of block b's self_attention (pre-hook);
# stream after the LAST block = input of the classifier (pre-hook).
STREAMS = {}
def make_pre(name):
    def fn(module, args, kwargs):
        STREAMS[name] = args[0].detach()
    return fn
pre_handles = []
n_blocks = model.num_attention_blocks
for bidx, block in enumerate(model.attention_blocks):
    pre_handles.append(block["self_attention"].register_forward_pre_hook(
        make_pre(f"depth{bidx}"), with_kwargs=True))
pre_handles.append(model.classifier.register_forward_pre_hook(
    make_pre(f"depth{n_blocks}"), with_kwargs=True))

def fwd_with_streams(x, types):
    with torch.no_grad():
        out = model(x[..., :5], types)
        # snapshot first: calling classifier() below re-fires its pre-hook and would
        # overwrite the final-depth stream with whatever we pass in
        snap = {d: STREAMS[f"depth{d}"] for d in range(n_blocks + 1)}
        lens = {d: model.classifier(snap[d]) for d in range(n_blocks + 1)}
    return out, lens

Xs, Ts = [], []
for _ in range(ARGS.batches):
    b = next(dl)
    types = b["types"]
    ok = ((types == NU).sum(1) == 1) & (((types == 0) | (types == 1)).sum(1) == 1)
    Xs.append(b["x"][ok].clone()); Ts.append(types[ok].clone())
X = torch.cat(Xs); T = torch.cat(Ts); N = len(X)
TRU = torch.round(X[..., -1]).long()
ar = torch.arange(N)
npos = (T == NU).float().argmax(1); lpos = ((T == 0) | (T == 1)).float().argmax(1)
lvbb = (TRU[ar, lpos] == 3).numpy()
boosted = (~torch.from_numpy(lvbb)) & (((T == LJ) & (TRU == 2)).sum(1) == 1)
hpos = ((T == LJ) & (TRU == 1)).float().argmax(1)          # H-ljet (truth-1)
has_h = ((T == LJ) & (TRU == 1)).sum(1) >= 1
wpos = ((T == LJ) & (TRU == 2)).float().argmax(1)          # W-ljet (boosted only)

# chunked forward, accumulate lens margins at the tokens of interest
lens_lep = {d: [] for d in range(n_blocks + 1)}
lens_nu = {d: [] for d in range(n_blocks + 1)}
lens_h = {d: [] for d in range(n_blocks + 1)}
lens_w_claim = {d: [] for d in range(n_blocks + 1)}        # argmax==W of the W-ljet
lens_w_margin = {d: [] for d in range(n_blocks + 1)}
agree = {d: [] for d in range(n_blocks + 1)}               # lens argmax: nu==lep
for c in range(0, N, 2048):
    sl = slice(c, min(c + 2048, N))
    xx, tt = X[sl], T[sl]
    out, lens = fwd_with_streams(xx[..., :7], tt)          # fwd uses [...,:5] internally
    car = torch.arange(len(xx))
    for d in range(n_blocks + 1):
        L = lens[d]
        mg = L[..., W] - L[..., 0]
        lens_lep[d].append(mg[car, lpos[sl]])
        lens_nu[d].append(mg[car, npos[sl]])
        lens_h[d].append(mg[car, hpos[sl]])
        am = L.argmax(-1)
        agree[d].append((am[car, npos[sl]] == am[car, lpos[sl]]).float())
        bsel = boosted[sl]
        lens_w_claim[d].append((am[car, wpos[sl]] == W)[bsel].float())
        lens_w_margin[d].append(mg[car, wpos[sl]][bsel])

print(f"model={ARGS.model}  events={N}  boosted={boosted.sum().item()}  depths=0..{n_blocks}")
print(f"\n{'depth':>6s} {'P(Wjet claims W)':>17s} {'med Wjet margin':>16s} "
      f"{'AUC ch|lep':>10s} {'AUC ch|nu':>10s} {'AUC ch|Hjet':>11s} {'P(nu=lep)':>10s}")
for d in range(n_blocks + 1):
    wcl = torch.cat(lens_w_claim[d]); wmg = torch.cat(lens_w_margin[d])
    lp = torch.cat(lens_lep[d]).numpy(); nu_ = torch.cat(lens_nu[d]).numpy()
    hh = torch.cat(lens_h[d]).numpy(); ag = torch.cat(agree[d])
    auc_l = roc_auc_score(lvbb, lp); auc_n = roc_auc_score(lvbb, nu_)
    auc_h = roc_auc_score(lvbb[has_h.numpy()], hh[has_h.numpy()])
    lab = "embed" if d == 0 else f"blk{d-1}+"
    print(f"  {lab:>5s} {wcl.mean():17.4f} {wmg.median():16.2f} "
          f"{auc_l:10.4f} {auc_n:10.4f} {max(auc_h,1-auc_h):11.4f} {ag.mean():10.4f}")
print("\n(AUC ch|token = how well that token's lens W-margin separates true lvbb vs qqbb;")
print(" 0.5 = token doesn't know the channel yet. H-jet AUC folded to >=0.5.)")

for h in handles + pre_handles: h.remove()
print("done.")
