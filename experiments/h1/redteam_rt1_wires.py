"""RT1 (+RT8): red-team the D4-lite "two wires" causal certificate (thesis model).

On a STRICTLY UNSEEN val slice (batches 13-18; waves used 1-12):
  - replicate the round-2e scalar census (AUCs of all 12 nu-scalars);
  - reproduce the wires-swap flip rate;
  - specificity controls never run in wave 2/3:
      * placebo: overwrite wires with OWN-class means (should do ~nothing);
      * swap each of the 12 scalars individually to opposite-class means;
      * swap the 3 non-wire block-2 scalars jointly;
      * swap ALL 10 non-wire scalars while CLAMPING the wires to their clean values
        (their account predicts ~no flips: verdict fully determined by the wires);
      * swap all 12;
      * norm-matched random-direction kick at nu (same L2 as the wire-swap delta,
        same blocks/positions) -- "is it just a big kick?" control;
      * norm-matched kick along NON-wire up-projection directions (b1h2, b2h3).
  - RT8: B1 XOR independence-null per stratum (how much of "XOR holds" is base rates).

All flip rates get 95% Wilson CIs.
"""
import os, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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
NONWIRES = [k for k in ALL_HEADS if k not in WIRES]
SKIP_BATCHES = 12
N_BATCHES = 6
torch.manual_seed(0)

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in range(510115, 510125)},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)
for _ in range(SKIP_BATCHES):
    next(dl)

model, _ = load_model("thesis-ent1-bn1-d152", checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=1)]
attn_modules = [blk["self_attention"] for blk in model.attention_blocks]
D = attn_modules[0].embed_dim

# override machinery: scalar overwrites per (blk,head) + raw vector kicks per blk
OV = {"active": False, "cfg": {}, "kick": {}, "pos": None}
def make_override(blk):
    def hook(module, args, kwargs, output):
        if not OV["active"]:
            return output
        heads = [(b, h) for (b, h) in OV["cfg"] if b == blk]
        kick = OV["kick"].get(blk)
        if not heads and kick is None:
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
        if kick is not None:
            out[bidx, pos] = out[bidx, pos] + kick
        return (out, w)
    return hook
for bi, m in enumerate(attn_modules):
    handles.append(m.register_forward_hook(make_override(bi), with_kwargs=True))

Xs, Ts = [], []
for _ in range(N_BATCHES):
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
resolved = (~lvbb) & (((T == SJ) & (TRU == 2)).sum(1) == 2)
jet_m = (T == LJ) | (T == SJ)
print(f"slice C (batches 13-18): events={N} (lvbb {lvbb.sum()}, boosted {boosted.sum()}, resolved {resolved.sum()})")

def wilson(k, n):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - hw), min(1.0, c + hw)

# ---------------- pass 1: baseline ----------------
msgs = {k: [] for k in ALL_HEADS}
pred_all = []
for c0 in range(0, N, 2048):
    sl = slice(c0, min(c0 + 2048, N))
    with torch.no_grad():
        out = model(X[sl][..., :5], T[sl])
    B = out.shape[0]; bidx = torch.arange(B)
    pred_all.append(out.argmax(-1))
    for (blk, h) in ALL_HEADS:
        bna = cache[f"block_{blk}_attention"]["bottleneck_activation"]
        msgs[(blk, h)].append(bna[bidx, h, npos[sl], 0])
pred_all = torch.cat(pred_all)
pred_nu = pred_all[ar, npos]; pred_lep = pred_all[ar, lpos]
M = {k: torch.cat(v) for k, v in msgs.items()}
pn = (pred_nu == W); pl = (pred_lep == W)
lv = lvbb.numpy()

print("\n=== census replication on UNSEEN slice (AUC vs truth / vs pred_nu) ===")
print("    [round-2e logged on batches 1-12: b2h2 0.987/0.9963, b1h3 0.978/0.9892]")
for blk in range(3):
    row = []
    for h in range(4):
        v = M[(blk, h)].numpy()
        a_t = roc_auc_score(lv, v); a_p = roc_auc_score(pn.numpy(), v)
        row.append(f"h{h}: {max(a_t,1-a_t):.4f}/{max(a_p,1-a_p):.4f}")
    print(f"  b{blk}  " + "   ".join(row))

# ---------------- RT8: XOR null ----------------
jet_claims = ((pred_all == W) & jet_m).sum(1) > 0
print("\n=== RT8: B1 XOR vs independence null (per stratum) ===")
for lab, m in (("lvbb", lvbb), ("boosted", boosted), ("resolved", resolved)):
    a = jet_claims[m].float(); l = (pred_lep[m] == W).float()
    xor = 1 - (a * l).mean() - ((1 - a) * (1 - l)).mean()
    pa, plep = a.mean(), l.mean()
    null = (pa * (1 - plep) + (1 - pa) * plep).item()
    # phi coefficient
    n11 = (a * l).mean(); n10 = (a * (1 - l)).mean(); n01 = ((1 - a) * l).mean(); n00 = ((1 - a) * (1 - l)).mean()
    den = torch.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) + 1e-12
    phi = (n11 * n00 - n10 * n01) / den
    print(f"  {lab:9s} XOR-ok={xor:.4f}  independence-null={null:.4f}  excess={xor-null:+.4f}  phi={phi:+.3f}")

# ---------------- D4-lite configs ----------------
mean_W = {k: M[k][pn].mean() for k in ALL_HEADS}
mean_N = {k: M[k][~pn].mean() for k in ALL_HEADS}
def swap_t(k):   return torch.where(pn, mean_N[k], mean_W[k]).float()
def same_t(k):   return torch.where(pn, mean_W[k], mean_N[k]).float()
def clamp_t(k):  return M[k].float()  # per-event clean value

CONFIGS = []
CONFIGS.append(("placebo own-class means (wires)", {k: same_t(k) for k in WIRES}, None))
CONFIGS.append(("swap wires [b1h3+b2h2]", {k: swap_t(k) for k in WIRES}, None))
for k in ALL_HEADS:
    tag = "WIRE" if k in WIRES else "ctrl"
    CONFIGS.append((f"swap b{k[0]}h{k[1]} alone [{tag}]", {k: swap_t(k)}, None))
CONFIGS.append(("swap blk2 non-wires [b2h0+b2h1+b2h3]", {k: swap_t(k) for k in [(2,0),(2,1),(2,3)]}, None))
CONFIGS.append(("swap ALL 10 non-wires, wires CLAMPED clean",
                {**{k: swap_t(k) for k in NONWIRES}, **{k: clamp_t(k) for k in WIRES}}, None))
CONFIGS.append(("swap ALL 12", {k: swap_t(k) for k in ALL_HEADS}, None))

# norm-matched kicks: per-event delta norms of the wire swap, per block
wup = {k: model.attention_blocks[k[0]]["bottleneck_up"][k[1]].weight[:, 0].detach() for k in ALL_HEADS}
dnorm = {k: ((swap_t(k) - M[k].float()).abs() * wup[k].norm()) for k in WIRES}  # [N]
g = torch.Generator().manual_seed(7)
rand_dir = {blk: torch.randn(N, D, generator=g) for blk in (1, 2)}
for blk in rand_dir:
    rand_dir[blk] = rand_dir[blk] / rand_dir[blk].norm(dim=-1, keepdim=True)
kick_rand = {1: rand_dir[1] * dnorm[(1, 3)].unsqueeze(-1), 2: rand_dir[2] * dnorm[(2, 2)].unsqueeze(-1)}
# kick along non-wire up directions, matched per-block norm, sign aligned with the swap's
udir = {k: (wup[k] / wup[k].norm()) for k in [(1, 2), (2, 3)]}
sgn = {k: torch.sign(swap_t(kw) - M[kw].float()) for k, kw in [((1, 2), (1, 3)), ((2, 3), (2, 2))]}
kick_nonwire = {1: udir[(1, 2)].unsqueeze(0) * (dnorm[(1, 3)] * sgn[(1, 2)]).unsqueeze(-1),
                2: udir[(2, 3)].unsqueeze(0) * (dnorm[(2, 2)] * sgn[(2, 3)]).unsqueeze(-1)}
CONFIGS.append(("norm-matched RANDOM-dir kick @blk1+2", {}, kick_rand))
CONFIGS.append(("norm-matched kick along b1h2+b2h3 dirs", {}, kick_nonwire))

print("\n=== RT1: overwrite configs at the NU position (flip rates vs baseline, 95% Wilson CI) ===")
print(f"  {'config':44s} {'nu flip (all)':>20s} {'nu lvbb':>8s} {'nu qqbb':>8s} {'lep flip':>9s}")
for name, cfg, kick in CONFIGS:
    flips_nu, flips_lep = [], []
    for c0 in range(0, N, 2048):
        sl = slice(c0, min(c0 + 2048, N))
        OV["active"] = True
        OV["pos"] = npos[sl]
        OV["cfg"] = {k: t[sl] for k, t in cfg.items()}
        OV["kick"] = {blk: kv[sl] for blk, kv in (kick or {}).items()}
        with torch.no_grad():
            out = model(X[sl][..., :5], T[sl])
        OV["active"] = False
        bidx = torch.arange(out.shape[0])
        flips_nu.append(out[bidx, npos[sl]].argmax(-1) != pred_nu[sl])
        flips_lep.append(out[bidx, lpos[sl]].argmax(-1) != pred_lep[sl])
    f_n = torch.cat(flips_nu); f_l = torch.cat(flips_lep)
    p, lo, hi = wilson(f_n.sum().item(), N)
    print(f"  {name:44s} {p:7.4f} [{lo:.4f},{hi:.4f}] {f_n[lvbb].float().mean():8.4f} "
          f"{f_n[~lvbb].float().mean():8.4f} {f_l.float().mean():9.4f}")

for h in handles: h.remove()
print("\ndone.")
