"""H1 wave 2, wires half (thesis model): C1 attention contrast (lep/nu -> W-ljet vs
H-ljet), C4 completion (scalar census into LEP), per-key decomposition of the verdict
messages (b1h3, b2h2) into nu, and D4-lite: overwrite/zero those scalars at the nu
position and watch the verdict.

Decomposition uses the hook conventions of interp/activations.py:
  bottleneck_activation[i,h,q,0] = down_w_h . ( sum_k a_h[q,k] * VO_h[k] ),
  VO_h[k] = (x_k @ Wv_h^T + bv_h) @ Wout_h  (out_proj bias excluded; down/up biases
  never applied by the hook). So per-key contribution = a_h[q,k] * s_val_h[k] with
  s_val_h[k] = down_w_h . VO_h[k]; sums are checked against the cache.

Preregistrations: (i) in boosted qqbb, the W-ljet dominates the qqbb-side value of
nu's b1h3/b2h2 scalars; (ii) overwriting BOTH wires at nu flips nu's verdict where
round-1 zero-ablations didn't, and BREAKS P(nu=lep) (lep untouched) -- demonstrating
the lockstep is 'parallel same evidence', not 'nu reads lep'.
"""
import os, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
torch.set_num_threads(4)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))
PAD, NU, LJ, SJ = 5, 2, 3, 4
W = 2
N_BATCHES = 6
stds = np.ones(7); stds[:4] = 1e5
WIRES = [(1, 3), (2, 2)]
EXTRA = [(2, 0), (2, 3)]

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
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
D = model.attention_blocks[0]["self_attention"].embed_dim
NH = model.attention_blocks[0]["self_attention"].num_heads
DH = D // NH

# ---- override machinery (registered AFTER the bottleneck hooks -> sees their output)
OV = {"active": False, "cfg": {}, "nupos": None}
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
        nupos = OV["nupos"]
        for (b, h) in heads:
            tgt = OV["cfg"][(b, h)]
            s_cur = cache[f"block_{blk}_attention"]["bottleneck_activation"][bidx, h, nupos, 0]
            w_up = model.attention_blocks[blk]["bottleneck_up"][h].weight[:, 0]
            out[bidx, nupos] = out[bidx, nupos] + (tgt - s_cur).unsqueeze(-1) * w_up
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
wpos = ((T == LJ) & (TRU == 2)).float().argmax(1)
has_h = ((T == LJ) & (TRU == 1)).sum(1) >= 1
hpos = ((T == LJ) & (TRU == 1)).float().argmax(1)
print(f"events={N} (lvbb {lvbb.sum()}, boosted {boosted.sum()})")

def sval_table(blk):
    """s_val_h[k] for all heads of a block, from cached input. [B,N] per head."""
    x_in = cache[f"block_{blk}_attention"]["input"][0]          # [B,N,D]
    attn = attn_modules[blk]
    Wv = attn.in_proj_weight[2 * D:3 * D]; bv = attn.in_proj_bias[2 * D:3 * D]
    v = x_in @ Wv.t() + bv                                      # [B,N,D]
    out = {}
    for h in range(NH):
        v_h = v[..., h * DH:(h + 1) * DH]
        Wout_h = attn.out_proj.weight[:, h * DH:(h + 1) * DH]   # [D,DH]
        VO = v_h @ Wout_h.t()                                   # [B,N,D]
        dw = model.attention_blocks[blk]["bottleneck_down"][h].weight[0]  # [D]
        out[h] = VO @ dw                                        # [B,N]
    return out

# ---------------- pass 1: predictions, scalars, C1, decomposition ----------------
msgs_nu = {(b, h): [] for b in range(3) for h in range(4)}
msgs_lep = {(b, h): [] for b in range(3) for h in range(4)}
pred_nu, pred_lep = [], []
c1 = {(tok, b, h): [] for tok in ("lep", "nu") for b in range(3) for h in range(4)}
contrib = {k: {} for k in WIRES}    # (blk,h) -> category -> list of per-event sums
CATS = ("W-ljet", "H-ljet", "other-jets", "lepton", "nu-self")
for k in WIRES:
    for c in CATS: contrib[k][c] = []
sanity_max = 0.0
for c0 in range(0, N, 2048):
    sl = slice(c0, min(c0 + 2048, N))
    xx, tt = X[sl], T[sl]
    with torch.no_grad():
        out = model(xx[..., :5], tt)
    B = len(xx); bidx = torch.arange(B)
    np_, lp_ = npos[sl], lpos[sl]
    pr = out.argmax(-1)
    pred_nu.append(pr[bidx, np_]); pred_lep.append(pr[bidx, lp_])
    for blk in range(3):
        bna = cache[f"block_{blk}_attention"]["bottleneck_activation"]
        aw = cache[f"block_{blk}_attention"]["attn_weights_per_head"]
        for h in range(4):
            msgs_nu[(blk, h)].append(bna[bidx, h, np_, 0])
            msgs_lep[(blk, h)].append(bna[bidx, h, lp_, 0])
            for tok, pos in (("lep", lp_), ("nu", np_)):
                c1[(tok, blk, h)].append(torch.stack(
                    [aw[bidx, h, pos, wpos[sl]], aw[bidx, h, pos, hpos[sl]]], -1))
    for (blk, h) in WIRES:
        sv = sval_table(blk)[h]                                  # [B,N]
        aw = cache[f"block_{blk}_attention"]["attn_weights_per_head"][:, h]  # [B,Q,K]
        a_nu = aw[bidx, np_]                                     # [B,N]
        per_key = a_nu * sv                                      # [B,N]
        s_rec = per_key.sum(1)
        s_ref = cache[f"block_{blk}_attention"]["bottleneck_activation"][bidx, h, np_, 0]
        sanity_max = max(sanity_max, (s_rec - s_ref).abs().max().item())
        tt_ = tt; tru_ = TRU[sl]
        masks = {"W-ljet": (tt_ == LJ) & (tru_ == 2), "H-ljet": (tt_ == LJ) & (tru_ == 1),
                 "other-jets": ((tt_ == LJ) | (tt_ == SJ)) & (tru_ != 2) & (tru_ != 1),
                 "lepton": (tt_ == 0) | (tt_ == 1), "nu-self": tt_ == NU}
        for cname, mm in masks.items():
            contrib[(blk, h)][cname].append((per_key * mm).sum(1))
pred_nu = torch.cat(pred_nu); pred_lep = torch.cat(pred_lep)
M_nu = {k: torch.cat(v).numpy() for k, v in msgs_nu.items()}
M_lep = {k: torch.cat(v).numpy() for k, v in msgs_lep.items()}
print(f"decomposition sanity: max|reconstructed - cached| = {sanity_max:.2e}")

lv = lvbb.numpy(); pn = (pred_nu == W).numpy(); pl = (pred_lep == W).numpy()
print("\n=== C4 completion: scalar census INTO THE LEPTON (AUC vs truth / vs pred_lep) ===")
for blk in range(3):
    row = []
    for h in range(4):
        a_t = roc_auc_score(lv, M_lep[(blk, h)]); a_p = roc_auc_score(pl, M_lep[(blk, h)])
        row.append(f"h{h}: {max(a_t,1-a_t):.3f}/{max(a_p,1-a_p):.3f}")
    print(f"  b{blk}  " + "   ".join(row))

print("\n=== C1: attention to W-ljet vs H-ljet (boosted, both ljets present) ===")
bsel = (boosted & has_h).numpy()
print(f"  n={bsel.sum()}; entries: mean a(->W-ljet) / mean a(->H-ljet) / P(W>H)")
for tok in ("lep", "nu"):
    rows = []
    for blk in range(3):
        for h in range(4):
            a = torch.cat(c1[(tok, blk, h)])[bsel]
            rows.append(f"b{blk}h{h}: {a[:,0].mean():.2f}/{a[:,1].mean():.2f}/{(a[:,0]>a[:,1]).float().mean():.2f}")
    print(f"  {tok}: " + "  ".join(rows[:6]))
    print(f"       " + "  ".join(rows[6:]))

print("\n=== per-key decomposition of nu's verdict scalars (mean contribution by sender) ===")
for k in WIRES:
    print(f"  b{k[0]}h{k[1]}:")
    print(f"    {'sender':>10s} {'lvbb':>8s} {'boosted':>9s}")
    for cname in CATS:
        v = torch.cat(contrib[k][cname])
        print(f"    {cname:>10s} {v[lvbb].mean():+8.3f} {v[boosted].mean():+9.3f}")

# ---------------- D4-lite: overwrite the wires at nu ----------------
mean_W = {k: torch.tensor(M_nu[k][pn].mean()) for k in WIRES + EXTRA}
mean_N = {k: torch.tensor(M_nu[k][~pn].mean()) for k in WIRES + EXTRA}
def targets_for(keys, mode):
    # per-event target: swap -> opposite-class mean; zero -> 0
    out = {}
    for k in keys:
        if mode == "zero":
            out[k] = torch.zeros(N)
        else:
            out[k] = torch.where(torch.from_numpy(pn), mean_N[k], mean_W[k])
    return out

CONFIGS = [("zero b1h3", targets_for([(1, 3)], "zero")),
           ("zero b2h2", targets_for([(2, 2)], "zero")),
           ("zero both wires", targets_for(WIRES, "zero")),
           ("swap b1h3", targets_for([(1, 3)], "swap")),
           ("swap b2h2", targets_for([(2, 2)], "swap")),
           ("swap both wires", targets_for(WIRES, "swap")),
           ("swap wires+b2h0+b2h3", targets_for(WIRES + EXTRA, "swap"))]
print("\n=== D4-lite: overwrite scalars AT THE NU POSITION only ===")
print(f"  {'config':22s} {'nu flip lvbb':>13s} {'nu flip qqbb':>13s} {'lep flip':>9s} {'P(nu=lep)':>10s}")
for name, tgts in CONFIGS:
    flips_nu, flips_lep, agree = [], [], []
    for c0 in range(0, N, 2048):
        sl = slice(c0, min(c0 + 2048, N))
        OV["active"] = True
        OV["nupos"] = npos[sl]
        OV["cfg"] = {k: t[sl] for k, t in tgts.items()}
        with torch.no_grad():
            out = model(X[sl][..., :5], T[sl])
        OV["active"] = False
        bidx = torch.arange(len(out))
        pr_n = out[bidx, npos[sl]].argmax(-1); pr_l = out[bidx, lpos[sl]].argmax(-1)
        flips_nu.append(pr_n != pred_nu[sl]); flips_lep.append(pr_l != pred_lep[sl])
        agree.append(pr_n == pr_l)
    f_n = torch.cat(flips_nu); f_l = torch.cat(flips_lep); ag = torch.cat(agree)
    print(f"  {name:22s} {f_n[lvbb].float().mean():13.4f} {f_n[~lvbb].float().mean():13.4f} "
          f"{f_l.float().mean():9.4f} {ag.float().mean():10.4f}")

for h in handles: h.remove()
print("\ndone.")
