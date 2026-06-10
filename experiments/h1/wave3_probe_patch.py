"""H1 wave 3: D6 probes (when is W-candidacy linearly decodable in the LJET's own
stream?) + D1c activation patching (overwrite the W-ljet's stream at depth L with the
same event's H-ljet stream; where does the swap stop mattering?). Thesis model.

Prereg: candidacy decodable by end of block 1 (claims crystallize blocks 0-1, D2);
patching at embed/post-blk0 kills the claim AND flips nu (the wires then read 'no W');
patching post-blk1 only partially flips nu (b1h3's block-1 message already delivered);
patching post-blk2 leaves nu untouched (claims vanish at the jet token only ->
bookkeeping violation events).
"""
import os, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
torch.set_num_threads(4)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
PAD, NU, LJ, SJ = 5, 2, 3, 4
W, H = 2, 1
stds = np.ones(7); stds[:4] = 1e5

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
n_blocks = model.num_attention_blocks

# ---- stream capture (depths 0..n_blocks; 0 = post-object_net) ----
STREAMS = {}
def make_pre(name):
    def fn(module, args, kwargs):
        STREAMS[name] = args[0].detach()
    return fn
for bidx_, block in enumerate(model.attention_blocks):
    handles.append(block["self_attention"].register_forward_pre_hook(make_pre(f"d{bidx_}"), with_kwargs=True))
handles.append(model.classifier.register_forward_pre_hook(make_pre(f"d{n_blocks}"), with_kwargs=True))

# ---- patch machinery ----
PATCH = {"active": False, "depth": None, "pos": None, "target": None}
def obj_net_hook(module, args, output):
    if PATCH["active"] and PATCH["depth"] == 0:
        output = output.clone()
        output[torch.arange(len(output)), PATCH["pos"]] = PATCH["target"]
    return output
handles.append(model.object_net.register_forward_hook(obj_net_hook))
def make_mlp_hook(b):
    def fn(module, args, output):
        if PATCH["active"] and PATCH["depth"] == b + 1:
            output = output.clone()
            bi = torch.arange(len(output))
            output[bi, PATCH["pos"]] = PATCH["target"] - args[0][bi, PATCH["pos"]]
        return output
    return fn
for b, block in enumerate(model.attention_blocks):
    handles.append(block["post_attention"].register_forward_hook(make_mlp_hook(b)))

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
has_h = ((T == LJ) & (TRU == 1)).sum(1) >= 1
wpos = ((T == LJ) & (TRU == 2)).float().argmax(1)
hpos = ((T == LJ) & (TRU == 1)).float().argmax(1)

# ---- pass 1: baseline preds, probe data, H-ljet stream targets per depth ----
probe_X = {d: [] for d in range(n_blocks + 1)}
probe_y = []
h_stream = {d: torch.zeros(N, model.attention_blocks[0]["self_attention"].embed_dim)
            for d in range(n_blocks + 1)}
pred_nu, pred_w, pred_lep = [], [], []
for c0 in range(0, N, 2048):
    sl = slice(c0, min(c0 + 2048, N))
    xx, tt = X[sl], T[sl]
    with torch.no_grad():
        out = model(xx[..., :5], tt)
    B = len(xx); bi = torch.arange(B)
    pr = out.argmax(-1)
    pred_nu.append(pr[bi, npos[sl]]); pred_w.append(pr[bi, wpos[sl]])
    pred_lep.append(pr[bi, lpos[sl]])
    lj = (tt == LJ)
    yv = TRU[sl][lj]
    probe_y.append(yv)
    for d in range(n_blocks + 1):
        s = STREAMS[f"d{d}"]
        probe_X[d].append(s[lj])
        h_stream[d][sl] = s[bi, hpos[sl]]
pred_nu = torch.cat(pred_nu); pred_w = torch.cat(pred_w); pred_lep = torch.cat(pred_lep)
probe_y = torch.cat(probe_y).numpy()

print(f"events={N} (boosted {boosted.sum()}); ljet tokens for probes: {len(probe_y)}")
print("\n=== D6: linear probe 'is this LJET the truth-W?' per depth (W vs H / W vs all) ===")
is_w = (probe_y == 2).astype(int)
is_h = (probe_y == 1)
half = len(probe_y) // 2
for d in range(n_blocks + 1):
    PX = torch.cat(probe_X[d]).numpy()
    tr, te = slice(0, half), slice(half, None)
    clf = LogisticRegression(max_iter=2000).fit(PX[tr], is_w[tr])
    sc = clf.decision_function(PX[te])
    auc_all = roc_auc_score(is_w[te], sc)
    wh = is_w[te].astype(bool) | is_h[te]
    auc_wh = roc_auc_score(is_w[te][wh], sc[wh])
    lab = "embed" if d == 0 else f"blk{d-1}+"
    print(f"  {lab:>6s}  AUC(W vs all ljets) {auc_all:.4f}   AUC(W vs H) {auc_wh:.4f}")

# ---- D1c: patch the W-ljet stream with the same event's H-ljet stream ----
sel = torch.where(boosted & has_h)[0]
print(f"\n=== D1c: patch W-ljet stream <- same-event H-ljet stream, per depth "
      f"(boosted&has_h, n={len(sel)}) ===")
print(f"  {'depth':>6s} {'P(Wjet claims)':>15s} {'nu flip':>8s} {'lep flip':>9s}")
base_claim = (pred_w[sel] == W).float().mean()
print(f"  {'none':>6s} {base_claim:15.4f} {'-':>8s} {'-':>9s}")
for d in range(n_blocks + 1):
    f_nu, f_lep, claims = [], [], []
    for c0 in range(0, len(sel), 2048):
        ii = sel[c0:c0 + 2048]
        PATCH.update(active=True, depth=d, pos=wpos[ii], target=h_stream[d][ii])
        with torch.no_grad():
            out = model(X[ii][..., :5], T[ii])
        PATCH["active"] = False
        bi = torch.arange(len(ii))
        pr = out.argmax(-1)
        claims.append(pr[bi, wpos[ii]] == W)
        f_nu.append(pr[bi, npos[ii]] != pred_nu[ii])
        f_lep.append(pr[bi, lpos[ii]] != pred_lep[ii])
    lab = "embed" if d == 0 else f"blk{d-1}+"
    print(f"  {lab:>6s} {torch.cat(claims).float().mean():15.4f} "
          f"{torch.cat(f_nu).float().mean():8.4f} {torch.cat(f_lep).float().mean():9.4f}")

for hh in handles: hh.remove()
print("\ndone.")
