"""H1 wave 3: A5 tag surgery + E2 candidacy feature-attribution table (thesis model).

A5 prereg: candidacy gate beyond kinematics is the TAG. (i) lvbb H-ljet with m->80
alone barely claims W (A2: 1.2%); with tag ALSO zeroed it should claim substantially.
(ii) boosted W-ljet with tag->1 (Xbb-like) loses candidacy; with tag->1 AND m->125
(full H disguise) the event loses its W candidate => lep flips toward W.

E2: one-table occlusion summary of WHICH features drive the W-ljet's own claim margin
(direction items are new; pT/m/tag rows cross-check waves 1-3 dose-responses).
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
DSIDS = list(range(510115, 510125))
PAD, NU, LJ, SJ = 5, 2, 3, 4
W, H = 2, 1
stds = np.ones(7); stds[:4] = 1e5
torch.manual_seed(0)

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

def fwd(x, types):
    outs = []
    with torch.no_grad():
        for c in range(0, len(x), 2048):
            outs.append(model(x[c:c + 2048, :, :5], types[c:c + 2048]))
    return torch.cat(outs)

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

# ---------------- step 0: tag forensics ----------------
print("=== tag value distribution by type & truth ===")
tag = X[..., 4]
for tt, nm in ((0, "electron"), (1, "muon"), (NU, "nu"), (LJ, "ljet"), (SJ, "sjet")):
    mm = T == tt
    vals, cnts = torch.unique(tag[mm], return_counts=True)
    s = "  ".join(f"{v.item():g}:{c.item()/mm.sum().item():.3f}" for v, c in zip(vals, cnts))
    print(f"  {nm:9s} {s}")
for tru_v, nm in ((1, "H-ljet"), (2, "W-ljet")):
    mm = (T == LJ) & (TRU == tru_v)
    vals, cnts = torch.unique(tag[mm], return_counts=True)
    s = "  ".join(f"{v.item():g}:{c.item()/mm.sum().item():.3f}" for v, c in zip(vals, cnts))
    print(f"  {nm:9s} {s}")
mm = (T == SJ) & (TRU == 2)
vals, cnts = torch.unique(tag[mm], return_counts=True)
print(f"  {'W-sjet':9s} " + "  ".join(f"{v.item():g}:{c.item()/mm.sum().item():.3f}" for v, c in zip(vals, cnts)))

def set_mass(X2, rows, pos, m_gev):
    p2 = (X2[rows, pos, :3] ** 2).sum(-1)
    X2[rows, pos, 3] = torch.sqrt(p2 + (m_gev / 100.0) ** 2)

# ---------------- A5a: lvbb H-ljet {m80, tag0, both} ----------------
sel = torch.where(lvbb & (((T == LJ) & (TRU == 1)).sum(1) == 1))[0]
hp = ((T[sel] == LJ) & (TRU[sel] == 1)).float().argmax(1)
sar = torch.arange(len(sel))
print(f"\n=== A5a. lvbb H-ljet surgery (n={len(sel)}) [prereg: m80+tag0 >> m80 alone] ===")
print(f"  {'variant':>16s} {'P(H claims W)':>14s} {'P(H pred H)':>12s} {'P(lep=W)':>10s}")
def run_h(label, m_gev=None, tag_to=None):
    X2 = X[sel].clone()
    if m_gev is not None: set_mass(X2, sar, hp, m_gev)
    if tag_to is not None: X2[sar, hp, 4] = tag_to
    o = fwd(X2, T[sel]); pr = o.argmax(-1)
    print(f"  {label:>16s} {(pr[sar, hp]==W).float().mean():14.4f} "
          f"{(pr[sar, hp]==H).float().mean():12.4f} {(pr[sar, lpos[sel]]==W).float().mean():10.4f}")
run_h("as-is")
run_h("m->80", m_gev=80)
run_h("tag->0", tag_to=0.0)
run_h("m->80 + tag->0", m_gev=80, tag_to=0.0)
run_h("tag->Wmed -2.6", tag_to=-2.6)
run_h("m80+tag->Wmed", m_gev=80, tag_to=-2.6)

# ---------------- A5b: boosted W-ljet {tag->1, +m125} ----------------
sel = torch.where(boosted)[0]
wp = ((T[sel] == LJ) & (TRU[sel] == 2)).float().argmax(1)
sar = torch.arange(len(sel))
print(f"\n=== A5b. boosted W-ljet surgery (n={len(sel)}) [prereg: tag->1 kills candidacy] ===")
print(f"  {'variant':>16s} {'P(W claims W)':>14s} {'P(W pred H)':>12s} {'P(lep=W)':>10s}")
def run_w(label, m_gev=None, tag_to=None):
    X2 = X[sel].clone()
    if m_gev is not None: set_mass(X2, sar, wp, m_gev)
    if tag_to is not None: X2[sar, wp, 4] = tag_to
    o = fwd(X2, T[sel]); pr = o.argmax(-1)
    print(f"  {label:>16s} {(pr[sar, wp]==W).float().mean():14.4f} "
          f"{(pr[sar, wp]==H).float().mean():12.4f} {(pr[sar, lpos[sel]]==W).float().mean():10.4f}")
run_w("as-is")
run_w("tag->1", tag_to=1.0)
run_w("m->125", m_gev=125)
run_w("tag->1 + m->125", m_gev=125, tag_to=1.0)
run_w("tag->Hmed +3.0", tag_to=3.0)
run_w("m125+tag->Hmed", m_gev=125, tag_to=3.0)

# ---------------- E2: occlusion table on the W-ljet claim margin ----------------
print(f"\n=== E2. W-ljet occlusion table (boosted, n={len(sel)}); claim margin = logitW-logit0 at the W-ljet ===")
o = fwd(X[sel], T[sel])
base_margin = (o[sar, wp, W] - o[sar, wp, 0])
base_claim = (o.argmax(-1)[sar, wp] == W).float().mean()
print(f"  {'occlusion':>22s} {'P(claims)':>10s} {'med dMargin':>12s}")
print(f"  {'as-is':>22s} {base_claim:10.4f} {'-':>12s}")
g = torch.Generator().manual_seed(2)
def occlude(label, fn):
    X2 = X[sel].clone()
    fn(X2)
    o2 = fwd(X2, T[sel])
    m2 = o2[sar, wp, W] - o2[sar, wp, 0]
    cl = (o2.argmax(-1)[sar, wp] == W).float().mean()
    print(f"  {label:>22s} {cl:10.4f} {(m2-base_margin).median():12.2f}")
def rot_phi(X2):
    th = torch.rand(len(sel), generator=g) * 2 * np.pi
    c, s = torch.cos(th), torch.sin(th)
    px, py = X2[sar, wp, 0].clone(), X2[sar, wp, 1].clone()
    X2[sar, wp, 0] = px * c - py * s; X2[sar, wp, 1] = px * s + py * c
def flip_eta(X2):
    X2[sar, wp, 2] = -X2[sar, wp, 2]
def zero_eta(X2):
    m2 = torch.clamp(X2[sar, wp, 3]**2 - (X2[sar, wp, :3]**2).sum(-1), min=0)
    X2[sar, wp, 2] = 0.0
    X2[sar, wp, 3] = torch.sqrt((X2[sar, wp, :3]**2).sum(-1) + m2)
def scale_pt(X2, s):
    m2 = torch.clamp(X2[sar, wp, 3]**2 - (X2[sar, wp, :3]**2).sum(-1), min=0)
    X2[sar, wp, :3] *= s
    X2[sar, wp, 3] = torch.sqrt((X2[sar, wp, :3]**2).sum(-1) + m2)
occlude("phi randomized", rot_phi)
occlude("eta -> -eta", flip_eta)
occlude("eta -> 0 (central)", zero_eta)
occlude("pT x0.5 (fixed m)", lambda X2: scale_pt(X2, 0.5))
occlude("pT x2 (fixed m)", lambda X2: scale_pt(X2, 2.0))
occlude("m -> 40", lambda X2: set_mass(X2, sar, wp, 40))
occlude("m -> 125", lambda X2: set_mass(X2, sar, wp, 125))
occlude("tag -> 1", lambda X2: X2.__setitem__((sar, wp, 4), torch.ones(len(sel))))
occlude("tag -> Hmed +3.0", lambda X2: X2.__setitem__((sar, wp, 4), torch.full((len(sel),), 3.0)))

for hh in handles: hh.remove()
print("\ndone.")
