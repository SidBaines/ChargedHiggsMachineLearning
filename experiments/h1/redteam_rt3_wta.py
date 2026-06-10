"""RT3: factorial test of "winner-take-all is pT-ranked" (thesis model, slice C).

Wave-2 A8 compared two REAL W-ljets: their masses barely differ (both ~m_W) while
their pTs differ a lot, so "winner has higher pT 87%" vs "closer-to-80 62%" does not
disentangle pT-ranking from mass-ranking. Here both candidates get fully controlled
(pT, m), tag fixed to the W-ljet median (-2.6), in boosted qqbb events:

  original truth-W ljet: kinematics surgically set IN PLACE (keeps in-event direction)
  competitor: inserted into a pad slot with a random other-event direction

Cells: pT decides at tied mass; mass decides at tied pT; pT-vs-mass head-to-head;
symmetric tie (coherence/direction check -- E2's "direction-blind" predicts ~50/50).
"""
import os, sys
import numpy as np
import torch
import argparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="thesis-ent1-bn1-d152")
ARGS = ap.parse_args()
torch.set_num_threads(6)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
PAD, NU, LJ, SJ = 5, 2, 3, 4
W = 2
WTAG = -2.6
stds = np.ones(7); stds[:4] = 1e5
torch.manual_seed(0)

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in range(510115, 510125)},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)
for _ in range(12):
    next(dl)

BN = 1 if "bn1" in ARGS.model else None
model, _ = load_model(ARGS.model, checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=BN)]

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

bsel = torch.where(boosted & ((T == PAD).sum(1) >= 1))[0]
wp = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
bar = torch.arange(len(bsel))
print(f"boosted events with a free slot: {len(bsel)}")

rng = np.random.default_rng(0)
donor_j = rng.integers(len(bsel), size=len(bsel))
donor_dir = X[bsel[donor_j], wp[donor_j], :3].clone()   # random other-event W direction

def set_kin(rows3, pt_gev, m_gev):
    """Scale a [n,3] momentum to target pT (direction preserved), return (p3, E)."""
    pt_now = torch.sqrt(rows3[:, 0] ** 2 + rows3[:, 1] ** 2).clamp(min=1e-6)
    p3 = rows3 * ((pt_gev / 100.0) / pt_now).unsqueeze(-1)
    E = torch.sqrt((p3 ** 2).sum(-1) + (m_gev / 100.0) ** 2)
    return p3, E

def build(cell):
    (pt_o, m_o), (pt_i, m_i) = cell
    X2 = X[bsel].clone(); T2 = T[bsel].clone()
    p3, E = set_kin(X2[bar, wp, :3], pt_o, m_o)
    X2[bar, wp, :3] = p3; X2[bar, wp, 3] = E; X2[bar, wp, 4] = WTAG
    rows = torch.zeros(len(bsel), 7)
    p3, E = set_kin(donor_dir, pt_i, m_i)
    rows[:, :3] = p3; rows[:, 3] = E; rows[:, 4] = WTAG
    slot = (T2 == PAD).float().argmax(1)
    X2[bar, slot] = rows; T2[bar, slot] = LJ
    return X2, T2, slot

CELLS = [
    ("a. pT decides? orig(600,80) vs ins(300,80)", ((600, 80), (300, 80))),
    ("b. mirror:      orig(300,80) vs ins(600,80)", ((300, 80), (600, 80))),
    ("c. head2head:   orig(300,80) vs ins(600,60)", ((300, 80), (600, 60))),
    ("d. mirror:      orig(600,60) vs ins(300,80)", ((600, 60), (300, 80))),
    ("e. tie:         orig(600,80) vs ins(600,80)", ((600, 80), (600, 80))),
    ("f. mass@tied-pT orig(450,80) vs ins(450,110)", ((450, 80), (450, 110))),
    ("g. mirror:      orig(450,110) vs ins(450,80)", ((450, 110), (450, 80))),
    ("h. h2h closer:  orig(450,80) vs ins(600,110)", ((450, 80), (600, 110))),
    ("i. mirror:      orig(600,110) vs ins(450,80)", ((600, 110), (450, 80))),
]
print(f"\n{'cell':48s} {'P(orig)':>8s} {'P(ins)':>8s} {'both':>6s} {'neither':>8s} {'orig wins|one':>14s} {'P(lep=W)':>9s}")
for name, cell in CELLS:
    X2, T2, slot = build(cell)
    o = fwd(X2, T2); pr = o.argmax(-1)
    ow = pr[bar, wp] == W; iw = pr[bar, slot] == W
    lw = pr[bar, lpos[bsel]] == W
    one = ow ^ iw
    owins = (ow[one]).float().mean() if one.sum() > 0 else float("nan")
    print(f"{name:48s} {ow.float().mean():8.4f} {iw.float().mean():8.4f} {(ow&iw).float().mean():6.4f} "
          f"{(~ow&~iw).float().mean():8.4f} {owins:14.4f} {lw.float().mean():9.4f}")

for h in handles: h.remove()
print("\ndone.")
