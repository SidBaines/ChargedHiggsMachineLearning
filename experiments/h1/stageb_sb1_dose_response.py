"""SB1 (Stage B): two-candidate pT-ratio dose-response — the comparator's shape.

Post-red-team prereg (docs/H1_ASK_THE_JETS_TEST_PLAN.md): the open quantities are
the WIDTH of the both-claim peak around r=1 (how big a score gap suppresses the
loser) and the sigmoid crossing offset from the coherence handicap. Two designs:

  asym (RT3-style):  original truth-W kept in place at (600,80); competitor
                     inserted at (r*600, 80) with a foreign direction.
  sym  (amendment):  original W masked to PAD (clean removal — key_padding_mask
                     semantics); TWO foreign candidates inserted into pad slots,
                     cand1 (600,80), cand2 (r*600,80). No coherence asymmetry by
                     construction -> crossing should sit at r=1; any residual offset
                     in the asym design measures the in-event coherence handicap.

Data: fresh val slice (batches 19+), boosted qqbb with >=2 pad slots (sym design).
Usage: .venv/bin/python experiments/h1/stageb_sb1_dose_response.py
       [--model thesis-ent1-bn1-d152] [--batches 6] [--skip 18]
"""
import os, sys, argparse
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="thesis-ent1-bn1-d152")
ap.add_argument("--batches", type=int, default=6)
ap.add_argument("--skip", type=int, default=18)
ap.add_argument("--ckpt-override", default=None)
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
for _ in range(ARGS.skip):
    next(dl)

BN = 1 if "bn1" in ARGS.model else None
model, _ = load_model(ARGS.model, checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False, checkpoint_override=ARGS.ckpt_override)
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
for _ in range(ARGS.batches):
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

bsel = torch.where(boosted & ((T == PAD).sum(1) >= 2))[0]   # 2 slots for the sym design
NB = len(bsel)
wp = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
bar = torch.arange(NB)
pads = (T[bsel] == PAD).float()
slot1 = pads.argmax(1)
pads2 = pads.clone(); pads2[bar, slot1] = 0
slot2 = pads2.argmax(1)
print(f"fresh-slice batches {ARGS.skip+1}-{ARGS.skip+ARGS.batches}: boosted+2slots n={NB}")

rng = np.random.default_rng(0)
donor_idx1 = rng.integers(NB, size=NB)
donor_idx2 = rng.integers(NB, size=NB)
donor_dir1 = X[bsel[donor_idx1], wp[donor_idx1], :3].clone()
donor_dir2 = X[bsel[donor_idx2], wp[donor_idx2], :3].clone()

def report_donor_guard(name, donor_idx):
    tt = T[bsel[donor_idx], wp[donor_idx]]
    yy = TRU[bsel[donor_idx], wp[donor_idx]]
    print(f"{name}: donor true-W-ljet fraction {(((tt == LJ) & (yy == 2)).float().mean()):.4f}; "
          f"type counts {[int((tt == k).sum()) for k in range(6)]}")

report_donor_guard("donor_dir1", donor_idx1)
report_donor_guard("donor_dir2", donor_idx2)

def set_kin(rows3, pt_gev, m_gev):
    pt_now = torch.sqrt(rows3[:, 0] ** 2 + rows3[:, 1] ** 2).clamp(min=1e-6)
    p3 = rows3 * ((pt_gev / 100.0) / pt_now).unsqueeze(-1)
    E = torch.sqrt((p3 ** 2).sum(-1) + (m_gev / 100.0) ** 2)
    return p3, E

def insert(X2, T2, pos, direction, pt, m):
    rows = torch.zeros(NB, 7)
    p3, E = set_kin(direction, pt, m)
    rows[:, :3] = p3; rows[:, 3] = E; rows[:, 4] = WTAG
    X2[bar, pos] = rows; T2[bar, pos] = LJ

def build_asym(r):
    X2 = X[bsel].clone(); T2 = T[bsel].clone()
    p3, E = set_kin(X2[bar, wp, :3], 600, 80)
    X2[bar, wp, :3] = p3; X2[bar, wp, 3] = E; X2[bar, wp, 4] = WTAG
    insert(X2, T2, slot1, donor_dir1, 600 * r, 80)
    return X2, T2, wp, slot1

def build_sym(r):
    X2 = X[bsel].clone(); T2 = T[bsel].clone()
    T2[bar, wp] = PAD; X2[bar, wp] = 0.0          # clean removal of the real W
    insert(X2, T2, slot1, donor_dir1, 600, 80)
    insert(X2, T2, slot2, donor_dir2, 600 * r, 80)
    return X2, T2, slot1, slot2

RS = [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.6, 2.0]
for design, builder in (("asym (orig in-event vs foreign ins)", build_asym),
                        ("sym (two foreign candidates)", build_sym)):
    print(f"\n=== {design} ===   c1 fixed (600,80); c2 at (600r, 80)")
    print(f"{'r':>5s} {'P(c1)':>7s} {'P(c2)':>7s} {'both':>7s} {'neither':>8s} "
          f"{'P(c2 wins|one)':>15s} {'P(lep=W)':>9s}")
    for r in RS:
        X2, T2, p1, p2 = builder(r)
        out = fwd(X2, T2); pr = out.argmax(-1)
        c1 = pr[bar, p1] == W; c2 = pr[bar, p2] == W
        one = c1 ^ c2
        c2w = (c2[one]).float().mean().item() if one.sum() else float("nan")
        lw = (pr[bar, lpos[bsel]] == W).float().mean()
        print(f"{r:5.2f} {c1.float().mean():7.4f} {c2.float().mean():7.4f} "
              f"{(c1&c2).float().mean():7.4f} {(~c1&~c2).float().mean():8.4f} "
              f"{c2w:15.4f} {lw:9.4f}")

for h in handles: h.remove()
print("\ndone.")
