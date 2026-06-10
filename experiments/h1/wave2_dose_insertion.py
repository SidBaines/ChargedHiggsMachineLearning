"""H1 wave 2, input-level half: A2 H-ljet mass dose-response, A1-pT pT dose-response,
A4 W-candidate insertion (sufficiency), A8 two-claimant tie-break.

Preregistrations in docs/H1_ASK_THE_JETS_TEST_PLAN.md (+ wave-1 updates).
Types: 3=LJET, 4=SJET. Units: MeV/1e5 (GeV value /100).

Usage: .venv/bin/python tmp_h1_wave2_dose.py [--model ent1-d20-2blk] [--batches 6]
       [--ckpt-override ...]
"""
import os, sys, argparse
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
torch.set_num_threads(4)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

p = argparse.ArgumentParser()
p.add_argument("--model", default="ent1-d20-2blk")
p.add_argument("--batches", type=int, default=6)
p.add_argument("--ckpt-override", default=None)
ARGS = p.parse_args()

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
print(f"model={ARGS.model}  events={N} (lvbb {lvbb.sum()}, boosted {boosted.sum()})")

def set_mass(X2, rows_idx, pos, m_gev):
    p2 = (X2[rows_idx, pos, :3] ** 2).sum(-1)
    X2[rows_idx, pos, 3] = torch.sqrt(p2 + (m_gev / 100.0) ** 2)

# ---------------- A2: H-ljet mass dose-response ----------------
# lvbb events with exactly one truth-1 ljet (drag the H toward/away from m_W)
sel = torch.where(lvbb & (((T == LJ) & (TRU == 1)).sum(1) == 1))[0]
hp = ((T[sel] == LJ) & (TRU[sel] == 1)).float().argmax(1)
sar = torch.arange(len(sel))
print(f"\n=== A2a. H-ljet mass dose-response in lvbb (n={len(sel)}) "
      "[H-A: H claims W near 80 => lep flips to none] ===")
print(f"  {'m_set':>6s} {'P(H-jet pred H)':>16s} {'P(H-jet claims W)':>18s} {'P(lep=W)':>10s}")
o = fwd(X[sel], T[sel]); pr = o.argmax(-1)
print(f"  {'as-is':>6s} {(pr[sar, hp]==H).float().mean():16.4f} {(pr[sar, hp]==W).float().mean():18.4f} "
      f"{(pr[sar, lpos[sel]]==W).float().mean():10.4f}")
for m_gev in (40, 60, 80, 90, 100, 110, 125, 150, 200):
    X2 = X[sel].clone(); set_mass(X2, sar, hp, m_gev)
    o = fwd(X2, T[sel]); pr = o.argmax(-1)
    print(f"  {m_gev:6d} {(pr[sar, hp]==H).float().mean():16.4f} {(pr[sar, hp]==W).float().mean():18.4f} "
          f"{(pr[sar, lpos[sel]]==W).float().mean():10.4f}")

# boosted qqbb with exactly one truth-1 ljet: drag H toward 80 => competition (A8-by-mass)
sel = torch.where(boosted & (((T == LJ) & (TRU == 1)).sum(1) == 1))[0]
hp = ((T[sel] == LJ) & (TRU[sel] == 1)).float().argmax(1)
wp = ((T[sel] == LJ) & (TRU[sel] == 2)).float().argmax(1)
sar = torch.arange(len(sel))
print(f"\n=== A2b. drag H-ljet mass in BOOSTED qqbb (n={len(sel)}) "
      "[competition: who claims?] ===")
print(f"  {'m_set':>6s} {'P(H claims W)':>14s} {'P(Wjet claims W)':>17s} {'P(both)':>8s} {'P(lep=W)':>10s}")
o = fwd(X[sel], T[sel]); pr = o.argmax(-1)
print(f"  {'as-is':>6s} {(pr[sar, hp]==W).float().mean():14.4f} {(pr[sar, wp]==W).float().mean():17.4f} "
      f"{((pr[sar, hp]==W)&(pr[sar, wp]==W)).float().mean():8.4f} {(pr[sar, lpos[sel]]==W).float().mean():10.4f}")
for m_gev in (80, 100, 125):
    X2 = X[sel].clone(); set_mass(X2, sar, hp, m_gev)
    o = fwd(X2, T[sel]); pr = o.argmax(-1)
    print(f"  {m_gev:6d} {(pr[sar, hp]==W).float().mean():14.4f} {(pr[sar, wp]==W).float().mean():17.4f} "
          f"{((pr[sar, hp]==W)&(pr[sar, wp]==W)).float().mean():8.4f} {(pr[sar, lpos[sel]]==W).float().mean():10.4f}")

# ---------------- A1-pT: W-ljet pT dose-response at fixed mass ----------------
sel = torch.where(boosted)[0]
wp = ((T[sel] == LJ) & (TRU[sel] == 2)).float().argmax(1)
sar = torch.arange(len(sel))
print(f"\n=== A1-pT. W-ljet pT scaling at fixed mass (boosted, n={len(sel)}) "
      "[B3 predicts: soft => missed] ===")
print(f"  {'scale':>6s} {'P(Wjet claims)':>15s} {'P(lep=W)':>10s}")
m_now2 = torch.clamp(X[sel][sar, wp, 3] ** 2 - (X[sel][sar, wp, :3] ** 2).sum(-1), min=0)
for s in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    X2 = X[sel].clone()
    X2[sar, wp, :3] *= s
    X2[sar, wp, 3] = torch.sqrt((X2[sar, wp, :3] ** 2).sum(-1) + m_now2)
    o = fwd(X2, T[sel]); pr = o.argmax(-1)
    print(f"  {s:6.2f} {(pr[sar, wp]==W).float().mean():15.4f} {(pr[sar, lpos[sel]]==W).float().mean():10.4f}")

# ---------------- A4: insertion into lvbb (sufficiency) ----------------
rng = np.random.default_rng(0)
lv_idx = torch.where(lvbb & ((T == PAD).sum(1) >= 1))[0][:3000]
bo_idx = torch.where(boosted)[0]
wrow_of = ((T[bo_idx] == LJ) & (TRU[bo_idx] == 2)).float().argmax(1)

def insert_rows(base_idx, make_row):
    X2 = X[base_idx].clone(); T2 = T[base_idx].clone()
    slot = (T2 == PAD).float().argmax(1)
    bar_ = torch.arange(len(base_idx))
    rows, types_new = make_row(len(base_idx))
    X2[bar_, slot] = rows; T2[bar_, slot] = types_new
    return X2, T2, slot

def donor_w_rows(n):
    j = rng.integers(len(bo_idx), size=n)
    rows = X[bo_idx[j], wrow_of[j]].clone(); rows[:, -1] = 0; rows[:, 5] = 0
    return rows, torch.full((n,), LJ, dtype=T.dtype)

def synth_rows(n, m_gev, pt_gev, tt=LJ):
    j = rng.integers(len(bo_idx), size=n)
    src = X[bo_idx[j], wrow_of[j]]
    pt_now = torch.sqrt(src[:, 0] ** 2 + src[:, 1] ** 2).clamp(min=1e-6)
    scale = (pt_gev / 100.0) / pt_now
    rows = torch.zeros(n, 7)
    rows[:, 0] = src[:, 0] * scale; rows[:, 1] = src[:, 1] * scale
    rows[:, 2] = src[:, 2] * scale
    rows[:, 3] = torch.sqrt((rows[:, :3] ** 2).sum(-1) + (m_gev / 100.0) ** 2)
    return rows, torch.full((n,), tt, dtype=T.dtype)

print(f"\n=== A4. insertion into lvbb (n={len(lv_idx)}) "
      "[H-A: real/synth W claims; lep flips; controls don't] ===")
print(f"  {'variant':>34s} {'P(ins claims W)':>16s} {'P(lep=W)':>10s} {'P(both W)':>10s}")
o = fwd(X[lv_idx], T[lv_idx]); pr = o.argmax(-1)
base_lepW = (pr[torch.arange(len(lv_idx)), lpos[lv_idx]] == W).float().mean()
print(f"  {'as-is (no insert)':>34s} {'-':>16s} {base_lepW:10.4f} {'-':>10s}")
for lab, mk in (("real W-ljet copy", donor_w_rows),
                ("synthetic ljet m=80 pT=600", lambda n: synth_rows(n, 80, 600)),
                ("control ljet m=125 pT=600", lambda n: synth_rows(n, 125, 600)),
                ("control soft sjet m=10 pT=50", lambda n: synth_rows(n, 10, 50, tt=SJ))):
    X2, T2, slot = insert_rows(lv_idx, mk)
    o = fwd(X2, T2); pr = o.argmax(-1)
    bar_ = torch.arange(len(lv_idx))
    ins_W = pr[bar_, slot] == W
    lep_W = pr[bar_, lpos[lv_idx]] == W
    print(f"  {lab:>34s} {ins_W.float().mean():16.4f} {lep_W.float().mean():10.4f} "
          f"{(ins_W & lep_W).float().mean():10.4f}")

# ---------------- A8: two claimants (insert 2nd real W into boosted) ----------------
bo2 = torch.where(boosted & ((T == PAD).sum(1) >= 1))[0][:3000]
wp2 = ((T[bo2] == LJ) & (TRU[bo2] == 2)).float().argmax(1)
X2, T2, slot = insert_rows(bo2, donor_w_rows)
o = fwd(X2, T2); pr = o.argmax(-1)
bar_ = torch.arange(len(bo2))
orig_W = pr[bar_, wp2] == W; ins_W = pr[bar_, slot] == W
lep_W = pr[bar_, lpos[bo2]] == W
print(f"\n=== A8. second real W-ljet inserted into boosted qqbb (n={len(bo2)}) "
      "[exactly one claims? who wins?] ===")
print(f"  P(orig claims)={orig_W.float().mean():.4f}  P(inserted claims)={ins_W.float().mean():.4f}")
print(f"  P(both)={ (orig_W&ins_W).float().mean():.4f}  P(neither)={(~orig_W&~ins_W).float().mean():.4f}  "
      f"P(exactly one)={(orig_W^ins_W).float().mean():.4f}  P(lep=W)={lep_W.float().mean():.4f}")
one = orig_W ^ ins_W
pt_o = torch.sqrt(X2[bar_, wp2, 0]**2 + X2[bar_, wp2, 1]**2)
pt_i = torch.sqrt(X2[bar_, slot, 0]**2 + X2[bar_, slot, 1]**2)
m_o = torch.sqrt(torch.clamp(X2[bar_, wp2, 3]**2 - (X2[bar_, wp2, :3]**2).sum(-1), min=0)) * 100
m_i = torch.sqrt(torch.clamp(X2[bar_, slot, 3]**2 - (X2[bar_, slot, :3]**2).sum(-1), min=0)) * 100
win_ins = ins_W[one]
print(f"  among exactly-one: P(winner has higher pT)   = "
      f"{torch.where(win_ins, pt_i[one] > pt_o[one], pt_o[one] > pt_i[one]).float().mean():.4f}")
print(f"  among exactly-one: P(winner closer to 80 GeV) = "
      f"{torch.where(win_ins, (m_i[one]-80).abs() < (m_o[one]-80).abs(), (m_o[one]-80).abs() < (m_i[one]-80).abs()).float().mean():.4f}")

for h in handles: h.remove()
print("\ndone.")
