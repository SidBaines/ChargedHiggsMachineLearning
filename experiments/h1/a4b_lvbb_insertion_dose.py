"""A4b: the lvbb-insertion dose-response — resolves the claim-2 asymmetry
(W-mask flips the lepton 77%, but real-donor W-insertion into lvbb flips only ~30%).

RESULT (2026-06-11, fresh slice, thesis model): nothing is gated/missing —
(a) broadcast intact: P(lep=W | ins claims)=0.06 vs 0.96 — lep-drop == ins-claim
    event-by-event; P(nu=lep)=0.98 throughout.
(b) the ~30% average is donor composition: flip rate by donor-pT/lepW-pT quartile
    = 0.001 / 0.05 / 0.39 / 0.83 (median lvbb leptonic-W pT is 575 GeV — random
    boosted-event donors are usually too soft for these events).
(c) dose at pT = r x pT(lepW): clean sigmoid, crossing r~1.4, saturating 0.93/0.96
    by r=2-3 — the verdict is a GRADED hadronic-vs-leptonic comparison (b1h3's
    job), the cross-system analogue of SB1's jet-vs-jet comparator; a hadronic
    candidate must beat the leptonic W by ~40% in pT to steal a true lvbb event.
(d) direction control: the event's own H-jet direction == foreign direction
    (no large cross-system coherence effect).

=> Claim 2 sharpened: "ask the jets" -> "weigh the best hadronic claim against the
leptonic-W evidence; the threshold scales with leptonic hardness". Explains the
mask/insert asymmetry: masking removes all hadronic evidence (lepton wins by
default); inserting must BEAT live leptonic evidence.
"""
import os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, REPO)
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
for _ in range(18):
    next(dl)

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

lsel = torch.where(lvbb & ((T == PAD).sum(1) >= 1))[0]
NL = len(lsel)
lar = torch.arange(NL)
slot = (T[lsel] == PAD).float().argmax(1)
lp, np_ = lpos[lsel], npos[lsel]
hp = ((T[lsel] == LJ) & (TRU[lsel] == 1)).float().argmax(1)
has_h = ((T[lsel] == LJ) & (TRU[lsel] == 1)).sum(1) >= 1

# leptonic-W proxy: lepton + neutrino transverse momentum
pvec = X[lsel][lar, lp, :2] + X[lsel][lar, np_, :2]
pt_lepW = torch.sqrt((pvec ** 2).sum(-1)) * 100   # GeV
print(f"lvbb with slot: n={NL}; med pT(lepW) {pt_lepW.median():.0f} GeV")

out0 = fwd(X[lsel], T[lsel]); pr0 = out0.argmax(-1)
lep_W0 = pr0[lar, lp] == W
print(f"baseline P(lep=W) = {lep_W0.float().mean():.4f}")

bsel = torch.where(boosted)[0]
wpb = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
rng = np.random.default_rng(0)
don = rng.integers(len(bsel), size=NL)
donor_rows = X[bsel[don], wpb[don]].clone(); donor_rows[:, 5:] = 0.0
donor_pt = torch.sqrt((donor_rows[:, :2] ** 2).sum(-1)) * 100

def set_kin(rows3, pt_gev_t, m_gev):
    pt_now = torch.sqrt(rows3[:, 0] ** 2 + rows3[:, 1] ** 2).clamp(min=1e-6)
    p3 = rows3 * ((pt_gev_t / 100.0) / pt_now).unsqueeze(-1)
    E = torch.sqrt((p3 ** 2).sum(-1) + (m_gev / 100.0) ** 2)
    return p3, E

# ---- (a)+(b): real-donor insertion, coupling + stratification ----
Xi = X[lsel].clone(); Ti = T[lsel].clone()
Xi[lar, slot] = donor_rows; Ti[lar, slot] = LJ
pri = fwd(Xi, Ti).argmax(-1)
ins_W = pri[lar, slot] == W
lep_W = pri[lar, lp] == W
drop = lep_W0 & ~lep_W
print(f"\n(a) real-donor insertion: P(ins=W) {ins_W.float().mean():.4f}, "
      f"P(lep drops|had W) {(drop[lep_W0]).float().mean():.4f}")
print(f"    P(lep=W | ins claims)   = {lep_W[ins_W].float().mean():.4f}")
print(f"    P(lep=W | ins no claim) = {lep_W[~ins_W].float().mean():.4f}")
print(f"    P(nu=lep) after insertion = {(pri[lar, np_] == pri[lar, lp]).float().mean():.4f}")

ratio = donor_pt / pt_lepW.clamp(min=1)
print(f"\n(b) flip rate by donor-pT / lepW-pT quartile:")
qs = torch.quantile(ratio, torch.tensor([0., .25, .5, .75, 1.]))
for i in range(4):
    m = (ratio >= qs[i]) & (ratio <= qs[i + 1])
    print(f"    r in [{qs[i]:5.2f},{qs[i+1]:5.2f}]: n={m.sum().item():5d}  "
          f"P(ins=W)={ins_W[m].float().mean():.3f}  P(lep drops)={drop[m].float().mean():.3f}")

# ---- (c) dose: synthetic candidate at pT = r x pT(lepW), random donor direction ----
print(f"\n(c) synthetic candidate (m=80, W-tag, foreign direction) at pT = r x pT(lepW):")
print(f"    {'r':>5s} {'P(ins=W)':>9s} {'P(lep=W)':>9s} {'P(lep drops)':>13s}")
for r in [0.5, 0.8, 1.0, 1.25, 1.6, 2.0, 3.0]:
    X2 = X[lsel].clone(); T2 = T[lsel].clone()
    rows = torch.zeros(NL, 7)
    p3, E = set_kin(donor_rows[:, :3], r * pt_lepW, 80)
    rows[:, :3] = p3; rows[:, 3] = E; rows[:, 4] = WTAG
    X2[lar, slot] = rows; T2[lar, slot] = LJ
    pr = fwd(X2, T2).argmax(-1)
    iw = pr[lar, slot] == W; lw = pr[lar, lp] == W
    print(f"    {r:5.2f} {iw.float().mean():9.4f} {lw.float().mean():9.4f} "
          f"{((lep_W0 & ~lw)[lep_W0]).float().mean():13.4f}")

# ---- (d) same dose with the event's own H-jet DIRECTION (coherence control) ----
print(f"\n(d) same, but direction = the event's own H-jet (has_h events, n={has_h.sum().item()}):")
print(f"    {'r':>5s} {'P(ins=W)':>9s} {'P(lep drops)':>13s}")
hsel = torch.where(has_h)[0]
for r in [1.0, 1.6, 2.0, 3.0]:
    X2 = X[lsel].clone(); T2 = T[lsel].clone()
    hdir = X[lsel][lar, hp, :3].clone()
    rows = torch.zeros(NL, 7)
    p3, E = set_kin(hdir, r * pt_lepW, 80)
    rows[:, :3] = p3; rows[:, 3] = E; rows[:, 4] = WTAG
    X2[lar, slot] = rows; T2[lar, slot] = LJ
    pr = fwd(X2, T2).argmax(-1)
    iw = (pr[lar, slot] == W)[hsel]; lw = (pr[lar, lp] == W)
    dr = ((lep_W0 & ~lw)[lep_W0 & has_h]).float().mean()
    print(f"    {r:5.2f} {iw.float().mean():9.4f} {dr:13.4f}")

for h in handles: h.remove()
print("\ndone.")
