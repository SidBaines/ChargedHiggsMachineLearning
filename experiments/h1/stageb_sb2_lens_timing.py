"""SB2 (+SB6 ride-along) (Stage B): logit-lens timing of the competition.

Preregistered in docs/H1_ASK_THE_JETS_TEST_PLAN.md. In two-candidate events
(same construction as SB3: original truth-W surgically set, competitor inserted
into a pad slot), decode every depth's residual stream with the final classifier
(exact lens, no LayerNorm) and track the W-claim margin of winner vs loser.

  B-i/B-ii predict: both candidates start W-like (embed lens) and diverge at the
  block where competition acts; B-iii: never together. SB3 found the only big
  jet->jet attention asymmetry at block 2 — if margins diverge only AFTER block 2,
  the b2 read carries the suppression; if earlier, b2 is readout.

Also timed: the SOLO rest x2 suppression (RT4 arm c) — shared-machinery (B-ii)
predicts it lands at the SAME depth as two-candidate suppression.

SB6: winner's margin in two-W vs the solo twin at matched kinematics.

Data: fresh val slice (batches 19+). Usage:
  .venv/bin/python experiments/h1/stageb_sb2_lens_timing.py
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
NBLK = model.num_attention_blocks

# residual-stream capture (wave1_logit_lens conventions: snapshot before re-decoding)
STREAMS = {}
def make_pre(name):
    def fn(module, args, kwargs):
        STREAMS[name] = args[0].detach()
    return fn
pre_handles = [blk["self_attention"].register_forward_pre_hook(make_pre(f"depth{i}"), with_kwargs=True)
               for i, blk in enumerate(model.attention_blocks)]
pre_handles.append(model.classifier.register_forward_pre_hook(make_pre(f"depth{NBLK}"), with_kwargs=True))

def fwd_lens(x, types, positions):
    """Chunked forward. Returns (final_logits, lens_margin[posname][depth] = [N],
    lens_claim[posname][depth] = [N] bool) at the given per-event positions."""
    outs = []
    mg = {nm: [[] for _ in range(NBLK + 1)] for nm in positions}
    cl = {nm: [[] for _ in range(NBLK + 1)] for nm in positions}
    with torch.no_grad():
        for c in range(0, len(x), 2048):
            sl = slice(c, min(c + 2048, len(x)))
            outs.append(model(x[sl][..., :5], types[sl]))
            snap = {d: STREAMS[f"depth{d}"] for d in range(NBLK + 1)}
            car = torch.arange(outs[-1].shape[0])
            for d in range(NBLK + 1):
                L = model.classifier(snap[d])
                m = L[..., W] - L[..., 0]
                a = L.argmax(-1)
                for nm, pos in positions.items():
                    mg[nm][d].append(m[car, pos[sl]])
                    cl[nm][d].append(a[car, pos[sl]] == W)
    return (torch.cat(outs),
            {nm: [torch.cat(v) for v in mg[nm]] for nm in mg},
            {nm: [torch.cat(v) for v in cl[nm]] for nm in cl})

# ---- data: fresh val slice (identical selection to SB3) ----
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

bsel = torch.where(boosted & ((T == PAD).sum(1) >= 1))[0]
NB = len(bsel)
wp = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
bar = torch.arange(NB)
slot = (T[bsel] == PAD).float().argmax(1)
real = T[bsel] != PAD
m2_all = torch.clamp(X[bsel][..., 3] ** 2 - (X[bsel][..., :3] ** 2).sum(-1), min=0)
print(f"fresh-slice batches {ARGS.skip+1}-{ARGS.skip+ARGS.batches}: boosted+slot n={NB}")

rng = np.random.default_rng(0)
donor_j = rng.integers(NB, size=NB)
donor_dir = X[bsel[donor_j], wp[donor_j], :3].clone()

def set_kin(rows3, pt_gev, m_gev):
    pt_now = torch.sqrt(rows3[:, 0] ** 2 + rows3[:, 1] ** 2).clamp(min=1e-6)
    p3 = rows3 * ((pt_gev / 100.0) / pt_now).unsqueeze(-1)
    E = torch.sqrt((p3 ** 2).sum(-1) + (m_gev / 100.0) ** 2)
    return p3, E

def build(pt_o, m_o, pt_i=None, m_i=None, rest_scale=1.0):
    X2 = X[bsel].clone(); T2 = T[bsel].clone()
    if rest_scale != 1.0:
        sc = torch.full(X2.shape[:2], 1.0); sc[real] = rest_scale; sc[bar, wp] = 1.0
        X2[..., :3] = X2[..., :3] * sc.unsqueeze(-1)
        X2[..., 3] = torch.sqrt((X2[..., :3] ** 2).sum(-1) + m2_all)
        X2[~real] = X[bsel][~real]
    p3, E = set_kin(X2[bar, wp, :3], pt_o, m_o)
    X2[bar, wp, :3] = p3; X2[bar, wp, 3] = E; X2[bar, wp, 4] = WTAG
    if pt_i is not None:
        rows = torch.zeros(NB, 7)
        p3, E = set_kin(donor_dir, pt_i, m_i)
        rows[:, :3] = p3; rows[:, 3] = E; rows[:, 4] = WTAG
        X2[bar, slot] = rows; T2[bar, slot] = LJ
    return X2, T2

def depth_lab(d): return "embed" if d == 0 else f"blk{d-1}+"

def report(title, mgs, cls, masks):
    """masks: list of (label, posname, bool mask)"""
    print(f"\n--- {title} ---")
    print(f"{'depth':>6s} " + " ".join(f"{lab+' mg':>12s} {'P(W)':>7s}" for lab, _, _ in masks))
    for d in range(NBLK + 1):
        cells = []
        for lab, nm, msk in masks:
            m_ = mgs[nm][d][msk]; c_ = cls[nm][d][msk].float()
            cells.append(f"{m_.median().item():12.2f} {c_.mean().item():7.4f}")
        print(f"{depth_lab(d):>6s} " + " ".join(cells))

# ============ solo references ============
solo_mg, solo_cl, solo_claim = {}, {}, {}
for nm, (po, mo, rs) in {"solo600": (600, 80, 1.0), "solo300": (300, 80, 1.0),
                         "solo600_restx2": (600, 80, 2.0)}.items():
    X2, T2 = build(po, mo, rest_scale=rs)
    out, mgs, cls = fwd_lens(X2, T2, {"w": wp})
    solo_mg[nm], solo_cl[nm] = mgs, cls
    solo_claim[nm] = out.argmax(-1)[bar, wp] == W
all_t = torch.ones(NB, dtype=torch.bool)
kept = solo_claim["solo600"] & solo_claim["solo600_restx2"]
lost = solo_claim["solo600"] & ~solo_claim["solo600_restx2"]
report("solo timing: solo600 vs solo300 vs solo600+rest x2 (all events)",
       {"a": solo_mg["solo600"]["w"], "b": solo_mg["solo300"]["w"], "c": solo_mg["solo600_restx2"]["w"]},
       {"a": solo_cl["solo600"]["w"], "b": solo_cl["solo300"]["w"], "c": solo_cl["solo600_restx2"]["w"]},
       [("solo600", "a", all_t), ("solo300", "b", all_t), ("restx2", "c", all_t)])
report("solo rest x2, suppressed events only (claim lost)",
       {"a": solo_mg["solo600"]["w"], "c": solo_mg["solo600_restx2"]["w"]},
       {"a": solo_cl["solo600"]["w"], "c": solo_cl["solo600_restx2"]["w"]},
       [("solo600", "a", lost), ("restx2", "c", lost)])

# ============ two-candidate cells ============
for title, (po, mo, pi, mi) in {
        "tie (600,80) v (600,80)": (600, 80, 600, 80),
        "asymA (600,80) v ins(300,80)": (600, 80, 300, 80)}.items():
    X2, T2 = build(po, mo, pi, mi)
    out, mgs, cls = fwd_lens(X2, T2, {"o": wp, "i": slot})
    pr = out.argmax(-1)
    ow = pr[bar, wp] == W; iw = pr[bar, slot] == W
    both = ow & iw; o_only = ow & ~iw; i_only = iw & ~ow
    masks = [("orig|both", "o", both), ("ins|both", "i", both),
             ("win|XOR(o)", "o", o_only), ("los|XOR(i)", "i", o_only)]
    if i_only.sum() >= 30:
        masks += [("win|XOR(i)", "i", i_only), ("los|XOR(o)", "o", i_only)]
    report(f"{title}   [n both={both.sum().item()} XOR-o={o_only.sum().item()} XOR-i={i_only.sum().item()}]",
           mgs, cls, masks)

# ============ SB6: winner margin, two-W vs solo twin ============
X2, T2 = build(600, 80, 600, 80)
out_tie, mgs_tie, _ = fwd_lens(X2, T2, {"o": wp})
fm_tie = mgs_tie["o"][NBLK]                      # final-depth = true output margin
fm_solo = solo_mg["solo600"]["w"][NBLK]
print(f"\n=== SB6: winner's margin vs solo twin (same kinematics 600/80) ===")
print(f"solo600 margin: med {fm_solo.median():.2f} mean {fm_solo.mean():.2f} | claim {solo_claim['solo600'].float().mean():.4f}")
print(f"tie orig margin: med {fm_tie.median():.2f} mean {fm_tie.mean():.2f} | claim {(out_tie.argmax(-1)[bar,wp]==W).float().mean():.4f}")
d_ = fm_tie - fm_solo
print(f"per-event delta (tie - solo): med {d_.median():.2f} mean {d_.mean():.2f} "
      f"(negative = competitor costs margin)")

for h in handles + pre_handles: h.remove()
print("\ndone.")
