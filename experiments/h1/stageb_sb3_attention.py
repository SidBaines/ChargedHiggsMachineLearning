"""SB3 (Stage B): jet<->jet attention forensics — weak B-i (directed suppression
edge) vs B-ii (context-relative scoring).

Preregistered in docs/H1_ASK_THE_JETS_TEST_PLAN.md (Stage B, post-red-team
amendments). In two-candidate events (RT3 insertion machinery: original truth-W
surgically set in place, competitor inserted into a pad slot with a foreign
direction, both tagged W-median), measure per block/head:

  a(loser->winner) vs a(winner->loser)        B-i: asymmetric edge; B-ii: symmetric
  vs single-W baselines a(W->H-jet), a(W->context)
  vs the SOLO rest x2 condition (RT4 arm c)   [RT amendment]: if the heads whose
     W->context attention mediates the solo suppression are the same heads that
     carry loser->winner attention, "competition" reduces to B-ii.

Cells: tie (600,80)v(600,80); asymA orig(600,80) v ins(300,80); asymB mirror.
Solo: orig at (600,80) / (300,80), no insertion; solo600 + rest x2.
Data: val batches 19+ (waves used 1-12, red-team 13-18) — fresh slice.

Usage: .venv/bin/python experiments/h1/stageb_sb3_attention.py
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
NH = model.attention_blocks[0]["self_attention"].num_heads

def fwd_rows(x, types, qpos):
    """Chunked forward. Returns (logits [N,15,3],
    rows[qname][blk] = attention row at that query position, [N, NH, 15])."""
    outs = []
    rows = {nm: [[] for _ in range(NBLK)] for nm in qpos}
    with torch.no_grad():
        for c in range(0, len(x), 2048):
            sl = slice(c, min(c + 2048, len(x)))
            outs.append(model(x[sl][..., :5], types[sl]))
            car = torch.arange(outs[-1].shape[0])
            for b in range(NBLK):
                AW = cache.store[f"block_{b}_attention"]["attn_weights_per_head"]
                for nm, qi in qpos.items():
                    rows[nm][b].append(AW[car, :, qi[sl], :])
    return torch.cat(outs), {nm: [torch.cat(r) for r in rows[nm]] for nm in rows}

def at(row_blk, kpos):
    """row_blk [N,NH,15], kpos [N] -> [N,NH]"""
    n = len(kpos)
    return row_blk.gather(2, kpos.view(n, 1, 1).expand(n, NH, 1)).squeeze(2)

# ---- data: fresh val slice ----
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
hp = ((T[bsel] == LJ) & (TRU[bsel] == 1)).float().argmax(1)
has_h = ((T[bsel] == LJ) & (TRU[bsel] == 1)).sum(1) >= 1
bar = torch.arange(NB)
slot = (T[bsel] == PAD).float().argmax(1)   # same slot in every cell (T2 built identically)
real = T[bsel] != PAD
m2_all = torch.clamp(X[bsel][..., 3] ** 2 - (X[bsel][..., :3] ** 2).sum(-1), min=0)
print(f"fresh-slice batches {ARGS.skip+1}-{ARGS.skip+ARGS.batches}: boosted+slot n={NB} (has_h {has_h.float().mean():.2f})")

rng = np.random.default_rng(0)
donor_j = rng.integers(NB, size=NB)
donor_dir = X[bsel[donor_j], wp[donor_j], :3].clone()

def set_kin(rows3, pt_gev, m_gev):
    pt_now = torch.sqrt(rows3[:, 0] ** 2 + rows3[:, 1] ** 2).clamp(min=1e-6)
    p3 = rows3 * ((pt_gev / 100.0) / pt_now).unsqueeze(-1)
    E = torch.sqrt((p3 ** 2).sum(-1) + (m_gev / 100.0) ** 2)
    return p3, E

def build(pt_o, m_o, pt_i=None, m_i=None, rest_scale=1.0):
    """Original truth-W set in place; optional competitor in the pad slot;
    optional scaling of all OTHER real objects (masses fixed)."""
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

def wilson(k, n):
    if n == 0: return ""
    p = k / n; z = 1.96
    c = (p + z*z/(2*n) + z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/1) / (1+z*z/n)
    lo = (p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n)
    return f"[{lo:.3f},{c:.3f}]"

# ============ two-candidate cells ============
CELLS = {
    "tie  (600,80)v(600,80)": (600, 80, 600, 80),
    "asymA(600,80)v(300,80)": (600, 80, 300, 80),
    "asymB(300,80)v(600,80)": (300, 80, 600, 80),
}
two = {}
print(f"\n=== claim bookkeeping (fresh slice — RT3 replication check) ===")
print(f"{'cell':24s} {'P(orig)':>8s} {'P(ins)':>8s} {'both':>7s} {'neither':>8s} {'orig wins|one':>14s}")
for nm, (po, mo, pi, mi) in CELLS.items():
    X2, T2 = build(po, mo, pi, mi)
    out, rows = fwd_rows(X2, T2, {"o": wp, "i": slot})
    pr = out.argmax(-1)
    ow = pr[bar, wp] == W; iw = pr[bar, slot] == W
    one = ow ^ iw
    two[nm] = dict(rows=rows, ow=ow, iw=iw, one=one)
    owins = (ow[one]).float().mean().item() if one.sum() else float("nan")
    print(f"{nm:24s} {ow.float().mean():8.4f} {iw.float().mean():8.4f} {(ow&iw).float().mean():7.4f} "
          f"{(~ow&~iw).float().mean():8.4f} {owins:14.4f}")

# ============ single-W baselines + solo rest x2 ============
solo = {}
for nm, (po, mo, rs) in {"solo600": (600, 80, 1.0), "solo300": (300, 80, 1.0),
                         "solo600_restx2": (600, 80, 2.0)}.items():
    X2, T2 = build(po, mo, rest_scale=rs)
    out, rows = fwd_rows(X2, T2, {"w": wp})
    solo[nm] = dict(rows=rows, claim=(out.argmax(-1)[bar, wp] == W))
print(f"\nsolo claim rates: " + "  ".join(
    f"{nm}={d['claim'].float().mean():.4f}" for nm, d in solo.items()))
kept = solo["solo600"]["claim"] & solo["solo600_restx2"]["claim"]
lost = solo["solo600"]["claim"] & ~solo["solo600_restx2"]["claim"]
print(f"solo600: restx2 kept={kept.sum().item()} lost={lost.sum().item()} "
      f"(suppression rate {lost.float().sum()/solo['solo600']['claim'].float().sum():.4f})")

# ============ per block/head attention tables ============
def fmt(v): return f"{v:6.3f}"

print(f"\n=== jet<->jet attention per block/head ===")
print("(tie: by position; asym: winner/loser defined by outcome among XOR events)")
hdr = "  ".join(f"b{b}h{h}" for b in range(NBLK) for h in range(NH))
print(f"{'quantity':42s} {hdr}")

def row_stat(rows_q, kpos, mask):
    """mean over masked events of attention at kpos: [NBLK*NH] flat."""
    out = []
    for b in range(NBLK):
        a = at(rows_q[b], kpos)[mask]          # [n, NH]
        out.append(a.mean(0) if len(a) else torch.full((NH,), float("nan")))
    return torch.cat(out)

def pline(label, vec):
    print(f"{label:42s} " + "  ".join(fmt(v.item()) for v in vec))

# tie cell, positional + outcome splits
d = two["tie  (600,80)v(600,80)"]
both = d["ow"] & d["iw"]
pline("tie: a(o->i) | both claim", row_stat(d["rows"]["o"], slot, both))
pline("tie: a(i->o) | both claim", row_stat(d["rows"]["i"], wp, both))
o_only = d["ow"] & ~d["iw"]; i_only = d["iw"] & ~d["ow"]
pline("tie: a(loser->winner) | XOR (i lost)", row_stat(d["rows"]["i"], wp, o_only))
pline("tie: a(winner->loser) | XOR (o won)", row_stat(d["rows"]["o"], slot, o_only))
pline("tie: a(loser->winner) | XOR (o lost)", row_stat(d["rows"]["o"], slot, i_only))
pline("tie: a(winner->loser) | XOR (i won)", row_stat(d["rows"]["i"], wp, i_only))

# asym cells: winner/loser by outcome (XOR events only)
for nm in ("asymA(600,80)v(300,80)", "asymB(300,80)v(600,80)"):
    d = two[nm]
    ow_x = d["ow"] & ~d["iw"]; iw_x = d["iw"] & ~d["ow"]
    # loser->winner: ins->orig where orig won, orig->ins where ins won
    l2w = torch.cat([row_stat(d["rows"]["i"], wp, ow_x).unsqueeze(0),
                     row_stat(d["rows"]["o"], slot, iw_x).unsqueeze(0)])
    w2l = torch.cat([row_stat(d["rows"]["o"], slot, ow_x).unsqueeze(0),
                     row_stat(d["rows"]["i"], wp, iw_x).unsqueeze(0)])
    n_ow, n_iw = ow_x.sum().item(), iw_x.sum().item()
    wts = torch.tensor([[n_ow], [n_iw]], dtype=torch.float)
    l2w = torch.nan_to_num(l2w); w2l = torch.nan_to_num(w2l)  # empty group has weight 0
    l2w = (l2w * wts).sum(0) / wts.sum(); w2l = (w2l * wts).sum(0) / wts.sum()
    pline(f"{nm[:5]}: a(loser->winner)  n={n_ow+n_iw}", l2w)
    pline(f"{nm[:5]}: a(winner->loser)", w2l)
    pline(f"{nm[:5]}: asymmetry (l2w - w2l)", l2w - w2l)

# single-W references
m_all = torch.ones(NB, dtype=torch.bool)
pline("solo600: a(W->H-jet) | has_h", row_stat(solo["solo600"]["rows"]["w"], hp, has_h))
ctx_w = []
for b in range(NBLK):
    a = solo["solo600"]["rows"]["w"][b]               # [N,NH,15]
    msk = real.clone(); msk[bar, wp] = False           # context = real, non-self
    ctx_w.append((a * msk.unsqueeze(1)).sum(-1).mean(0))
pline("solo600: a(W->all context)", torch.cat(ctx_w))

# ============ solo suppression mediation [RT amendment] ============
print(f"\n=== solo rest x2: which heads' W->context attention moves? ===")
print("(delta = restx2 - solo600 of a(W->context); split kept vs lost claim)")
for label, mask in (("kept", kept), ("lost", lost)):
    dvec = []
    for b in range(NBLK):
        msk = real.clone(); msk[bar, wp] = False
        a0 = (solo["solo600"]["rows"]["w"][b] * msk.unsqueeze(1)).sum(-1)
        a2 = (solo["solo600_restx2"]["rows"]["w"][b] * msk.unsqueeze(1)).sum(-1)
        d_ = (a2 - a0)[mask]
        dvec.append(d_.mean(0) if len(d_) else torch.full((NH,), float("nan")))
    pline(f"delta a(W->ctx) | claim {label} (n={mask.sum().item()})", torch.cat(dvec))

# head ranking overlap
d = two["asymA(600,80)v(300,80)"]
ow_x = d["ow"] & ~d["iw"]
asym_score = (row_stat(d["rows"]["i"], wp, ow_x) - row_stat(d["rows"]["o"], slot, ow_x)).abs()
dvec = []
for b in range(NBLK):
    msk = real.clone(); msk[bar, wp] = False
    a0 = (solo["solo600"]["rows"]["w"][b] * msk.unsqueeze(1)).sum(-1)
    a2 = (solo["solo600_restx2"]["rows"]["w"][b] * msk.unsqueeze(1)).sum(-1)
    dvec.append((a2 - a0)[lost].mean(0))
solo_score = torch.cat(dvec).abs()
names = [f"b{b}h{h}" for b in range(NBLK) for h in range(NH)]
rk = lambda s: [names[i] for i in s.argsort(descending=True)[:5]]
print(f"\ntop-5 heads by |competition asymmetry|: {rk(asym_score)}")
print(f"top-5 heads by |solo-suppression delta|: {rk(solo_score)}")

for h in handles: h.remove()
print("\ndone.")
