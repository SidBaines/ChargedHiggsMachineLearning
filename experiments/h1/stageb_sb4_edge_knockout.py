"""SB4 (Stage B): causal edge knockout of the jet<->jet attention.

SB3 found the only big jet->jet attention asymmetry at block 2 (b2h0/b2h2/b2h3,
loser->winner), the same heads that mediate solo rest-x2 suppression; SB2 found
tie-breaking lands AT block 2 (both candidates claim after blk1; loser flips during
blk2), while large-gap and solo-context suppression act earlier (blk0-1).

Here we knock out attention EDGES (zero selected post-softmax entries, renormalize
the row by default) inside a faithful by-hand recompute of each attention block
(per-head out-proj + bottleneck math mirrored from interp/activations.py; inactive
knockout is validated against the library hook to 1e-4).

Conditions (tie cell (600,80)v(600,80), fresh val slice):
  KO b2 o<->i (all heads / per head / each direction), KO b0+b1 o<->i control,
  no-renorm robustness. Readout: claim rates, restoration of baseline-XOR losers,
  winner margin, P(lep=W) sanity.
Solo conditions: KO W->context at b2 vs b0+b1, on solo600 (control: does KO break
  normal candidacy?) and solo600+rest x2 (does KO undo the relative-hardness
  suppression — shared-machinery test).

Usage: .venv/bin/python experiments/h1/stageb_sb4_edge_knockout.py
       [--model thesis-ent1-bn1-d152] [--batches 6] [--skip 18]
"""
import os, sys, argparse, math
import numpy as np
import torch
import torch.nn.functional as F

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
NBLK = model.num_attention_blocks
NH = model.attention_blocks[0]["self_attention"].num_heads

# ---------- surgical attention: by-hand recompute with optional edge knockout ----------
# KO spec (set per forward chunk): active, blocks (set), heads (tuple), renorm,
# q [G,n] long, kmask [G,n,15] bool — for each edge group g, zero
# A[ev, h, q[g,ev], kmask[g,ev]] for h in heads, then optionally renormalize the row.
KO = {"active": False}

def fwd_ko(x, types):
    """Plain forward through the surgical hooks with knockout inactive."""
    outs = []
    KO["active"] = False
    with torch.no_grad():
        for c in range(0, len(x), 2048):
            outs.append(model(x[c:c + 2048, :, :5], types[c:c + 2048]))
    return torch.cat(outs)

# ---------- data (identical selection to SB2/SB3) ----------
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
lp_b = lpos[bsel]
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

def onehot(pos):
    m = torch.zeros(NB, 15, dtype=torch.bool); m[bar, pos] = True; return m

def fwd_ko_multi(x, types, blocks, heads, edge_groups, renorm=True):
    """edge_groups: list of (q [N], kmask [N,15]) — all knocked out together."""
    outs = []
    with torch.no_grad():
        for c in range(0, len(x), 2048):
            sl = slice(c, min(c + 2048, len(x)))
            KO.update(active=True, blocks=blocks, heads=heads, renorm=renorm,
                      q=torch.stack([g[0][sl] for g in edge_groups]),
                      kmask=torch.stack([g[1][sl] for g in edge_groups]))
            outs.append(model(x[sl][..., :5], types[sl]))
    KO["active"] = False
    return torch.cat(outs)

def make_surgical(block_idx, block):
    bdown = block["bottleneck_down"] if BN is not None else None
    bup = block["bottleneck_up"] if BN is not None else None

    def hook_fn(module, inputs, kwargs, output):
        q0 = inputs[0]
        kpm = kwargs.get("key_padding_mask", None)
        kpm = F._canonical_mask(mask=kpm, mask_name="key_padding_mask",
                                other_type=F._none_or_dtype(kpm), other_name="",
                                target_type=q0.dtype)
        q = k = v = q0.transpose(1, 0)
        tgt_len, bsz, E = q.shape
        q, k, v = F._in_projection_packed(q, k, v, module.in_proj_weight, module.in_proj_bias)
        hd = E // NH
        q = q.view(tgt_len, bsz * NH, hd).transpose(0, 1)
        k = k.view(tgt_len, bsz * NH, hd).transpose(0, 1)
        v = v.view(tgt_len, bsz * NH, hd).transpose(0, 1)
        q_scaled = q * math.sqrt(1.0 / float(hd))
        if kpm is not None:
            kpm_e = (kpm.view(bsz, 1, 1, tgt_len).expand(-1, NH, -1, -1)
                     .reshape(bsz * NH, 1, tgt_len))
            A = torch.baddbmm(kpm_e, q_scaled, k.transpose(-2, -1))
        else:
            A = torch.bmm(q_scaled, k.transpose(-2, -1))
        A = F.softmax(A, dim=-1).view(bsz, NH, tgt_len, tgt_len)

        if KO["active"] and block_idx in KO["blocks"]:
            car = torch.arange(bsz)
            for g in range(KO["q"].shape[0]):
                qg, kg = KO["q"][g], KO["kmask"][g]
                for h in KO["heads"]:
                    row = A[car, h, qg, :]
                    row = row * (~kg).float()
                    if KO["renorm"]:
                        row = row / row.sum(-1, keepdim=True).clamp(min=1e-9)
                    A[car, h, qg, :] = row

        A2 = A.view(bsz * NH, tgt_len, tgt_len)
        ob = torch.bmm(A2, v)
        ob = ob.transpose(0, 1).contiguous().view(tgt_len * bsz, E)
        per_head = torch.empty(bsz, NH, tgt_len, module.out_proj.weight.shape[0])
        for h in range(NH):
            W_rows = module.out_proj.weight.transpose(0, 1)[h * hd:(h + 1) * hd]
            ho = torch.matmul(ob[:, h * hd:(h + 1) * hd], W_rows)
            per_head[:, h] = ho.contiguous().view(tgt_len, bsz, -1).transpose(0, 1)
        if BN is not None:
            for h in range(NH):
                s = torch.matmul(per_head[:, h].reshape(-1, E), bdown[h].weight.t())
                per_head[:, h] = torch.matmul(s, bup[h].weight.t()).view(bsz, tgt_len, E)
        return per_head.sum(dim=1) + module.out_proj.bias, output[1]
    return hook_fn

# ---------- validation: inactive surgical hook == library bottleneck hook ----------
Xv, Tv = build(600, 80, 600, 80)
cache = ActivationCache()
lib_handles = [m.register_forward_hook(fn, with_kwargs=True)
               for m, fn in hook_attention_heads(model, cache, detach=True,
                     SINGLE_ATTENTION=False, bottleneck_attention_output=BN)]
with torch.no_grad():
    ref = model(Xv[:256, :, :5], Tv[:256])
for h in lib_handles: h.remove()
sur_handles = [blk["self_attention"].register_forward_hook(
                   make_surgical(i, blk), with_kwargs=True)
               for i, blk in enumerate(model.attention_blocks)]
KO["active"] = False
with torch.no_grad():
    got = model(Xv[:256, :, :5], Tv[:256])
assert torch.allclose(ref, got, atol=1e-4), f"surgical hook mismatch: {(ref-got).abs().max()}"
print("surgical hook validated against library hook (max dev "
      f"{(ref-got).abs().max():.2e})")

# ---------- conditions ----------
oh_slot, oh_wp = onehot(slot), onehot(wp)
ctx = real.clone(); ctx[bar, wp] = False   # context keys for solo KO
ALLH = tuple(range(NH))

def claims(out):
    pr = out.argmax(-1)
    return pr[bar, wp] == W, pr[bar, slot] == W, pr[bar, lp_b] == W

print("\n=== tie cell (600,80) v (600,80) ===")
out0 = fwd_ko(Xv, Tv)
ow0, iw0, lw0 = claims(out0)
xor0 = ow0 ^ iw0
loser_pos = torch.where(ow0, slot, wp)     # baseline loser's position (XOR events)
print(f"{'condition':44s} {'P(orig)':>8s} {'P(ins)':>8s} {'both':>7s} {'P(lep=W)':>9s} {'loser restored|XOR':>19s}")
def report(nm, out):
    ow, iw, lw = claims(out)
    pr = out.argmax(-1)
    lr = (pr[bar, loser_pos] == W)[xor0].float().mean()
    print(f"{nm:44s} {ow.float().mean():8.4f} {iw.float().mean():8.4f} "
          f"{(ow&iw).float().mean():7.4f} {lw.float().mean():9.4f} {lr:19.4f}")
report("baseline", out0)
E_OI = [(wp, oh_slot)]; E_IO = [(slot, oh_wp)]; E_BI = E_OI + E_IO
LAST = NBLK - 1
EARLY = set(range(LAST))
report(f"KO b{LAST} (last) o<->i all heads", fwd_ko_multi(Xv, Tv, {LAST}, ALLH, E_BI))
report(f"KO b{LAST} o->i all heads", fwd_ko_multi(Xv, Tv, {LAST}, ALLH, E_OI))
report(f"KO b{LAST} i->o all heads", fwd_ko_multi(Xv, Tv, {LAST}, ALLH, E_IO))
for h in range(NH):
    report(f"KO b{LAST} o<->i head {h} only", fwd_ko_multi(Xv, Tv, {LAST}, (h,), E_BI))
for blk in sorted(EARLY):
    report(f"KO b{blk} o<->i all heads", fwd_ko_multi(Xv, Tv, {blk}, ALLH, E_BI))
report(f"KO early ({sorted(EARLY)}) o<->i all heads", fwd_ko_multi(Xv, Tv, EARLY, ALLH, E_BI))
report("KO ALL blocks o<->i all heads", fwd_ko_multi(Xv, Tv, set(range(NBLK)), ALLH, E_BI))
report(f"KO b{LAST} o<->i all heads NO-renorm", fwd_ko_multi(Xv, Tv, {LAST}, ALLH, E_BI, renorm=False))

print("\n=== solo conditions: KO W->context ===")
E_CTX = [(wp, ctx)]
print(f"{'condition':44s} {'P(W claims)':>12s} {'P(lep=W)':>9s}")
for nm, (rs,) in {"solo600": (1.0,), "solo600 rest x2": (2.0,)}.items():
    Xs_, Ts_ = build(600, 80, rest_scale=rs)
    solo_conds = [("none", None), (f"KO b{LAST} W->ctx", {LAST}),
                  ("KO early W->ctx", EARLY)]
    if len(EARLY) > 1:
        solo_conds += [(f"KO b{blk} W->ctx", {blk}) for blk in sorted(EARLY)]
    for ko_nm, blocks in solo_conds:
        out = (fwd_ko(Xs_, Ts_) if blocks is None
               else fwd_ko_multi(Xs_, Ts_, blocks, ALLH, E_CTX))
        pr = out.argmax(-1)
        print(f"{nm+' | '+ko_nm:44s} {(pr[bar,wp]==W).float().mean():12.4f} "
              f"{(pr[bar,lp_b]==W).float().mean():9.4f}")

for h in sur_handles: h.remove()
print("\ndone.")
