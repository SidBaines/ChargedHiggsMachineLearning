"""LW1: the leptonic-W side of the verdict comparison — where is it read, and
what is it computed from?

Follows A4b (the verdict = graded hadronic-vs-leptonic comparison, crossing at
candidate-pT ~ 1.4 x pT(lepW)). Questions:
 1) WHERE does the comparison happen — which positions/blocks/edges carry the
    leptonic evidence (a) into the candidate jet's own claim, (b) into lep/nu?
 2) WHAT leptonic variables feed it — individual magnitudes (lepton pT, MET),
    their vector sum, or W-like structure (dphi, mT)?

Preregistered predictions (from thesis robustness "nu angles irrelevant" + round-2
"lepton+MET magnitude" + SB4's context-absorption picture):
 - magnitudes matter; dphi/mT structure does NOT (nu phi-rotation ~ no-op);
 - b1h3's leptonic side is read from the lepton and nu keys directly;
 - the candidate's suppression in lvbb flows through its OWN reads of lep/nu at
   blocks 0-1 (same two-stage shape as SB4), and the lepton's concession through
   its reads of the jet at blocks 1-2 (the known reader heads b1h3/b2h2).

Sections:
 A. observational: what do the b1h3/b2h2 scalars at nu track in clean lvbb?
 B. per-key decomposition of b1h3 (and b2h2) at nu/lep in lvbb, and in the A4b
    insertion setting (does the candidate's key flip the comparator?)
 C. input-side surgery at FIXED candidate (pT_c = 1.2 x baseline pT(lepW)):
    lep x1.3 | nu x1.3 | both x1.3 | both x0.77 | nu phi+90deg | nu phi+180deg |
    swap lep/nu pT magnitudes | nu x0.1
 D. circuit-side edge KO (SB4 surgical hooks):
    r=1.0 (candidate suppressed): cut ins->{lep,nu} per block (vs ins->jets ctrl)
    r=2.0 (lepton concedes):      cut lep->ins per block, and nu->ins

Fresh val slice (batches 19-24), thesis model.
Usage: .venv/bin/python experiments/h1/lw1_leptonic_evidence.py
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="thesis-ent1-bn1-d152")
ap.add_argument("--batches", type=int, default=6)
ap.add_argument("--skip", type=int, default=18)
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

model, _ = load_model(ARGS.model, checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
lib_handles = [m.register_forward_hook(fn, with_kwargs=True)
               for m, fn in hook_attention_heads(model, cache, detach=True,
                     SINGLE_ATTENTION=False, bottleneck_attention_output=1)]
NBLK = model.num_attention_blocks
attn_modules = [blk["self_attention"] for blk in model.attention_blocks]
D = attn_modules[0].embed_dim
NH = attn_modules[0].num_heads
DH = D // NH

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

lsel = torch.where(lvbb & ((T == PAD).sum(1) >= 1))[0]
NL = len(lsel)
lar = torch.arange(NL)
slot = (T[lsel] == PAD).float().argmax(1)
lp, np_ = lpos[lsel], npos[lsel]
real_l = T[lsel] != PAD

# leptonic kinematics (units of 100 GeV unless noted)
plep = X[lsel][lar, lp]; pnu = X[lsel][lar, np_]
pt_lep = torch.sqrt((plep[:, :2] ** 2).sum(-1))
pt_nu = torch.sqrt((pnu[:, :2] ** 2).sum(-1))
vsum = plep[:, :2] + pnu[:, :2]
pt_lepW = torch.sqrt((vsum ** 2).sum(-1)).clamp(min=0.01)
dphi = torch.atan2(plep[:, 1], plep[:, 0]) - torch.atan2(pnu[:, 1], pnu[:, 0])
dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi
mt = torch.sqrt(torch.clamp(2 * pt_lep * pt_nu * (1 - torch.cos(dphi)), min=0))
isjet = ((T[lsel] == LJ) | (T[lsel] == SJ)).float()
pt_all = torch.sqrt((X[lsel][..., :2] ** 2).sum(-1))
ht_jets = (pt_all * isjet).sum(1)
ljet_lead = (pt_all * (T[lsel] == LJ).float()).max(1).values
print(f"lvbb+slot n={NL} | med pT(lep) {pt_lep.median()*100:.0f} GeV, MET {pt_nu.median()*100:.0f}, "
      f"pT(lepW) {pt_lepW.median()*100:.0f}, dphi {dphi.abs().median():.2f}")

# ---------- A + B: scalars at nu/lep + per-key decomposition (clean lvbb) ----------
def sval_table(blk):
    x_in = cache[f"block_{blk}_attention"]["input"][0]
    attn = attn_modules[blk]
    Wv = attn.in_proj_weight[2 * D:3 * D]; bv = attn.in_proj_bias[2 * D:3 * D]
    v = x_in @ Wv.t() + bv
    out = {}
    for h in range(NH):
        v_h = v[..., h * DH:(h + 1) * DH]
        Wout_h = attn.out_proj.weight[:, h * DH:(h + 1) * DH]
        VO = v_h @ Wout_h.t()
        dw = model.attention_blocks[blk]["bottleneck_down"][h].weight[0]
        out[h] = VO @ dw
    return out

WIRES = [(1, 3), (2, 2)]
CATS = ("lepton", "nu-self", "ljets", "sjets")
sc = {k: [] for k in WIRES}
contrib = {(k, tok): {c: [] for c in CATS} for k in WIRES for tok in ("nu", "lep")}
san = 0.0
for c0 in range(0, NL, 2048):
    sl = slice(c0, min(c0 + 2048, NL))
    xx, tt = X[lsel][sl], T[lsel][sl]
    with torch.no_grad():
        model(xx[..., :5], tt)
    B = len(xx); bidx = torch.arange(B)
    for (blk, h) in WIRES:
        bna = cache[f"block_{blk}_attention"]["bottleneck_activation"]
        aw = cache[f"block_{blk}_attention"]["attn_weights_per_head"]
        sv = sval_table(blk)[h]
        sc[(blk, h)].append(bna[bidx, h, np_[sl], 0])
        for tok, pos in (("nu", np_[sl]), ("lep", lp[sl])):
            con = aw[bidx, h, pos, :] * sv                       # [B,15]
            cats = {"lepton": con[bidx, lp[sl]],
                    "nu-self": con[bidx, np_[sl]],
                    "ljets": (con * (tt == LJ)).sum(-1),
                    "sjets": (con * (tt == SJ)).sum(-1)}
            for c, v_ in cats.items():
                contrib[((blk, h), tok)][c].append(v_)
            if tok == "nu":
                san = max(san, (con.sum(-1) - bna[bidx, h, pos, 0]).abs().max().item())
print(f"decomposition sanity (sum of keys vs cached scalar): max dev {san:.2e}")

print("\n=== A. what do the wire scalars at nu track? (Spearman rho, clean lvbb) ===")
feats = {"pT(lep)": pt_lep, "MET": pt_nu, "pT(lepW)": pt_lepW, "mT(lep,nu)": mt,
         "|dphi(lep,nu)|": dphi.abs(), "HT(jets)": ht_jets, "lead-ljet pT": ljet_lead,
         "pT(lepW)/HT(jets)": pt_lepW / ht_jets.clamp(min=0.01)}
print(f"{'feature':>20s} {'b1h3@nu':>9s} {'b2h2@nu':>9s}")
for nm, f in feats.items():
    r1 = spearmanr(f.numpy(), torch.cat(sc[(1, 3)]).numpy()).statistic
    r2 = spearmanr(f.numpy(), torch.cat(sc[(2, 2)]).numpy()).statistic
    print(f"{nm:>20s} {r1:9.3f} {r2:9.3f}")

print("\n=== B. per-key decomposition in clean lvbb (mean contribution) ===")
print(f"{'wire@token':>14s} " + " ".join(f"{c:>9s}" for c in CATS))
for (k, tok), d in contrib.items():
    print(f"b{k[0]}h{k[1]}@{tok:>4s}   " + " ".join(f"{torch.cat(d[c]).mean():9.3f}" for c in CATS))

# ---------- C. input-side surgery at fixed candidate ----------
rng = np.random.default_rng(0)
bsel = torch.where(boosted)[0]
wpb = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
don = rng.integers(len(bsel), size=NL)
ddir = X[bsel[don], wpb[don], :3].clone()

def set_kin(rows3, pt_t, m_gev):
    pt_now = torch.sqrt(rows3[:, 0] ** 2 + rows3[:, 1] ** 2).clamp(min=1e-6)
    p3 = rows3 * (pt_t / pt_now).unsqueeze(-1)
    E = torch.sqrt((p3 ** 2).sum(-1) + (m_gev / 100.0) ** 2)
    return p3, E

PT_C = 1.2 * pt_lepW                                   # fixed candidate, all arms

def rot_phi(rows, ang):
    c_, s_ = math.cos(ang), math.sin(ang)
    out = rows.clone()
    out[:, 0] = rows[:, 0] * c_ - rows[:, 1] * s_
    out[:, 1] = rows[:, 0] * s_ + rows[:, 1] * c_
    return out

def scale_tok(rows, f):
    out = rows.clone()
    m2 = torch.clamp(rows[:, 3] ** 2 - (rows[:, :3] ** 2).sum(-1), min=0)
    out[:, :3] = rows[:, :3] * (f if not torch.is_tensor(f) else f.unsqueeze(-1))
    out[:, 3] = torch.sqrt((out[:, :3] ** 2).sum(-1) + m2)
    return out

def arm(name, lep_fn=None, nu_fn=None):
    X2 = X[lsel].clone(); T2 = T[lsel].clone()
    if lep_fn is not None: X2[lar, lp] = lep_fn(X2[lar, lp])
    if nu_fn is not None: X2[lar, np_] = nu_fn(X2[lar, np_])
    rows = torch.zeros(NL, 7)
    p3, E = set_kin(ddir, PT_C, 80)
    rows[:, :3] = p3; rows[:, 3] = E; rows[:, 4] = WTAG
    X2[lar, slot] = rows; T2[lar, slot] = LJ
    out = fwd(X2, T2)
    # b1h3 at nu under this arm (last chunk only would be wrong; recompute fully)
    s13 = []
    with torch.no_grad():
        for c0 in range(0, NL, 2048):
            sl = slice(c0, min(c0 + 2048, NL))
            model(X2[sl][..., :5], T2[sl])
            bidx = torch.arange(sl.stop - sl.start)
            s13.append(cache["block_1_attention"]["bottleneck_activation"][bidx, 3, np_[sl], 0])
    pr = out.argmax(-1)
    iw = (pr[lar, slot] == W).float().mean()
    lw = (pr[lar, lp] == W).float().mean()
    print(f"{name:>34s} {iw:9.4f} {lw:9.4f} {torch.cat(s13).mean():10.3f}")

print("\n=== C. leptonic-side surgery at FIXED candidate (pT_c = 1.2 x pT(lepW)_orig) ===")
print(f"{'arm':>34s} {'P(ins=W)':>9s} {'P(lep=W)':>9s} {'b1h3@nu':>10s}")
arm("baseline (candidate only)")
arm("lep x1.3", lep_fn=lambda r: scale_tok(r, 1.3))
arm("nu (MET) x1.3", nu_fn=lambda r: scale_tok(r, 1.3))
arm("both x1.3", lep_fn=lambda r: scale_tok(r, 1.3), nu_fn=lambda r: scale_tok(r, 1.3))
arm("both x0.77", lep_fn=lambda r: scale_tok(r, 0.77), nu_fn=lambda r: scale_tok(r, 0.77))
arm("nu phi +90deg", nu_fn=lambda r: rot_phi(r, math.pi / 2))
arm("nu phi +180deg", nu_fn=lambda r: rot_phi(r, math.pi))
arm("swap lep/nu pT magnitudes",
    lep_fn=lambda r: scale_tok(r, (pt_nu / pt_lep.clamp(min=1e-3))),
    nu_fn=lambda r: scale_tok(r, (pt_lep / pt_nu.clamp(min=1e-3))))
arm("nu x0.1 (kill MET)", nu_fn=lambda r: scale_tok(r, 0.1))

# ---------- D. circuit-side edge knockouts ----------
KO = {"active": False}
def make_surgical(block_idx, block):
    bdown, bup = block["bottleneck_down"], block["bottleneck_up"]
    def hook_fn(module, inputs, kwargs, output):
        q0 = inputs[0]
        kpm = kwargs.get("key_padding_mask", None)
        kpm = F._canonical_mask(mask=kpm, mask_name="key_padding_mask",
                                other_type=F._none_or_dtype(kpm), other_name="",
                                target_type=q0.dtype)
        q = k = v = q0.transpose(1, 0)
        tgt, bsz, E_ = q.shape
        q, k, v = F._in_projection_packed(q, k, v, module.in_proj_weight, module.in_proj_bias)
        hd = E_ // NH
        q = q.view(tgt, bsz * NH, hd).transpose(0, 1)
        k = k.view(tgt, bsz * NH, hd).transpose(0, 1)
        v = v.view(tgt, bsz * NH, hd).transpose(0, 1)
        qs = q * math.sqrt(1.0 / float(hd))
        kpm_e = kpm.view(bsz, 1, 1, tgt).expand(-1, NH, -1, -1).reshape(bsz * NH, 1, tgt)
        A = torch.baddbmm(kpm_e, qs, k.transpose(-2, -1))
        A = F.softmax(A, dim=-1).view(bsz, NH, tgt, tgt)
        if KO["active"] and block_idx in KO["blocks"]:
            car = torch.arange(bsz)
            for g in range(KO["q"].shape[0]):
                for h in KO["heads"]:
                    row = A[car, h, KO["q"][g], :] * (~KO["kmask"][g]).float()
                    A[car, h, KO["q"][g], :] = row / row.sum(-1, keepdim=True).clamp(min=1e-9)
        ob = torch.bmm(A.view(bsz * NH, tgt, tgt), v).transpose(0, 1).contiguous().view(tgt * bsz, E_)
        per_head = torch.empty(bsz, NH, tgt, E_)
        for h in range(NH):
            Wr = module.out_proj.weight.transpose(0, 1)[h * hd:(h + 1) * hd]
            per_head[:, h] = torch.matmul(ob[:, h * hd:(h + 1) * hd], Wr).view(tgt, bsz, -1).transpose(0, 1)
        for h in range(NH):
            s = torch.matmul(per_head[:, h].reshape(-1, E_), bdown[h].weight.t())
            per_head[:, h] = torch.matmul(s, bup[h].weight.t()).view(bsz, tgt, E_)
        return per_head.sum(dim=1) + module.out_proj.bias, output[1]
    return hook_fn

for h in lib_handles: h.remove()
sur_handles = [blk["self_attention"].register_forward_hook(make_surgical(i, blk), with_kwargs=True)
               for i, blk in enumerate(model.attention_blocks)]

def build_r(r):
    X2 = X[lsel].clone(); T2 = T[lsel].clone()
    rows = torch.zeros(NL, 7)
    p3, E = set_kin(ddir, r * pt_lepW, 80)
    rows[:, :3] = p3; rows[:, 3] = E; rows[:, 4] = WTAG
    X2[lar, slot] = rows; T2[lar, slot] = LJ
    return X2, T2

def fwd_ko(x, types, blocks=None, heads=None, edges=None):
    outs = []
    with torch.no_grad():
        for c in range(0, len(x), 2048):
            sl = slice(c, min(c + 2048, len(x)))
            if blocks is None:
                KO["active"] = False
            else:
                KO.update(active=True, blocks=blocks, heads=heads,
                          q=torch.stack([g[0][sl] for g in edges]),
                          kmask=torch.stack([g[1][sl] for g in edges]))
            outs.append(model(x[sl][..., :5], types[sl]))
    KO["active"] = False
    return torch.cat(outs)

def km(*poss):
    m = torch.zeros(NL, 15, dtype=torch.bool)
    for p in poss: m[lar, p] = True
    return m
km_lepnu = km(lp, np_); km_ins = km(slot)
km_jets = ((T[lsel] == LJ) | (T[lsel] == SJ)); km_jets = km_jets.clone()

ALLH = tuple(range(NH))
print("\n=== D1. r=1.0 candidate (suppressed): cut the candidate's reads ===")
X2, T2 = build_r(1.0)
print(f"{'condition':>40s} {'P(ins=W)':>9s} {'P(lep=W)':>9s}")
for nm, blocks, edges in (
        ("baseline", None, None),
        ("KO ins->(lep,nu) b0+b1", {0, 1}, [(slot, km_lepnu)]),
        ("KO ins->(lep,nu) b2", {2}, [(slot, km_lepnu)]),
        ("KO ins->(lep,nu) all blocks", {0, 1, 2}, [(slot, km_lepnu)]),
        ("KO ins->(other jets) all blocks (ctrl)", {0, 1, 2}, [(slot, km_jets)]),
):
    o = fwd_ko(X2, T2) if blocks is None else fwd_ko(X2, T2, blocks, ALLH, edges)
    pr = o.argmax(-1)
    print(f"{nm:>40s} {(pr[lar, slot]==W).float().mean():9.4f} {(pr[lar, lp]==W).float().mean():9.4f}")

print("\n=== D2. r=2.0 candidate (lepton concedes): cut lep/nu's reads of it ===")
X2, T2 = build_r(2.0)
print(f"{'condition':>40s} {'P(ins=W)':>9s} {'P(lep=W)':>9s} {'P(nu=W)':>9s}")
for nm, blocks, heads, edges in (
        ("baseline", None, None, None),
        ("KO lep->ins all blocks", {0, 1, 2}, ALLH, [(lp, km_ins)]),
        ("KO lep->ins b1 only", {1}, ALLH, [(lp, km_ins)]),
        ("KO lep->ins b2 only", {2}, ALLH, [(lp, km_ins)]),
        ("KO lep->ins b1h3 only", {1}, (3,), [(lp, km_ins)]),
        ("KO lep->ins b2h2 only", {2}, (2,), [(lp, km_ins)]),
        ("KO nu->ins all blocks", {0, 1, 2}, ALLH, [(np_, km_ins)]),
        ("KO lep+nu->ins all blocks", {0, 1, 2}, ALLH, [(lp, km_ins), (np_, km_ins)]),
):
    o = fwd_ko(X2, T2) if blocks is None else fwd_ko(X2, T2, blocks, heads, edges)
    pr = o.argmax(-1)
    print(f"{nm:>40s} {(pr[lar, slot]==W).float().mean():9.4f} "
          f"{(pr[lar, lp]==W).float().mean():9.4f} {(pr[lar, np_]==W).float().mean():9.4f}")

for h in sur_handles: h.remove()
print("\ndone.")
