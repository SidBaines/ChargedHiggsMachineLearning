"""H1 wave 1, behavioral half: B1 claim bookkeeping, B2 shared-score margins,
B3 error autopsy, A1 W-ljet mass dose-response, A6 permutation control.

Preregistrations in docs/H1_ASK_THE_JETS_TEST_PLAN.md. Types: 3=LJET, 4=SJET.

Usage: .venv/bin/python tmp_h1_wave1_behav.py [--model ent1-d20-2blk] [--batches 6]
       [--ckpt-override /path/to/chkpt.pth]
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
W = 2
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
jet_m = (T == LJ) | (T == SJ)
boosted = (~lvbb) & (((T == LJ) & (TRU == 2)).sum(1) == 1)
resolved = (~lvbb) & (((T == SJ) & (TRU == 2)).sum(1) == 2)
STRATA = (("lvbb", lvbb), ("qq-boosted", boosted), ("qq-resolved", resolved))

out = fwd(X, T)
pred = out.argmax(-1)
margin = out[..., W] - out[..., 0]                     # per-token W-vs-none margin
lep_W = pred[ar, lpos] == W
nu_W = pred[ar, npos] == W
jet_claims = ((pred == W) & jet_m)                     # per-token bool
n_jet_claims = jet_claims.sum(1)
jet_sys_claims = n_jet_claims > 0
evt_correct = lep_W == lvbb                            # lep verdict correct

print(f"model={ARGS.model}  events={N} (lvbb {lvbb.sum()}, boosted {boosted.sum()}, resolved {resolved.sum()})")

# ---------------- B1: claim bookkeeping ----------------
print("\n=== B1. claim bookkeeping ===")
print("  P(lep=W vs jet-system-claims) joint, per stratum [XOR prediction: ~all mass on 'exactly one']")
for lab, m in STRATA:
    a = jet_sys_claims[m].float(); l = lep_W[m].float()
    both = (a * l).mean(); neither = ((1 - a) * (1 - l)).mean()
    print(f"  {lab:12s} P(jet-sys W)={a.mean():.4f} P(lep W)={l.mean():.4f} "
          f"both={both:.4f} neither={neither:.4f} XOR-ok={1-both-neither:.4f}")
print("  # claiming jets | jet-system claims  [predict: boosted->1, resolved->2]")
for lab, m in STRATA:
    mm = m & jet_sys_claims
    if mm.sum() == 0: continue
    c = n_jet_claims[mm]
    dist = [(c == k).float().mean().item() for k in (1, 2, 3)]
    print(f"  {lab:12s} n={mm.sum().item():5d}  1:{dist[0]:.3f} 2:{dist[1]:.3f} 3+:{(c>=3).float().mean():.3f}")
print("  same, split by lep-verdict correctness (errors = claim transfers?):")
for lab, m in (("correct", evt_correct), ("error", ~evt_correct)):
    a = jet_sys_claims[m].float(); l = lep_W[m].float()
    print(f"  {lab:12s} n={m.sum().item():5d}  XOR-ok={1-(a*l).mean()-((1-a)*(1-l)).mean():.4f}")
print(f"  P(nu=lep) overall: {(nu_W == lep_W).float().mean():.4f}")

# ---------------- B2: shared-score margins ----------------
print("\n=== B2. lep margin vs best-jet margin [predict: anti-correlated within channel] ===")
jm = margin.masked_fill(~jet_m, -1e9)
best_jet_margin = jm.max(1).values
lep_margin = margin[ar, lpos]
def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)
print(f"  all events:   r = {pearson(lep_margin, best_jet_margin):+.4f}")
for lab, m in STRATA:
    print(f"  {lab:12s}  r = {pearson(lep_margin[m], best_jet_margin[m]):+.4f}  (n={m.sum().item()})")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 5))
for lab, m, c in (("lvbb", lvbb, "tab:blue"), ("qq-boosted", boosted, "tab:orange"),
                  ("qq-resolved", resolved, "tab:green")):
    sel = torch.where(m)[0][:3000]
    ax.scatter(best_jet_margin[sel], lep_margin[sel], s=2, alpha=0.25, label=lab, color=c)
ax.set_xlabel("best jet W-margin"); ax.set_ylabel("lep W-margin"); ax.legend()
ax.set_title(f"B2 shared-score test ({ARGS.model})")
os.makedirs(f"{REPO}/tmp_plots", exist_ok=True)
fig.tight_layout(); fig.savefig(f"{REPO}/tmp_plots/wave1_B2_{ARGS.model}.png", dpi=120)
print(f"  saved tmp_plots/wave1_B2_{ARGS.model}.png")

# ---------------- B3: error autopsy (qqbb, lep wrongly = W) ----------------
print("\n=== B3. error autopsy [predict: errors = W-detector misses, off-window W-jet] ===")
m_obj = torch.sqrt(torch.clamp(X[..., 3] ** 2 - (X[..., :3] ** 2).sum(-1), min=0)) * 100.0
pt_obj = torch.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2) * 100.0
for lab, m in (("qq-boosted", boosted), ("qq-resolved", resolved)):
    err = m & lep_W; cor = m & ~lep_W
    print(f"  {lab}: error rate {err.sum().item()/m.sum().item():.4f} (n_err={err.sum().item()})")
    print(f"    P(no jet claims W) | error: {(~jet_sys_claims[err]).float().mean():.4f}   | correct: {(~jet_sys_claims[cor]).float().mean():.4f}")
    if lab == "qq-boosted":
        wpos_mask = (T == LJ) & (TRU == 2)
        wm = (m_obj * wpos_mask).sum(1); wpt = (pt_obj * wpos_mask).sum(1)
        for nm, sel in (("error", err), ("correct", cor)):
            if sel.sum() > 5:
                print(f"    W-ljet mass | {nm}: median {wm[sel].median():.1f} GeV "
                      f"(16-84%: {np.percentile(wm[sel].numpy(),16):.0f}-{np.percentile(wm[sel].numpy(),84):.0f})   "
                      f"pT median {wpt[sel].median():.0f} GeV")
    else:
        mjj = torch.zeros(N)
        for i in torch.where(m)[0]:
            rows = torch.where((T[i] == SJ) & (TRU[i] == 2))[0]
            p4 = X[i, rows, :4].sum(0)
            mjj[i] = torch.sqrt(torch.clamp(p4[3]**2 - (p4[:3]**2).sum(), min=0)) * 100
        for nm, sel in (("error", err), ("correct", cor)):
            if sel.sum() > 5:
                print(f"    m(jj) | {nm}: median {mjj[sel].median():.1f} GeV "
                      f"(16-84%: {np.percentile(mjj[sel].numpy(),16):.0f}-{np.percentile(mjj[sel].numpy(),84):.0f})")

# ---------------- A1: W-ljet mass dose-response (boosted) ----------------
print("\n=== A1. W-ljet mass dose-response (boosted qqbb) "
      "[predict: claim bump ~80-90; lep flips mirror it both sides] ===")
bsel = torch.where(boosted)[0]
wpos = ((T[bsel] == LJ) & (TRU[bsel] == 2)).float().argmax(1)
bar = torch.arange(len(bsel))
print(f"  events: {len(bsel)}; as-is W-ljet mass median "
      f"{m_obj[bsel, wpos].median():.1f} GeV")
print(f"  {'m_set GeV':>10s} {'P(Wjet claims)':>15s} {'P(lep=W)':>10s} {'med lep margin':>15s}")
o = fwd(X[bsel], T[bsel])
pr = o.argmax(-1)
print(f"  {'as-is':>10s} {(pr[bar, wpos]==W).float().mean():15.4f} "
      f"{(pr[bar, lpos[bsel]]==W).float().mean():10.4f} "
      f"{(o[bar, lpos[bsel], W]-o[bar, lpos[bsel], 0]).median():15.2f}")
for m_gev in (5, 20, 40, 60, 70, 80, 90, 100, 110, 125, 150, 175, 200, 250, 300):
    X2 = X[bsel].clone()
    p2 = (X2[bar, wpos, :3] ** 2).sum(-1)
    X2[bar, wpos, 3] = torch.sqrt(p2 + (m_gev / 100.0) ** 2)
    o = fwd(X2, T[bsel])
    pr = o.argmax(-1)
    print(f"  {m_gev:10d} {(pr[bar, wpos]==W).float().mean():15.4f} "
          f"{(pr[bar, lpos[bsel]]==W).float().mean():10.4f} "
          f"{(o[bar, lpos[bsel], W]-o[bar, lpos[bsel], 0]).median():15.2f}")

# ---------------- A6: permutation control ----------------
print("\n=== A6. slot-permutation control [predict: zero change] ===")
g = torch.Generator().manual_seed(1)
perm = torch.argsort(torch.rand(N, 15, generator=g), dim=1)
Xp = torch.gather(X, 1, perm.unsqueeze(-1).expand(-1, -1, 7))
Tp = torch.gather(T, 1, perm)
op = fwd(Xp, Tp)
op_back = torch.zeros_like(op)
op_back.scatter_(1, perm.unsqueeze(-1).expand(-1, -1, 3), op)
real = (T != PAD)
dlogit = (op_back - out).abs()[real]
mismatch = (op_back.argmax(-1) != pred)[real].float().mean()
print(f"  pred mismatch rate (real objects): {mismatch:.6f}   max|dlogit|: {dlogit.max():.2e}")

for h in handles: h.remove()
print("\ndone.")
