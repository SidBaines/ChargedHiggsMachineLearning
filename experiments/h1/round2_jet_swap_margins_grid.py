"""H1 round 2: WHAT drives the lep/nu "W vs none" (channel) decision?

Round 1 refuted the naive lepton->[head]->nu circuit: the decision is global and
redundant. Round-2 experiments (thesis model AND the d20 organism):

  A. Margin analysis: do corruption flips concentrate at low logit margin?
  B. JET-SYSTEM SWAP (decisive): transplant the full jet system from an event of the
     OPPOSITE true channel, leptons/nu untouched. If lep/nu predictions follow the
     jets, the channel decision is computed from the jet system.
  C. Corruption grid: untested single-feature corruptions (sjet/ljet masking,
     lepton/MET magnitude scaling, tag zeroing), flip rates split by true channel.

Usage: .venv/bin/python tmp_h1_round2.py [--model thesis-ent1-bn1-d152] [--batches 6]
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
p.add_argument("--model", default="thesis-ent1-bn1-d152")
p.add_argument("--batches", type=int, default=6)
ARGS = p.parse_args()

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))
PAD, NU = 5, 2
W_CLASS = 2          # model class for W products (truth 2 and 3 both map here)
stds = np.ones(7); stds[:4] = 1e5
torch.manual_seed(0)

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)
batches = []
for _ in range(ARGS.batches):
    b = next(dl)
    batches.append({"x": b["x"].clone(), "types": b["types"].clone()})

BN = 1 if "bn1" in ARGS.model else None
model, _ = load_model(ARGS.model, checkpoint_root=os.path.join(REPO, "tmp_checkpoints"),
                      register_bottleneck_hook=False)
model.eval()
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=BN)]

def fwd(x, types):
    with torch.no_grad():
        return model(x[..., :5], types)

def lepnu_idx(types):
    nu_m = types == NU; lep_m = (types == 0) | (types == 1)
    ok = (nu_m.sum(1) == 1) & (lep_m.sum(1) == 1)
    ii = torch.arange(len(types))[ok]
    npos = nu_m[ok].float().argmax(1); lpos = lep_m[ok].float().argmax(1)
    return ii, npos, lpos

# ---------- baseline pass: store events + preds + margins ----------
EV = []   # one dict per usable event
for b in batches:
    x, types = b["x"], b["types"]
    out = fwd(x, types)
    tru = torch.round(x[..., -1]).long()
    pred = out.argmax(-1)
    ii, npos, lpos = lepnu_idx(types)
    for k in range(len(ii)):
        i, np_, lp_ = ii[k].item(), npos[k].item(), lpos[k].item()
        EV.append(dict(
            x=x[i], types=types[i], nupos=np_, leppos=lp_,
            lvbb=(tru[i, lp_].item() == 3),
            p_nu=pred[i, np_].item(), p_lep=pred[i, lp_].item(),
            m_nu=(out[i, np_, W_CLASS] - out[i, np_, 0]).item(),
            m_lep=(out[i, lp_, W_CLASS] - out[i, lp_, 0]).item()))
n_lv = sum(e["lvbb"] for e in EV); n_qq = len(EV) - n_lv
acc_nu = np.mean([e["p_nu"] == (W_CLASS if e["lvbb"] else 0) for e in EV])
agree = np.mean([e["p_nu"] == e["p_lep"] for e in EV])
print(f"model={ARGS.model}  events={len(EV)} (lvbb {n_lv} / qqbb {n_qq})")
print(f"baseline: nu_acc={acc_nu:.4f}  P(nu=lep)={agree:.4f}")
m = np.array([e["m_nu"] for e in EV]); lv = np.array([e["lvbb"] for e in EV])
print(f"nu margin (logitW-logit0): lvbb mean {m[lv].mean():+.2f}  qqbb mean {m[~lv].mean():+.2f}")

# ---------- A. margin analysis under lepton-phi randomization ----------
flips, margins = [], []
for b in batches:
    x, types = b["x"].clone(), b["types"]
    lep_m = (types == 0) | (types == 1)
    th = torch.rand(x.shape[0]) * 2 * np.pi
    c, s = torch.cos(th).unsqueeze(1), torch.sin(th).unsqueeze(1)
    px, py = x[..., 0].clone(), x[..., 1].clone()
    x[..., 0] = torch.where(lep_m, px * c - py * s, px)
    x[..., 1] = torch.where(lep_m, px * s + py * c, py)
    out = fwd(x, types)
    pred = out.argmax(-1)
    ii, npos, lpos = lepnu_idx(types)
    base_out = fwd(b["x"], types)
    base_pred = base_out.argmax(-1)
    flips.append((pred[ii, npos] != base_pred[ii, npos]).numpy())
    margins.append(np.abs((base_out[ii, npos, W_CLASS] - base_out[ii, npos, 0]).numpy()))
flips = np.concatenate(flips); margins = np.concatenate(margins)
print(f"\n=== A. nu-flip rate under lepton-phi randomization, by baseline |margin| decile ===")
qs = np.quantile(margins, np.linspace(0, 1, 11))
for d in range(10):
    sel = (margins >= qs[d]) & (margins <= qs[d + 1])
    print(f"  decile {d} (|m| {qs[d]:6.2f}-{qs[d+1]:6.2f}): flip rate {flips[sel].mean():.4f}  (n={sel.sum()})")
try:
    from sklearn.metrics import roc_auc_score
    print(f"  AUC(|margin| predicts NO-flip) = {roc_auc_score(1 - flips, margins):.4f}  overall flip {flips.mean():.4f}")
except Exception:
    pass

# ---------- B. jet-system swap ----------
def hybrid(lep_src, jet_src):
    x = torch.zeros(15, 7); t = torch.full((15,), PAD, dtype=lep_src["types"].dtype)
    k = 0
    for r in range(15):
        if lep_src["types"][r].item() in (0, 1, NU):
            x[k] = lep_src["x"][r]; t[k] = lep_src["types"][r]; k += 1
    for r in range(15):
        if jet_src["types"][r].item() in (3, 4) and k < 15:
            x[k] = jet_src["x"][r]; t[k] = jet_src["types"][r]; k += 1
    return x, t

def run_hybrids(lep_events, jet_events, label, n_max=1500):
    rng = np.random.default_rng(0)
    n = min(n_max, len(lep_events))
    xs, ts = [], []
    for k in range(n):
        ls = lep_events[k]
        js = ls if jet_events is None else jet_events[rng.integers(len(jet_events))]
        x, t = hybrid(ls, js)
        xs.append(x); ts.append(t)
    X = torch.stack(xs); T = torch.stack(ts)
    preds = []
    for c in range(0, n, 1024):
        preds.append(fwd(X[c:c + 1024], T[c:c + 1024]).argmax(-1))
    pred = torch.cat(preds)
    nu_m = T == NU; lep_m = (T == 0) | (T == 1)
    npos = nu_m.float().argmax(1); lpos = lep_m.float().argmax(1)
    ar = torch.arange(n)
    p_nu, p_lep = pred[ar, npos], pred[ar, lpos]
    pW_nu = (p_nu == W_CLASS).float().mean().item()
    pW_lep = (p_lep == W_CLASS).float().mean().item()
    ag = (p_nu == p_lep).float().mean().item()
    print(f"  {label:42s} P(nu=W)={pW_nu:.4f}  P(lep=W)={pW_lep:.4f}  P(nu=lep)={ag:.4f}  (n={n})")

LV = [e for e in EV if e["lvbb"]]; QQ = [e for e in EV if not e["lvbb"]]
print(f"\n=== B. jet-system swap (lep/nu kept, ALL jets transplanted) ===")
print("  in-situ reference:")
print(f"  {'lvbb events as-is':42s} P(nu=W)={np.mean([e['p_nu']==W_CLASS for e in LV]):.4f}  "
      f"P(lep=W)={np.mean([e['p_lep']==W_CLASS for e in LV]):.4f}")
print(f"  {'qqbb events as-is':42s} P(nu=W)={np.mean([e['p_nu']==W_CLASS for e in QQ]):.4f}  "
      f"P(lep=W)={np.mean([e['p_lep']==W_CLASS for e in QQ]):.4f}")
run_hybrids(LV, None, "lvbb leps + OWN jets (repack control)")
run_hybrids(LV, LV,   "lvbb leps + jets from OTHER lvbb (control)")
run_hybrids(LV, QQ,   "lvbb leps + jets from qqbb  <-- KEY")
run_hybrids(QQ, None, "qqbb leps + OWN jets (repack control)")
run_hybrids(QQ, QQ,   "qqbb leps + jets from OTHER qqbb (control)")
run_hybrids(QQ, LV,   "qqbb leps + jets from lvbb  <-- KEY")

# ---------- C. corruption grid, flip rates split by channel ----------
def corrupt_and_measure(name, fn):
    res = {"lv_flip": [0, 0], "qq_flip": [0, 0], "agree": [0, 0]}
    for b in batches:
        x, types = b["x"].clone(), b["types"].clone()
        x, types = fn(x, types)
        out = fwd(x, types)
        pred = out.argmax(-1)
        bx, btypes = b["x"], b["types"]
        base_pred = fwd(bx, btypes).argmax(-1)
        tru = torch.round(bx[..., -1]).long()
        ii, npos, lpos = lepnu_idx(btypes)   # positions unchanged by these corruptions
        lvb = tru[ii, lpos] == 3
        f = pred[ii, npos] != base_pred[ii, npos]
        res["lv_flip"][0] += f[lvb].sum().item(); res["lv_flip"][1] += lvb.sum().item()
        res["qq_flip"][0] += f[~lvb].sum().item(); res["qq_flip"][1] += (~lvb).sum().item()
        ag = pred[ii, npos] == pred[ii, lpos]
        res["agree"][0] += ag.sum().item(); res["agree"][1] += len(ii)
    print(f"  {name:34s} nu-flip lvbb {res['lv_flip'][0]/max(res['lv_flip'][1],1):.4f}  "
          f"qqbb {res['qq_flip'][0]/max(res['qq_flip'][1],1):.4f}  "
          f"P(nu=lep) {res['agree'][0]/res['agree'][1]:.4f}")

def mask_type(tt):
    def fn(x, types):
        m = types == tt
        x[m] = 0.0; types[m] = PAD
        return x, types
    return fn

def scale_type(tt_list, s):
    def fn(x, types):
        m = torch.zeros_like(types, dtype=torch.bool)
        for tt in tt_list: m |= types == tt
        x[..., :4] = torch.where(m.unsqueeze(-1), x[..., :4] * s, x[..., :4])
        return x, types
    return fn

def zero_tag(tt_list):
    def fn(x, types):
        m = torch.zeros_like(types, dtype=torch.bool)
        for tt in tt_list: m |= types == tt
        x[..., 4] = torch.where(m, torch.zeros_like(x[..., 4]), x[..., 4])
        return x, types
    return fn

print(f"\n=== C. corruption grid (nu pred-flip rate by true channel) ===")
corrupt_and_measure("mask ALL ljets (type 3)", mask_type(3))
corrupt_and_measure("mask ALL sjets (type 4)", mask_type(4))
corrupt_and_measure("zero tag on all jets", zero_tag([3, 4]))
corrupt_and_measure("zero tag on ljets only", zero_tag([3]))
corrupt_and_measure("lepton 4-mom x2", scale_type([0, 1], 2.0))
corrupt_and_measure("lepton 4-mom x0.5", scale_type([0, 1], 0.5))
corrupt_and_measure("nu(MET) 4-mom x2", scale_type([NU], 2.0))
corrupt_and_measure("nu(MET) 4-mom x0.5", scale_type([NU], 0.5))
corrupt_and_measure("lepton+nu 4-mom x2", scale_type([0, 1, NU], 2.0))
corrupt_and_measure("lepton+nu 4-mom x0.5", scale_type([0, 1, NU], 0.5))

for h in handles: h.remove()
print("\ndone.")
