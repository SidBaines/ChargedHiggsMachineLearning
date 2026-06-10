"""H1 (neutrino-pairing circuit) — Steps 1+2 on the thesis model, 20250321v1 val signal.

Step 1: behavioral — does pred(nu) track pred(lepton)?
Step 2: attention — which heads carry nu<-lep / lep<-nu, conditioned on true channel?
Preview of step 4: does the b2h3 scalar message into nu encode the lepton's class?
"""
import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))
PAD, NU, LEP_TYPES = 5, 2, (0, 1)
N_BLOCKS, N_HEADS = 3, 4
stds = np.ones(7); stds[:4] = 1e5

dl = ProportionalMemoryMappedDataset(
    N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
    memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
    max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
    n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
    means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True,
)

model, _ = load_model("thesis-ent1-bn1-d152", checkpoint_root=os.path.join(REPO, "tmp_checkpoints"), register_bottleneck_hook=False)
cache = ActivationCache()
handles = [m.register_forward_hook(fn, with_kwargs=True)
           for m, fn in hook_attention_heads(model, cache, detach=True,
                 SINGLE_ATTENTION=False, bottleneck_attention_output=1)]

agree = {"all": [0, 0]}
follow = np.zeros((3, 3), dtype=np.int64)   # pred_lep x pred_nu
attn_nu2lep = np.zeros((2, N_BLOCKS, N_HEADS)); attn_nu2lep_n = np.zeros(2)
attn_lep2nu = np.zeros((2, N_BLOCKS, N_HEADS))
bn_msg, bn_label = [], []                    # b2h3 scalar into nu; lepton truth==W?
tru_eq = [0, 0]

N_BATCHES = 25
for bi in range(N_BATCHES):
    b = next(dl)
    x, types = b["x"], b["types"]
    with torch.no_grad():
        out = model(x[..., :5], types)
    tru = torch.round(x[..., -1]).long()
    pred = out.argmax(-1)

    has_nu = (types == NU).sum(1) == 1
    has_lep = ((types == 0) | (types == 1)).sum(1) == 1
    ok = has_nu & has_lep
    idx = torch.arange(len(types))[ok]
    nu_pos = (types[ok] == NU).float().argmax(1)
    lep_pos = ((types[ok] == 0) | (types[ok] == 1)).float().argmax(1)

    p_nu = pred[idx, nu_pos]; p_lep = pred[idx, lep_pos]
    t_nu = tru[idx, nu_pos]; t_lep = tru[idx, lep_pos]
    tru_eq[0] += (t_nu == t_lep).sum().item(); tru_eq[1] += len(idx)
    agree["all"][0] += (p_nu == p_lep).sum().item(); agree["all"][1] += len(idx)
    for pl in range(3):
        for pn in range(3):
            follow[pl, pn] += ((p_lep == pl) & (p_nu == pn)).sum().item()

    is_wlv = (t_lep == 3).numpy().astype(int)  # true leptonic-W event (lep is W product; class 2 = W)
    for blk in range(N_BLOCKS):
        aw = cache[f"block_{blk}_attention"]["attn_weights_per_head"]  # [B,H,Q,K]
        a = aw[idx]
        for h in range(N_HEADS):
            n2l = a[torch.arange(len(idx)), h, nu_pos, lep_pos].numpy()
            l2n = a[torch.arange(len(idx)), h, lep_pos, nu_pos].numpy()
            for w in (0, 1):
                attn_nu2lep[w, blk, h] += n2l[is_wlv == w].sum()
                attn_lep2nu[w, blk, h] += l2n[is_wlv == w].sum()
        if blk == 2:
            bna = cache["block_2_attention"]["bottleneck_activation"]  # [B,H,N,1]
            bn_msg.append(bna[idx, 3, nu_pos, 0].numpy())
            bn_label.append(is_wlv)
    for w in (0, 1):
        attn_nu2lep_n[w] += (is_wlv == w).sum()

print(f"events used: {tru_eq[1]}   truth(nu)==truth(lep): {tru_eq[0]/tru_eq[1]:.4f}")
print(f"\n=== Step 1: behavioral ===")
print(f"P(pred_nu == pred_lep) = {agree['all'][0]/agree['all'][1]:.4f}")
print("joint counts pred_lep(rows) x pred_nu(cols):")
print(follow)
rn = follow / follow.sum(1, keepdims=True)
print("row-normalised:"); print(np.round(rn, 3))

print(f"\n=== Step 2: attention nu->lep (mean), by true channel ===")
print(f"{'':12s}" + "".join(f"  b{b}h{h}" for b in range(N_BLOCKS) for h in range(N_HEADS)))
for w, lab in ((1, "W_lv (lep=W)"), (0, "other       ")):
    row = attn_nu2lep[w] / attn_nu2lep_n[w]
    print(f"{lab}" + "".join(f" {row[b,h]:5.2f}" for b in range(N_BLOCKS) for h in range(N_HEADS)))
print(f"--- attention lep->nu (mean), by true channel ---")
for w, lab in ((1, "W_lv (lep=W)"), (0, "other       ")):
    row = attn_lep2nu[w] / attn_nu2lep_n[w]
    print(f"{lab}" + "".join(f" {row[b,h]:5.2f}" for b in range(N_BLOCKS) for h in range(N_HEADS)))

# step-4 preview: does the b2h3 scalar message into nu separate W_lv vs other?
msg = np.concatenate(bn_msg); lab = np.concatenate(bn_label)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(lab, msg)
print(f"\n=== Step 4 preview: b2h3 scalar message INTO the neutrino ===")
print(f"separation of true-channel by that single number: AUC = {max(auc,1-auc):.4f}")
print(f"mean msg | W_lv: {msg[lab==1].mean():+.3f}   | other: {msg[lab==0].mean():+.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
for w, lab_, c in ((1, "true W→lν (lep is W product)", "tab:blue"), (0, "other channels", "tab:orange")):
    ax.hist(msg[lab == w], bins=80, alpha=0.6, density=True, label=lab_, color=c)
ax.set_xlabel("b2h3 bottleneck scalar at the neutrino position"); ax.set_ylabel("density")
ax.legend(); ax.set_title(f"The single number head b2h3 sends the neutrino (AUC={max(auc,1-auc):.3f})")
fig.tight_layout(); fig.savefig(f"{REPO}/tmp_plots/h1_b2h3_message_hist.png", dpi=130)
print("saved: tmp_plots/h1_b2h3_message_hist.png")
for h in handles: h.remove()
