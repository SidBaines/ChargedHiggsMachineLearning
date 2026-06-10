"""H1 Step 3: head ablations on the thesis model — does killing the channel-conditional
heads selectively destroy the NEUTRINO's assignment?

Ablation = subtract that head's (post-bottleneck) contribution from the attention output
during the forward pass, so downstream blocks see the ablated stream.
Truth mapping: file {0:none,1:H,2:Whad,3:Wlep} -> model classes {0,1,2,2}.
"""
import os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
from models.registry import load_model
from interp.activations import ActivationCache, hook_attention_heads
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))
PAD, NU = 5, 2
stds = np.ones(7); stds[:4] = 1e5
TRUTH_MAP = torch.tensor([0, 1, 2, 2])

def make_loader():
    return ProportionalMemoryMappedDataset(
        N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7,
        memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
        max_objs_in_memmap=15, batch_size=2048, device="cpu", is_train=False,
        n_splits=2, validation_split_idx=0, n_targets=3, shuffle=False, shuffle_batch=False,
        means=None, stds=stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)

model, _ = load_model("thesis-ent1-bn1-d152", checkpoint_root=os.path.join(REPO, "tmp_checkpoints"), register_bottleneck_hook=False)
cache = ActivationCache()
pairs = hook_attention_heads(model, cache, detach=True, SINGLE_ATTENTION=False,
                             bottleneck_attention_output=1)
attn_modules = [m for m, _ in pairs if type(m).__name__ == "MultiheadAttention"]

base_handles = [m.register_forward_hook(fn, with_kwargs=True) for m, fn in pairs]

ABLATE = {}  # block -> list of heads, mutated per config

def make_ablation_hook(blk):
    def hook(module, args, kwargs, output):
        heads = ABLATE.get(blk, [])
        if not heads:
            return output
        out, w = output
        per_head = cache[f"block_{blk}_attention"]["attn_output_per_head"]
        for h in heads:
            out = out - per_head[:, h]
        return (out, w)
    return hook

for blk, m in enumerate(attn_modules):
    base_handles.append(m.register_forward_hook(make_ablation_hook(blk), with_kwargs=True))

CONFIGS = [("baseline", {})]
CONFIGS += [(f"b{b}h{h}", {b: [h]}) for b in range(3) for h in range(4)]
CONFIGS += [("b2h0+b2h3", {2: [0, 3]}),
            ("b1h3+b2h0+b2h3", {1: [3], 2: [0, 3]}),
            ("b0h0 (always-on)", {0: [0]})]

N_BATCHES = 6
print(f"{'config':18s} {'nu_acc':>7s} {'lep_acc':>8s} {'jet_acc':>8s} {'P(nu=lep)':>10s} {'evt_perfect':>12s}")
for name, abl in CONFIGS:
    ABLATE.clear(); ABLATE.update(abl)
    dl = make_loader()
    nu_c = nu_n = lep_c = lep_n = jet_c = jet_n = agree = pairs_n = perf = evt = 0
    for _ in range(N_BATCHES):
        b = next(dl)
        x, types = b["x"], b["types"]
        with torch.no_grad():
            out = model(x[..., :5], types)
        tru = TRUTH_MAP[torch.round(x[..., -1]).long().clamp(0, 3)]
        pred = out.argmax(-1)
        real = types != PAD
        nu_m = types == NU; lep_m = (types == 0) | (types == 1); jet_m = (types == 3) | (types == 4)
        nu_c += (pred[nu_m] == tru[nu_m]).sum().item(); nu_n += nu_m.sum().item()
        lep_c += (pred[lep_m] == tru[lep_m]).sum().item(); lep_n += lep_m.sum().item()
        jet_c += (pred[jet_m] == tru[jet_m]).sum().item(); jet_n += jet_m.sum().item()
        ok = (nu_m.sum(1) == 1) & (lep_m.sum(1) == 1)
        ii = torch.arange(len(types))[ok]
        npos = nu_m[ok].float().argmax(1); lpos = lep_m[ok].float().argmax(1)
        agree += (pred[ii, npos] == pred[ii, lpos]).sum().item(); pairs_n += len(ii)
        perfect = ((pred == tru) | ~real).all(1)
        perf += perfect.sum().item(); evt += len(perfect)
    print(f"{name:18s} {nu_c/nu_n:7.4f} {lep_c/lep_n:8.4f} {jet_c/jet_n:8.4f} "
          f"{agree/pairs_n:10.4f} {perf/evt:12.4f}")
