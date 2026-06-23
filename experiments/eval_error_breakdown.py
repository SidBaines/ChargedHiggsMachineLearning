"""Per-category ERROR-TYPE breakdown for a trained reco model.

Beyond PerfectRecoPct, this asks *what kind* of mistake a model makes on the events it
gets wrong: hallucinate a resolved Higgs (small-R jets -> H), invent a hadronic W
(a jet -> W), lose the true H jet, swap the two large-R jets (H<->W), or botch the
lepton/neutrino. Weighted by MC_Wts, sliced by true reco category.

Object types {0:e,1:mu,2:nu,3:ljet(large-R),4:sjet(small-R),5:pad}; labels {0:none,1:H,2:W}.
truth x[...,-1] in {0:none,1:H,2:W-had,3:W-lep} -> collapsed ts {0,1,2}.

CLI:  .venv/bin/python experiments/eval_error_breakdown.py <tag|out_dir> ... [--batches N] [--save]
      --batches default 150 representative batches (0 = full val); --save writes
      error_profile.json into each run's output dir.
Reusable:  from eval_error_breakdown import profile_outdir   # returns the profile dict
"""
import os, sys, json, re, glob, argparse
import numpy as np, torch
torch.set_num_threads(2)  # be polite to a concurrently-running training sweep
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO)
from models.models import TestNetwork
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from utils.utils import check_category

DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/"); DSIDS = list(range(510115, 510125))
_stds = np.ones(7); _stds[:4] = 1e5
# flags reported per category (weighted fraction OF that category's MISSED events)
FLAGS = {4: ["sjet->H (resolved-H)", "jet->W (hadronic-W)", "lost true-H jet", "lepton/nu W err"],
         5: ["H<->W jet swap", "sjet->H (resolved-H)", "sjet->W (resolved-W)", "lost true-H jet"]}


def _build(cfg):
    return TestNetwork(hidden_dim_attn=cfg["d_attn"], hidden_dim=cfg["d_model"], feature_set=cfg["feature_set"],
        bottleneck_attention=cfg["bottleneck_attention"], include_mlp=cfg["include_mlp"],
        num_attention_blocks=cfg["num_blocks"], hidden_dim_mlp=cfg["d_mlp"], num_heads=cfg["num_heads"],
        embedding_size=cfg["embedding_size"], num_classes=cfg["num_classes"],
        use_lorentz_invariant_features=cfg["use_lorentz_invariant_features"], dropout_p=cfg["dropout_p"],
        num_particle_types=cfg["num_particle_types"], num_object_net_layers=cfg["num_object_net_layers"],
        is_layer_norm=cfg["is_layer_norm"]).to("cpu")


def profile_outdir(out_dir, batches=150):
    """Load the final checkpoint in out_dir and return the per-category error profile dict."""
    cfg = json.load(open(f"{out_dir}/config.json")); nctx, mi = cfg["num_particle_types"], cfg["model_input_vars"]
    ck = sorted(glob.glob(f"{out_dir}/models/*/chkpt*.pth"), key=lambda p: int(re.search(r"chkpt(\d+)", p).group(1)))[-1]
    m = _build(cfg); m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
    dl = ProportionalMemoryMappedDataset(is_train=False, shuffle=False, shuffle_batch=True,
        N_Real_Vars_In_File=7, N_Real_Vars_To_Return=7, memmap_paths={d: f"{DATA}dsid_{d}.memmap" for d in DSIDS},
        max_objs_in_memmap=15, batch_size=cfg["batch_size"], device="cpu", n_splits=2, validation_split_idx=0,
        n_targets=3, means=None, stds=_stds, objs_to_output=15, signal_only=True, has_eventNumbers=True)
    dl._reset_indices(); nb = len(dl) if batches == 0 else min(batches, len(dl))
    wtot = {4: 0.0, 5: 0.0}; wperf = {4: 0.0, 5: 0.0}; wmiss = {4: 0.0, 5: 0.0}
    fl = {C: {f: 0.0 for f in FLAGS[C]} for C in (4, 5)}
    with torch.no_grad():
        for _ in range(nb):
            b = next(dl); x, types, w = b["x"], b["types"], b["MC_Wts"].float()
            truth = x[..., -1]; pred = m(x[..., :mi], types).squeeze().argmax(-1)
            ts = (truth == 1).long() + (truth > 1).long() * 2; nonpad = types != (nctx - 1)
            perfect = ((pred == ts) | ~nonpad).all(-1)
            cat = check_category(types, truth, nctx - 1, use_torch=True)
            sjet, ljet, lepnu = (types == 4), (types == 3), ((types == 0) | (types == 1) | (types == 2))
            ev = {"sjet->H (resolved-H)": (sjet & nonpad & (pred == 1)).any(-1),
                  "jet->W (hadronic-W)": ((sjet | ljet) & nonpad & (pred == 2)).any(-1),
                  "sjet->W (resolved-W)": (sjet & nonpad & (pred == 2)).any(-1),
                  "lost true-H jet": ((ts == 1) & nonpad & (pred != 1)).any(-1),
                  "lepton/nu W err": (lepnu & (ts == 2) & nonpad & (pred != 2)).any(-1),
                  "H<->W jet swap": ((ljet & (ts == 1) & (pred == 2)) | (ljet & (ts == 2) & (pred == 1))).any(-1)}
            for C in (4, 5):
                inC = (cat == C); miss = inC & ~perfect
                wtot[C] += (w * inC).sum().item(); wperf[C] += (w * (inC & perfect)).sum().item()
                wmiss[C] += (w * miss).sum().item()
                for f in fl[C]:
                    fl[C][f] += (w * (miss & ev[f])).sum().item()
    prof = {"out_dir": out_dir, "d_model": cfg["d_model"], "blocks": cfg["num_blocks"],
            "role": "boosted specialist" if cfg.get("keep_cats") == [4, 5] else "all-cats generalist",
            "batches": nb}
    for C in (4, 5):
        prof[f"cat{C}"] = {"perfect_reco": (wperf[C] / wtot[C]) if wtot[C] else None,
                           "miss_rate": (wmiss[C] / wtot[C]) if wtot[C] else None,
                           "miss_errors": {f: (fl[C][f] / wmiss[C]) if wmiss[C] else None for f in fl[C]}}
    return prof


def _print(tag, p):
    print(f"\n=== {tag}  ({p['role']}; {p['d_model']}d/{p['blocks']}b; {p['batches']} batches) ===")
    for C in (4, 5):
        c = p[f"cat{C}"]
        print(f"  cat{C}: perfect-reco={c['perfect_reco']:.4f}  (miss {c['miss_rate']:.4f})  — of MISSES:")
        for f in FLAGS[C]:
            v = c["miss_errors"][f]
            print(f"       {(v*100 if v is not None else 0):5.1f}%  {f}")


def _resolve(arg):
    if os.path.isfile(os.path.join(arg, "config.json")):
        return arg
    return re.search(r"^out: (.+)$", open(f"tmp_suite/{arg}.log").read(), re.M).group(1).strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tags", nargs="+")
    ap.add_argument("--batches", type=int, default=150); ap.add_argument("--save", action="store_true")
    A = ap.parse_args()
    for tag in A.tags:
        od = _resolve(tag); p = profile_outdir(od, A.batches); _print(tag, p)
        if A.save:
            json.dump(p, open(f"{od}/error_profile.json", "w"), indent=1); print(f"   saved {od}/error_profile.json")
