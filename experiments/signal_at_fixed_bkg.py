"""Signal yield at a fixed retained-background working point, per H+ mass point.

For each trained model in a joint real-bkg run_root, score the validation set, find
the threshold where the retained background yield == a target (default 200 events),
and report the retained SIGNAL yield per H+ mass DSID. Makes the "signal kept when N
background events survive" comparison that lines up with the old pipeline (which
pre-cuts events before classification).

Weights: SIGNED MC_Wts (physical-intent per-event weights from the memmap, summed →
expected yields). Val is ~half the dataset (position split) -> yields scaled by
--split-factor (default 2) to full-dataset equivalents.

CAVEATS (R6): the absolute scale depends on the ntuple weight normalization (signal
is reference-cross-section normalized); the cross-model / cross-mass *shape* is the
robust comparison. The retained-bkg target is approximate to the same degree.

Usage:
  python signal_at_fixed_bkg.py [run_root] [--bkg-targets 100,200,500] [--device mps]
                                [--eval-batches N] [--plot out.png]
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

EXP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXP)
REPO = os.path.dirname(EXP); sys.path.insert(0, REPO)
from joint_primitives import event_score_from_object_logits
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from models.models import TestNetwork

DSID_MASS_TEV = {510115: 0.8, 510116: 0.9, 510117: 1.0, 510118: 1.2, 510119: 1.4,
                 510120: 1.6, 510121: 1.8, 510122: 2.0, 510123: 2.5, 510124: 3.0}
SIG_DSIDS = sorted(DSID_MASS_TEV)
DATA_DEFAULT = os.path.join(REPO, "tmp_data_20250321v1_all") + "/"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_root", nargs="?", default=None,
                   help="output/joint_real_bkg_* dir (default: newest)")
    p.add_argument("--data-dir", default=DATA_DEFAULT)
    p.add_argument("--bkg-targets", default="100,200,500")
    p.add_argument("--device", choices=["cpu", "mps"], default="mps")
    p.add_argument("--eval-batches", type=int, default=0, help="0 = full val set")
    p.add_argument("--split-factor", type=float, default=2.0,
                   help="scale val yields to full dataset (position split -> ~2)")
    p.add_argument("--plot", default=None, help="path to save the per-mass plot PNG")
    p.add_argument("--report-target", type=float, default=200.0,
                   help="which bkg target the per-mass table + plot use")
    return p.parse_args()


def newest_run_root():
    c = sorted(glob.glob(os.path.join(REPO, "output", "joint_real_bkg_*")))
    return c[-1] if c else None


def discover_runs(run_root):
    runs = []
    for lg in sorted(glob.glob(os.path.join(run_root, "logs", "*.log"))):
        name = os.path.basename(lg)[:-4]
        out = None
        with open(lg) as f:
            for line in f:
                if line.startswith("out: "):
                    out = line.split("out: ", 1)[1].strip(); break
        if not out:
            continue
        cfg = os.path.join(out, "config.json")
        cks = sorted(glob.glob(os.path.join(out, "models", "*", "chkpt*.pth")),
                     key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
        if os.path.exists(cfg) and cks:
            runs.append((name, cfg, cks[-1]))
    return runs


def build_model(cfg, device):
    return TestNetwork(
        hidden_dim_attn=cfg.get("d_attn"), hidden_dim=cfg["d_model"], feature_set=cfg["feature_set"],
        bottleneck_attention=cfg.get("bottleneck_attention"), include_mlp=cfg["include_mlp"],
        num_attention_blocks=cfg["num_blocks"], hidden_dim_mlp=cfg["d_mlp"], num_heads=cfg["num_heads"],
        embedding_size=cfg["embedding_size"], num_classes=cfg["num_classes"],
        use_lorentz_invariant_features=cfg["use_lorentz_invariant_features"], dropout_p=cfg["dropout_p"],
        num_particle_types=cfg["num_particle_types"], num_object_net_layers=cfg["num_object_net_layers"],
        is_layer_norm=cfg["is_layer_norm"], is_reconstruction_model=True,
        add_event_head=cfg.get("add_event_head", False), num_event_classes=cfg.get("num_event_classes", 3),
    ).to(device)


def make_val_dl(cfg, data_dir, device):
    if not data_dir.endswith("/"):
        data_dir += "/"
    dsids = sorted(int(os.path.basename(p)[5:11]) for p in glob.glob(f"{data_dir}dsid_*.memmap"))
    stds = np.ones(cfg["n_real_vars_in_file"]); stds[:4] = cfg.get("scale_data_std", 1e5)
    mk = dict(N_Real_Vars_In_File=cfg["n_real_vars_in_file"], N_Real_Vars_To_Return=cfg["n_real_vars_in_file"],
              memmap_paths={d: f"{data_dir}dsid_{d}.memmap" for d in dsids}, max_objs_in_memmap=cfg["max_n_objs"],
              batch_size=cfg["batch_size"], device=device, n_splits=cfg["n_splits"],
              validation_split_idx=cfg["validation_split_idx"], n_targets=3, means=None, stds=stds,
              objs_to_output=cfg["max_n_objs"], signal_only=False,
              has_eventNumbers=cfg.get("has_eventNumbers", False))
    return ProportionalMemoryMappedDataset(is_train=False, shuffle=False, shuffle_batch=False, **mk)


def sanitize_padding(x, types, pad):
    m = (types == pad).unsqueeze(-1)
    safe = torch.zeros_like(x)
    safe[..., 0] = 1e-3; safe[..., 1] = 1e-3
    if safe.shape[-1] > 3:
        safe[..., 3] = 2e-3
    return torch.where(m, safe, x)


def score_model(cfg, ckpt, val_dl, n_batches, device):
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    pad = cfg["num_particle_types"] - 1
    nin = cfg["model_input_vars"]
    use_event = (cfg["mode"] == "twohead") and (cfg.get("lambda_event", 1.0) > 0)
    val_dl._reset_indices()
    nb = n_batches if n_batches > 0 else len(val_dl)
    scores, wts, dsids = [], [], []
    with torch.no_grad():
        for _ in range(nb):
            b = next(val_dl)
            x = sanitize_padding(b["x"], b["types"], pad); types = b["types"]
            out = model(x[..., :nin], types)
            reco = out["reco"] if isinstance(out, dict) else out
            if use_event:
                sc = 1.0 - torch.softmax(out["event"], dim=-1)[:, 0]
            else:
                sc = event_score_from_object_logits(reco, types, pad, method="softor")
            scores.append(sc.detach().cpu().numpy())
            wts.append(b["MC_Wts"].detach().cpu().numpy())
            dsids.append(b["dsids"].detach().cpu().numpy().astype(int))
    return np.concatenate(scores), np.concatenate(wts), np.concatenate(dsids)


def working_points(scores, wts, dsids, targets, split):
    """Return {target: {thr, bkg, sig_total, per_dsid:{dsid:yield}, eff:{dsid:eff}}}.

    eff = retained signal yield / that DSID's TOTAL signal yield (divides out the
    reference-cross-section normalization, so it's the clean cross-mass / old-vs-new
    comparison)."""
    is_bkg = ~np.isin(dsids, SIG_DSIDS)
    order = np.argsort(scores, kind="mergesort")[::-1]
    s, w, d, bk = scores[order], wts[order], dsids[order], is_bkg[order]
    cum_bkg = np.cumsum(np.where(bk, w, 0.0)) * split
    tot = {sd: float(w[d == sd].sum() * split) for sd in SIG_DSIDS}  # full signal yield/DSID
    out = {}
    for t in targets:
        idx = int(np.searchsorted(cum_bkg, t))      # first index where retained bkg >= t
        idx = min(idx, len(s) - 1)
        sel = slice(0, idx + 1)
        per, eff = {}, {}
        for sd in SIG_DSIDS:
            per[sd] = float(w[sel][d[sel] == sd].sum() * split)
            eff[sd] = (per[sd] / tot[sd]) if tot[sd] > 0 else float("nan")
        out[t] = dict(thr=float(s[idx]), bkg=float(cum_bkg[idx]),
                      sig_total=float(sum(per.values())), per_dsid=per, eff=eff, tot=tot)
    return out


def main():
    a = parse_args()
    rr = a.run_root or newest_run_root()
    if not rr or not os.path.isdir(rr):
        print(f"no run_root ({rr})"); return
    device = a.device if (a.device == "cpu" or torch.backends.mps.is_available()) else "cpu"
    targets = [float(t) for t in a.bkg_targets.split(",")]
    runs = discover_runs(rr)
    print(f"run_root: {os.path.basename(rr)}   models: {len(runs)}   device: {device}")
    print(f"weights: SIGNED MC_Wts x split_factor {a.split_factor} (full-dataset yields). "
          f"Absolute scale uncalibrated (R6); compare shapes/relative.\n")

    val_dl = None
    results = {}
    for name, cfgp, ckpt in runs:
        with open(cfgp) as f:
            cfg = json.load(f)
        if val_dl is None:
            val_dl = make_val_dl(cfg, a.data_dir, device)
        sc, w, d = score_model(cfg, ckpt, val_dl, a.eval_batches, device)
        wp = working_points(sc, w, d, targets, a.split_factor)
        method = "event-head" if (cfg["mode"] == "twohead" and cfg.get("lambda_event", 1) > 0) else "softor-readout"
        results[name] = dict(wp=wp, method=method)
        line = "  ".join(f"bkg={int(t)}:Sig={wp[t]['sig_total']:.1f}(thr {wp[t]['thr']:.3f})" for t in targets)
        print(f"{name:24s} [{method:14s}]  {line}")

    rt = a.report_target
    masses = [DSID_MASS_TEV[s] for s in SIG_DSIDS]
    print(f"\n=== Signal YIELD per H+ mass @ retained bkg = {int(rt)} (full-dataset, signed-wt; R6-uncal abs scale) ===")
    hdr = "model".ljust(24) + " | " + " | ".join(f"{m:>5.1f}" for m in masses) + " |  total"
    print(hdr); print("-" * len(hdr))
    for name, r in results.items():
        if rt not in r["wp"]:
            continue
        per = r["wp"][rt]["per_dsid"]
        print(name.ljust(24) + " | " + " | ".join(f"{per[s]:5.0f}" for s in SIG_DSIDS) + f" | {r['wp'][rt]['sig_total']:6.0f}")

    print(f"\n=== Signal EFFICIENCY (% of each mass's total signal kept) @ retained bkg = {int(rt)} ===")
    print(hdr); print("-" * len(hdr))
    for name, r in results.items():
        if rt not in r["wp"]:
            continue
        eff = r["wp"][rt]["eff"]
        print(name.ljust(24) + " | " + " | ".join(f"{100*eff[s]:5.1f}" for s in SIG_DSIDS) + " |")
    print(f"(columns = H+ mass in TeV; efficiency divides out per-sample normalization → clean cross-mass / old-vs-new comparison)")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, (axy, axe) = plt.subplots(1, 2, figsize=(15, 6))
            for name, r in results.items():
                if rt not in r["wp"]:
                    continue
                lbl = f"{name} ({r['method']})"
                axy.plot(masses, [r["wp"][rt]["per_dsid"][s] for s in SIG_DSIDS], marker="o", label=lbl)
                axe.plot(masses, [100 * r["wp"][rt]["eff"][s] for s in SIG_DSIDS], marker="o", label=lbl)
            for ax in (axy, axe):
                ax.set_xlabel("H+ mass [TeV]"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2)
            axy.set_ylabel(f"Retained signal yield @ {int(rt)} bkg"); axy.set_title("Signal yield (R6-uncal abs scale)")
            axe.set_ylabel(f"Signal efficiency [%] @ {int(rt)} bkg"); axe.set_title("Signal efficiency (normalization-free)")
            fig.suptitle(f"Signal kept when {int(rt)} background events survive — full dataset, real bkg, d20/b2/h4 8ep")
            fig.tight_layout(); fig.savefig(a.plot, dpi=130)
            print(f"\nplot saved: {a.plot}")
        except Exception as e:
            print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
