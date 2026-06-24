"""Fig-4-style comparison: signal kept at 200 retained background (Sb200), per H+ mass,
OLD method (hand-cut reco + high-level NN) vs NEW method (our transformer reco+classifier),
each on its NATIVE event set.

- OLD numbers: read from the stored pickles (sig_rems_{qqbb,lvbb}.pkl) — already computed,
  on the old method's cut-based-reconstructed native sample.
- THESIS-NEW reference: "Low-level NN (transformer-reco inputs)" from the same pickles
  (the thesis's own transformer reco+classification) — sanity reference for our models.
- OUR NEW models: computed here from tonight's checkpoints. Native set = events our model's
  RECO head reconstructs into one of the 6 valid topologies (utils.check_valid). Reco-invalid
  events are set to pure-background [1,0,0] so they're rejected (not yield-scaled away),
  keeping the per-DSID expected-yield normalization correct. Sb200 via the SAME
  lowlevelmetrics.HEPMetrics.compute_signal_selection_metrics used for the stored numbers.

Usage: sb200_oldvsnew.py [run_root] [--models twohead_joint,twohead_int,...] [--device mps]
                         [--eval-batches N] [--plot out.png]
"""
import argparse, glob, json, os, sys, pickle
import numpy as np
import torch

# lowlevelmetrics.compute_auc uses np.trapz, removed in newer numpy -> shim to np.trapezoid
# (identical function) so the metric runs exactly as when the stored numbers were generated.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

EXP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, EXP)
REPO = os.path.dirname(EXP); sys.path.insert(0, REPO)
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset
from models.models import TestNetwork
from metrics import lowlevelmetrics
from utils.utils import check_valid
from joint_primitives import event_score_from_object_logits

DSID_MASS_TEV = {510115: 0.8, 510116: 0.9, 510117: 1.0, 510118: 1.2, 510119: 1.4,
                 510120: 1.6, 510121: 1.8, 510122: 2.0, 510123: 2.5, 510124: 3.0}
SIG_DSIDS = sorted(DSID_MASS_TEV)
MASSES = [DSID_MASS_TEV[d] for d in SIG_DSIDS]
DATA_DEFAULT = os.path.join(REPO, "tmp_data_20250321v1_all") + "/"
# Models to read from the pickles: old baselines + thesis-new reference.
OLD_KEYS = ["High-level Joint NN", "High-level Parametrised NN", "Low-level NN (transformer-reco inputs)"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_root", nargs="?", default=None)
    p.add_argument("--data-dir", default=DATA_DEFAULT)
    p.add_argument("--models", default="twohead_joint,twohead_int,single_softor,twohead_seqfrozen,twohead_seqfull,twohead_intdetach")
    p.add_argument("--extra", default="", help="extra models as 'label=<TrainingOutputDir>' comma-list "
                                               "(reads {dir}/config.json + highest chkpt); for models outside the run_root")
    p.add_argument("--device", choices=["cpu", "mps"], default="mps")
    p.add_argument("--eval-batches", type=int, default=0, help="0 = full val")
    p.add_argument("--bkg-level", type=float, default=200.0)
    p.add_argument("--plot", default=None)
    return p.parse_args()


def newest_run_root():
    c = sorted(glob.glob(os.path.join(REPO, "output", "joint_real_bkg_*")))
    return c[-1] if c else None


def discover(run_root, wanted):
    out = {}
    for lg in sorted(glob.glob(os.path.join(run_root, "logs", "*.log"))):
        name = os.path.basename(lg)[:-4]
        if name not in wanted:
            continue
        odir = None
        with open(lg) as f:
            for line in f:
                if line.startswith("out: "):
                    odir = line.split("out: ", 1)[1].strip(); break
        if not odir:
            continue
        cfg = os.path.join(odir, "config.json")
        cks = sorted(glob.glob(os.path.join(odir, "models", "*", "chkpt*.pth")),
                     key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
        if os.path.exists(cfg) and cks:
            out[name] = (cfg, cks[-1])
    return out


def resolve_extra(spec):
    """Parse 'label=<TrainingOutputDir>' comma-list -> {label: (config.json, highest_chkpt)}."""
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        label, d = item.split("=", 1)
        cfg = os.path.join(d, "config.json")
        cks = sorted(glob.glob(os.path.join(d, "models", "*", "chkpt*.pth")),
                     key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
        if os.path.exists(cfg) and cks:
            out[label.strip()] = (cfg, cks[-1])
        else:
            print(f"WARN: extra model '{label}' not found in {d}")
    return out


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
    m = (types == pad).unsqueeze(-1); safe = torch.zeros_like(x)
    safe[..., 0] = 1e-3; safe[..., 1] = 1e-3
    if safe.shape[-1] > 3:
        safe[..., 3] = 2e-3
    return torch.where(m, safe, x)


def compute_new_model(cfg, ckpt, val_dl, n_batches, device, bkg_level):
    """Return {mass: {'qqbb':sig, 'lvbb':sig}} at the given retained-bkg level, plus
    diagnostics (validity fractions)."""
    model = build_model(cfg, device); model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
    pad = cfg["num_particle_types"] - 1; nin = cfg["model_input_vars"]
    # single-head ("background = all-none"): no event head -> channel comes from the RECO
    # (check_valid valid_qqbb/valid_lvbb) and the discriminant is the softor readout, P(bkg)=1-softor.
    # two-head: channel from the event classifier (metric-internal); reco only gates validity.
    single = (cfg.get("mode") == "single") or (not cfg.get("add_event_head", False))
    val_dl._reset_indices()
    nb = n_batches if n_batches > 0 else len(val_dl)
    met = lowlevelmetrics.HEPMetrics(num_classes=3, max_buffer_len=int(val_dl.get_total_samples()) + cfg["batch_size"],
                                     max_bkg_levels=[int(bkg_level)], signal_acceptance_levels=[1000],
                                     total_weights_per_dsid=val_dl.weight_sums, mHLimits=[(0, 1e10)])
    n_valid = n_tot = 0
    with torch.no_grad():
        for _ in range(nb):
            b = next(val_dl)
            x = sanitize_padding(b["x"], b["types"], pad); types = b["types"]
            out = model(x[..., :nin], types)
            reco = out["reco"] if isinstance(out, dict) else out
            valid, chan = check_valid(types, reco, pad, categorical=True, returnTypes=True)
            chan = chan.to(reco.device)  # 0=reco-invalid, 1=lvbb, 2=qqbb
            if single:
                s = event_score_from_object_logits(reco, types, pad, "softor").to(reco.device).clamp(0, 1)
                probs = torch.zeros((len(s), 3), device=reco.device, dtype=s.dtype)
                probs[:, 0] = 1.0  # default pure-bkg (covers reco-invalid chan==0)
                ql = (chan == 2); ll = (chan == 1)
                probs[ql, 0] = 1.0 - s[ql]; probs[ql, 2] = s[ql]
                probs[ll, 0] = 1.0 - s[ll]; probs[ll, 1] = s[ll]
            else:
                if not (isinstance(out, dict) and "event" in out):
                    raise SystemExit("two-head model expected an event head")
                probs = torch.softmax(out["event"], dim=-1).clone()  # [bkg, lvbb, qqbb]
                invalid = (chan == 0)
                if invalid.any():
                    probs[invalid] = torch.tensor([1.0, 0.0, 0.0], device=probs.device, dtype=probs.dtype)
            n_valid += int((chan > 0).sum()); n_tot += len(chan)
            met.update(probs, b["y"].argmax(-1), b["MC_Wts"], b["mWh_qqbb"], b["mWh_lvbb"], b["dsids"], b["mH"])
    rv = met.compute_and_log(0, "val_MC", log_level=2, save=False, verbose=False, calc_all=True)
    res = {}
    bl = int(bkg_level)
    for d in SIG_DSIDS:
        m = DSID_MASS_TEV[d]
        qpref = f"val_MC_ByMassAcceptance_FixedBkg/sig_qqbb_expected/{bl}_{m}_"
        lpref = f"val_MC_ByMassAcceptance_FixedBkg/sig_lvbb_expected/{bl}_{m}_"
        kq = next((k for k in rv if k.startswith(qpref)), None)
        kl = next((k for k in rv if k.startswith(lpref)), None)
        res[m] = {"qqbb": float(rv[kq]) if kq else float("nan"),
                  "lvbb": float(rv[kl]) if kl else float("nan")}
    return res, (n_valid / max(n_tot, 1)), met


def compute_max_available(met):
    """Max possible signal = full truth-channel yield (NO reco/selection), per mass, using the
    metric's own per-DSID expected-yield normalization (total/processed). Truth-channel = the
    true decay mode (target==2 qqbb, ==1 lvbb), set before any reco. Model-independent."""
    n = met.current_update_point
    t = met.all_targets[:n]; w = met.all_weights[:n]; d = met.all_dsids[:n].astype(int)
    res = {}
    for dsid in SIG_DSIDS:
        proc = met.processed_weight_sums_per_dsid.get(dsid, 0.0)
        mult = (met.total_weights_per_dsid[dsid] / proc) if proc else 0.0
        sel = (d == dsid)
        res[DSID_MASS_TEV[dsid]] = {"qqbb": float(w[sel & (t == 2)].sum() * mult),
                                    "lvbb": float(w[sel & (t == 1)].sum() * mult)}
    return res


def main():
    a = parse_args()
    rr = a.run_root or newest_run_root()
    device = a.device if (a.device == "cpu" or torch.backends.mps.is_available()) else "cpu"
    wanted = [m.strip() for m in a.models.split(",") if m.strip()]
    runs = {**discover(rr, wanted), **resolve_extra(a.extra)}
    qq_old = pickle.load(open(os.path.join(REPO, "sig_rems_qqbb.pkl"), "rb"))
    lv_old = pickle.load(open(os.path.join(REPO, "sig_rems_lvbb.pkl"), "rb"))
    print(f"run_root: {os.path.basename(rr)}  our models: {list(runs)}  device: {device}  bkg={int(a.bkg_level)}\n")

    # OUR models
    new_qq, new_lv = {}, {}
    val_dl = None
    max_avail = None
    for name, (cfgp, ckpt) in runs.items():
        cfg = json.load(open(cfgp))
        if val_dl is None:
            val_dl = make_val_dl(cfg, a.data_dir, device)
        res, vfrac, met = compute_new_model(cfg, ckpt, val_dl, a.eval_batches, device, a.bkg_level)
        new_qq[name] = [res[m]["qqbb"] for m in MASSES]
        new_lv[name] = [res[m]["lvbb"] for m in MASSES]
        if max_avail is None:  # truth-channel totals are model-independent; compute once
            max_avail = compute_max_available(met)
        print(f"{name:22s} reco-valid frac={vfrac:.3f}")
    max_qq = [max_avail[m]["qqbb"] for m in MASSES]
    max_lv = [max_avail[m]["lvbb"] for m in MASSES]

    def table(title, old_d, new_d, max_vals):
        print(f"\n=== {title}: signal kept @ {int(a.bkg_level)} bkg, per H+ mass [TeV] (native set each) ===")
        hdr = "model".ljust(34) + " | " + " | ".join(f"{m:>5.1f}" for m in MASSES)
        print(hdr); print("-" * len(hdr))
        print("MAX available (truth channel)".ljust(34) + " | " + " | ".join(f"{x:5.0f}" for x in max_vals))
        for k in OLD_KEYS:
            if k in old_d:
                print(("OLD " + k).ljust(34) + " | " + " | ".join(f"{float(x):5.0f}" for x in old_d[k]))
        for k, v in new_d.items():
            print(("NEW(ours) " + k).ljust(34) + " | " + " | ".join(f"{x:5.0f}" for x in v))
    table("QQBB", qq_old, new_qq, max_qq)
    table("LVBB", lv_old, new_lv, max_lv)

    if a.plot:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
            for ax, ch, old_d, new_d, mx in [(axes[0], "qqbb", qq_old, new_qq, max_qq),
                                             (axes[1], "lvbb", lv_old, new_lv, max_lv)]:
                ax.plot(MASSES, mx, color="k", lw=2.6, ls="-", marker=None,
                        label="MAX available (all truth-channel signal)")
                ax.fill_between(MASSES, mx, alpha=0.05, color="k")
                for k in OLD_KEYS:
                    if k in old_d:
                        sty = dict(ls="--", marker="s") if k.startswith("High-level") else dict(ls=":", marker="^")
                        ax.plot(MASSES, [float(x) for x in old_d[k]], label=k, **sty)
                for k, v in new_d.items():
                    ax.plot(MASSES, v, marker="o", lw=2.2, label=f"NEW ours: {k}")
                ax.set_xlabel("H+ mass [TeV]"); ax.set_ylabel(f"signal kept @ {int(a.bkg_level)} bkg")
                ax.set_title(f"{ch}  (each method on its native event set)")
                ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
            fig.suptitle(f"Sb{int(a.bkg_level)}: OLD (hand-cut reco + high-level NN) vs NEW (transformer reco+classifier) "
                         f"vs MAX available. Per-DSID expected-yield normalized; reco-validity = native selection.")
            fig.tight_layout(); fig.savefig(a.plot, dpi=130); print(f"\nplot saved: {a.plot}")
        except Exception as e:
            print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
