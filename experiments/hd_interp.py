"""H-D interp: does the joint (reco+classification) objective make the channel /
signal-vs-background structure MORE explicit in the representation, or muddier?

We compare two matched d152/b3/h4/mlp400 models (30-ep winner recipe, seed 0,
no bottleneck) by re-running quantitative linear-probe sweeps on their residual
stream, layer by layer:

  (A) CHANNEL decodability (lvbb vs qqbb), on SIGNAL events with exactly 1 lepton
      + 1 neutrino, probed at three loci — the lepton token, the neutrino token,
      and the padding-masked mean pool (the event head's input). H1 established
      the lep/nu "W vs none" verdict *is* the global channel decision, delivered
      to lep/nu; so "is channel linearly readable, and from which layer" is a
      direct measure of how explicit that circuit is.

  (B) SIGNAL-vs-BACKGROUND decodability, on signal+background events, probed at
      the mean pool per layer (= the free-discrimination / H-C measurement, now
      resolved by depth). Tells us whether a reco-only backbone already separates
      sig/bkg (it never saw background) and whether/where joint training sharpens
      it.

Probe = StandardScaler + LogisticRegression, clean internal train/test split;
we report test ROC-AUC (threshold-free, robust to imbalance) per layer.

CAVEATS (documented, not fatal for a representation-geometry comparison):
  * The two models differ in train/val split (reco-only=eventNumber, joint=position)
    so a signal event held-out for one may have been train for the other. At
    677k params / 2.8M events event-level memorization is negligible; and leakage
    can only *inflate* the leaked model's decodability, so an equal-or-lower joint
    number is robust to it. Flagged per run.
  * "Joint" bundles three changes vs reco-only: the event head, the event loss,
    and training on real background (reco-only is signal-only). That bundle *is*
    "the joint model as deployed" — the intended H-D contrast.
  * Minor: reco-only config has learning_rate_log_decay=true vs joint false; both
    use lr_schedule=cosine so this is likely a no-op legacy flag.

Usage:
  .venv/bin/python experiments/hd_interp.py \
      --joint output/20260624-163646_TrainingOutput \
      --reco  output/20260613-234627_TrainingOutput \
      --device mps --out output/hd_interp
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from models.models import TestNetwork
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

PAD_TYPE = 5            # object types: 0=e 1=mu 2=nu 3=ljet 4=sjet 5=pad
LEP_TYPES = (0, 1)
NU_TYPE = 2


# --------------------------------------------------------------------------- #
# model + data loading (mirrors experiments/sb200_oldvsnew.py)
# --------------------------------------------------------------------------- #
def build_model(cfg, device):
    m = TestNetwork(
        hidden_dim_attn=cfg.get("d_attn"),
        hidden_dim=cfg["d_model"],
        feature_set=cfg["feature_set"],
        bottleneck_attention=cfg.get("bottleneck_attention"),
        include_mlp=cfg["include_mlp"],
        num_attention_blocks=cfg["num_blocks"],
        hidden_dim_mlp=cfg["d_mlp"],
        num_heads=cfg["num_heads"],
        embedding_size=cfg["embedding_size"],
        num_classes=cfg["num_classes"],
        use_lorentz_invariant_features=cfg["use_lorentz_invariant_features"],
        dropout_p=cfg["dropout_p"],
        num_particle_types=cfg["num_particle_types"],
        num_object_net_layers=cfg["num_object_net_layers"],
        is_layer_norm=cfg["is_layer_norm"],
        is_reconstruction_model=True,
        add_event_head=cfg.get("add_event_head", False),
        num_event_classes=cfg.get("num_event_classes", 3),
    ).to(device)
    return m


def latest_ckpt(train_dir, n_splits, val_idx):
    d = os.path.join(train_dir, "models", f"Nplits{n_splits}_ValIdx{val_idx}")
    cks = glob.glob(os.path.join(d, "chkpt*.pth"))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {d}")
    # chkpt{epoch}_{step}.pth -> sort by epoch then step
    def key(p):
        b = os.path.basename(p)[len("chkpt"):-len(".pth")]
        e, s = b.split("_")
        return (int(e), int(s))
    return sorted(cks, key=key)[-1]


def load_trained(train_dir, device):
    cfg = json.load(open(os.path.join(train_dir, "config.json")))
    model = build_model(cfg, device)
    ckpt = latest_ckpt(train_dir, cfg["n_splits"], cfg["validation_split_idx"])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, cfg, ckpt


def make_loader(cfg, data_dir, device, signal_only, position_split, batch_size=2048):
    stds = np.ones(cfg["n_real_vars_in_file"])
    stds[:4] = cfg.get("scale_data_std", 1e5)
    dsids = sorted(int(os.path.basename(p)[5:11])
                   for p in glob.glob(os.path.join(data_dir, "dsid_*.memmap")))
    if signal_only:
        dsids = [d for d in dsids if 510000 < d < 600000]
    memmap_paths = {d: os.path.join(data_dir, f"dsid_{d}.memmap") for d in dsids}
    return ProportionalMemoryMappedDataset(
        N_Real_Vars_In_File=cfg["n_real_vars_in_file"],
        N_Real_Vars_To_Return=cfg["n_real_vars_in_file"],
        memmap_paths=memmap_paths,
        max_objs_in_memmap=cfg["max_n_objs"],
        batch_size=batch_size, device=device, is_train=False,
        n_splits=cfg["n_splits"], validation_split_idx=cfg["validation_split_idx"],
        n_targets=3, means=None, stds=stds, objs_to_output=cfg["max_n_objs"],
        signal_only=signal_only,
        has_eventNumbers=(not position_split),
        shuffle=False, shuffle_batch=False,
    )


# --------------------------------------------------------------------------- #
# residual-stream extraction (mirrors TestNetwork._backbone_features so it works
# despite the blocks being a manually-applied ModuleList of ModuleDicts)
# --------------------------------------------------------------------------- #
def backbone_residuals(model, object_features, object_types):
    """Return [emb, after_block0, after_block1, ...] each [B, n_obj, d_model]."""
    type_emb = model.type_embedding(object_types)
    if model.use_lorentz_invariant_features:
        inv = model.invariant_features(object_features[..., :4])
    else:
        inv = object_features[..., :4]
    combined = torch.cat([inv, object_features[..., 4:], type_emb], dim=-1)
    of = model.object_net(combined)
    res = [of]
    kpm = (object_types == (model.num_particle_types - 1))
    for block in model.attention_blocks:
        identity = of
        proj = block["input_projection"](of) if model.hidden_dim_attn is not None else of
        attn_out, _ = block["self_attention"](proj, proj, proj, key_padding_mask=kpm)
        if model.hidden_dim_attn is not None:
            attn_out = block["output_projection"](attn_out)
        if model.include_mlp:
            residual = identity + attn_out
            of = residual + block["post_attention"](residual)
        else:
            of = identity + attn_out
        res.append(of)
    return res


def layer_names(n_blocks):
    return ["emb"] + [f"b{i}" for i in range(n_blocks)]


# --------------------------------------------------------------------------- #
# (A) channel probe — collect per-locus residual vectors over a signal slice
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_channel(model, cfg, dl, n_batches, device):
    nlayers = cfg["num_blocks"] + 1
    lep = [[] for _ in range(nlayers)]
    nu = [[] for _ in range(nlayers)]
    pooled = [[] for _ in range(nlayers)]
    chan = []
    nin = cfg["model_input_vars"]
    for _ in range(n_batches):
        b = next(dl)
        x, types, y = b["x"], b["types"], b["y"]
        is_sig = y.argmax(-1) != 0
        ok = is_sig & ((types == NU_TYPE).sum(1) == 1) \
            & (((types == LEP_TYPES[0]) | (types == LEP_TYPES[1])).sum(1) == 1)
        if ok.sum() == 0:
            continue
        x, types, y = x[ok], types[ok], y[ok]
        res = backbone_residuals(model, x[..., :nin], types)
        ar = torch.arange(len(x), device=x.device)
        lpos = ((types == LEP_TYPES[0]) | (types == LEP_TYPES[1])).float().argmax(1)
        npos = (types == NU_TYPE).float().argmax(1)
        nonpad = (types != PAD_TYPE).unsqueeze(-1).float()
        denom = nonpad.sum(1).clamp(min=1)
        for li, R in enumerate(res):
            lep[li].append(R[ar, lpos].float().cpu().numpy())
            nu[li].append(R[ar, npos].float().cpu().numpy())
            pooled[li].append(((R * nonpad).sum(1) / denom).float().cpu().numpy())
        # channel: y.argmax 1=lvbb, 2=qqbb  -> qqbb=1
        chan.append((y.argmax(-1) == 2).long().cpu().numpy())
    cat = lambda L: [np.concatenate(x) for x in L]
    return cat(lep), cat(nu), cat(pooled), np.concatenate(chan)


# --------------------------------------------------------------------------- #
# (B) sig/bkg probe — collect pooled residual + labels + weights over all-data
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_sigbkg(model, cfg, dl, n_batches, device):
    nlayers = cfg["num_blocks"] + 1
    pooled = [[] for _ in range(nlayers)]
    is_bkg, wts = [], []
    nin = cfg["model_input_vars"]
    for _ in range(n_batches):
        b = next(dl)
        x, types, y = b["x"], b["types"], b["y"]
        res = backbone_residuals(model, x[..., :nin], types)
        nonpad = (types != PAD_TYPE).unsqueeze(-1).float()
        denom = nonpad.sum(1).clamp(min=1)
        for li, R in enumerate(res):
            pooled[li].append(((R * nonpad).sum(1) / denom).float().cpu().numpy())
        is_bkg.append((y.argmax(-1) == 0).long().cpu().numpy())
        wts.append(b["MC_Wts"].abs().float().cpu().numpy())
    cat = lambda L: [np.concatenate(x) for x in L]
    return cat(pooled), np.concatenate(is_bkg), np.concatenate(wts)


# --------------------------------------------------------------------------- #
# probing
# --------------------------------------------------------------------------- #
def probe(X, y, sample_weight=None, seed=0):
    """StandardScaler + logistic; return test AUC, test acc (+ weighted AUC)."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.4, random_state=seed, stratify=y)
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(X[tr]), y[tr])
    p = clf.predict_proba(sc.transform(X[te]))[:, 1]
    out = {"auc": float(roc_auc_score(y[te], p)),
           "acc": float(accuracy_score(y[te], (p > 0.5).astype(int))),
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "pos_frac": float(y.mean())}
    if sample_weight is not None:
        out["auc_weighted"] = float(roc_auc_score(y[te], p, sample_weight=sample_weight[te]))
    return out


def run_channel(model, cfg, name, data_dir, device, n_batches):
    dl = make_loader(cfg, data_dir, device, signal_only=True,
                     position_split=False, batch_size=2048)
    lep, nu, pooled, chan = collect_channel(model, cfg, dl, n_batches, device)
    names = layer_names(cfg["num_blocks"])
    res = {"n_events": int(len(chan)), "qqbb_frac": float(chan.mean()),
           "lep": {}, "nu": {}, "pooled": {}}
    for locus, L in (("lep", lep), ("nu", nu), ("pooled", pooled)):
        for nm, X in zip(names, L):
            res[locus][nm] = probe(X, chan)
    print(f"[{name}] channel probe: {res['n_events']} signal 1lep+1nu events, "
          f"qqbb frac {res['qqbb_frac']:.3f}")
    return res


def run_sigbkg(model, cfg, name, data_dir, device, n_batches):
    dl = make_loader(cfg, data_dir, device, signal_only=False,
                     position_split=True, batch_size=2048)
    pooled, is_bkg, wts = collect_sigbkg(model, cfg, dl, n_batches, device)
    names = layer_names(cfg["num_blocks"])
    res = {"n_events": int(len(is_bkg)), "bkg_frac": float(is_bkg.mean()), "pooled": {}}
    for nm, X in zip(names, pooled):
        res["pooled"][nm] = probe(X, is_bkg, sample_weight=wts)
    print(f"[{name}] sig/bkg probe: {res['n_events']} events, bkg frac {res['bkg_frac']:.3f}")
    return res


# --------------------------------------------------------------------------- #
def plot(results, names, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    panels = [("channel", "lep", "Channel @ lepton"),
              ("channel", "nu", "Channel @ neutrino"),
              ("channel", "pooled", "Channel @ mean-pool"),
              ("sigbkg", "pooled", "Sig-vs-bkg @ mean-pool")]
    colors = {"reco": "tab:blue", "joint": "tab:red"}
    x = np.arange(len(names))
    for ax, (task, locus, title) in zip(axes, panels):
        for key in ("reco", "joint"):
            d = results[key][task][locus]
            y = [d[nm]["auc"] for nm in names]
            ax.plot(x, y, "-o", color=colors[key], label=key)
            if task == "sigbkg":
                yw = [d[nm].get("auc_weighted", np.nan) for nm in names]
                ax.plot(x, yw, "--s", color=colors[key], alpha=0.55,
                        label=f"{key} (wt)")
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_title(title); ax.set_xlabel("layer"); ax.set_ylabel("test ROC-AUC")
        ax.set_ylim(0.45, 1.01); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("H-D: channel & sig/bkg linear-probe decodability by layer "
                 "(joint vs reco-only, d152)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved plot -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", required=True, help="joint TrainingOutput dir")
    ap.add_argument("--reco", required=True, help="reco-only TrainingOutput dir")
    ap.add_argument("--signal-data", default=os.path.join(REPO, "tmp_data_20250321v1_signal"))
    ap.add_argument("--all-data", default=os.path.join(REPO, "tmp_data_20250321v1_all"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--channel-batches", type=int, default=10)
    ap.add_argument("--sigbkg-batches", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(REPO, "output", "hd_interp"))
    a = ap.parse_args()

    dev = a.device
    results = {}
    meta = {}
    for key, train_dir in (("reco", a.reco), ("joint", a.joint)):
        model, cfg, ckpt = load_trained(train_dir, dev)
        meta[key] = {"dir": train_dir, "ckpt": ckpt,
                     "d_model": cfg["d_model"], "num_blocks": cfg["num_blocks"],
                     "add_event_head": cfg.get("add_event_head", False),
                     "has_eventNumbers": cfg.get("has_eventNumbers")}
        print(f"\n=== {key}: {os.path.basename(train_dir)} "
              f"(d{cfg['d_model']} b{cfg['num_blocks']} "
              f"event_head={cfg.get('add_event_head', False)}) ===")
        ch = run_channel(model, cfg, key, a.signal_data, dev, a.channel_batches)
        sb = run_sigbkg(model, cfg, key, a.all_data, dev, a.sigbkg_batches)
        results[key] = {"channel": ch, "sigbkg": sb}
        del model
        if dev == "mps":
            torch.mps.empty_cache()

    names = layer_names(meta["joint"]["num_blocks"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump({"meta": meta, "results": results, "layers": names}, f, indent=1)
    print(f"\nsaved json -> {a.out}.json")
    plot(results, names, a.out + ".png")

    # console summary
    print("\n================ SUMMARY (test ROC-AUC by layer) ================")
    for task, loci in (("channel", ("lep", "nu", "pooled")), ("sigbkg", ("pooled",))):
        print(f"\n--- {task} ---")
        for locus in loci:
            print(f" locus={locus}")
            for key in ("reco", "joint"):
                d = results[key][task][locus]
                row = "  ".join(f"{nm}:{d[nm]['auc']:.3f}" for nm in names)
                print(f"   {key:5s} {row}")


if __name__ == "__main__":
    main()
