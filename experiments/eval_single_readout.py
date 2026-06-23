"""Post-hoc fair readout evaluator for single-head joint checkpoints."""

import argparse
import json
import os
import sys

import numpy as np
import torch

EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPERIMENTS)
from joint_primitives import (  # noqa: E402
    EVENT_SUMMARY_FEATURE_NAMES,
    best_asimov_z,
    event_score_from_object_logits,
    event_summary_features,
    fit_fair_readout,
    weighted_roc_auc,
)

REPO = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, REPO)
from dataloaders.lowleveldataloader import ProportionalMemoryMappedDataset  # noqa: E402
from models.models import TestNetwork  # noqa: E402


DATA = os.path.join(REPO, "tmp_data_20250321v1_signal/")
DSIDS = list(range(510115, 510125))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint .pth state_dict saved by train_joint.py")
    parser.add_argument("--config", default=None, help="config.json; defaults to walking up from --ckpt")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--pseudo-bkg-dsid", type=int, default=None)
    parser.add_argument("--fit-batches", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def find_config(ckpt_path):
    current = os.path.abspath(os.path.dirname(ckpt_path))
    while True:
        candidate = os.path.join(current, "config.json")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(f"could not find config.json by walking up from {ckpt_path!r}")


def resolve_device(requested):
    if requested == "mps" and not torch.backends.mps.is_available():
        print("requested --device mps but MPS is unavailable; using cpu")
        return "cpu"
    return requested


def build_model(config, device):
    model = TestNetwork(
        hidden_dim_attn=config.get("d_attn"),
        hidden_dim=config["d_model"],
        feature_set=config["feature_set"],
        bottleneck_attention=config.get("bottleneck_attention"),
        include_mlp=config["include_mlp"],
        num_attention_blocks=config["num_blocks"],
        hidden_dim_mlp=config["d_mlp"],
        num_heads=config["num_heads"],
        embedding_size=config["embedding_size"],
        num_classes=config["num_classes"],
        use_lorentz_invariant_features=config["use_lorentz_invariant_features"],
        dropout_p=config["dropout_p"],
        num_particle_types=config["num_particle_types"],
        num_object_net_layers=config["num_object_net_layers"],
        is_layer_norm=config["is_layer_norm"],
        is_reconstruction_model=True,
        add_event_head=config.get("add_event_head", False),
        num_event_classes=config.get("num_event_classes", 3),
    )
    return model.to(device)


def make_dataloaders(config, batch_size, device):
    stds = np.ones(config["n_real_vars_in_file"])
    stds[:4] = config.get("scale_data_std", 1.0e5)
    mk = dict(
        N_Real_Vars_In_File=config["n_real_vars_in_file"],
        N_Real_Vars_To_Return=config["n_real_vars_in_file"],
        memmap_paths={dsid: f"{DATA}dsid_{dsid}.memmap" for dsid in DSIDS},
        max_objs_in_memmap=config["max_n_objs"],
        batch_size=batch_size,
        device=device,
        n_splits=config["n_splits"],
        validation_split_idx=config["validation_split_idx"],
        n_targets=3,
        means=None,
        stds=stds,
        objs_to_output=config["max_n_objs"],
        signal_only=True,
        has_eventNumbers=config["has_eventNumbers"],
    )
    train_dl = ProportionalMemoryMappedDataset(
        is_train=True,
        shuffle=config.get("shuffle_objects", True),
        shuffle_batch=True,
        **mk,
    )
    val_dl = ProportionalMemoryMappedDataset(
        is_train=False,
        shuffle=False,
        shuffle_batch=True,
        **mk,
    )
    return train_dl, val_dl


def sanitize_padding(x, types, padding_token):
    """Padding objects are masked out, but still need finite benign inputs."""
    pad = (types == padding_token).unsqueeze(-1)
    safe = torch.zeros_like(x)
    if safe.shape[-1] > 0:
        safe[..., 0] = 1.0e-3
    if safe.shape[-1] > 1:
        safe[..., 1] = 1.0e-3
    if safe.shape[-1] > 3:
        safe[..., 3] = 2.0e-3
    return torch.where(pad, safe, x)


def batch_signal_labels(batch, pseudo_bkg_dsid):
    if pseudo_bkg_dsid is not None:
        is_bkg = batch["dsids"] == pseudo_bkg_dsid
    else:
        is_bkg = batch["y"].argmax(-1) == 0
    return (~is_bkg).long()


def per_object_logits(model, x, types, model_input_vars):
    out = model(x[..., :model_input_vars], types)
    return out["reco"] if isinstance(out, dict) else out


def collect_fit_inputs(model, dataloader, n_batches, config, padding_token, pseudo_bkg_dsid):
    feature_chunks, label_chunks, weight_chunks = [], [], []
    for _ in range(min(max(n_batches, 0), len(dataloader))):
        batch = next(dataloader)
        x = sanitize_padding(batch["x"], batch["types"], padding_token)
        types = batch["types"]
        reco_logits = per_object_logits(model, x, types, config["model_input_vars"])
        feature_chunks.append(event_summary_features(reco_logits, types, padding_token).detach().cpu())
        label_chunks.append(batch_signal_labels(batch, pseudo_bkg_dsid).detach().cpu())
        weight_chunks.append(batch["MC_Wts"].detach().cpu())

    if not feature_chunks:
        return torch.empty((0, len(EVENT_SUMMARY_FEATURE_NAMES))), torch.empty(0), torch.empty(0)
    return torch.cat(feature_chunks), torch.cat(label_chunks), torch.cat(weight_chunks)


def collect_eval_inputs(model, dataloader, n_batches, config, padding_token, pseudo_bkg_dsid):
    feature_chunks, softor_chunks, max_chunks, label_chunks, weight_chunks = [], [], [], [], []
    for _ in range(min(max(n_batches, 0), len(dataloader))):
        batch = next(dataloader)
        x = sanitize_padding(batch["x"], batch["types"], padding_token)
        types = batch["types"]
        reco_logits = per_object_logits(model, x, types, config["model_input_vars"])
        feature_chunks.append(event_summary_features(reco_logits, types, padding_token).detach().cpu())
        softor_chunks.append(
            event_score_from_object_logits(reco_logits, types, padding_token, method="softor").detach().cpu()
        )
        max_chunks.append(
            event_score_from_object_logits(reco_logits, types, padding_token, method="max").detach().cpu()
        )
        label_chunks.append(batch_signal_labels(batch, pseudo_bkg_dsid).detach().cpu())
        weight_chunks.append(batch["MC_Wts"].detach().cpu())

    if not feature_chunks:
        empty_features = torch.empty((0, len(EVENT_SUMMARY_FEATURE_NAMES)))
        empty = torch.empty(0)
        return empty_features, empty, empty, empty.long(), empty
    return (
        torch.cat(feature_chunks),
        torch.cat(softor_chunks),
        torch.cat(max_chunks),
        torch.cat(label_chunks),
        torch.cat(weight_chunks),
    )


def metrics_for(scores, labels, weights):
    scores_np = np.asarray(scores).reshape(-1)
    labels_np = np.asarray(labels).reshape(-1)
    weights_np = np.abs(np.asarray(weights).reshape(-1))
    if labels_np.size == 0 or np.unique(labels_np).size < 2:
        return float("nan"), float("nan"), float("nan")
    auc = weighted_roc_auc(scores_np, labels_np, weights_np)
    best_z, best_threshold = best_asimov_z(scores_np, labels_np, weights_np)
    return auc, best_z, best_threshold


def print_metric_line(name, scores, labels, weights):
    auc, best_z, threshold = metrics_for(scores, labels, weights)
    print(f"{name}: AUC={auc:.4f}  best_asimov_z={best_z:.4f}  thr={threshold:.4f}")


def main():
    args = parse_args()
    ckpt_path = os.path.abspath(args.ckpt)
    config_path = os.path.abspath(args.config) if args.config else find_config(ckpt_path)
    device = resolve_device(args.device)

    with open(config_path) as handle:
        config = json.load(handle)
    batch_size = args.batch_size or config["batch_size"]
    torch.manual_seed(config.get("seed", 0))
    np.random.seed(config.get("seed", 0))

    model = build_model(config, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    train_dl, val_dl = make_dataloaders(config, batch_size, device)
    train_dl._reset_indices()
    val_dl._reset_indices()

    padding_token = config["num_particle_types"] - 1
    print(f"checkpoint: {ckpt_path}")
    print(f"config: {config_path}")
    print(f"device: {device}")
    print(f"batch_size: {batch_size}")
    print(f"pseudo_bkg_dsid: {args.pseudo_bkg_dsid}")

    with torch.no_grad():
        fit_features, fit_labels, fit_weights = collect_fit_inputs(
            model, train_dl, args.fit_batches, config, padding_token, args.pseudo_bkg_dsid
        )
        fair_readout = fit_fair_readout(fit_features, fit_labels, fit_weights)
        eval_features, softor_scores, max_scores, eval_labels, eval_weights = collect_eval_inputs(
            model, val_dl, args.eval_batches, config, padding_token, args.pseudo_bkg_dsid
        )

    eval_labels_np = eval_labels.numpy()
    print(
        f"fit events: {fit_labels.numel()}  eval events: {eval_labels.numel()}  "
        f"eval signal/background: {int((eval_labels_np == 1).sum())}/{int((eval_labels_np == 0).sum())}"
    )
    if eval_labels_np.size == 0 or np.unique(eval_labels_np).size < 2:
        print("note: eval labels contain fewer than two classes; AUC and thresholded Z are undefined.")

    print_metric_line("parameter-free softor", softor_scores.numpy(), eval_labels_np, eval_weights.numpy())
    print_metric_line("parameter-free max", max_scores.numpy(), eval_labels_np, eval_weights.numpy())

    if fair_readout is None:
        print("fair-readout logistic: AUC=nan  best_asimov_z=nan  thr=nan")
        print("note: fair readout was not fit because fit labels contain fewer than two classes.")
    else:
        fair_scores = fair_readout.score(eval_features)
        print_metric_line("fair-readout logistic", fair_scores, eval_labels_np, eval_weights.numpy())
        print("fair-readout coefficients:")
        for name, coef in fair_readout.coefficients.items():
            print(f"  {name}: {coef:+.6g}")


if __name__ == "__main__":
    main()
