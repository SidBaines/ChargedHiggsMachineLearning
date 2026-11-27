"""Evaluate explicit physics and pooled readouts of a frozen joint checkpoint.

Run with python -m experiments.physics_readouts. The only inputs to feature
extraction are the event objects and frozen network outputs. Four disjoint
row residues within each DSID supply fit, calibration, evaluation and
confirmation events, with the common frozen reconstruction defining the
validity gate and channel assignment.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from experiments.eval_single_readout import build_model, find_config, sanitize_padding
from experiments.physics_samples import operating_points, sample_events
from utils.utils import check_category, check_valid


CHANNELS = {1: "lvbb", 2: "qqbb"}
SPLITS = {"fit": 0, "calibration": 2, "evaluation": 4, "confirmation": 6}
MASS_TEV = dict(zip(range(510115, 510125), [0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]))
FEATURE_NAMES = (
    "higgs_present", "higgs_is_resolved", "higgs_mass_gev", "higgs_pt_over_jet_ht",
    "higgs_max_tag", "higgs_mean_tag", "higgs_prong_delta_r",
    "higgs_min_assigned_logit_margin", "higgs_best_unassigned_logit_margin",
    "higgs_alternative_exists",
    "w_present", "w_is_resolved", "w_is_leptonic", "w_mass_gev", "w_pt_over_jet_ht",
    "w_max_jet_tag", "w_mean_jet_tag", "w_prong_delta_r",
    "w_min_assigned_logit_margin", "w_best_unassigned_logit_margin",
    "w_alternative_exists", "both_present", "wh_mass_gev", "wh_delta_r",
    "wh_abs_delta_phi", "w_pt_fraction_of_wh_scalar_pt", "log_jet_ht_gev",
    "n_large_r_jets", "n_small_r_jets",
)
WIDTH_TARGETS = {"physics": 12000, "pooled": 12000, "physics_pooled": 12000, "physics_wide": 48000}
DEFAULT_ARMS = ("physics", "pooled", "physics_pooled", "pooled_linear", "physics_wide")
LINEAR_CONTROLS = ("physics_linear", "physics_pooled_linear")


def p4_observables(p4):
    px, py, pz, energy = np.moveaxis(p4, -1, 0)
    pt = np.hypot(px, py)
    mass = np.sqrt(np.maximum(energy ** 2 - pt ** 2 - pz ** 2, 0))
    eta = np.arcsinh(pz / np.maximum(pt, 1.e-6))
    return pt, mass, eta, np.arctan2(py, px)


def delta_phi(phi_1, phi_2):
    return (phi_1 - phi_2 + np.pi) % (2 * np.pi) - np.pi


def prong_distance(mask, object_eta, object_phi):
    has_two = mask.sum(axis=1) == 2
    first = mask.argmax(axis=1)
    last = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    row = np.arange(len(mask))
    deta = object_eta[row, first] - object_eta[row, last]
    dphi = delta_phi(object_phi[row, first], object_phi[row, last])
    return np.where(has_two, np.hypot(deta, dphi), 0)


def candidate_evidence(mask, logits, kind, eligible, tags, jets, present):
    assigned_margin = logits[..., kind] - logits[..., [i for i in range(3) if i != kind]].max(axis=-1)
    winner = np.where(mask, assigned_margin, np.inf).min(axis=1)
    alternative_mask = eligible & ~mask
    alternative = np.where(alternative_mask, assigned_margin, -np.inf).max(axis=1)
    tagged = mask & jets
    max_tag = np.where(tagged, tags, -np.inf).max(axis=1)
    mean_tag = np.where(tagged, tags, 0).sum(axis=1) / np.maximum(tagged.sum(axis=1), 1)
    return (
        np.where(present & tagged.any(axis=1), max_tag, 0),
        np.where(present, mean_tag, 0),
        np.where(present, winner, 0),
        np.where(alternative_mask.any(axis=1), alternative, 0),
        alternative_mask.any(axis=1),
    )


def physics_features(x, types, logits):
    """Use exact hard assignments, zero-fill invalid candidates, and flag absence.

    A Higgs candidate consists of either one large-R or two small-R jets. A W
    candidate uses either one large-R jet, two small-R jets, or a charged
    lepton and a neutrino. The W leptonic mass uses the stored neutrino p4.
    Individual valid candidates can be present in an invalid complete event.
    """
    prediction = logits.argmax(axis=-1)
    nonpad = types != 5
    large = types == 3
    small = types == 4
    jets = large | small
    lepton = (types == 0) | (types == 1)
    neutrino = types == 2
    h = (prediction == 1) & nonpad
    w = (prediction == 2) & nonpad
    nh, nw = h.sum(axis=1), w.sum(axis=1)
    h_boosted = (nh == 1) & (h & large).any(axis=1)
    h_resolved = (h & small).sum(axis=1) == 2
    h_resolved &= nh == 2
    w_boosted = (nw == 1) & (w & large).any(axis=1)
    w_resolved = ((w & small).sum(axis=1) == 2) & (nw == 2)
    w_lep = (nw == 2) & (w & lepton).any(axis=1) & (w & neutrino).any(axis=1)
    has_h = h_boosted | h_resolved
    has_w = w_boosted | w_resolved | w_lep
    both = has_h & has_w
    h &= has_h[:, None]
    w &= has_w[:, None]
    p4 = np.where(nonpad[..., None], x[..., :4], 0).astype(np.float64) * 100
    tags = x[..., 4]
    pt, _, eta, phi = p4_observables(p4)
    ht = np.where(jets, pt, 0).sum(axis=1)
    h4 = np.where(h[..., None], p4, 0).sum(axis=1)
    w4 = np.where(w[..., None], p4, 0).sum(axis=1)
    h_pt, h_mass, h_eta, h_phi = p4_observables(h4)
    w_pt, w_mass, w_eta, w_phi = p4_observables(w4)
    _, wh_mass, _, _ = p4_observables(h4 + w4)
    wh_dphi = np.abs(delta_phi(w_phi, h_phi))
    h_tag, h_mean_tag, h_margin, h_alt, h_alt_exists = candidate_evidence(h, logits, 1, jets, tags, jets, has_h)
    w_tag, w_mean_tag, w_margin, w_alt, w_alt_exists = candidate_evidence(w, logits, 2, nonpad, tags, jets, has_w)
    features = np.stack([
        has_h, h_resolved, h_mass, h_pt / np.maximum(ht, 1), h_tag, h_mean_tag,
        prong_distance(h, eta, phi), h_margin, h_alt, h_alt_exists,
        has_w, w_resolved, w_lep, w_mass, w_pt / np.maximum(ht, 1), w_tag, w_mean_tag,
        prong_distance(w, eta, phi), w_margin, w_alt, w_alt_exists,
        both, np.where(both, wh_mass, 0),
        np.where(both, np.hypot(w_eta - h_eta, wh_dphi), 0),
        np.where(both, wh_dphi, 0),
        np.where(both, w_pt / np.maximum(h_pt + w_pt, 1), 0),
        np.log1p(ht), large.sum(axis=1), small.sum(axis=1),
    ], axis=1).astype(np.float32)
    assert features.shape[1] == len(FEATURE_NAMES)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite reconstructed physics features")
    return features


def significance_score(logits):
    """Signal-to-background log odds, monotone with 1–P(background)."""
    return (torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]).cpu().numpy()


@torch.no_grad()
def freeze_features(model, config, events, device, batch_size):
    output = {key: [] for key in ("physics", "pooled", "native", "channel", "perfect", "category")}
    padding = config["num_particle_types"] - 1
    for start in range(0, len(events["label"]), batch_size):
        selection = slice(start, start + batch_size)
        x = events["x"][selection]
        types = events["types"][selection]
        truth = events["truth"][selection]
        types_tensor = torch.as_tensor(types, device=device, dtype=torch.long)
        sanitized = sanitize_padding(torch.as_tensor(x, device=device), types_tensor, padding)
        feats = model._backbone_features(sanitized[..., :config["model_input_vars"]], types_tensor)
        logits = model.classifier(feats)
        nonpad = (types_tensor != padding).unsqueeze(-1)
        pooled = (feats * nonpad).sum(dim=1) / nonpad.sum(dim=1).clamp(min=1)
        native = model.event_classifier(pooled)
        _, channel = check_valid(types_tensor, logits, padding, categorical=True, returnTypes=True)
        pred = logits.argmax(-1).cpu().numpy()
        collapsed = (truth == 1).astype(np.int8) + (truth > 1).astype(np.int8) * 2
        output["physics"].append(physics_features(x, types, logits.cpu().numpy()))
        output["pooled"].append(pooled.cpu().numpy())
        output["native"].append(significance_score(native))
        output["channel"].append(channel.cpu().numpy())
        output["perfect"].append(((pred == collapsed) | (types == padding)).all(axis=1))
        output["category"].append(check_category(types, truth, padding))
    merged = {key: np.concatenate(parts) for key, parts in output.items()}
    if not np.isfinite(merged["pooled"]).all() or not np.isfinite(merged["native"]).all():
        raise ValueError("Non-finite frozen network outputs")
    return merged


def fit_weights(fit):
    """Give signal and background equal total fit weight, as in joint training."""
    source = np.abs(fit.get("train_weight", fit["weight"])).copy()
    signal = fit["label"] > 0
    for mask in (signal, ~signal):
        group_sum = source[mask].sum()
        if group_sum <= 0:
            raise ValueError("Fit split has no positive training weight for one binary class")
        source[mask] *= len(source) / (2 * group_sum)
    return source.astype(np.float32)


def readout_inputs(frozen, name):
    name = name.removesuffix("_linear")
    if name == "physics_pooled":
        return np.column_stack((frozen["physics"], frozen["pooled"]))
    return frozen["pooled"] if name.startswith("pooled") else frozen["physics"]


def fit_readout(fit_events, splits, name, seed, epochs, batch_size):
    torch.manual_seed(seed)
    fit = readout_inputs(splits["fit"], name)
    mean = fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = fit.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.e-7] = 1
    x = torch.from_numpy((fit - mean) / scale)
    y = torch.as_tensor(fit_events["label"], dtype=torch.long)
    weights = torch.from_numpy(fit_weights(fit_events))
    features = x.shape[1]
    if name.endswith("_linear"):
        model = nn.Linear(features, 3)
    else:
        width = (WIDTH_TARGETS[name] - 3) // (features + 4)
        model = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1.e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0002)
    shuffle = torch.Generator().manual_seed(seed)
    fit_total_weight = weights.sum().item()
    history = []
    for _ in range(epochs):
        totals = 0
        for batch in torch.randperm(len(y), generator=shuffle).split(batch_size):
            loss = nn.functional.cross_entropy(model(x[batch]), y[batch], reduction="none")
            weighted_loss = (loss * weights[batch]).sum()
            optimizer.zero_grad()
            (weighted_loss / len(batch)).backward()
            optimizer.step()
            totals += weighted_loss.item()
        history.append(totals / fit_total_weight)
        scheduler.step()
    with torch.no_grad():
        scores = {
            split: np.concatenate([
                significance_score(model(torch.from_numpy((batch - mean) / scale)))
                for batch in np.array_split(readout_inputs(frozen, name),
                                            max(1, len(frozen["native"]) // 4096))
            ])
            for split, frozen in splits.items() if split != "fit"
        }
    return scores, {"parameters": sum(p.numel() for p in model.parameters()), "fit_loss_by_epoch": history}


def weight_summary(weight, mask):
    selected = weight[mask].astype(np.float64)
    return {"n": len(selected), "yield": float(selected.sum()), "sumw2": float(np.square(selected).sum())}


def perfect_reconstruction(events, features):
    weights = events["weight"]
    signal = events["label"] > 0
    categories = {"overall": signal}
    categories.update({str(category): signal & (features["category"] == category) for category in range(6)})
    categories["unclassified"] = signal & (features["category"] == -1)
    out = {}
    for name, mask in categories.items():
        perfect = weight_summary(weights, mask & features["perfect"])
        total = weight_summary(weights, mask)
        out[name] = {
            "perfect": perfect, "total": total,
            "fraction": perfect["yield"] / total["yield"] if total["yield"] else None,
        }
    return out


def auc(score, label, weight):
    if len(np.unique(label)) < 2 or min(weight[label].sum(), weight[~label].sum()) <= 0:
        return None
    return roc_auc_score(label, score, sample_weight=weight)


def score_metrics(events, frozen, score):
    y = events["label"] > 0
    channel = frozen["channel"]
    weights = np.abs(events["weight"])
    valid = channel > 0
    gated_score = np.where(valid, score, -np.inf)
    # sklearn accepts only finite scores. Put invalid events in a shared tie below every valid score.
    gated_score[~valid] = score.min() - 1
    by_channel = {}
    for cid, name in CHANNELS.items():
        selected = channel == cid
        by_channel[name] = auc(score[selected], y[selected], weights[selected])
    return {
        "auc_all_abs_weight": auc(score, y, weights),
        "auc_with_invalid_tied_at_bottom_abs_weight": auc(gated_score, y, weights),
        "auc_valid_abs_weight": auc(score[valid], y[valid], weights[valid]),
        "auc_by_reco_channel_abs_weight": by_channel,
    }


def ambiguity(features):
    h = features[:, FEATURE_NAMES.index("higgs_min_assigned_logit_margin")]
    w = features[:, FEATURE_NAMES.index("w_min_assigned_logit_margin")]
    return np.minimum(h, w)


def tail_slices(events, frozen, score, working_points, ambiguity_threshold):
    out = {}
    bg = events["label"] == 0
    score = np.asarray(score, dtype=float)
    is_ambiguous = ambiguity(frozen["physics"]) <= ambiguity_threshold
    dsids = events["dsid"]
    w = events["weight"]
    for cid, channel in CHANNELS.items():
        threshold = working_points[channel]["threshold"]
        in_channel = frozen["channel"] == cid
        same_channel_signal = events["label"] == cid
        kept = in_channel & (score >= threshold) if threshold is not None else np.zeros_like(in_channel)
        slices = {}
        for label, mask in {"lower_margin_half": is_ambiguous, "upper_margin_half": ~is_ambiguous}.items():
            slices[label] = {
                "background_retained": weight_summary(w, kept & mask & bg),
                "background_available": weight_summary(w, in_channel & mask & bg),
                "signal_by_mass": {
                    str(dsid): {
                        "retained": weight_summary(w, kept & mask & same_channel_signal & (dsids == dsid)),
                        "available": weight_summary(w, in_channel & mask & same_channel_signal & (dsids == dsid)),
                    }
                    for dsid in MASS_TEV
                },
            }
        processes = {}
        for process, mask in {
            "top_410xxx": dsids // 1000 == 410,
            "vjets_700xxx": dsids // 1000 == 700,
            "diboson_363xxx": dsids // 1000 == 363,
            "other": ~np.isin(dsids // 1000, (410, 700, 363)),
        }.items():
            processes[process] = {
                "retained": weight_summary(w, kept & bg & mask),
                "available": weight_summary(w, in_channel & bg & mask),
            }
        selected_bg = kept & bg
        out[channel] = {
            "by_ambiguity": slices, "background_by_process": processes,
            "signal_crossfeed": weight_summary(w, kept & ~bg & ~same_channel_signal),
            "background_retained_by_dsid": {
                str(dsid): weight_summary(w, selected_bg & (dsids == dsid))
                for dsid in np.unique(dsids[selected_bg])
            },
        }
    return out


def readout_report(events, frozen, scores, split, ambiguity_threshold):
    cal, test = events["calibration"], events[split]
    cal_channel, test_channel = frozen["calibration"]["channel"], frozen[split]["channel"]
    work = operating_points(cal, test, scores["calibration"], scores[split], cal_channel, test_channel)
    return {
        **score_metrics(test, frozen[split], scores[split]),
        "operating_points": work,
        "tail_slices": tail_slices(test, frozen[split], scores[split], work, ambiguity_threshold),
    }


def cache_id(args):
    inputs = [args.ckpt.resolve(), Path(find_config(args.ckpt)).resolve(),
              *sorted(args.data_dir.resolve().glob("dsid_*.memmap*"))]
    digest = hashlib.sha256()
    for path in inputs:
        stat = path.stat()
        digest.update(repr((str(path), stat.st_size, stat.st_mtime_ns)).encode())
    digest.update(repr((FEATURE_NAMES, args.device, args.inference_batch)).encode())
    return digest.hexdigest()[:20]


def load_or_extract(args, model, cfg, split, n_sig, n_bg, cache_key):
    residue = SPLITS[split]
    if args.cache_dir is not None:
        cache_file = args.cache_dir / f"readouts_{cache_key}_{residue}_{n_sig}_{n_bg}_{args.sample_seed}.npz"
        if cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as data:
                events = {key[3:]: data[key] for key in data.files if key.startswith("ev_")}
                frozen = {key[3:]: data[key] for key in data.files if key.startswith("fr_")}
            print(f"{split} loaded {len(events['label']):,} cached events", flush=True)
            return events, frozen
    started = time.monotonic()
    events = sample_events(args.data_dir, residue, n_sig, n_bg, seed=args.sample_seed)
    print(f"{split}: sample {len(events['label']):,} events in {time.monotonic() - started:.1f}s", flush=True)
    frozen = freeze_features(model, cfg, events, args.device, args.inference_batch)
    print(f"{split}: extracted physics and pooled features in {time.monotonic() - started:.1f}s", flush=True)
    events = {key: val for key, val in events.items() if key not in {"x", "types", "truth"}}
    if args.cache_dir is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(".tmp.npz")
        np.savez(temporary, **{f"ev_{key}": val for key, val in events.items()},
                 **{f"fr_{key}": val for key, val in frozen.items()})
        temporary.replace(cache_file)
    return events, frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--inference-batch", type=int, default=1024)
    parser.add_argument("--fit-batch", type=int, default=1024)
    parser.add_argument("--n-fit", type=int, nargs=2, metavar=("SIGNAL", "BACKGROUND"), default=(12000, 30000))
    parser.add_argument("--n-calibration", type=int, nargs=2, metavar=("SIGNAL", "BACKGROUND"), default=(12000, 100000))
    parser.add_argument("--n-test", type=int, nargs=2, metavar=("SIGNAL", "BACKGROUND"), default=(12000, 100000))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--arms", nargs="+", choices=DEFAULT_ARMS + LINEAR_CONTROLS, default=DEFAULT_ARMS)
    parser.add_argument("--sample-seed", type=int, default=2701)
    parser.add_argument("--seeds", type=int, nargs="+", default=(1421, 1422, 1423))
    parser.add_argument("--pilot", action="store_true", help="Skip confirmation and explicitly label output pilot")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    if args.epochs < 1 or min(args.n_fit + args.n_calibration + args.n_test) < 1:
        parser.error("epoch and sample sizes must be positive")

    with open(find_config(args.ckpt)) as handle:
        cfg = json.load(handle)
    model = build_model(cfg, args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device, weights_only=True))
    model.eval()
    sizes = {"fit": args.n_fit, "calibration": args.n_calibration, "evaluation": args.n_test}
    if not args.pilot:
        sizes["confirmation"] = args.n_test
    events, frozen = {}, {}
    cache_key = cache_id(args) if args.cache_dir is not None else None
    for split, (n_sig, n_bg) in sizes.items():
        events[split], frozen[split] = load_or_extract(args, model, cfg, split, n_sig, n_bg, cache_key)
        assert np.all(events[split]["row"] % 8 == SPLITS[split])
    fit_valid = frozen["fit"]["channel"] > 0
    if not fit_valid.any():
        raise ValueError("At least one fit event must pass frozen reconstruction")
    ambiguity_threshold = float(np.median(ambiguity(frozen["fit"]["physics"])[fit_valid]))
    test_splits = ("evaluation",) if args.pilot else ("evaluation", "confirmation")
    result = {
        "config": {
            "pilot": args.pilot, "ckpt": str(args.ckpt), "data": str(args.data_dir),
            "device": args.device, "threads": args.threads, "inference_batch": args.inference_batch,
            "fit_batch": args.fit_batch, "epochs": args.epochs, "seeds": list(args.seeds),
            "sample_seed": args.sample_seed, "requested_signal_background_counts": sizes,
            "row_residues": {split: SPLITS[split] for split in sizes},
            "feature_names": FEATURE_NAMES, "signal_mass_tev_by_dsid": MASS_TEV,
            "validity_and_channel": "common utils.check_valid on frozen object argmax; 1=lvbb, 2=qqbb",
            "score": "logsumexp(lvbb,qqbb) - background logit",
            "training": "3-class cross entropy, equal total fit weight for signal and background, "
                        "inclusion-corrected abs training weights when present, AdamW cosine schedule",
            "ambiguity": "smaller of minimum H and minimum W assigned-object margins; "
                         "split at median of reco-valid fit events (unweighted)",
            "ambiguity_logit_margin_threshold": ambiguity_threshold,
            "capacity_parameter_targets": WIDTH_TARGETS,
            "fitted_arms": args.arms,
        },
        "sample_counts": {split: {
            "signal": int((es["label"] > 0).sum()),
            "background": int((es["label"] == 0).sum()),
            "valid_signal": int(((es["label"] > 0) & (frozen[split]["channel"] > 0)).sum()),
            "valid_background": int(((es["label"] == 0) & (frozen[split]["channel"] > 0)).sum()),
        } for split, es in events.items()},
        "reconstruction": {split: perfect_reconstruction(events[split], frozen[split]) for split in test_splits},
        "readouts": {},
    }
    native = {split: fr["native"] for split, fr in frozen.items()}
    result["readouts"]["native"] = {split: readout_report(events, frozen, native, split, ambiguity_threshold)
                                    for split in test_splits}
    for name in args.arms:
        result["readouts"][name] = {}
        for seed in args.seeds:
            started = time.monotonic()
            scores, head = fit_readout(events["fit"], frozen, name, seed, args.epochs, args.fit_batch)
            arms = {split: readout_report(events, frozen, scores, split, ambiguity_threshold)
                    for split in test_splits}
            result["readouts"][name][str(seed)] = {**head, **arms}
            print(f"{name} seed={seed}: {time.monotonic() - started:.1f}s "
                  f"valid AUC " + ", ".join(f"{split}={arms[split]['auc_valid_abs_weight']}" for split in test_splits),
                  flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
