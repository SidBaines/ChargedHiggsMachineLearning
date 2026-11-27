"""Paired frozen jet edits and clean-head restoration on held-out real events.

Run using the original data and training output directories, e.g.
python experiments/physics_interventions.py --data-dir DATA --joint JOINT --reco RECO
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

from hd_interp import load_trained
from physics_samples import operating_points, sample_events
from utils.utils import check_category


W_MASS = 80.4
LARGE_JET = 3
PAD = 5
DONOR_SCALE = np.array([0.15, 0.30, 12.0])
ANGLE = 0.65
JET_ARMS = ("identity", "mass_half", "mass_full", "pt_down", "pt_up",
            "tag_neutral", "tag_down", "tag_up")
HEADS = tuple((block, head) for block in range(3) for head in range(4))
GEOM_ARMS = ("identity", "geometry", "identity_patch", "b1_all",
             *(f"b{block}h{head}" for block, head in HEADS))


def kinematics(vectors):
    p4 = vectors[..., :4].astype(np.float64)
    px, py, pz, energy = np.moveaxis(p4, -1, 0)
    pt = np.hypot(px, py)
    return (
        pt * 100.0,
        np.arcsinh(pz / np.maximum(pt, 1e-14)),
        np.arctan2(py, px),
        np.sqrt(np.maximum(energy ** 2 - (px ** 2 + py ** 2 + pz ** 2), 0)) * 100.0,
    )


def wrap_phi(dphi):
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def select_jet_pair(x, types):
    """Select the closest and second-closest boosted W-mass candidates."""
    pt, eta, _, mass = kinematics(x)
    accepted = (types == LARGE_JET) & (pt >= 200) & (np.abs(eta) <= 2)
    accepted &= (mass >= 50) & (mass <= 125)
    ranked = np.argsort(np.where(accepted, np.abs(mass - W_MASS), np.inf),
                        axis=1, kind="stable")
    rows = np.flatnonzero(accepted.sum(axis=1) >= 2)
    return rows, ranked[rows, 1], ranked[rows, 0]


def select_geometry_pair(x, types):
    """Select mutually exclusive boosted H and W proxies from mass and tag."""
    pt, eta, phi, mass = kinematics(x)
    tag = x[..., 4]
    large_jet = (types == LARGE_JET) & (pt >= 200) & (np.abs(eta) <= 2)
    h_allowed = large_jet & (mass >= 80) & (mass <= 165) & (tag >= 1.0)
    w_allowed = large_jet & (mass >= 50) & (mass <= 120) & (tag <= 0.5)
    h = np.where(h_allowed, tag, -np.inf).argmax(1)
    w = np.where(w_allowed, np.abs(mass - W_MASS), np.inf).argmin(1)
    rows = np.arange(len(x))
    dphi = wrap_phi(phi[rows, w] - phi[rows, h])
    suitable_separation = (np.abs(dphi) >= 1.1) & (np.abs(dphi) <= 2.45)
    rows = np.flatnonzero(h_allowed.any(1) & w_allowed.any(1) & suitable_separation)
    return rows, w[rows], h[rows], np.sign(dphi[rows])


def match_donors(target_x, donors):
    donor_rows, donor_cols = np.where(donors["types"] == LARGE_JET)
    bank = donors["x"][donor_rows, donor_cols]
    donor_pt, donor_eta, _, donor_mass = kinematics(bank)
    valid = (donor_pt >= 150) & (np.abs(donor_eta) <= 2.5) & (donor_mass >= 35)
    bank, donor_rows, donor_cols = bank[valid], donor_rows[valid], donor_cols[valid]
    features = np.column_stack((np.log(donor_pt[valid]), donor_eta[valid], donor_mass[valid]))
    pt, eta, _, mass = kinematics(target_x)
    query = np.column_stack((np.log(pt), eta, mass))
    tree = cKDTree(features / DONOR_SCALE)
    matched = {}
    for arm in ("tag_neutral", "tag_down", "tag_up"):
        matched[arm] = dict(
            tag=np.full(len(query), np.nan, dtype=np.float32),
            dsid=np.full(len(query), -1, dtype=np.int64),
            row=np.full(len(query), -1, dtype=np.int64),
            column=np.full(len(query), -1, dtype=np.int64),
            differences=np.full((len(query), 3), np.nan),
        )
    for idx, neighbours in enumerate(tree.query_ball_point(query / DONOR_SCALE, 1.0)):
        neighbours = np.asarray(neighbours, dtype=int)
        shift = bank[neighbours, 4] - target_x[idx, 4]
        distance = np.linalg.norm((features[neighbours] - query[idx]) / DONOR_SCALE, axis=1)
        order = np.lexsort((neighbours, distance))
        conditions = {
            "tag_neutral": np.abs(shift) <= 0.05,
            "tag_down": (shift >= -2.0) & (shift <= -0.5),
            "tag_up": (shift >= 0.5) & (shift <= 2.0),
        }
        for arm, condition in conditions.items():
            eligible = order[condition[order]]
            if not len(eligible):
                continue
            idx_donor = neighbours[eligible[0]]
            source = donor_rows[idx_donor]
            result = matched[arm]
            result["tag"][idx] = bank[idx_donor, 4]
            result["dsid"][idx] = donors["dsid"][source]
            result["row"][idx] = donors["row"][source]
            result["column"][idx] = donor_cols[idx_donor]
            result["differences"][idx] = features[idx_donor] - query[idx]
    common = np.logical_and.reduce([np.isfinite(result["tag"]) for result in matched.values()])
    return matched, common


def edit_jet(x, target, arm, donor=None):
    edited = x.copy()
    if arm == "identity":
        return edited
    rows = np.arange(len(x))
    original = x[rows, target]
    jet = original.astype(np.float64)
    p_squared = np.square(jet[:, :3]).sum(1)
    if arm.startswith("mass"):
        original_mass = kinematics(original)[3]
        fraction = {"mass_half": 0.5, "mass_full": 1.0}[arm]
        mass = (original_mass + fraction * (W_MASS - original_mass)) / 100.0
        jet[:, 3] = np.sqrt(p_squared + mass ** 2)
    elif arm.startswith("pt"):
        factor = {"pt_down": 0.80, "pt_up": 1.25}[arm]
        mass_squared = jet[:, 3] ** 2 - p_squared
        jet[:, :3] *= factor
        jet[:, 3] = np.sqrt(factor ** 2 * p_squared + mass_squared)
    else:
        jet[:, 4] = donor
    edited[rows, target] = jet
    return edited


def rotate_jet(x, target, angle):
    edited = x.copy()
    rows = np.arange(len(x))
    px, py = x[rows, target, :2].astype(np.float64).T
    cosine, sine = np.cos(angle), np.sin(angle)
    edited[rows, target, 0] = cosine * px - sine * py
    edited[rows, target, 1] = sine * px + cosine * py
    return edited


def head_context(attention, x, pad):
    """Context per query and head, immediately before the native output projection."""
    if not attention._qkv_same_embed_dim or attention.in_proj_bias is None:
        raise ValueError("Head restoration expects packed, biased self-attention")
    n, tokens, width = x.shape
    nhead = attention.num_heads
    dim = width // nhead
    q, k, v = F.linear(x, attention.in_proj_weight, attention.in_proj_bias).chunk(3, -1)
    q, k, v = (z.reshape(n, tokens, nhead, dim).transpose(1, 2) for z in (q, k, v))
    score = (q * (dim ** -0.5)) @ k.transpose(-2, -1)
    return score.masked_fill(pad[:, None, None, :], -torch.inf).softmax(-1) @ v


@contextmanager
def head_hooks(model, blocks, pad, captured=None, replacements=None, parity=None):
    def hook_for(block_idx):
        def hook(attention, inputs, output):
            ctx = head_context(attention, inputs[0], pad)
            if parity is not None:
                rebuilt = F.linear(ctx.transpose(1, 2).flatten(2), attention.out_proj.weight,
                                   attention.out_proj.bias)
                parity[block_idx] = max(parity.get(block_idx, 0),
                                        float((rebuilt - output[0]).abs().max()))
            if captured is not None:
                captured[block_idx] = ctx
            if replacements is not None:
                delta = torch.zeros_like(ctx)
                for head, clean in replacements[block_idx].items():
                    delta[:, head] = clean - ctx[:, head]
                correction = F.linear(delta.transpose(1, 2).flatten(2),
                                      attention.out_proj.weight)
                return output[0] + correction, output[1]
            return None
        return hook

    handles = [model.attention_blocks[block]["self_attention"].register_forward_hook(hook_for(block))
               for block in blocks]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def as_numpy(output, include_event):
    result = {"reco": output["reco"].numpy()}
    if include_event:
        result["event"] = output["event"].numpy()
    return result


@torch.inference_mode()
def infer_single(model, x, types, batch_size, include_event):
    result = []
    for start in range(0, len(x), batch_size):
        s = slice(start, start + batch_size)
        result.append(as_numpy(model(torch.from_numpy(x[s]), torch.from_numpy(types[s])),
                               include_event))
    return {key: np.concatenate([batch[key] for batch in result]) for key in result[0]}


@torch.inference_mode()
def infer_geometry(model, x, types, target, signs, batch_size, include_event):
    collected = {arm: [] for arm in GEOM_ARMS}
    parity = {}
    for start in range(0, len(x), batch_size):
        s = slice(start, start + batch_size)
        tx = torch.from_numpy(x[s])
        tt = torch.from_numpy(types[s])
        pad = tt == PAD
        clean_ctx = {}
        with head_hooks(model, range(3), pad, captured=clean_ctx, parity=parity):
            clean = model(tx, tt)
        corrupted_x = torch.from_numpy(rotate_jet(x[s], target[s], ANGLE * signs[s]))
        corrupted_ctx = {}
        with head_hooks(model, (1,), pad, captured=corrupted_ctx):
            corrupted = model(corrupted_x, tt)
        arms = {"identity": clean, "geometry": corrupted}
        patches = {f"b{b}h{h}": {b: {h: clean_ctx[b][:, h]}} for b, h in HEADS}
        patches["b1_all"] = {1: {h: clean_ctx[1][:, h] for h in range(4)}}
        patches["identity_patch"] = {1: {0: corrupted_ctx[1][:, 0]}}
        for arm, patch in patches.items():
            with head_hooks(model, patch.keys(), pad, replacements=patch):
                arms[arm] = model(corrupted_x, tt)
        for arm, output in arms.items():
            collected[arm].append(as_numpy(output, include_event))
    return {
        arm: {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
        for arm, batches in collected.items()
    }, parity


def softmax(values):
    p = np.exp(values - values.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def event_summary(outputs, types, truth, target=None, other=None):
    reco = outputs["reco"]
    pred = reco.argmax(-1)
    cat = check_category(types, pred, PAD)
    result = {
        "perfect": ((pred == np.minimum(truth, 2)) | (types == PAD)).all(1),
        "valid": cat >= 0,
        "channel": np.where(cat < 0, 0, np.where(np.isin(cat, (1, 4)), 1, 2)),
        "reco_objects": pred.astype(np.int8),
    }
    if "event" in outputs:
        p = softmax(outputs["event"])
        result.update({"p_background": p[:, 0], "p_lvbb": p[:, 1], "p_qqbb": p[:, 2],
                       "p_signal": 1.0 - p[:, 0]})
    if target is None:
        return result

    rows = np.arange(len(types))
    lep = (types == 0) | (types == 1)
    nu = types == 2
    single_pair = (lep.sum(1) == 1) & (nu.sum(1) == 1)
    lep_W = np.sum((lep | nu) & (pred == 2), axis=1)
    w_margin = reco[..., 2] - reco[..., :2].max(-1)
    tagged = reco[rows, target]
    p = softmax(tagged)
    result.update({
        "target_w_margin": w_margin[rows, target],
        "other_w_margin": w_margin[rows, other],
        "pair_w_margin": w_margin[rows, target] - w_margin[rows, other],
        "pair_winner_target": w_margin[rows, target] > w_margin[rows, other],
        "target_w_claim": pred[rows, target] == 2,
        "other_w_claim": pred[rows, other] == 2,
        "both_w_claim": (pred[rows, target] == 2) & (pred[rows, other] == 2),
        "target_w_vs_h": tagged[:, 2] - tagged[:, 1],
        "target_w_vs_none": tagged[:, 2] - tagged[:, 0],
        "target_p_none": p[:, 0],
        "target_p_h": p[:, 1],
        "target_p_w": p[:, 2],
        "lep_pair": single_pair,
        "lep_verdict": lep_W == 2,
        "lep_disagree": lep_W == 1,
    })
    return result


class Estimates:
    """Event-paired means, with conditional MC uncertainty within sampled DSIDs."""

    def __init__(self, sample, indices):
        self.weights = sample["weight"][indices].astype(float)
        self.labels = sample["label"][indices]
        self.dsids = sample["dsid"][indices]

    def mean(self, values, where=None):
        if where is None:
            where = np.ones(len(self.weights), dtype=bool)
        v = values[where].astype(float)
        w = self.weights[where]
        dsid = self.dsids[where]
        n = len(v)
        result = {"n": n, "sum_weight": float(w.sum()), "n_negative_weights": int((w < 0).sum()),
                  "mean": None, "ci95": None, "signed_mean": None, "signed_ci95": None}
        if n == 0:
            return result
        avg = v.mean()
        result["mean"] = float(avg)
        if n > 1:
            se = v.std(ddof=1) / np.sqrt(n)
            result["ci95"] = [float(avg - 1.96 * se), float(avg + 1.96 * se)]
        denom = w.sum()
        if abs(denom) > 1e-12 * np.abs(w).sum():
            signed = np.dot(w, v) / denom
            result["signed_mean"] = float(signed)
            influence = w * (v - signed)
            var = 0.0
            for d in np.unique(dsid):
                group = influence[dsid == d]
                if len(group) > 1:
                    var += len(group) * group.var(ddof=1)
            if n > 1:
                se = np.sqrt(var) / abs(denom)
                result["signed_ci95"] = [float(signed - 1.96 * se), float(signed + 1.96 * se)]
        return result

    def describe(self, values, mask=None):
        if mask is None:
            mask = np.ones(len(self.weights), dtype=bool)
        return {
            label: self.mean(values, mask & label_mask)
            for label, label_mask in (
                ("all", np.ones(len(self.labels), dtype=bool)),
                ("background", self.labels == 0),
                ("lvbb", self.labels == 1),
                ("qqbb", self.labels == 2),
            )
        }


def comparisons(current, reference, estimates, mask, types):
    """Compute every arm's raw deltas before population summaries."""
    metrics = {}
    for key in current:
        if key in ("reco_objects", "channel", "lep_verdict", "lep_pair", "lep_disagree"):
            continue
        if key == "perfect":
            metrics[key] = estimates.describe(current[key].astype(float) - reference[key],
                                               mask & (estimates.labels != 0))
        else:
            metrics[key] = estimates.describe(current[key].astype(float) - reference[key], mask)
    metrics["winner_swap"] = estimates.describe(current["pair_winner_target"] !=
                                               reference["pair_winner_target"], mask)
    metrics["became_pair_winner"] = estimates.describe(current["pair_winner_target"] &
                                                       ~reference["pair_winner_target"], mask)
    metrics["lost_pair_winner"] = estimates.describe(~current["pair_winner_target"] &
                                                     reference["pair_winner_target"], mask)
    metrics["reco_channel_change"] = estimates.describe(current["channel"] != reference["channel"],
                                                       mask)
    changes = (current["reco_objects"] != reference["reco_objects"]) & (types != PAD)
    metrics["any_object_assignment_change"] = estimates.describe(changes.any(1), mask)
    paired = mask & reference["lep_pair"]
    for key in ("lep_verdict", "lep_disagree"):
        metrics[f"{key}_change"] = estimates.describe(current[key] != reference[key], paired)
        metrics[f"{key}_net"] = estimates.describe(current[key].astype(float) - reference[key], paired)
    return metrics


def recovery(outputs, signatures, types, estimates):
    clean = signatures["identity"]
    corrupted = signatures["geometry"]

    def errors(arm):
        current = signatures[arm]
        reco = outputs[arm]["reco"]
        clean_reco = outputs["identity"]["reco"]
        nonpad = types != PAD
        result = {
            "target_w_margin": np.abs(current["target_w_margin"] - clean["target_w_margin"]),
            "pair_w_margin": np.abs(current["pair_w_margin"] - clean["pair_w_margin"]),
            "winner": current["pair_winner_target"] != clean["pair_winner_target"],
            "production_channel": current["channel"] != clean["channel"],
            "leptonic_verdict": current["lep_verdict"] != clean["lep_verdict"],
            "fraction_object_assignments_changed":
                ((current["reco_objects"] != clean["reco_objects"]) & nonpad).sum(1)
                / nonpad.sum(1),
            "reco_probability_total_variation":
                (0.5 * np.abs(softmax(reco) - softmax(clean_reco)).sum(-1) * nonpad).sum(1)
                / nonpad.sum(1),
        }
        if "p_signal" in clean:
            result["event_signal_probability"] = np.abs(current["p_signal"] - clean["p_signal"])
            result["event_probability_total_variation"] = 0.5 * sum(
                np.abs(current[key] - clean[key]) for key in ("p_background", "p_lvbb", "p_qqbb")
            )
        return result

    base_error = errors("geometry")
    candidate_error = errors("b1h0")
    report = {}
    for arm in outputs:
        if arm == "identity":
            continue
        stats = {}
        for key, error in errors(arm).items():
            mask = clean["lep_pair"] if key == "leptonic_verdict" else None
            gain = base_error[key].astype(float) - error
            base = estimates.mean(base_error[key], mask)
            effect = estimates.mean(gain, mask)
            detail = {"corrupted_error": base, "restoration_gain": effect,
                      "fraction_of_mean_error_recovered":
                          effect["mean"] / base["mean"] if base["mean"] else None}
            if arm not in ("b1h0", "geometry"):
                detail["b1h0_gain_minus_this_arm_gain"] = estimates.mean(
                    error.astype(float) - candidate_error[key], mask
                )
            if key in ("winner", "production_channel", "leptonic_verdict"):
                relevant = np.ones(len(error), dtype=bool) if mask is None else mask
                detail["n_corrupted"] = int((base_error[key] & relevant).sum())
                detail["n_restored"] = int((base_error[key] & ~error & relevant).sum())
                detail["n_new_errors"] = int((~base_error[key] & error & relevant).sum())
            stats[key] = detail
        report[arm] = stats
    return report


def check_edits(clean, edited, target, arm):
    rows = np.arange(len(clean))
    other = np.ones(clean.shape[:2], dtype=bool)
    other[rows, target] = False
    np.testing.assert_array_equal(edited[other], clean[other])
    base = clean[rows, target]
    change = edited[rows, target]
    p0, e0, a0, m0 = kinematics(base)
    p1, e1, a1, m1 = kinematics(change)
    close = np.testing.assert_allclose
    if arm == "identity":
        np.testing.assert_array_equal(edited, clean)
    elif arm.startswith("tag"):
        np.testing.assert_array_equal(change[:, :4], base[:, :4])
    elif arm.startswith("mass"):
        np.testing.assert_array_equal(change[:, :3], base[:, :3])
        np.testing.assert_array_equal(change[:, 4], base[:, 4])
        fraction = 0.5 if arm == "mass_half" else 1.0
        close(m1, m0 + fraction * (W_MASS - m0), atol=0.1)
    else:
        close(m1, m0, atol=0.1)
        close(e1, e0, atol=1e-6)
        if arm.startswith("pt"):
            np.testing.assert_array_equal(change[:, 4], base[:, 4])
            close(a1, a0, atol=1e-6)
            close(p1 / p0, 0.80 if arm == "pt_down" else 1.25, atol=1e-6)
        elif arm == "geometry":
            np.testing.assert_array_equal(change[:, 2:], base[:, 2:])
            close(p1 / p0, 1.0, atol=1e-6)


def perfect_reconstruction(sample, signature):
    stats = Estimates(sample, np.arange(len(sample["label"])))
    signal = sample["label"] != 0
    truth_cat = check_category(sample["types"], sample["truth"], PAD)
    return {"all_signal": stats.mean(signature["perfect"], signal),
            **{f"cat{c}": stats.mean(signature["perfect"], signal & (truth_cat == c))
               for c in range(6)}}


def ids(sample, indices):
    return {key: sample[key][indices] for key in ("dsid", "row", "label", "weight", "mc", "truth", "types")}


def save_raw(path, arrays, signatures):
    save = arrays.copy()
    for arm, signature in signatures.items():
        save.update({f"{arm}__{key}": value for key, value in signature.items()})
    np.savez_compressed(path, **save)


def paired_model_comparisons(signatures, tag_common, estimates):
    result = {}
    joint, reco = signatures["joint"], signatures["reco"]
    for arm in JET_ARMS:
        use = tag_common if arm.startswith("tag") else np.ones(len(tag_common), dtype=bool)
        result[arm] = {}
        for metric in ("target_w_margin", "pair_w_margin", "pair_winner_target", "target_w_claim"):
            joint_diff = joint[arm][metric].astype(float) - joint["identity"][metric]
            reco_diff = reco[arm][metric].astype(float) - reco["identity"][metric]
            result[arm][metric] = estimates.describe(joint_diff - reco_diff, use)
        flips = [(curve[arm]["lep_verdict"] != curve["identity"]["lep_verdict"]).astype(float)
                 for curve in (joint, reco)]
        result[arm]["lep_verdict_change"] = estimates.describe(
            flips[0] - flips[1], use & joint["identity"]["lep_pair"]
        )
        result[arm]["winner_swap"] = estimates.describe(
            (joint[arm]["pair_winner_target"] != joint["identity"]["pair_winner_target"]).astype(float)
            - (reco[arm]["pair_winner_target"] != reco["identity"]["pair_winner_target"]), use
        )
    return result


def run_model(name, checkpoint_dir, sample, donors, basic, geometry, batch_size, output):
    model, cfg, checkpoint = load_trained(checkpoint_dir, "cpu")
    if cfg["has_eventNumbers"] or cfg["validation_split_idx"] != 0 or cfg["n_splits"] != 2:
        raise ValueError("Expected even row positions held out by both checkpoints")
    if model.bottleneck_attention is not None:
        raise ValueError("Expected ordinary attention in both checkpoints")
    joint = name == "joint"
    print(f"{name}: {checkpoint}", flush=True)
    x, types, truth = (sample[k] for k in ("x", "types", "truth"))
    native = infer_single(model, x, types, batch_size, joint)
    baseline = event_summary(native, types, truth)
    result = {"checkpoint": checkpoint, "perfect_reconstruction": perfect_reconstruction(sample, baseline)}
    if joint:
        calibration = infer_single(model, donors["x"], donors["types"], batch_size, joint)
        cal = event_summary(calibration, donors["types"], donors["truth"])
        result["baseline_sb200"] = operating_points(donors, sample, cal["p_signal"],
                                                    baseline["p_signal"],
                                                    cal["channel"], baseline["channel"])
    else:
        result["event_head"] = "lambda_event=0, omitted"
    print(f"{name}: full baseline complete", flush=True)

    indices, target, alternative, matched, tag_common = basic
    paired_types, paired_truth = types[indices], truth[indices]
    paired_x = x[indices]
    clean = {key: values[indices] for key, values in native.items()}
    signatures = {"identity": event_summary(clean, paired_types, paired_truth, target, alternative)}
    jet_estimates = Estimates(sample, indices)
    pop = {}
    empty = np.ones(len(indices), dtype=bool)
    for arm in JET_ARMS:
        use = tag_common if arm.startswith("tag") else empty
        donor_tag = None
        if arm.startswith("tag"):
            original_tag = paired_x[np.arange(len(target)), target, 4]
            donor_tag = np.where(tag_common, matched[arm]["tag"], original_tag)
        edited = edit_jet(paired_x, target, arm, donor_tag)
        check_edits(paired_x, edited, target, arm)
        # Match the indices and batch boundaries for the numerical identity comparison.
        if arm == "identity":
            clean_repeated = infer_single(model, x.copy(), types, batch_size, joint)
            errors = {key: float(np.max(np.abs(value - clean_repeated[key])))
                      for key, value in native.items()}
            if any(value != 0 for value in errors.values()):
                raise RuntimeError(f"{name} native identity mismatch: {errors}")
            result["native_identity_max_abs_logit_error"] = errors
        else:
            changed = infer_single(model, edited, paired_types, batch_size, joint)
            signatures[arm] = event_summary(changed, paired_types, paired_truth, target, alternative)
        pop[arm] = comparisons(signatures[arm], signatures["identity"], jet_estimates, use, paired_types)
        print(f"{name}: {arm:12s} n={int(use.sum()):5d} "
              f"target W margin change={pop[arm]['target_w_margin']['all']['mean']:+.4f}", flush=True)
    result["jet"] = pop
    result["jet_baseline"] = {
        key: jet_estimates.describe(signatures["identity"][key])
        for key in ("target_w_margin", "other_w_margin", "pair_w_margin", "target_w_claim",
                    "other_w_claim", "both_w_claim", "pair_winner_target")
    }
    jet_arrays = ids(sample, indices)
    jet_arrays.update(target=target, alternative=alternative, input=paired_x, tag_common=tag_common)
    for arm, match in matched.items():
        jet_arrays.update({f"donor_{arm}_{key}": value for key, value in match.items()})
    save_raw(output / f"jet_{name}.npz", jet_arrays, signatures)
    print(f"{name}: saved paired jet results", flush=True)

    rows, geom_w, geom_h, signs = geometry
    g_x, g_types, g_truth = x[rows], types[rows], truth[rows]
    geom_estimates = Estimates(sample, rows)
    g_base = {key: value[rows] for key, value in native.items()}
    result["geometry"] = {}
    for label, direction in (("toward", -1), ("away", 1)):
        check_edits(g_x, rotate_jet(g_x, geom_w, ANGLE * direction * signs), geom_w, "geometry")
        outputs, parity = infer_geometry(model, g_x, g_types, geom_w, direction * signs, batch_size, joint)
        identity_error = {key: float(np.max(np.abs(value - outputs["identity_patch"][key])))
                          for key, value in outputs["geometry"].items()}
        batch_error = {key: float(np.max(np.abs(value - outputs["identity"][key])))
                       for key, value in g_base.items()}
        if any(value > 2e-4 for value in parity.values()) or any(identity_error.values()):
            raise RuntimeError(f"{name} geometry attention verification failed: {parity}, {identity_error}")
        if any(value > 2e-4 for value in batch_error.values()):
            raise RuntimeError(f"{name} geometry native verification failed: {batch_error}")
        g_signatures = {arm: event_summary(out, g_types, g_truth, geom_w, geom_h)
                        for arm, out in outputs.items()}
        detail = recovery(outputs, g_signatures, g_types, geom_estimates)
        result["geometry"][label] = {
            "attention_reconstruction_max_abs_per_block": parity,
            "identity_patch_max_abs_logit_error": identity_error,
            "clean_forward_max_abs_logit_error_with_new_batch": batch_error,
            "geometry_corruption": comparisons(g_signatures["geometry"], g_signatures["identity"],
                                                geom_estimates, np.ones(len(rows), dtype=bool), g_types),
            "patch_recovery": detail,
        }
        g_arrays = ids(sample, rows)
        g_arrays.update(target=geom_w, alternative=geom_h, azimuth_offset=ANGLE * direction * signs,
                        input=g_x)
        save_raw(output / f"geometry_{label}_{name}.npz", g_arrays, g_signatures)
        print(f"{name}: geometry {label} n={len(rows)} identity={identity_error} "
              f"b1h0 object-probability recovery="
              f"{detail['b1h0']['reco_probability_total_variation']['fraction_of_mean_error_recovered']:+.4f}",
              flush=True)
    (output / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return signatures


def run(args):
    torch.set_num_threads(args.threads)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    sample = sample_events(args.data_dir, 4, args.n_signal, args.n_background, args.seed)
    donors = sample_events(args.data_dir, 2, args.n_signal, args.n_background, args.seed)
    x, types = sample["x"], sample["types"]
    indices, target, other = select_jet_pair(x, types)
    if len(indices) == 0:
        raise ValueError("No eligible events for the jet interventions")
    matched, common = match_donors(x[indices, target], donors)
    if not common.any():
        raise ValueError("No events with all three tag donor matches")
    geometry = select_geometry_pair(x, types)
    g_indices, geom_w, geom_h, _ = geometry
    if len(g_indices) == 0:
        raise ValueError("No eligible events for the geometry interventions")
    print(f"Assessment n={len(x)}. Jet n={len(indices)}, common tag n={common.sum()}, "
          f"geometry n={len(g_indices)}.", flush=True)
    jet_stats = Estimates(sample, indices)
    geo_stats = Estimates(sample, g_indices)
    geo_x = x[g_indices]
    n = np.arange(len(g_indices))
    truth = sample["truth"][g_indices]
    geo_diag = {
        "h_proxy_truth_H": geo_stats.describe(truth[n, geom_h] == 1),
        "w_proxy_truth_hadronic_W": geo_stats.describe(truth[n, geom_w] == 2),
        "both_proxies_match_truth": geo_stats.describe((truth[n, geom_h] == 1) &
                                                       (truth[n, geom_w] == 2)),
    }
    for label, direction in (("toward", -1), ("away", 1)):
        def pair_distance(vectors):
            _, eta, phi, _ = kinematics(vectors)
            return np.hypot(eta[n, geom_w] - eta[n, geom_h],
                            wrap_phi(phi[n, geom_w] - phi[n, geom_h]))
        rotated = rotate_jet(geo_x, geom_w, ANGLE * direction * geometry[3])
        geo_diag[f"deltaR_change_{label}"] = geo_stats.describe(pair_distance(rotated) -
                                                              pair_distance(geo_x))
    report = {
        "run_parameters": vars(args),
        "split": {"evaluation_row_mod_8": 4, "donors_and_calibration_row_mod_8": 2,
                  "confirmation_row_mod_8_untouched": 6},
        "rules": {
            "jet": "At least two large-R jets with pT >= 200 GeV, |eta| <= 2 and mass in [50,125] GeV. Closest to 80.4 GeV is the alternative and second closest is the edited target. Tie by token index.",
            "mass": "Halfway or fully toward 80.4 GeV, with fixed px,py,pz and tag.",
            "pt": "Scales all three momentum components by 0.8 or 1.25, recomputes energy at fixed mass, and preserves direction and tag.",
            "matched_tag": "Nearest real large-R jet within unit ellipsoid in (log pT, eta, m) using scales (0.15, 0.30, 12 GeV), across distinct split. Increase or decrease the tag by 0.5 to 2.0, neutral difference <=0.05. Restrict all three tag arms to events with matches for all three.",
            "geometry": "Highest-tag pT>=200 GeV |eta|<=2 large-R jet with tag>=1 and 80–165 GeV mass is the H proxy. Remaining W proxy is the closest to 80.4 GeV with tag<=0.5, same pT and eta ranges and 50–120 GeV mass. Require nominal absolute interproxy delta phi in [1.1,2.45]. Rotate W proxy only in azimuth by 0.65 rad toward and away from H in distinct edits.",
            "patch": "Restore each head's clean pre-output-projection context at all query tokens, or all four heads at block 1. Identity patch restores the corrupted b1h0 context. Track clean-referenced absolute errors and paired error reductions.",
            "scores": "W margin = W logit minus maximum of H and none logits. Rank the two candidate jets by this margin. Leptonic verdict = joint W assignment to both the one lepton and one neutrino. Native signal score = 1 - P(background).",
            "statistics": "Paired event differences, including zeros. Approximate normal 95% intervals; signed-MC weighted intervals use within-DSID linearized ratio variances and omit singleton strata. Unadjusted intervals for descriptive comparisons. Errors and gains use the identical event for each head.",
        },
        "assessment_counts": {str(label): int((sample["label"] == label).sum()) for label in (0, 1, 2)},
        "jet_counts": jet_stats.describe(np.ones(len(indices))),
        "tag_common_counts": jet_stats.describe(np.ones(len(indices)), common),
        "geometry_counts": geo_stats.describe(np.ones(len(g_indices))),
        "geometry_diagnostics": geo_diag,
        "tag_donors": {},
    }
    for arm, matches in matched.items():
        report["tag_donors"][arm] = {
            "n_matched_before_common_restriction": int(np.isfinite(matches["tag"]).sum()),
            "tag_shift": jet_stats.describe(matches["tag"] - x[indices, target, 4], common),
            "mean_abs_delta_logpt_eta_massgev":
                np.abs(matches["differences"][common]).mean(0).tolist(),
        }
    report["jet_diagnostics"] = {
        "target_mass_gev": jet_stats.describe(kinematics(x[indices, target])[3]),
        "alternative_mass_gev": jet_stats.describe(kinematics(x[indices, other])[3]),
        "target_pt_gev": jet_stats.describe(kinematics(x[indices, target])[0]),
        "alternative_pt_gev": jet_stats.describe(kinematics(x[indices, other])[0]),
    }

    signatures = {}
    for name in ("joint", "reco"):
        signatures[name] = run_model(name, getattr(args, name), sample, donors,
                                     (indices, target, other, matched, common), geometry,
                                     args.batch_size, output)
    report["paired_joint_minus_reconstruction"] = paired_model_comparisons(signatures, common, jet_stats)
    (output / "paired.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Complete: {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("data-dir", "joint", "reco"):
        parser.add_argument(f"--{path}", required=True)
    parser.add_argument("--out", default="output/physics_interp/interventions")
    parser.add_argument("--n-signal", type=int, default=40_000)
    parser.add_argument("--n-background", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
