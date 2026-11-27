"""Fit one sparse kinematic scalar and replace that wire in a joint student."""
import numpy as np
import torch
from sklearn.linear_model import OrthogonalMatchingPursuit

from physics_samples import operating_points, sample_events, yield_summary
from physics_students import (PAD, SEEDS, hard_losses, performance, predict, probabilities,
                              save_json, save_predictions, student, tensors)
from utils.utils import check_category

TOPOLOGIES = ("H resolved / W resolved", "H resolved / W leptonic", "H resolved / W merged",
              "H merged / W resolved", "H merged / W leptonic", "H merged / W merged")
JET_TYPES = (3, 4)
MASS_DOSES = (40, 60, 80.4, 105, 130, 150)
TAG_DOSES = (0.25, 0.5, 0.75, 1.0)


def score_reconstruction(probs, sample):
    types = sample["types"]
    reco = probs["reco"].argmax(-1)
    category = check_category(types, reco, PAD)
    channel = np.where(category == -1, 0, np.where(np.isin(category, (1, 4)), 1, 2))
    return (1 - probs["event"][:, 0]).astype(float), channel


def fit_points(cal, test, cal_probs, test_probs):
    cal_score, cal_channel = score_reconstruction(cal_probs, cal)
    test_score, test_channel = score_reconstruction(test_probs, test)
    try:
        return operating_points(cal, test, cal_score, test_score, cal_channel, test_channel)
    except ValueError:
        # Keep a populated channel even when the model never reconstructs the other on calibration.
        result = {}
        for channel, name in ((1, "lvbb"), (2, "qqbb")):
            selected = (cal["label"] == 0) & (cal_channel == channel)
            if not selected.any():
                result[name] = {"reason": "no valid calibration background", "n": 0}
                continue
            other = 3 - channel
            padding = {key: np.concatenate((vals, vals[:1])) for key, vals in cal.items()}
            padding["label"][-1] = 0
            padding["weight"][-1] = 0
            value = operating_points(padding, test, np.r_[cal_score, 0.],
                                     test_score, np.r_[cal_channel, other], test_channel)
            result[name] = value[name]
        return result


def evaluate_saved(args, samples):
    scores = {}
    saved = {}
    paths = sorted(args.out.glob("*_r2.npz"))
    for path in paths:
        name = path.stem.removesuffix("_r2")
        probs = {}
        for residue in (2, 4, 6):
            with np.load(args.out / f"{name}_r{residue}.npz") as data:
                for column in ("dsid", "row"):
                    np.testing.assert_array_equal(samples[residue][column], data[column])
                probs[residue] = {head: data[f"{head}_prob"] for head in ("reco", "event")}
        saved[name] = probs
        scores[name] = {}
        for residue in (4, 6):
            sample = samples[residue]
            types, truth, label, weights = (sample[key] for key in ("types", "truth", "label", "weight"))
            signal = label != 0
            topology = check_category(types, truth, PAD)
            reco = probs[residue]["reco"].argmax(-1)
            correct = ((reco == truth.clip(0, 2)) | (types == PAD)).all(1)
            _, reconstructed_channel = score_reconstruction(probs[residue], sample)
            scores[name][residue] = {
                **performance(probs[residue], sample),
                "valid_reco_count_fraction": float((reconstructed_channel != 0).mean()),
                "topologies": {name: {"all": yield_summary(weights[signal & (topology == index)]),
                                     "perfect": yield_summary(weights[signal & (topology == index) & correct])}
                               for index, name in enumerate(TOPOLOGIES)},
                "other_truth_topology": yield_summary(weights[signal & (topology == -1)]),
                "sb200": fit_points(samples[2], sample, probs[2], probs[residue]),
            }
    save_json(args.out / "evaluation.json", scores)
    differences = {}
    for seed in SEEDS:
        hard, soft = saved[f"{seed}_labels"], saved[f"{seed}_soft_and_labels"]
        differences[seed] = {}
        for residue in (4, 6):
            sample = samples[residue]
            signal = sample["label"] != 0
            weights = sample["weight"][signal]
            truth = sample["truth"][signal].clip(0, 2)
            pad = sample["types"][signal] == PAD
            hard_correct = ((hard[residue]["reco"][signal].argmax(-1) == truth) | pad).all(1)
            soft_correct = ((soft[residue]["reco"][signal].argmax(-1) == truth) | pad).all(1)
            changes = soft_correct.astype(int) - hard_correct.astype(int)
            delta = float(np.average(changes, weights=weights))
            # Independent-event weighted ratio error, with the predictions paired by (DSID,row).
            terms = weights * (changes - delta)
            hard_values, soft_values = (scores[f"{seed}_{arm}"][residue] for arm in ("labels", "soft_and_labels"))
            differences[seed][residue] = {
                "soft_minus_hard_perfect_signed": delta,
                "approx_mc_se_of_paired_difference": float(np.linalg.norm(terms) / abs(weights.sum())),
                "gain_count": int((changes > 0).sum()),
                "loss_count": int((changes < 0).sum()),
                "soft_minus_hard_auc_abs_yield_weighted": soft_values["auc_abs_yield_weighted"] -
                hard_values["auc_abs_yield_weighted"],
                "soft_minus_hard_event_accuracy_count_fraction": soft_values["event_accuracy_count_fraction"] -
                hard_values["event_accuracy_count_fraction"],
            }
    save_json(args.out / "paired_differences.json", differences)


def kinematics(x):
    momentum = x[..., :3].astype(np.float64)
    pt = np.hypot(momentum[..., 0], momentum[..., 1])
    mass = np.sqrt(np.maximum(x[..., 3].astype(np.float64) ** 2 - (momentum ** 2).sum(-1), 0))
    eta = np.arcsinh(momentum[..., 2] / np.maximum(pt, 1e-6))
    phi = np.arctan2(momentum[..., 1], momentum[..., 0])
    return pt, mass, eta, phi


def physics_basis(x, types):
    """Fixed dimensionless kinematic basis. All values come from inference inputs."""
    pt, mass, eta, phi = kinematics(x)
    pad = types == PAD
    signal_jet = np.isin(types, JET_TYPES)
    jet_tag = np.where(signal_jet, x[..., 4], 0)
    total_pt = np.where(pad, 0, pt).sum(1, keepdims=True).clip(1e-6)
    large = types == 3
    small = types == 4
    lep = types <= 2
    w_like = np.exp(-0.5 * ((mass - .804) / .28) ** 2) * large
    h_like = np.exp(-0.5 * ((mass - 1.25) / .35) ** 2) * large
    delta_phi = phi[:, :, None] - phi[:, None, :]
    delta_eta = eta[:, :, None] - eta[:, None, :]
    dr = np.sqrt(delta_eta ** 2 + (np.arctan2(np.sin(delta_phi), np.cos(delta_phi))) ** 2)
    proximity = np.exp(-0.5 * dr ** 2) * (~np.eye(types.shape[1], dtype=bool))[None, :, :]
    shape = pt.shape
    basis = {
        "log(1+pT)": np.log1p(pt),
        "pT/HT": pt / total_pt,
        "eta": eta,
        "abs(eta)": np.abs(eta),
        "sin(phi)": np.sin(phi),
        "cos(phi)": np.cos(phi),
        "jet_tag": jet_tag,
        "large_jet_mass": mass * large,
        "large_jet_mass_squared": mass ** 2 * large,
        "large_jet_W_mass_kernel": w_like,
        "large_jet_H_mass_kernel": h_like,
        "large_jet_W_mass_kernel*tag": w_like * jet_tag,
        "large_jet_H_mass_kernel*tag": h_like * jet_tag,
        "weighted_nearby_lepton_pT/HT": (proximity @ np.where(lep, pt, 0)[..., None]).squeeze(-1) / total_pt,
        "weighted_nearby_large_jet_pT/HT": (proximity @ np.where(large, pt, 0)[..., None]).squeeze(-1) / total_pt,
        "weighted_nearby_small_jet_pT/HT": (proximity @ np.where(small, pt, 0)[..., None]).squeeze(-1) / total_pt,
        "weighted_nearby_W_mass_kernel": (proximity @ w_like[..., None]).squeeze(-1),
        "weighted_nearby_H_mass_kernel": (proximity @ h_like[..., None]).squeeze(-1),
        "weighted_nearby_tag": (proximity @ jet_tag[..., None]).squeeze(-1),
        "log(1+HT)": np.broadcast_to(np.log1p(total_pt), shape),
        "leptonic_pT/HT": np.broadcast_to(np.where(lep, pt, 0).sum(1, keepdims=True) / total_pt, shape),
        "large_jet_pT/HT": np.broadcast_to(np.where(large, pt, 0).sum(1, keepdims=True) / total_pt, shape),
        "small_jet_pT/HT": np.broadcast_to(np.where(small, pt, 0).sum(1, keepdims=True) / total_pt, shape),
        "n_large_jets": np.broadcast_to(large.sum(1, keepdims=True), shape),
        "n_small_jets": np.broadcast_to(small.sum(1, keepdims=True), shape),
        "sum_large_jet_W_mass_kernel": np.broadcast_to(w_like.sum(1, keepdims=True), shape),
        "sum_large_jet_H_mass_kernel": np.broadcast_to(h_like.sum(1, keepdims=True), shape),
        "sum_large_jet_tags": np.broadcast_to((large * jet_tag).sum(1, keepdims=True), shape),
        "sum_small_jet_tags": np.broadcast_to((small * jet_tag).sum(1, keepdims=True), shape),
        "max_large_jet_tag": np.broadcast_to(np.where(large, jet_tag, -np.inf).max(1, keepdims=True), shape),
    }
    basis["max_large_jet_tag"] = np.nan_to_num(basis["max_large_jet_tag"], neginf=0.)
    raw = tuple(basis.items())
    for type_label, mask in (("large_jet", large), ("small_jet", small), ("lepton_or_met", lep)):
        basis[type_label] = mask
        for name, values in raw:
            basis[f"{type_label}*{name}"] = mask * values
    return list(basis), np.stack(list(basis.values()), axis=-1).astype(np.float32)


def selection_model(out, key, device):
    model = student(device).eval()
    model.load_state_dict(torch.load(out / f"{key}.pt", map_location=device, weights_only=True))
    return model


@torch.no_grad()
def run_model(model, data, block_idx=None, head_idx=None, replacement=None, capture=False):
    handles = []
    collected = []
    for index, block in enumerate(model.attention_blocks):
        if capture:
            collected.append([])

            def collect(module, inputs, values, position=index):
                collected[position].append(values.detach().cpu().numpy())

            handles.append(block["scalar_messages"].register_forward_hook(collect))
    if replacement is not None:
        samples = 0

        def substitute(module, inputs, values):
            nonlocal samples
            edited = values.clone()
            if np.ndim(replacement):
                edited[..., head_idx] = torch.as_tensor(replacement[samples:samples + len(values)],
                                                        device=values.device, dtype=values.dtype)
            else:
                edited[..., head_idx] = float(replacement)
            samples += len(values)
            return edited

        handles.append(model.attention_blocks[block_idx]["scalar_messages"].register_forward_hook(substitute))
    try:
        predictions = predict(model, data)
    finally:
        for handle in handles:
            handle.remove()
    activations = np.stack([np.concatenate(parts) for parts in collected], axis=0) if capture else None
    return predictions, activations


def slice_sample(sample, index):
    return {name: value[index] for name, value in sample.items()}


def select_head(model, development, device):
    data = tensors(development, device)
    original, messages = run_model(model, data, capture=True)
    baseline = np.array([float(value) for value in hard_losses(original, data)])
    measurements = []
    for block, layers in enumerate(messages):
        for head in range(layers.shape[-1]):
            scalar = layers[..., head]
            average = float(scalar[development["types"] != PAD].mean())
            output, _ = run_model(model, data, block, head, average)
            difference = np.array([float(value) for value in hard_losses(output, data)]) - baseline
            measurements.append({"block": block, "head": head, "mean": average,
                                 "reco_ce_change": float(difference[0]), "event_ce_change": float(difference[1]),
                                 "sum_ce_change": float(difference.sum())})
    return max(measurements, key=lambda row: row["sum_ce_change"]), measurements


def fit_formula(model, development, route, device):
    mask = development["types"] != PAD
    block, head = route["block"], route["head"]
    _, messages = run_model(model, tensors(development, device), capture=True)
    target = messages[block, ..., head][mask]
    names, basis = physics_basis(development["x"], development["types"])
    data = basis[mask].astype(float)
    mean, std = data.mean(0), data.std(0)
    active = std > 1e-6
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=12).fit((data[:, active] - mean[active]) / std[active], target)
    coefficients = np.zeros(len(names))
    coefficients[active] = model.coef_ / std[active]
    intercept = float(model.intercept_ - mean @ coefficients)
    terms = {name: float(value) for name, value in zip(names, coefficients) if value}
    return {"intercept": intercept, "terms": terms, "max_terms": 12}


def formula_value(formula, sample):
    chunks = []
    coefficients = np.array(list(formula["terms"].values()))
    for start in range(0, len(sample["x"]), 512):
        batch = slice(start, start + 512)
        names, basis = physics_basis(sample["x"][batch], sample["types"][batch])
        indices = [names.index(name) for name in formula["terms"]]
        chunks.append(formula["intercept"] + basis[..., indices] @ coefficients)
    return np.concatenate(chunks)


def measure_formula(model, route, formula, sample, device):
    block, head = route["block"], route["head"]
    inputs = tensors(sample, device)
    original, messages = run_model(model, inputs, capture=True)
    value = formula_value(formula, sample)
    substituted, _ = run_model(model, inputs, block, head, value)
    mean_only, _ = run_model(model, inputs, block, head, route["mean"])
    identity, _ = run_model(model, inputs, block, head, messages[block, ..., head])
    keep = sample["types"] != PAD
    target = messages[block, ..., head][keep]
    residual = value[keep] - target
    comparison = {"r2": 1 - float(np.mean(residual ** 2) / target.var()),
                  "scalar_rmse": float(np.sqrt(np.mean(residual ** 2))),
                  "scalar_std": float(target.std()),
                  "identity_max_abs_logit_delta": {key: float((original[key] - identity[key]).abs().max())
                                                    for key in original}}
    outputs = {}
    for name, output in (("intact", original), ("formula", substituted), ("mean_only", mean_only)):
        probs = probabilities(output)
        outputs[name] = probs
        matches = (output["reco"].argmax(-1) == original["reco"].argmax(-1)).cpu().numpy()
        comparison[name] = {**performance(probs, sample),
                            "intact_event_argmax_agreement": float((output["event"].argmax(-1) ==
                                                                   original["event"].argmax(-1)).float().mean()),
                            "intact_full_reco_agreement": float((matches | ~keep).all(1).mean()),
                            "intact_object_argmax_agreement": float(matches[keep].mean())}
    return comparison, outputs


def probe_population(sample, limit=384):
    pt, mass, eta, _ = kinematics(sample["x"])
    large = sample["types"] == 3
    plausible = large & (pt >= 1.5) & (mass >= .4) & (mass <= 1.5)
    plot_rows = np.flatnonzero(plausible.sum(1) >= 2)
    # Input-only; choose the W-mass-nearest jet among the two candidates with masses in range.
    slot = np.where(plausible, np.abs(mass - .804), np.inf).argmin(1)
    all_rows, all_slots = np.nonzero(large & np.isfinite(sample["x"][..., 4]))
    donor_tag = sample["x"][all_rows, all_slots, 4]
    if not len(donor_tag):
        raise ValueError("No finite large-jet donors")
    qlo, qhi = np.quantile(donor_tag, [.25, .75])
    # Take the stable first events after the sampler's seeded permutation.
    plot_rows = plot_rows[:limit]
    targets = slot[plot_rows]
    target_tag = sample["x"][plot_rows, targets, 4]
    distance = (
        ((np.log(pt[plot_rows, targets, None]) - np.log(np.maximum(pt[all_rows, all_slots], 1e-6))) / .25) ** 2
        + ((eta[plot_rows, targets, None] - eta[all_rows, all_slots]) / .5) ** 2
        + ((mass[plot_rows, targets, None] - mass[all_rows, all_slots]) / .25) ** 2
    )
    eligible = (plot_rows[:, None] != all_rows) & (distance <= 9.)
    donors = {}
    for sign, donor_mask in (("low_tag", donor_tag <= qlo), ("high_tag", donor_tag >= qhi)):
        delta = donor_tag - target_tag[:, None]
        direction = delta < -1e-5 if sign == "low_tag" else delta > 1e-5
        matching = np.where(eligible & donor_mask & direction, distance, np.inf)
        donor = matching.argmin(1)
        match = np.isfinite(matching[np.arange(len(donor)), donor])
        donors[sign] = {"match": match, "tag": donor_tag[donor],
                        "dsid": sample["dsid"][all_rows[donor]], "row": sample["row"][all_rows[donor]],
                        "slot": all_slots[donor], "distance": matching[np.arange(len(donor)), donor]}
    return plot_rows, targets, donors


def summary_pair(intact, formula):
    effect_error = formula - intact
    correlation = np.corrcoef(intact, formula)[0, 1] if intact.std() and formula.std() else None
    changed = np.abs(intact) > 1e-6
    return {"intact_mean": float(intact.mean()), "formula_mean": float(formula.mean()) ,
            "intact_mean_abs": float(np.abs(intact).mean()), "formula_mean_abs": float(np.abs(formula).mean()),
            "paired_mae": float(np.abs(effect_error).mean()),
            "paired_rmse": float(np.sqrt(np.mean(effect_error ** 2))),
            "paired_correlation": float(correlation) if correlation is not None else None,
            "changed": int(changed.sum()),
            "sign_agreement_on_changed": float((np.sign(intact[changed]) == np.sign(formula[changed])).mean())
            if changed.any() else None}


def response(model, route, formula, sample, device, artifact):
    rows, slots, donors = probe_population(sample)
    probe = slice_sample(sample, rows)
    index = np.arange(len(rows))
    block, head = route["block"], route["head"]
    stored = {"dsid": probe["dsid"], "row": probe["row"], "slot": slots,
              "types": probe["types"], "truth": probe["truth"], "label": probe["label"],
              "weight": probe["weight"], "input_target_p4_and_tag": probe["x"][index, slots]}
    for name, measurements in donors.items():
        for key, data in measurements.items():
            stored[f"{name}_donor_{key}"] = data

    def measure(data, indices, candidate_slots):
        tensor_data = tensors(data, device)
        logits, messages = run_model(model, tensor_data, capture=True)
        target_scalar = messages[block, ..., head]
        formula_scalar = formula_value(formula, data)
        substituted, _ = run_model(model, tensor_data, block, head, formula_scalar)
        base = probabilities(logits)
        new = probabilities(substituted)
        mean_only = probabilities(run_model(model, tensor_data, block, head, route["mean"])[0])
        views = {}
        for mode, probabilities_, scalar in (
            ("intact", base, target_scalar), ("formula", new, formula_scalar),
            ("mean_only", mean_only, np.full_like(target_scalar, route["mean"])),
        ):
            views[mode] = {
                "scalar": scalar[indices, candidate_slots],
                "scalar_all_tokens": scalar,
                "reco": probabilities_["reco"],
                "event": probabilities_["event"],
            }
        return views

    def record(name, metrics, indices, baseline):
        chosen = slots[indices]
        result = {"n": len(indices)}
        values = {mode: {"scalar": views["scalar"] - baseline[mode]["scalar"][indices],
                         "scalar_all_tokens": views["scalar_all_tokens"] - baseline[mode]["scalar_all_tokens"][indices],
                         "reco": views["reco"] - baseline[mode]["reco"][indices],
                         "event": views["event"] - baseline[mode]["event"][indices]}
                  for mode, views in metrics.items()}
        for mode, view in metrics.items():
            for key, array in view.items():
                stored[f"{name}_{mode}_{key}"] = array.astype(np.float32)
        if not len(indices):
            return result
        selected = np.arange(len(indices))
        result["scalar_delta"] = summary_pair(values["intact"]["scalar"], values["formula"]["scalar"])
        types = probe["types"][indices]
        result["scalar_deltas_by_token_type"] = {}
        for kind, mask in (("lepton_or_met", types <= 2), ("large_jet", types == 3), ("small_jet", types == 4)):
            if mask.any():
                result["scalar_deltas_by_token_type"][kind] = summary_pair(
                    values["intact"]["scalar_all_tokens"][mask], values["formula"]["scalar_all_tokens"][mask])
        result["target_object_class_deltas"] = {
            key: summary_pair(values["intact"]["reco"][selected, chosen, i],
                              values["formula"]["reco"][selected, chosen, i])
            for i, key in enumerate(("none", "H", "W"))
        }
        result["event_class_deltas"] = {key: summary_pair(values["intact"]["event"][:, i], values["formula"]["event"][:, i])
                                       for i, key in enumerate(("background", "lvbb", "qqbb"))}
        result["mean_only_target_W_probability_delta"] = summary_pair(
            values["intact"]["reco"][selected, chosen, 2], values["mean_only"]["reco"][selected, chosen, 2])
        intact = metrics["intact"]["reco"].argmax(-1)
        replaced = metrics["formula"]["reco"].argmax(-1)
        intact_before = baseline["intact"]["reco"][indices].argmax(-1)
        replaced_before = baseline["formula"]["reco"][indices].argmax(-1)
        changed_intact = ((intact != intact_before) & (probe["types"][indices] != PAD)).any(1)
        changed_formula = ((replaced != replaced_before) & (probe["types"][indices] != PAD)).any(1)
        true = probe["truth"][indices].clip(0, 2)
        signal = probe["label"][indices] != 0
        correct_intact = ((intact == true) | (probe["types"][indices] == PAD)).all(1)
        correct_formula = ((replaced == true) | (probe["types"][indices] == PAD)).all(1)
        event_changes = {mode: metrics[mode]["event"].argmax(1) != baseline[mode]["event"][indices].argmax(1)
                         for mode in ("intact", "formula")}
        result.update(reco_truth_intact=float(correct_intact.mean()),
                      reco_truth_formula=float(correct_formula.mean()),
                      signal_n=int(signal.sum()),
                      signal_reco_truth_intact=float(correct_intact[signal].mean()) if signal.any() else None,
                      signal_reco_truth_formula=float(correct_formula[signal].mean()) if signal.any() else None,
                      event_truth_intact=float((metrics["intact"]["event"].argmax(1) == probe["label"][indices]).mean()),
                      event_truth_formula=float((metrics["formula"]["event"].argmax(1) == probe["label"][indices]).mean()),
                      changed_any_label_intact=int(changed_intact.sum()),
                      changed_any_label_formula=int(changed_formula.sum()),
                      changed_any_label_agreement=float((changed_intact == changed_formula).mean()),
                      changed_event_label_intact=int(event_changes["intact"].sum()),
                      changed_event_label_formula=int(event_changes["formula"].sum()),
                      changed_event_label_agreement=float((event_changes["intact"] == event_changes["formula"]).mean()))
        return result

    results = {"selection": "at least two large-R jets with pT >= 150 GeV and 40 <= mass <= 150 GeV; edit closest to 80.4 GeV",
               "kinematic_units": "four-momentum divided by 100 GeV",
               "donor_match": "same large-R type from another held-out collision; lower/upper tag quartile and distance^2 <= 9 using log(pT)/.25, eta/.5 and mass/.25",
               "n": len(rows), "responses": {}}
    if not len(rows):
        np.savez_compressed(artifact, **stored)
        return results
    baseline = measure(probe, index, slots)
    identity = measure(dict(probe, x=probe["x"].copy()), index, slots)
    results["responses"]["identity"] = record("identity", identity, index, baseline)
    for mass in MASS_DOSES:
        data = dict(probe, x=probe["x"].copy())
        p4 = data["x"][index, slots, :3].astype(float)
        data["x"][index, slots, 3] = np.sqrt((p4 ** 2).sum(1) + (mass / 100.) ** 2)
        name = f"mass_{mass}"
        results["responses"][name] = record(name, measure(data, index, slots), index, baseline)
    for kind, donor in donors.items():
        matched = np.flatnonzero(donor["match"])
        stored[f"{kind}_probe_index"] = matched
        if not len(matched):
            results["responses"][kind] = {"n": 0}
            continue
        for dose in TAG_DOSES:
            data = slice_sample(probe, matched)
            selected = np.arange(len(matched))
            tag = data["x"][selected, slots[matched], 4]
            data["x"][selected, slots[matched], 4] = tag + dose * (donor["tag"][matched] - tag)
            name = f"{kind}_{dose}"
            results["responses"][name] = record(name, measure(data, selected, slots[matched]), matched, baseline)
    np.savez_compressed(artifact, **stored)
    return results


def run_formulas(args, training_record, device):
    config = training_record["config"]
    samples = {residue: sample_events(args.data, residue=residue, n_signal=config["eval_signal"],
                                      n_background=config["eval_background"]) for residue in (2, 4, 6)}
    evaluate_saved(args, samples)
    development = samples[2]
    screening = slice_sample(development, slice(None, 1072))
    fit_data = slice_sample(development, slice(1072, 4144))
    candidates = {key: record for key, record in training_record["models"].items() if record["seed"] != SEEDS[-1]}
    key = min(candidates, key=lambda key: candidates[key]["curve"][-1]["validation_hard_reco_ce"] +
              candidates[key]["curve"][-1]["validation_hard_event_ce"])
    confirmation_key = f"{SEEDS[-1]}_{candidates[key]['arm']}"
    distilled_key = f"{candidates[key]['seed']}_soft_and_labels"
    distilled_confirmation_key = f"{SEEDS[-1]}_soft_and_labels"
    record = {
        "student_selection": "min sum of hard validation reco and event CE over discovery seeds and both arms",
        "head_selection": "greatest increase in sum of hard reco and event CE under scalar replacement by its mean, on first 1072 dev events",
        "formula_fit": "orthogonal matching pursuit with at most 12 terms, on next 3072 dev events, no intervention or truth in fit",
        "confirmation": "discovery model and formula also tested on residue 6; same recipe applied to unseen seed, independently fitted on residue 2 and tested on residue 6",
        "distilled_followup": "apply fixed formula recipe to soft-target arms of the same discovery and confirmation seeds",
        "students": {},
    }
    selected = {key: (4, 6), confirmation_key: (6,), distilled_key: (4, 6), distilled_confirmation_key: (6,)}
    for model_key, test_residues in selected.items():
        model = selection_model(args.out, model_key, device)
        route, screening_results = select_head(model, screening, device)
        formula = fit_formula(model, fit_data, route, device)
        link = {"route": route, "screening_all_heads": screening_results, "formula": formula, "splits": {}}
        record["students"][model_key] = link
        for residue in test_residues:
            # Scoring fractions use full samples; limit the batched feature basis footprint.
            sample = samples[residue]
            cal = slice_sample(development, slice(4144, 8144))
            metrics, predictions = measure_formula(model, route, formula, sample, device)
            _, cal_predictions = measure_formula(model, route, formula, cal, device)
            for mode, probs in predictions.items():
                save_predictions(args.out / f"formula_{model_key}_{mode}_r{residue}.npz", sample, probs)
                metrics[mode]["sb200"] = fit_points(cal, sample, cal_predictions[mode], probs)
            metrics["probes"] = response(model, route, formula, sample, device,
                                         args.out / f"probes_{model_key}_r{residue}.npz")
            link["splits"][residue] = metrics
            save_json(args.out / "formula.json", record)
            print(f"{model_key} r{residue} b{route['block']}h{route['head']} formula r2={metrics['r2']:.3f} "
                  f"n_probes={metrics['probes']['n']}", flush=True)
