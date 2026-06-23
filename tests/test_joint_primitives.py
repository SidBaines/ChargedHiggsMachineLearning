import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from joint_primitives import (
    event_score_from_object_logits,
    event_summary_features,
    fit_fair_readout,
    weighted_roc_auc,
    asimov_z,
    best_asimov_z,
)

import math
import traceback

import numpy as np
import torch


def test_event_score_high_signal_none_and_shape():
    padding_token = -1
    reco_logits = torch.tensor(
        [
            [[-30.0, 30.0, -30.0], [30.0, -30.0, -30.0], [-30.0, 30.0, -30.0]],
            [[30.0, -30.0, -30.0], [30.0, -30.0, -30.0], [-30.0, 30.0, -30.0]],
        ],
        dtype=torch.float64,
    )
    object_types = torch.tensor(
        [
            [0, padding_token, padding_token],
            [0, 1, padding_token],
        ],
        dtype=torch.long,
    )

    for method in ("max", "softor"):
        scores = event_score_from_object_logits(reco_logits, object_types, padding_token, method=method)
        assert scores.shape == (2,)
        assert torch.allclose(scores[0], torch.tensor(1.0, dtype=torch.float64), atol=1.0e-10)
        assert torch.allclose(scores[1], torch.tensor(0.0, dtype=torch.float64), atol=1.0e-10)


def test_event_score_padding_slots_are_ignored():
    padding_token = 99
    base_logits = torch.tensor([[[0.0, 0.0, -40.0]]], dtype=torch.float64)
    base_types = torch.tensor([[1]], dtype=torch.long)
    padded_logits = torch.tensor(
        [[[0.0, 0.0, -40.0], [-40.0, 40.0, -40.0], [-40.0, -40.0, 40.0]]],
        dtype=torch.float64,
    )
    padded_types = torch.tensor([[1, padding_token, padding_token]], dtype=torch.long)

    for method in ("max", "softor"):
        base_score = event_score_from_object_logits(base_logits, base_types, padding_token, method=method)
        padded_score = event_score_from_object_logits(padded_logits, padded_types, padding_token, method=method)
        assert torch.allclose(base_score, padded_score, atol=1.0e-12)


def test_event_score_softor_ge_max_for_moderate_objects():
    padding_token = -1
    reco_logits = torch.tensor([[[0.0, 0.0, -40.0], [0.0, 0.0, -40.0]]], dtype=torch.float64)
    object_types = torch.tensor([[1, 1]], dtype=torch.long)

    max_score = event_score_from_object_logits(reco_logits, object_types, padding_token, method="max")
    softor_score = event_score_from_object_logits(reco_logits, object_types, padding_token, method="softor")

    assert torch.allclose(max_score, torch.tensor([0.5], dtype=torch.float64), atol=1.0e-12)
    assert torch.allclose(softor_score, torch.tensor([0.75], dtype=torch.float64), atol=1.0e-12)
    assert torch.all(softor_score >= max_score)


def test_event_score_all_padding_is_zero():
    padding_token = -1
    reco_logits = torch.tensor([[[-30.0, 30.0, -30.0], [-30.0, -30.0, 30.0]]], dtype=torch.float64)
    object_types = torch.tensor([[padding_token, padding_token]], dtype=torch.long)

    for method in ("max", "softor"):
        scores = event_score_from_object_logits(reco_logits, object_types, padding_token, method=method)
        assert torch.allclose(scores, torch.zeros(1, dtype=torch.float64), atol=0.0)


def test_event_summary_features_shape_and_padding():
    padding_token = 5
    reco_logits = torch.tensor(
        [
            [[-10.0, 10.0, -10.0], [10.0, -10.0, -10.0]],
            [[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]],
        ],
        dtype=torch.float64,
    )
    object_types = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
    features = event_summary_features(reco_logits, object_types, padding_token)

    assert features.shape == (2, 8)
    assert features[0, 0] > 0.999
    assert features[0, 3] > 0.999
    assert features[0, 1] < 1.0e-6
    assert features[0, 4] < 1.0e-6

    padding_logits = torch.tensor(
        [
            [[-100.0, 100.0, -100.0], [-100.0, -100.0, 100.0]],
            [[100.0, -100.0, -100.0], [-100.0, 100.0, -100.0]],
        ],
        dtype=torch.float64,
    )
    padded_logits = torch.cat([reco_logits, padding_logits], dim=1)
    padded_types = torch.cat(
        [object_types, torch.full((2, 2), padding_token, dtype=torch.long)],
        dim=1,
    )
    padded_features = event_summary_features(padded_logits, padded_types, padding_token)

    assert torch.allclose(features, padded_features, atol=1.0e-12)
    softor = event_score_from_object_logits(reco_logits, object_types, padding_token, method="softor")
    assert torch.allclose(features[:, 7], softor, atol=1.0e-12)

    all_padding_features = event_summary_features(
        padding_logits[:1],
        torch.full((1, 2), padding_token, dtype=torch.long),
        padding_token,
    )
    assert torch.allclose(all_padding_features, torch.zeros((1, 8), dtype=torch.float64), atol=0.0)


def test_fair_readout_beats_single_feature_when_combo_helps():
    train_features = np.array(
        [[0.9, 0.9]] * 40
        + [[0.9, 0.1]] * 40
        + [[0.1, 0.9]] * 40,
        dtype=float,
    )
    train_labels = np.array([1] * 40 + [0] * 80)
    train_weights = np.ones_like(train_labels, dtype=float)
    train_weights[::7] = -2.0

    test_features = np.array(
        [[0.8, 0.8]] * 20
        + [[0.8, 0.2]] * 20
        + [[0.2, 0.8]] * 20,
        dtype=float,
    )
    test_labels = np.array([1] * 20 + [0] * 40)
    test_weights = np.ones_like(test_labels, dtype=float)

    readout = fit_fair_readout(train_features, train_labels, train_weights)
    assert readout is not None

    fair_auc = weighted_roc_auc(readout.score(test_features), test_labels, test_weights)
    best_single_auc = max(
        weighted_roc_auc(test_features[:, 0], test_labels, test_weights),
        weighted_roc_auc(test_features[:, 1], test_labels, test_weights),
    )

    assert math.isclose(best_single_auc, 0.75, rel_tol=0.0, abs_tol=1.0e-12)
    assert fair_auc > best_single_auc + 0.20
    assert fit_fair_readout(train_features, np.ones_like(train_labels)) is None


def test_weighted_roc_auc_perfectly_separated():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert weighted_roc_auc(scores, labels) == 1.0


def test_weighted_roc_auc_perfectly_antiseparated():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [0, 0, 1, 1]
    assert weighted_roc_auc(scores, labels) == 0.0


def test_weighted_roc_auc_single_class_is_nan():
    scores = torch.tensor([0.1, 0.2, 0.3])
    labels = torch.tensor([1, 1, 1])
    assert math.isnan(weighted_roc_auc(scores, labels))


def test_weighted_roc_auc_sample_weights_change_tied_result():
    scores = np.array([0.9, 0.4, 0.4, 0.1])
    labels = np.array([1, 1, 0, 0])
    weights = np.array([1.0, 10.0, 1.0, 1.0])

    unweighted_auc = weighted_roc_auc(scores, labels)
    weighted_auc = weighted_roc_auc(scores, labels, weights)

    assert math.isclose(unweighted_auc, 0.875, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(weighted_auc, 17.0 / 22.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert not math.isclose(unweighted_auc, weighted_auc, rel_tol=0.0, abs_tol=1.0e-12)


def test_asimov_z_small_signal_limit():
    assert math.isclose(asimov_z(1.0, 10000.0), 1.0 / 100.0, rel_tol=0.01)


def test_asimov_z_equal_signal_background():
    expected = math.sqrt(2.0 * 100.0 * (2.0 * math.log(2.0) - 1.0))
    assert math.isclose(asimov_z(100.0, 100.0), expected, rel_tol=0.01)


def test_asimov_z_monotonic_in_signal_for_fixed_background():
    values = asimov_z(np.array([1.0, 2.0, 5.0]), 10.0)
    assert np.all(np.diff(values) > 0.0)


def test_asimov_z_nonpositive_background_is_finite():
    values = asimov_z(np.array([1.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))
    assert np.all(np.isfinite(values))
    assert values[0] > 0.0
    assert values[1] == 0.0
    assert values[2] == 0.0


def test_best_asimov_z_perfect_separation():
    scores = torch.tensor([0.95, 0.90, 0.20, 0.10])
    labels = torch.tensor([1, 1, 0, 0])
    weights = torch.tensor([2.0, 3.0, 10.0, 10.0])

    best_z, best_threshold = best_asimov_z(scores, labels, weights)

    assert isinstance(best_z, float)
    assert isinstance(best_threshold, float)
    assert math.isfinite(best_z)
    assert math.isclose(best_threshold, 0.90, rel_tol=0.0, abs_tol=1.0e-6)
    assert math.isclose(best_z, asimov_z(5.0, 0.0), rel_tol=0.0, abs_tol=1.0e-12)


if __name__ == "__main__":
    failures = 0
    for name, test_fn in sorted(globals().items()):
        if name.startswith("test_") and callable(test_fn):
            try:
                test_fn()
            except Exception:
                failures += 1
                print(f"{name}: FAIL")
                traceback.print_exc()
            else:
                print(f"{name}: PASS")
    if failures:
        raise SystemExit(1)
