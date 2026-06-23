import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from joint_primitives import event_score_from_object_logits, weighted_roc_auc, asimov_z, best_asimov_z

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
