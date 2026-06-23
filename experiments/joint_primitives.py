"""Small helper primitives for joint reconstruction/classification studies."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def event_score_from_object_logits(reco_logits, object_types, padding_token, method="softor"):
    """Reduce per-object reconstruction logits to one event signal score.

    Logit classes are assumed to be 0=none, 1=H, 2=W. Padding objects, marked
    by ``object_types == padding_token``, do not contribute to the aggregation.
    """
    probs = torch.softmax(reco_logits, dim=-1)
    p_sig = probs[..., 1] + probs[..., 2]
    non_padding = object_types != padding_token
    p_sig = torch.where(non_padding, p_sig, torch.zeros_like(p_sig))

    if method == "max":
        return p_sig.max(dim=1).values
    if method == "softor":
        return 1.0 - torch.prod(1.0 - p_sig, dim=1)
    raise ValueError(f"Unknown event-score method: {method!r}")


def weighted_roc_auc(scores, labels, weights=None):
    """Return weighted ROC-AUC for 1-D event scores and binary labels.

    ``weights`` are optional per-event sample weights passed directly to
    ``sklearn.metrics.roc_auc_score``. If only one label class is present, AUC is
    undefined and ``nan`` is returned.
    """
    if torch.is_tensor(scores):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    if torch.is_tensor(labels):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    if weights is None:
        weights_np = None
    elif torch.is_tensor(weights):
        weights_np = weights.detach().cpu().numpy()
    else:
        weights_np = np.asarray(weights)

    scores_np = np.asarray(scores_np).reshape(-1)
    labels_np = np.asarray(labels_np).reshape(-1)
    if weights_np is not None:
        weights_np = np.asarray(weights_np).reshape(-1)

    if np.unique(labels_np).size < 2:
        return float("nan")
    return float(roc_auc_score(labels_np, scores_np, sample_weight=weights_np))


def asimov_z(s, b):
    """Return no-uncertainty Asimov discovery significance for expected yields.

    Uses ``sqrt(2*((s+b)*log(1+s/b) - s))``. For exactly zero background with
    positive signal, an epsilon background floor is used to return a large finite
    approximation to the zero-background limit; negative background returns 0.
    """
    s_arr, b_arr = np.broadcast_arrays(
        np.asarray(s, dtype=float),
        np.asarray(b, dtype=float),
    )
    result = np.zeros_like(s_arr, dtype=float)
    eps = 1.0e-12
    active = (s_arr > 0.0) & (b_arr >= 0.0)

    if np.any(active):
        s_active = s_arr[active]
        b_active = np.where(b_arr[active] > 0.0, np.maximum(b_arr[active], eps), eps)
        radicand = 2.0 * ((s_active + b_active) * np.log1p(s_active / b_active) - s_active)
        result[active] = np.sqrt(np.maximum(radicand, 0.0))

    if result.ndim == 0:
        return float(result)
    return result


def best_asimov_z(scores, labels, weights):
    """Scan score thresholds and return the best Asimov Z and threshold.

    ``weights`` are expected event yields; the caller is responsible for any
    luminosity or MC normalization. Thresholds are the unique score values, with
    events selected by ``score >= threshold``.
    """
    if torch.is_tensor(scores):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    if torch.is_tensor(labels):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    if torch.is_tensor(weights):
        weights_np = weights.detach().cpu().numpy()
    else:
        weights_np = np.asarray(weights)

    scores_np = np.asarray(scores_np, dtype=float).reshape(-1)
    labels_np = np.asarray(labels_np).reshape(-1)
    weights_np = np.asarray(weights_np, dtype=float).reshape(-1)

    thresholds = np.unique(scores_np)
    if thresholds.size == 0:
        return float("nan"), float("nan")

    best_z = -np.inf
    best_threshold = float("nan")
    signal = labels_np == 1
    background = labels_np == 0

    for threshold in thresholds:
        selected = scores_np >= threshold
        s_yield = weights_np[selected & signal].sum()
        b_yield = weights_np[selected & background].sum()
        z_value = float(asimov_z(s_yield, b_yield))
        if z_value > best_z:
            best_z = z_value
            best_threshold = float(threshold)

    return float(best_z), best_threshold
