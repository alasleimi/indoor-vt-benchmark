"""Evaluator-only standard set and localization metrics."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def source_metrics(predicted, truth, cutoff=1.0, gate=0.5):
    predicted, truth = np.asarray(predicted).reshape(-1, 3), np.asarray(truth).reshape(-1, 3)
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("Source metrics require finite coordinates")
    n, m = len(predicted), len(truth)
    if n and m:
        distances = cdist(predicted, truth)
        cap = np.minimum(distances, cutoff)
        a, b = linear_sum_assignment(cap)
        ospa = (cap[a, b].sum() + cutoff * abs(n - m)) / max(n, m)
        # Minimize the number of invalid edges, not raw distance: maximum gated cardinality.
        a, b = linear_sum_assignment((distances > gate).astype(int))
        valid = distances[a, b] <= gate
        matches = list(zip(a[valid].tolist(), b[valid].tolist()))
    else:
        ospa, matches = (cutoff if n or m else 0.0), []
    tp = len(matches)
    return {
        "ospa": float(ospa),
        "f1": 2 * tp / (n + m) if n + m else 1.0,
        "tp": tp,
        "fp": n - tp,
        "fn": m - tp,
        "matches": matches,
        "ospa_p": 1,
        "ospa_c_m": cutoff,
        "f1_gate_m": gate,
    }


def matched_source_metrics(predicted, truth, cutoff=1.0, gate=0.5):
    """Maximum gated cardinality, then minimum squared error among those matches."""
    metrics = source_metrics(predicted, truth, cutoff, gate)
    predicted, truth = np.asarray(predicted).reshape(-1, 3), np.asarray(truth).reshape(-1, 3)
    squared_error = 0.0
    if metrics["tp"]:
        distances = cdist(predicted, truth)
        k = min(len(predicted), len(truth))
        costs = np.where(distances <= gate, (np.minimum(distances, gate) / gate)**2, k + 1.)
        a, b = linear_sum_assignment(costs)
        valid = distances[a, b] <= gate
        a, b = a[valid], b[valid]
        metrics["matches"] = list(zip(a.tolist(), b.tolist()))
        squared_error = float(np.sum(distances[a, b]**2))
    metrics["matched_squared_error_m2"] = squared_error
    metrics["matched_rmse_m"] = float(np.sqrt(squared_error / metrics["tp"])) if metrics["tp"] else None
    return metrics


def ap_source_metrics(predicted, truth, aps=("A", "B"), cutoff=1.0, gate=0.5):
    """Match within each AP, then pool costs/counts; never allow cross-AP credit.

    OSPA pools the per-AP capped assignment numerators and max(n_ap, m_ap)
    denominators. F1 pools TP/FP/FN. RMSE pools squared errors and TP. Returned
    match indices refer to arrays concatenated in `aps` order, for map drawing.
    """
    numerator, normalizer, squared_error = 0., 0, 0.
    tp, npred, ntruth, matches = 0, 0, 0, []
    for ap in aps:
        p, t = predicted[ap], truth[ap]
        measured = matched_source_metrics(p, t, cutoff, gate)
        weight = max(len(p), len(t))
        numerator += measured["ospa"] * weight
        normalizer += weight
        squared_error += measured["matched_squared_error_m2"]
        matches.extend((i + npred, j + ntruth) for i, j in measured["matches"])
        tp += measured["tp"]
        npred, ntruth = npred + len(p), ntruth + len(t)
    return {"ospa": float(numerator / normalizer) if normalizer else 0.,
            "f1": 2 * tp / (npred + ntruth) if npred + ntruth else 1.,
            "tp": tp, "fp": npred - tp, "fn": ntruth - tp, "matches": matches,
            "ospa_p": 1, "ospa_c_m": cutoff, "f1_gate_m": gate,
            "matched_squared_error_m2": squared_error,
            "matched_rmse_m": float(np.sqrt(squared_error / tp)) if tp else None,
            "matching_scope": "within_ap", "ospa_normalizer": normalizer}
