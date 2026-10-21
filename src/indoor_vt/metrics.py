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
