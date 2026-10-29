"""Merge, purge, and urge peak matching with optional size unification."""

from dataclasses import dataclass
import numpy as np
from numba import njit
from .costs import absolute_difference


@njit(cache=True)
def _subset(pair_cost):
    m, n = pair_cost.shape
    cost = np.full((m + 1, n + 1), np.inf)
    take = np.zeros((m + 1, n + 1), np.bool_)
    cost[0] = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            choose = cost[i - 1, j - 1] + pair_cost[i - 1, j - 1]
            if choose <= cost[i, j - 1]:
                cost[i, j], take[i, j] = choose, True
            else:
                cost[i, j] = cost[i, j - 1]
    result = np.empty(m, np.int64)
    i, j = m, n
    while i:
        if take[i, j]:
            result[i - 1] = j - 1
            i -= 1
        j -= 1
    return result


@njit(cache=True)
def _windows(values, p):
    n, width = len(values), 2 * p + 1
    scale = max(values[-1] - values[0], np.max(np.abs(values)), 1.0)
    tiny = np.finfo(np.float64).eps * scale * 16
    result = np.zeros((n, width))
    for i in range(n):
        for offset in range(-p, p + 1):
            j = i + offset
            if j < 0:
                value = values[0] + j * tiny
            elif j >= n:
                value = values[-1] + (j - n + 1) * tiny
            else:
                value = values[j]
            result[i, offset + p] = value - values[i]
    return result


@njit(cache=True)
def _delta(a, b, p, alpha, normalize, separation, tolerance):
    # At alpha=1 the pattern is inactive, including the half-window.
    if alpha == 1:
        p = 0
    wa, wb = _windows(a, p), _windows(b, p)
    width = 2 * p + 1
    result = np.empty((len(a), len(b)))
    difference = np.empty(width)
    divisor = width * (width - 1) / 2 if normalize and p else 1.0
    for i in range(len(a)):
        for j in range(len(b)):
            if separation >= 0 and (
                abs(a[i] - b[j]) > separation + tolerance or separation > a[i] + b[j] + tolerance
            ):
                result[i, j] = np.inf
                continue
            # Windows are sorted. Their pair-distance discrepancy is therefore
            # sum_{l<r} |d[l]-d[r]|, where d = wa[i]-wb[j]. After sorting d,
            # gap k is crossed by k*(width-k) pairs. Insertion sort is small
            # (at most 15 values in this study) and reuses one scratch array.
            for k in range(width):
                value = wa[i, k] - wb[j, k]
                slot = k
                while slot > 0 and difference[slot - 1] > value:
                    difference[slot] = difference[slot - 1]
                    slot -= 1
                difference[slot] = value
            pattern = 0.0
            for k in range(1, width):
                pattern += k * (width - k) * (difference[k] - difference[k - 1])
            result[i, j] = alpha * abs(a[i] - b[j]) + (1 - alpha) * pattern / divisor
    return result


@njit(cache=True)
def _crosses(a0, a1, b0, b1):
    return (a0 - b0) * (a1 - b1) < 0


@njit(cache=True)
def _merge_purge_urge(a0, b0, p, alpha, normalize, separation, tolerance):
    ia, ib = np.argsort(a0, kind="mergesort"), np.argsort(b0, kind="mergesort")
    a, b = a0[ia], b0[ib]
    out = np.empty((min(len(a), len(b)), 3))
    count, trace_count, round_id = 0, 0, 0
    trace = np.empty((min(len(a), len(b)), 5))
    while len(a) and len(b):
        round_id += 1
        delta = _delta(a, b, p, alpha, normalize, separation, tolerance)
        left_best = np.empty(len(a), np.int64)
        right_best = np.empty(len(b), np.int64)
        for i in range(len(a)):
            left_best[i] = np.argmin(delta[i])
        for j in range(len(b)):
            right_best[j] = np.argmin(delta[:, j])
        candidates = []
        for i in range(len(a)):
            j = left_best[i]
            if right_best[j] == i and np.isfinite(delta[i, j]):
                priority = delta[i, j]
                candidates.append((priority, int(ia[i]), int(ib[j]), i, j))
        if not len(candidates):
            break
        candidates.sort()
        keep_a, keep_b = np.ones(len(a), np.bool_), np.ones(len(b), np.bool_)
        for cost, ai, bi, i, j in candidates:
            keep_a[i], keep_b[j] = False, False
            good = True
            for previous in range(count):
                if _crosses(
                    a0[ai], b0[bi], a0[int(out[previous, 0])], b0[int(out[previous, 1])]
                ):
                    good = False
                    break
            trace[trace_count] = np.array([ai, bi, delta[i, j], round_id, 1.0 if good else 0.0])
            trace_count += 1
            if good:
                out[count, 0], out[count, 1], out[count, 2] = ai, bi, delta[i, j]
                count += 1
        a, ia, b, ib = a[keep_a], ia[keep_a], b[keep_b], ib[keep_b]
    return out[:count], trace[:trace_count]


@dataclass(frozen=True)
class MPUrgeMatcher:
    half_window: int = 2
    alpha: float = 0.5
    normalized_pattern: bool = False
    size_unification_norm: object = absolute_difference
    triangle_tolerance_m: float | None = None

    def __post_init__(self):
        if self.half_window < 0 or not 0 <= self.alpha <= 1:
            raise ValueError("Require a nonnegative half-window and alpha in [0,1]")
        if self.triangle_tolerance_m is not None and self.triangle_tolerance_m < 0:
            raise ValueError("Triangle tolerance must be nonnegative")

    def match(self, left, right, *, separation=None):
        left, right = np.asarray(left, float), np.asarray(right, float)
        ia, ib = np.argsort(left, kind="stable"), np.argsort(right, kind="stable")
        a, b = left[ia], right[ib]
        if self.size_unification_norm is not None and len(a) and len(b):
            if len(a) > len(b):
                keep = _subset(self.size_unification_norm(b[:, None], a[None, :]))
                a, ia = a[keep], ia[keep]
            elif len(b) > len(a):
                keep = _subset(self.size_unification_norm(a[:, None], b[None, :]))
                b, ib = b[keep], ib[keep]
        distance = -1.0
        if self.triangle_tolerance_m is not None:
            if separation is None:
                raise ValueError("Triangle checking requires receiver separation")
            distance = float(separation)
        if len(a) and len(b):
            matches, trace = _merge_purge_urge(a, b, self.half_window, self.alpha,
                self.normalized_pattern, distance, self.triangle_tolerance_m or 0.0)
            for table in (matches, trace):
                table[:, 0] = ia[table[:, 0].astype(int)]
                table[:, 1] = ib[table[:, 1].astype(int)]
        else:
            matches, trace = np.empty((0, 3)), np.empty((0, 5))
        return {"matches": matches, "trace": trace, "left_indices": ia, "right_indices": ib}

    def match_array(self, left, right, **kwargs):
        return self.match(left, right, **kwargs)["matches"]


@dataclass(frozen=True)
class CrossingAllowedMPUrgeMatcher(MPUrgeMatcher):
    """Accept every reciprocal candidate, including crossings in range order.

    Standard MPUrge purges both endpoints even for rejected crossings. Thus its
    complete candidate trace has exactly the same windows, costs and rounds as
    the crossing-allowed variant. Accepting every trace row changes only the
    crossing veto, preserving unification, endpoint uniqueness and tie rules.
    The distinct component type keeps this variant's cache separate while the
    unchanged standard matcher retains all existing cache keys.
    """
    def match(self, left, right, *, separation=None):
        result = super().match(left, right, separation=separation)
        result["matches"] = result["trace"][:, :3].copy()
        result["trace"][:, 4] = 1.
        return result
