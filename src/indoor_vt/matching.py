"""Interchangeable score, geometry gate and Merge/Purge/Urge matcher.

Source: MPUrge and MPUrge-MAP, with the ordered cross-check and coverage
division interpretation used by the source-checked SAPP implementation.
Compiled numerical loops preserve original observation indices and impose no
peak-count cap. Normalization belongs to the score, not the grouping module.
"""

from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass(frozen=True)
class MatcherConfig:
    half_window: int = 2
    alpha: float = 0.5
    normalized_pattern: bool = False
    size_norm: str = "l1"
    cross_mode: str = "order"
    conflict_priority: str = "dissimilarity"

    def __post_init__(self):
        if self.half_window < 0 or not 0 <= self.alpha <= 1:
            raise ValueError("Invalid pattern window or alpha")
        if self.size_norm not in ("l1", "l2") or self.cross_mode not in ("order", "printed"):
            raise ValueError("Unknown matcher interpretation")
        if self.conflict_priority not in ("dissimilarity", "distance"):
            raise ValueError("Unknown conflict priority")


@dataclass(frozen=True)
class PeakMatch:
    left: int
    right: int
    cost: float


@njit(cache=True)
def _subset(longer, target, squared):
    n, m = len(longer), len(target)
    cost = np.full((m + 1, n + 1), np.inf)
    take = np.zeros((m + 1, n + 1), np.bool_)
    cost[0] = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            difference = abs(longer[j - 1] - target[i - 1])
            if squared:
                difference *= difference
            choose = cost[i - 1, j - 1] + difference
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
    wa, wb = _windows(a, p), _windows(b, p)
    width = 2 * p + 1
    result = np.empty((len(a), len(b)))
    divisor = width * (width - 1) / 2 if normalize and p else 1.0
    for i in range(len(a)):
        for j in range(len(b)):
            if separation >= 0 and (
                abs(a[i] - b[j]) > separation + tolerance or separation > a[i] + b[j] + tolerance
            ):
                result[i, j] = np.inf
                continue
            pattern = 0.0
            for left in range(width):
                for right in range(left + 1, width):
                    pattern += abs(
                        abs(wa[i, left] - wa[i, right]) - abs(wb[j, left] - wb[j, right])
                    )
            result[i, j] = alpha * abs(a[i] - b[j]) + (1 - alpha) * pattern / divisor
    return result


@njit(cache=True)
def _crosses(a0, a1, b0, b1, printed):
    return (a0 - a1) * (b0 - b1) < 0 if printed else (a0 - b0) * (a1 - b1) < 0


@njit(cache=True)
def _match(a0, b0, p, alpha, normalize, squared, printed, separation, tolerance, distance_priority):
    ia, ib = np.argsort(a0, kind="mergesort"), np.argsort(b0, kind="mergesort")
    a, b = a0[ia], b0[ib]
    if len(a) > len(b):
        keep = _subset(a, b, squared)
        a, ia = a[keep], ia[keep]
    elif len(b) > len(a):
        keep = _subset(b, a, squared)
        b, ib = b[keep], ib[keep]
    out = np.empty((min(len(a), len(b)), 3))
    count = 0
    while len(a) and len(b):
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
                priority = abs(a[i] - b[j]) if distance_priority else delta[i, j]
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
                    a0[ai], b0[bi], a0[int(out[previous, 0])], b0[int(out[previous, 1])], printed
                ):
                    good = False
                    break
            if good:
                out[count, 0], out[count, 1], out[count, 2] = ai, bi, delta[i, j]
                count += 1
        a, ia, b, ib = a[keep_a], ia[keep_a], b[keep_b], ib[keep_b]
    return out[:count]


class PatternMatcher:
    def __init__(self, config=MatcherConfig()):
        self.config = config

    def match_array(self, a, b, *, separation=None, tolerance=0.0):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if not len(a) or not len(b):
            return np.empty((0, 3))
        c = self.config
        return _match(
            a,
            b,
            c.half_window,
            c.alpha,
            c.normalized_pattern,
            c.size_norm == "l2",
            c.cross_mode == "printed",
            -1.0 if separation is None else separation,
            tolerance,
            c.conflict_priority == "distance",
        )

    def __call__(self, a, b, **kwargs):
        return [
            PeakMatch(int(i), int(j), float(cost))
            for i, j, cost in self.match_array(a, b, **kwargs)
        ]
