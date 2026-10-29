"""Score source proposals against ranges, optionally refit, then select greedily."""

from dataclasses import dataclass
import numpy as np

from .models import SourceMap


def nearest_peaks(point, xyz, rows, available=None):
    """One nearest available peak per survey position; empty rows have inf error."""
    predicted = np.linalg.norm(xyz - point, axis=1)
    errors, indices = np.full(len(xyz), np.inf), np.full(len(xyz), -1, dtype=int)
    for i in range(len(xyz)):
        ids = np.arange(len(rows[i])) if available is None else np.flatnonzero(available[i])
        if len(ids):
            residuals = abs(rows[i][ids] - predicted[i])
            j = np.argmin(residuals)
            errors[i], indices[i] = residuals[j], ids[j]
    return errors, indices


def support_terms(errors, tolerance, smooth_scale_factor):
    if tolerance <= 0 or smooth_scale_factor <= 0:
        raise ValueError("Support and smooth scales must be positive")
    support = np.count_nonzero(errors <= tolerance)
    closeness = np.exp(-.5 * (errors / (tolerance * smooth_scale_factor))**2).sum()
    return support, closeness


@dataclass(frozen=True)
class SupportFirstScorer:
    inlier_tolerance_m: float = .18
    smooth_scale_factor: float = 1.0
    tie_break_weight: float = 1.0

    def __call__(self, errors):
        if not 0 <= self.tie_break_weight <= 1:
            raise ValueError("Tie-break weight must be in [0,1]")
        support, closeness = support_terms(errors, self.inlier_tolerance_m, self.smooth_scale_factor)
        return float(support + self.tie_break_weight * closeness / (len(errors) + 1))


@dataclass(frozen=True)
class SupportAndResidualScorer:
    inlier_tolerance_m: float = .18
    smooth_scale_factor: float = 1.0
    smooth_weight: float = .35
    median_penalty_weight: float = .10
    median_clip_factor: float = 4.0

    def __call__(self, errors):
        support, closeness = support_terms(errors, self.inlier_tolerance_m, self.smooth_scale_factor)
        penalty = (np.median(np.minimum(errors, self.median_clip_factor * self.inlier_tolerance_m))
                   / self.inlier_tolerance_m) if len(errors) else 0.
        return float(support + self.smooth_weight * closeness - self.median_penalty_weight * penalty)


def range_curve_rms(a, b, xyz):
    return float(np.sqrt(np.mean((np.linalg.norm(xyz - a, axis=1) - np.linalg.norm(xyz - b, axis=1))**2)))


@dataclass(frozen=True)
class NearestPeakRefitter:
    estimator: object
    weights: object
    association_tolerance_m: float = .36

    def fit(self, point, xyz, rows, available):
        errors, slots = nearest_peaks(point, xyz, rows, available)
        ids = np.flatnonzero(errors <= self.association_tolerance_m)
        ranges = np.array([rows[i][slots[i]] for i in ids])
        updated, detail = self.estimator.fit(xyz[ids], ranges, initial=point, weights=self.weights(errors[ids]))
        return updated, {"members": np.column_stack((ids, slots[ids])), **detail}


@dataclass(frozen=True)
class GreedySourceSelector:
    scorer: object
    refitter: object = None
    min_support_positions: int = 5
    min_support_fraction: float = .08
    duplicate_distance: object = range_curve_rms
    duplicate_tolerance_m: float = .12
    consume_support: bool = True
    max_sources: int | None = None

    def select(self, survey, candidates):
        output, audit = {}, {}
        for ap, proposals in candidates.by_ap.items():
            rows, xyz = survey.channels[ap], survey.xyz
            available = [np.ones(len(rows[i]), dtype=bool) for i in range(len(rows))]
            remaining = list(range(len(proposals)))
            required = max(self.min_support_positions, int(np.ceil(self.min_support_fraction * len(xyz))))
            accepted, history = [], []
            while remaining and (self.max_sources is None or len(accepted) < self.max_sources):
                scores = [self.scorer(nearest_peaks(proposals[i], xyz, rows, available)[0]) for i in remaining]
                best = int(np.argmax(scores))
                index = remaining.pop(best)
                point = proposals[index].copy()
                errors, slots = nearest_peaks(point, xyz, rows, available)
                if np.count_nonzero(errors <= self.scorer.inlier_tolerance_m) < required:
                    history.append({"proposal": index, "score": scores[best], "status": "stop_insufficient_support"})
                    break
                refit = None
                if self.refitter is not None:
                    updated, refit = self.refitter.fit(point, xyz, rows, available)
                    if updated is not None:
                        point = updated
                errors, slots = nearest_peaks(point, xyz, rows, available)
                support = np.flatnonzero(errors <= self.scorer.inlier_tolerance_m)
                duplicate = any(self.duplicate_distance(point, p, xyz) <= self.duplicate_tolerance_m for p in accepted)
                status = "insufficient_support" if len(support) < required else "duplicate" if duplicate else "accepted"
                detail = {"proposal": index, "score": scores[best], "point": point,
                          "support": np.column_stack((support, slots[support])), "errors": errors,
                          "refit": refit, "status": status}
                history.append(detail)
                if status != "accepted":
                    continue
                accepted.append(point)
                if self.consume_support:
                    for i in support:
                        available[i][slots[i]] = False
                duplicates = [i for i in remaining if self.duplicate_distance(proposals[i], point, xyz)
                              <= self.duplicate_tolerance_m]
                detail["suppressed_proposals"] = np.asarray(duplicates, dtype=int)
                blocked = set(duplicates)
                remaining = [i for i in remaining if i not in blocked]
            output[ap] = np.asarray(accepted).reshape(-1, 3)
            audit[ap] = {"required_support": required, "history": history}
        return SourceMap(output, {"selection": audit, "input": candidates})
