"""Unknown-assignment range least squares with an externally supplied source count.

The API accepts measured ranges, receiver coordinates, and a count only. Source
truth, room walls, AP coordinates, and path identities are not optimization inputs.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from .models import Ragged


def assign_ranges(xyz, ranges, sources):
    """Return receiver-by-source observed ranges (NaN for unobserved sources)."""
    predicted = np.linalg.norm(xyz[:, None, :] - sources[None, :, :], axis=2)
    assigned = np.full(predicted.shape, np.nan)
    indices = np.full(predicted.shape, -1, dtype=np.int64)
    for s in range(len(xyz)):
        observed = ranges[s]
        rows, columns = linear_sum_assignment((observed[:, None] - predicted[s])**2)
        assigned[s, columns] = observed[rows]
        indices[s, columns] = rows
    return assigned, indices


def range_sse(xyz, assigned, sources):
    predicted = np.linalg.norm(xyz[:, None, :] - sources[None, :, :], axis=2)
    return float(np.nansum((predicted - assigned)**2))


def refit_sources(xyz, assigned, sources, max_nfev):
    """Independent ordinary nonlinear least squares for fixed assignments."""
    result = sources.copy()
    for k in range(len(sources)):
        mask = np.isfinite(assigned[:, k])
        if mask.sum() < 4:
            continue
        positions, target = xyz[mask], assigned[mask, k]
        def residual(q):
            return np.linalg.norm(q - positions, axis=1) - target
        def jacobian(q):
            delta = q - positions
            norm = np.linalg.norm(delta, axis=1)
            return np.divide(delta, norm[:, None], out=np.zeros_like(delta), where=norm[:, None] > 0)
        fitted = least_squares(residual, sources[k], jac=jacobian, method="lm",
                               max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        # Even a truncated inner solve is useful if it decreases the actual objective.
        if np.isfinite(fitted.x).all() and np.sum(residual(fitted.x)**2) <= np.sum(residual(sources[k])**2):
            result[k] = fitted.x
    return result


@dataclass(frozen=True)
class RangeRMSE:
    starts: int = 16
    max_iterations: int = 150
    max_nfev: int = 60
    relative_tolerance: float = 1e-8
    seed: int = 20261027

    def fit(self, xyz, ranges: Ragged, count: int, *, progress=None):
        xyz = np.asarray(xyz, float)
        if count < 1 or len(xyz) != len(ranges) or self.starts < 1:
            raise ValueError("Positive source/start counts and aligned observations are required")
        lengths = np.diff(ranges.offsets)
        if np.any(lengths > count) or not np.any(lengths):
            raise ValueError("This model requires at most one observed peak per source")
        if not np.isfinite(xyz).all() or not np.isfinite(ranges.values).all():
            raise ValueError("Finite measurements are required")
        rng = np.random.default_rng(self.seed)
        start_time = time.perf_counter()
        best = None
        runs = []
        for start in range(self.starts):
            # Shell radii are measured at a random receiver. Missing radii use only
            # the observed range distribution; no clean fingerprints are accessed.
            anchor = int(rng.integers(len(xyz)))
            radii = np.sort(ranges[anchor])
            if len(radii) < count:
                radii = np.r_[radii, rng.choice(ranges.values, count - len(radii))]
            directions = rng.normal(size=(count, 3))
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            sources = xyz[anchor] + directions * radii[:, None]
            initialization = "random measured-range shells"
            if start == 0 and np.all(lengths == count):
                # A reproducible rank baseline is one of the starts, not truth.
                assigned = np.vstack([np.sort(ranges[s]) for s in range(len(xyz))])
                center = xyz.mean(0)
                local = xyz - center
                design = np.column_stack((-2 * local, np.ones(len(xyz))))
                rhs = assigned**2 - np.sum(local**2, axis=1)[:, None]
                sources = center + np.linalg.lstsq(design, rhs, rcond=None)[0][:3].T
                sources = refit_sources(xyz, assigned, sources, self.max_nfev)
                initialization = "equal-rank algebraic multilateration"
            assigned, indices = assign_ranges(xyz, ranges, sources)
            history = [range_sse(xyz, assigned, sources)]
            stop = "iteration_limit"
            for iteration in range(self.max_iterations):
                sources = refit_sources(xyz, assigned, sources, self.max_nfev)
                assigned, next_indices = assign_ranges(xyz, ranges, sources)
                loss = range_sse(xyz, assigned, sources)
                history.append(loss)
                stable = np.array_equal(indices, next_indices)
                indices = next_indices
                improvement = history[-2] - loss
                if improvement <= self.relative_tolerance * max(history[-2], 1e-12):
                    stop = "assignments_stable" if stable else "relative_objective_tolerance"
                    break
            record = {"start": start, "initialization": initialization,
                      "iterations": iteration + 1, "stop": stop,
                      "range_rmse_m": float(np.sqrt(history[-1] / lengths.sum())),
                      "sse_history": np.asarray(history)}
            runs.append(record)
            if best is None or history[-1] < best[0]:
                best = (history[-1], sources.copy(), indices.copy(), start)
            if progress:
                progress(record)
        return best[1], {"known_source_count": count, "observation_count": int(lengths.sum()),
                         "missing_observations": int(len(xyz) * count - lengths.sum()),
                         "selected_start": best[3], "range_rmse_m": float(np.sqrt(best[0] / lengths.sum())),
                         "assignments": best[2], "starts": runs,
                         "seconds": time.perf_counter() - start_time}


def range_rmse(parameters):
    """Construct the numerical solver; true count is supplied separately to fit()."""
    return RangeRMSE(**parameters)
