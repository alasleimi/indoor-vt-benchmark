"""Fixed predicted-K, delay-only coordinate refinement with analytic gradients."""

from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from .contracts import SourceMap


@dataclass(frozen=True)
class DiffAssignConfig:
    scale_m: float = 0.5
    temperature: float = 0.1
    reverse_weight: float = 0.5
    unmatched_cost: float = 2.0
    max_iterations: int = 100


def objective(flat, positions, rows, config):
    sources = flat.reshape(-1, 3)
    gradient, total, count = np.zeros_like(sources), 0.0, 0
    c = config
    for position, row in zip(positions, rows):
        if not len(row):
            continue
        delta = sources - position
        predicted = np.linalg.norm(delta, axis=1)
        difference = predicted[:, None] - row[None, :]
        absolute = np.sqrt(difference * difference + 1e-12)
        cost = absolute / c.scale_m
        logits = -cost / c.temperature
        # Log-mean-exp, followed by a fixed clutter cap. No observation is labelled
        # as direct/reflected; no path identities, K oracle or geometry bounds.
        forward = -c.temperature * (logsumexp(logits, axis=0) - np.log(len(sources)))
        reverse = -c.temperature * (logsumexp(logits, axis=1) - np.log(len(row)))
        total += np.minimum(forward, c.unmatched_cost).mean()
        total += c.reverse_weight * np.minimum(reverse, c.unmatched_cost).mean()
        weight = softmax(logits, axis=0) * (forward < c.unmatched_cost)[None, :] / len(row)
        weight += (
            c.reverse_weight
            * softmax(logits, axis=1)
            * (reverse < c.unmatched_cost)[:, None]
            / len(sources)
        )
        radial_gradient = (weight * difference / absolute / c.scale_m).sum(axis=1)
        unit = np.divide(
            delta, predicted[:, None], out=np.zeros_like(delta), where=predicted[:, None] > 0
        )
        gradient += radial_gradient[:, None] * unit
        count += 1
    return total / max(count, 1), gradient.ravel() / max(count, 1)


class DiffAssign:
    version = "diffassign.predicted_k.delay_only.v1"

    def __init__(self, config=DiffAssignConfig()):
        if config.scale_m <= 0 or config.temperature <= 0 or config.unmatched_cost <= 0:
            raise ValueError("Positive scale, temperature and unmatched penalty required")
        self.config = config

    def fit(self, survey, initial):
        output, audit = {}, {}
        for ap, sources in initial.by_ap.items():
            rows = [o.channels.get(ap, np.empty(0)) for o in survey.observations]
            if not len(sources) or not any(len(r) for r in rows):
                output[ap], audit[ap] = sources.copy(), {"status": "unchanged_empty"}
                continue
            args = (survey.xyz, rows, self.config)
            before = objective(sources.ravel(), *args)[0]
            result = minimize(
                objective,
                sources.ravel(),
                args=args,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.config.max_iterations, "ftol": 1e-9},
            )
            # Nonconvergence retains initialization; never select using evaluator truth.
            accepted = bool(result.success and np.isfinite(result.fun) and result.fun <= before)
            output[ap] = result.x.reshape(-1, 3) if accepted else sources.copy()
            audit[ap] = {
                "status": "ok" if accepted else "retained_initialization",
                "k": len(sources),
                "before": before,
                "after": float(result.fun),
                "iterations": int(result.nit),
                "solver_message": str(result.message),
            }
        return SourceMap(
            output,
            {
                "initial": initial.diagnostics,
                "refinement": audit,
                "config": asdict(self.config),
                "version": self.version,
            },
        )
