"""Signed 3D source/receiver estimation from measured ranges."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from .costs import squared_loss


@dataclass(frozen=True)
class Multilateration3D:
    loss: object = squared_loss
    max_evaluations: int = 100
    loss_scale_m: float = 1.0

    def __post_init__(self):
        assert self.loss_scale_m > 0, "loss_scale_m must be positive"

    def fit(self, positions, ranges, *, initial=None, weights=None):
        positions, ranges = np.asarray(positions, float), np.asarray(ranges, float)
        if len(positions) < 4 or np.linalg.matrix_rank(positions - positions.mean(0)) < 3:
            return None, {"status": "insufficient_3d_geometry"}
        if initial is None:
            center = positions.mean(0)
            local = positions - center
            matrix = np.column_stack((-2 * local, np.ones(len(local))))
            rhs = ranges**2 - np.sum(local**2, axis=1)
            initial = center + np.linalg.lstsq(matrix, rhs, rcond=None)[0][:3]
        root_weights = np.ones(len(ranges)) if weights is None else np.sqrt(weights)
        def residual(x):
            return root_weights * (np.linalg.norm(x - positions, axis=1) - ranges)
        def jacobian(x):
            delta = x - positions
            length = np.linalg.norm(delta, axis=1)
            return root_weights[:, None] * np.divide(delta, length[:, None],
                out=np.zeros_like(delta), where=length[:, None] > 0)
        linear = self.loss is squared_loss
        result = least_squares(residual, initial, jac=jacobian,
                               loss="linear" if linear else self.loss,
                               method="lm" if linear else "trf",
                               f_scale=self.loss_scale_m,
                               max_nfev=self.max_evaluations)
        detail = {"status": "ok" if result.success else "solver_failure",
                  "evaluations": result.nfev, "residuals": residual(result.x)}
        return (result.x if result.success and np.isfinite(result.x).all() else None), detail

    __call__ = fit
