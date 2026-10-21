"""Range-only, signed 3D multilateration with no room or AP truth bounds."""

import numpy as np
from scipy.optimize import least_squares


def fit_source(
    positions, ranges, initial=None, weights=None, max_nfev=100, loss="linear", f_scale=1.0
):
    positions, ranges = np.asarray(positions, float), np.asarray(ranges, float)
    if len(positions) < 4 or np.linalg.matrix_rank(positions - positions.mean(0)) < 3:
        return None, {"status": "insufficient_3d_geometry"}
    if initial is None:
        center = positions.mean(0)
        local = positions - center
        matrix = np.column_stack((-2 * local, np.ones(len(local))))
        coefficient = np.linalg.lstsq(
            matrix, ranges * ranges - np.sum(local * local, axis=1), rcond=None
        )[0]
        initial = center + coefficient[:3]
    root_weight = np.ones(len(ranges)) if weights is None else np.sqrt(weights)

    def residual(x):
        return root_weight * (np.linalg.norm(x - positions, axis=1) - ranges)

    def jacobian(x):
        d = x - positions
        norm = np.linalg.norm(d, axis=1)
        return root_weight[:, None] * np.divide(
            d, norm[:, None], out=np.zeros_like(d), where=norm[:, None] > 0
        )

    # Unconstrained three-coordinate linear-loss fits use MINPACK LM; robust
    # published consensus losses use TRF, which supports non-linear loss.
    method = "lm" if loss == "linear" else "trf"
    fit = least_squares(
        residual,
        initial,
        jac=jacobian,
        max_nfev=max_nfev,
        loss=loss,
        f_scale=f_scale,
        method=method,
    )
    if not np.isfinite(fit.x).all() or not fit.success:
        return None, {"status": "solver_failure", "nfev": int(fit.nfev)}
    return fit.x, {
        "status": "ok",
        "rms_m": float(np.sqrt(np.mean(residual(fit.x) ** 2))),
        "nfev": int(fit.nfev),
        "solver": method,
    }
