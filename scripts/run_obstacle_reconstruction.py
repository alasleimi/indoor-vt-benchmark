import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(key, "1")
from indoor_vt.execution.obstacle_reconstruction import run_obstacle_reconstruction  # noqa: E402 - numerical libraries must see thread limits before import

if __name__ == "__main__":
    run_obstacle_reconstruction("data/redesign/rectangle-20261025",
        "runs/redesign/rectangle-clean-selection-v4", "runs/redesign/rectangle-obstacle-reconstruction-v1")
