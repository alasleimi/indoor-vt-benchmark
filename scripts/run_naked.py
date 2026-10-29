"""Run Zayets/RANSAC without finishers, including stagewise and joint SMART-LEA."""
import argparse
from pathlib import Path

from indoor_vt.execution.composition_experiment import run_compositions
from indoor_vt.worker_memory import DEFAULT_LIMITS, install_limits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
    parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-naked-selection-v1"))
    parser.add_argument("--base", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
    args = parser.parse_args()
    limits = install_limits(**DEFAULT_LIMITS)
    try:
        run_compositions(args.data, args.run, args.base, methods=("zayets_naked", "ransac_naked"))
    finally:
        del limits
