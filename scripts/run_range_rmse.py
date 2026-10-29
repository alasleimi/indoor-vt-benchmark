"""Run the fixed-compute, unknown-assignment true-count range baseline."""
import argparse
from pathlib import Path

from indoor_vt.execution.range_rmse_experiment import run_range_rmse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
parser.add_argument("--base", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-range-rmse-true-count-v1"))
args = parser.parse_args()
run_range_rmse(args.data, args.base, args.run)
