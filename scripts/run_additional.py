"""Run the additional finished reconstructors and their tuned localization pipelines."""
import argparse
from pathlib import Path
from indoor_vt.execution.additional_experiment import run_additional

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-additional-selection-v1"))
parser.add_argument("--base", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
args = parser.parse_args()
run_additional(args.data, args.run, args.base)
