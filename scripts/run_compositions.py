"""Run finisher swaps, MPUrge triangle and their stagewise/joint pipelines."""
import argparse
from pathlib import Path
from indoor_vt.execution.composition_experiment import run_compositions

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-compositions-v1"))
parser.add_argument("--base", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
args = parser.parse_args()
run_compositions(args.data, args.run, args.base)
