"""Inspect one-room results and shared typed evidence locally."""

import argparse
from pathlib import Path
from indoor_vt.execution.viewer import serve

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
parser.add_argument("--port", type=int, default=8770)
args = parser.parse_args()
serve(args.data, args.run, args.port)
