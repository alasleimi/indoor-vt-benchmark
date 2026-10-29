"""Run the priority searches; resume from per-trial and per-condition receipts."""

import argparse
from pathlib import Path

from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.execution.experiment import run_experiment
from indoor_vt.worker_memory import install_limits, DEFAULT_LIMITS

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/redesign.json"))
parser.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
args = parser.parse_args()
limits = install_limits(**DEFAULT_LIMITS)
try:
    run_experiment(args.data, args.run, read_json(args.config))
except Exception as error:
    status_path = args.run / "status.json"
    status = read_json(status_path) if status_path.exists() else {}
    write_json(status_path, {**status, "state": "failed", "error": str(error)})
    raise
