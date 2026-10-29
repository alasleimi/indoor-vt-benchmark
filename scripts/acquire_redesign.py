"""Run only the new frozen native acquisition, independently of optimization."""

import argparse
from pathlib import Path

from indoor_vt.acquisition.dataset import acquire, prepare_dataset
from indoor_vt.acquisition.records import read_json

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/redesign.json"))
parser.add_argument("--root", type=Path, default=Path("data/redesign/rectangle-20261025"))
parser.add_argument("--qd-root", type=Path, default=Path("../simulator/nist-qd-realization"))
parser.add_argument("--matlab", default="C:/Program Files/MATLAB/R2026a/bin/matlab.exe")
parser.add_argument("--prepare-only", action="store_true")
args = parser.parse_args()
config = read_json(args.config)
if args.prepare_only:
    result = prepare_dataset(args.root, config, args.qd_root)
    print(f"Prepared {len(result['jobs'])} native chunks")
else:
    acquire(args.root, config, args.qd_root.resolve(), args.matlab)
