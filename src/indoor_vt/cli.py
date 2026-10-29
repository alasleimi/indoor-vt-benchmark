"""Acquisition, priority experiments, and the local evidence viewer."""

import argparse
from pathlib import Path

from .acquisition.records import read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="VT benchmark redesign")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "acquire", "optimize", "view"):
        current = sub.add_parser(command)
        current.add_argument("--config", type=Path, default=Path("configs/redesign.json"))
        current.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
        if command in ("prepare", "acquire"):
            current.add_argument("--qd-root", type=Path, default=Path("../simulator/nist-qd-realization"))
            current.add_argument("--matlab", default="C:/Program Files/MATLAB/R2026a/bin/matlab.exe")
        else:
            current.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
        if command == "view":
            current.add_argument("--port", type=int, default=8770)
    args = parser.parse_args(argv)
    if args.command == "view":
        from .execution.viewer import serve
        serve(args.data, args.run, args.port)
    elif args.command == "optimize":
        from .execution.experiment import run_experiment
        from .worker_memory import install_limits, DEFAULT_LIMITS
        limits = install_limits(**DEFAULT_LIMITS)
        try:
            run_experiment(args.data, args.run, read_json(args.config))
        except Exception as error:
            path = args.run / "status.json"
            state = read_json(path) if path.exists() else {}
            write_json(path, {**state, "state": "failed", "error": str(error)})
            raise
        finally:
            # Keep the Windows job handle alive for the worker's entire lifetime.
            del limits
    else:
        from .acquisition.dataset import prepare_dataset, acquire
        config = read_json(args.config)
        if args.command == "prepare":
            manifest = prepare_dataset(args.data, config, args.qd_root.resolve())
            print(f"Prepared {len(manifest['jobs'])} native chunks", flush=True)
        else:
            acquire(args.data, config, args.qd_root.resolve(), args.matlab)


if __name__ == "__main__":
    main()
