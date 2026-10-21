from __future__ import annotations
import argparse
from pathlib import Path
import json
import shutil

from .generation import generate_design
from .provenance import SeedBook, read_json
from .simulation import prepare_scenario, run_scenario


def main():
    parser = argparse.ArgumentParser(description="Local reproducible VT benchmark")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("design")
    gen.add_argument("--namespace", choices=["pilot", "development", "publication"], default="development")
    sim = sub.add_parser("simulate")
    sim.add_argument("--namespace", choices=["pilot", "development", "publication"], default="pilot")
    sim.add_argument("--qd-root", type=Path, required=True)
    sim.add_argument("--matlab", default=shutil.which("matlab"))
    sim.add_argument("--room", default=None)
    sim.add_argument("--layout", default=None)
    sim.add_argument("--ap", choices=["A", "B"], default=None)
    sim.add_argument("--frames", type=int, default=None, help="Pilot-only native timing sample")
    args = parser.parse_args()
    root = args.root.resolve()
    namespace = args.namespace
    design_path = root / "data" / namespace / "design.json"
    design = generate_design(root/"configs/dataset.json", root/"configs/seeds.json", namespace, design_path)
    if args.command == "design":
        print(json.dumps({"design": str(design_path), "rooms": len(design["rooms"]),
                          "hash": design["design_hash"]}), flush=True)
        return
    if args.frames and namespace != "pilot":
        parser.error("Subsampling is only allowed for the separately seeded pilot")
    if not args.matlab:
        parser.error("MATLAB executable required")
    seeds = SeedBook(design["seeds"]["root"], namespace)
    simroot = root/"data"/namespace/"scenarios"
    for room in design["rooms"]:
        if args.room and room["id"] != args.room:
            continue
        for ap in ("A", "B"):
            if args.ap and ap != args.ap:
                continue
            for layout in [None, *room["layouts"]]:
                label = "survey" if layout is None else layout["id"]
                if args.layout and label != args.layout:
                    continue
                indices = None
                if args.frames:
                    n = len(room["survey_xyz"] if layout is None else layout["queries_xyz"])
                    import numpy as np
                    indices = np.linspace(0, n-1, min(args.frames, n), dtype=int)
                manifest = prepare_scenario(simroot, room, ap, layout, design["config"], seeds,
                                            args.qd_root, indices)
                print(f"Simulating {manifest['key']} ({manifest['frames']} frames)", flush=True)
                receipt = run_scenario(simroot/manifest["key"], args.qd_root, args.matlab)
                print(json.dumps({"key": manifest["key"], "elapsed_s": receipt["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
