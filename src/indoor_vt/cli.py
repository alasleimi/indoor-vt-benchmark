from __future__ import annotations
import argparse
from pathlib import Path
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from .generation import generate_design
from .provenance import SeedBook
from .simulation import prepare_scenario, run_scenario


def main():
    parser = argparse.ArgumentParser(description="Local reproducible VT benchmark")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("design")
    gen.add_argument(
        "--namespace", choices=["pilot", "development", "publication"], default="development"
    )
    sim = sub.add_parser("simulate")
    sim.add_argument(
        "--namespace", choices=["pilot", "development", "publication"], default="pilot"
    )
    sim.add_argument("--qd-root", type=Path, required=True)
    sim.add_argument("--matlab", default=shutil.which("matlab"))
    sim.add_argument("--room", default=None)
    sim.add_argument("--layout", default=None)
    sim.add_argument("--ap", choices=["A", "B"], default=None)
    sim.add_argument("--frames", type=int, default=None, help="Pilot-only native timing sample")
    sim.add_argument("--workers", type=int, default=1)
    derive = sub.add_parser("observations")
    derive.add_argument(
        "--namespace", choices=["pilot", "development", "publication"], default="development"
    )
    truth = sub.add_parser("truth")
    truth.add_argument(
        "--namespace", choices=["pilot", "development", "publication"], default="development"
    )
    calibration = sub.add_parser("calibrate")
    calibration.add_argument(
        "--namespace", choices=["development", "publication"], default="development"
    )
    calibration.add_argument("--track", choices=["direct", "receiver"], required=True)
    calibration.add_argument("--regime", choices=["component", "end_to_end"], required=True)
    calibration.add_argument("--initial", required=True)
    calibration.add_argument("--vt", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--frozen", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument(
        "--namespace", choices=["pilot", "development", "publication"], default="development"
    )
    verify.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "evaluate":
        from .campaign import evaluate_frozen

        print(evaluate_frozen(root, args.frozen), flush=True)
        return
    namespace = args.namespace
    design_path = root / "data" / namespace / "design.json"
    design = generate_design(
        root / "configs/dataset.json", root / "configs/seeds.json", namespace, design_path
    )
    if args.command == "design":
        print(
            json.dumps(
                {
                    "design": str(design_path),
                    "rooms": len(design["rooms"]),
                    "hash": design["design_hash"],
                }
            ),
            flush=True,
        )
        return
    if args.command == "observations":
        from .dataset import materialize

        seeds = SeedBook(design["seeds"]["root"], namespace)
        for receipt in sorted((root / "data" / namespace / "scenarios").glob("*/*/*/receipt.json")):
            records = materialize(receipt.parent, design["config"], seeds)
            print(
                json.dumps({"scenario": str(receipt.parent), "conditions": len(records)}),
                flush=True,
            )
        return
    if args.command == "truth":
        from .truth import generate_truth, plot_truth

        destination = root / "data" / namespace / "evaluation"
        references = generate_truth(design, destination / "source_truth.json")
        plot_truth(design, references, destination / "source_catalogue.png")
        print(
            json.dumps(
                {"truth_hash": references["truth_hash"], "rooms": len(references["sources"])}
            ),
            flush=True,
        )
        return
    if args.command == "calibrate":
        from .campaign import calibrate

        print(
            calibrate(root, namespace, args.track, args.regime, args.initial, args.vt), flush=True
        )
        return
    if args.command == "verify":
        from .audit import audit_dataset

        result = audit_dataset(root, namespace, not args.allow_incomplete)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("completed_scenarios", "expected_scenarios", "frames")
                }
            ),
            flush=True,
        )
        return
    if args.frames and namespace != "pilot":
        parser.error("Subsampling is only allowed for the separately seeded pilot")
    if not args.matlab:
        parser.error("MATLAB executable required")
    seeds = SeedBook(design["seeds"]["root"], namespace)
    simroot = root / "data" / namespace / "scenarios"
    jobs = []
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

                    indices = np.linspace(0, n - 1, min(args.frames, n), dtype=int)
                manifest = prepare_scenario(
                    simroot, room, ap, layout, design["config"], seeds, args.qd_root, indices
                )
                jobs.append(manifest)
    if not jobs:
        parser.error("No scenarios match the requested room/layout filters")

    def execute(manifest):
        print(f"Simulating {manifest['key']} ({manifest['frames']} frames)", flush=True)
        receipt = run_scenario(simroot / manifest["key"], args.qd_root, args.matlab)
        return {"key": manifest["key"], "elapsed_s": receipt["elapsed_s"]}

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute, manifest) for manifest in jobs]
        for future in as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                failures.append(str(error))
                print(json.dumps({"error": str(error)}), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} acquisitions failed; rerun to resume. {failures[0]}")


if __name__ == "__main__":
    main()
