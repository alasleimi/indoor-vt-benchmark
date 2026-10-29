"""Run independent composition families in bounded worker processes.

Each worker uses the unchanged experiment runner and owns its logs. The parent
publishes immutable receipts/outcomes and atomic log snapshots to the viewer.
This changes scheduling only; parameter lists, seeds and winner rules are fixed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import os
import shutil
import time


def worker(method, data, root, base):
    from indoor_vt.execution import composition_experiment as experiment
    from indoor_vt.execution.compositions import VARIANTS as all_variants
    # Independent families have no shared choices. Restrict only the runner's
    # iteration; constructors and parameter spaces remain exactly the recorded ones.
    original = experiment.VARIANTS
    try:
        experiment.VARIANTS = {method: all_variants[method]}
        return len(experiment.run_compositions(data, root, base))
    finally:
        experiment.VARIANTS = original


def atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".publishing")
    shutil.copyfile(source, temporary)
    # Windows readers can briefly hold the destination against replacement.
    # A busy viewer must not stop publication or the experiment coordinator.
    for attempt in range(100):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def run(args):
    from indoor_vt.execution.compositions import VARIANTS
    from indoor_vt.execution.composition_experiment import composition_config, source_schedule
    from indoor_vt.execution.experiment import configurations
    from indoor_vt.execution.artifacts import file_lock
    from indoor_vt.acquisition.records import Seeds, read_json, write_json
    config = composition_config(read_json(args.base / "config.json"))
    seeds = Seeds(config["optimization"]["seed"])
    source_counts = {m: len(source_schedule(s, config["compositions"]["source_trials"],
        seeds.seed("composition-source", m))) for m, s in config["compositions"]["spaces"].items()}
    target = sum(source_counts[m] * (2 if m.startswith("mpurge") else 1) for m in VARIANTS)
    modes = len(config["optimization"]["unification"])
    target += len(VARIANTS) * modes * (len(configurations(config["optimization"]["smart_lea"]))
                                     + config["compositions"]["joint_trials"])
    selections = sum(modes if m.startswith("mpurge") else 1 for m in VARIANTS)
    selections += 2 * len(VARIANTS) * modes
    expected_outcomes = selections * (1 + len(config["observations"]["noise_levels"]))
    args.run.mkdir(parents=True, exist_ok=True)
    with file_lock(args.run / "run.lock"):
        write_json(args.run / "orchestration.json", {
            "script": str(Path(__file__).name), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "workers": args.workers, "families": list(VARIANTS), "target_trials": target,
            "note": "Independent family shards; numerical experiment implementation is unchanged"})
        shards = {m: args.run / "shards" / m for m in VARIANTS}
        # Resume any already-computed serial trials; never write a worker's live log.
        for method, shard in shards.items():
            for directory in ("trials", "selected", "evaluations", "outcomes"):
                for path in (args.run / directory).glob("*"):
                    if directory in ("trials", "selected") and not path.name.startswith(method + "_"):
                        continue
                    if directory in ("evaluations", "outcomes"):
                        continue  # Published completed outcomes are regenerated from exact cache if needed.
                    target_path = shard / directory / path.name
                    if not target_path.exists():
                        atomic_copy(path, target_path)
        copied = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {m: pool.submit(worker, m, args.data, shard, args.base) for m, shard in shards.items()}
            while True:
                states, results, completed, failures = [], [], 0, []
                for method, shard in shards.items():
                    for name in ("config.json", "implementation.json"):
                        path = shard / name
                        if path.exists():
                            value = read_json(path)
                            destination = args.run / name
                            if destination.exists() and read_json(destination) != value:
                                raise ValueError("Cannot combine incompatible worker metadata")
                            if not destination.exists():
                                write_json(destination, value)
                    for directory in ("trials", "selected", "evaluations", "outcomes"):
                        for path in (shard / directory).glob("*"):
                            if path.suffix not in (".json", ".jsonl", ".npz"):
                                continue
                            stat = path.stat()
                            signature = (stat.st_size, stat.st_mtime_ns)
                            if copied.get(path) != signature:
                                atomic_copy(path, args.run / directory / path.name)
                                copied[path] = signature
                    if (shard / "results.json").exists():
                        results.extend(read_json(shard / "results.json"))
                    if (shard / "status.json").exists():
                        status = read_json(shard / "status.json")
                        completed += status.get("completed_trials", 0)
                        states.append({"family": method, **status})
                    future = futures[method]
                    if future.done() and future.exception() is not None:
                        failures.append(method + ": " + str(future.exception()))
                write_json(args.run / "results.json", results)
                done = all(f.done() for f in futures.values())
                active = [s for s in states if s["state"] != "complete"]
                if done and not failures and (completed != target or len(results) != expected_outcomes):
                    failures.append("The suite is missing scheduled trials or outcomes")
                write_json(args.run / "status.json", {
                    "state": "failed" if failures else "complete" if done else "running",
                    "completed_trials": completed, "target_trials": target,
                    "selection_noise_m": 0., "updated": time.time(),
                    "current": {"method": "; ".join(s["current"]["method"] for s in active if s.get("current"))},
                    "workers": states, **({"error": failures} if failures else {})})
                if done:
                    if failures:
                        raise RuntimeError("; ".join(failures))
                    print(f"Completed {completed} trials and {len(results)} retained outcomes", flush=True)
                    return
                time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
    parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-compositions-v1"))
    parser.add_argument("--base", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    run(parser.parse_args())
