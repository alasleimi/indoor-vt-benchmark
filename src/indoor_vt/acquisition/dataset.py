"""Versioned dense acquisitions with deterministic survey views."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .geometry import room_design
from .nist import prepare_scenario, run_scenario, simulator_provenance
from .records import Seeds, read_json, write_json
from ..execution.store import Store
from .truth import source_catalogue


def survey_indices(room, spacing_m=0.5, phase_xy=(2, 2)):
    ratio = spacing_m / room["survey_spacing_m"]
    stride = int(round(ratio))
    phase = np.asarray(phase_xy, dtype=int)
    if stride < 1 or not np.isclose(stride, ratio) or phase.shape != (2,):
        raise ValueError("Spacing must be a positive integer multiple of the base spacing")
    if np.any(phase < 0) or np.any(phase >= stride):
        raise ValueError("Phase must be two lattice indices smaller than the stride")
    return np.flatnonzero(np.all(room["survey_lattice"][:, :2] % stride == phase, axis=1))


def prepare_dataset(root, config, qd_root):
    root, qd_root = Path(root), Path(qd_root).resolve()
    store = Store(root / "store")
    seeds = Seeds(config["seed"])
    rooms = [room_design(f, "scene", i, config["scene"], seeds)
             for f in config["room_families"] for i in range(config["rooms_per_family"])]
    simulator = {k: config["simulator"][k] for k in ("reflections", "diffuse", "materials")}
    scene_value = store.put(rooms)
    identity = {"scene": scene_value, "simulator": simulator,
                "simulator_version": simulator_provenance(qd_root)["tree_hash"],
                "seed": config["seed"], "chunk_positions": config["simulator"]["chunk_positions"]}
    manifest_path = root / "dataset.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["identity"] != identity:
            raise ValueError("Acquisition inputs changed; use a new dataset directory")
        return old
    truth = {room["id"]: {ap: source_catalogue(room, ap)
                          for ap in room["access_points_xyz"]} for room in rooms}
    jobs = []
    size = config["simulator"]["chunk_positions"]
    for room in rooms:
        coarse = survey_indices(room, **config["survey"])
        remaining = np.setdiff1d(np.arange(len(room["survey_xyz"])), coarse)
        for ap in room["access_points_xyz"]:
            batches = [(0, None, coarse)]
            batches += [(1, layout, np.arange(len(layout["queries_xyz"])))
                        for layout in room["layouts"]]
            batches.append((2, None, remaining))
            chunk = 0
            for priority, layout, indices in batches:
                for start in range(0, len(indices), size):
                    selected = indices[start:start + size]
                    scenario = prepare_scenario(root / "native", room, ap, layout,
                                                simulator, seeds, qd_root, selected, chunk)
                    jobs.append({"key": scenario["key"], "room": room["id"], "ap": ap,
                                 "group": scenario["layout"], "priority": priority,
                                 "position_ids": selected.tolist()})
                    chunk += 1
    jobs.sort(key=lambda j: (j["priority"], j["key"]))
    manifest = {"schema": "vt.dataset.2", "identity": identity, "scene_value": scene_value,
                "truth_value": store.put(truth), "jobs": jobs}
    write_json(manifest_path, manifest)
    return manifest


def acquire(root, config, qd_root, matlab):
    root = Path(root)
    manifest = prepare_dataset(root, config, qd_root)
    store = Store(root / "store")
    status = {"state": "running", "completed": 0, "total": len(manifest["jobs"]), "failed": []}
    def run(job):
        receipt = run_scenario(root / "native" / job["key"], Path(qd_root), matlab, store)
        return job["key"], receipt["frames"]
    write_json(root / "acquisition_status.json", status)
    with ThreadPoolExecutor(max_workers=config["simulator"]["workers"]) as pool:
        pending = {pool.submit(run, job): job for job in manifest["jobs"]}
        for task in as_completed(pending):
            try:
                key, frames = task.result()
                status["completed"] += 1
                print(f"Acquired {status['completed']}/{status['total']}: {key} ({frames} frames)",
                      flush=True)
            except Exception as error:
                status["failed"].append({"key": pending[task]["key"], "error": str(error)})
                print(f"Acquisition failed: {pending[task]['key']}: {error}", flush=True)
                for waiting in pending:
                    waiting.cancel()
                status["state"] = "failed"
                write_json(root / "acquisition_status.json", status)
                raise
            write_json(root / "acquisition_status.json", status)
    status["state"] = "failed" if status["failed"] else "complete"
    write_json(root / "acquisition_status.json", status)
    if status["failed"]:
        raise RuntimeError(f"{len(status['failed'])} native chunks failed")
    return manifest
