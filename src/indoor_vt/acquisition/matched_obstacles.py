"""Acquire saved obstacle layouts on one common subset of the original survey."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from pathlib import Path

import numpy as np

from .dataset import survey_indices
from .nist import prepare_scenario, run_scenario, simulator_provenance
from .observations import Observations
from .records import Seeds, digest, read_json, write_json
from ..execution.explorer import inside_obstacles
from ..execution.store import Store


def common_survey_mask(room, survey_config):
    original = survey_indices(room, **survey_config)
    xyz = room["survey_xyz"][original]
    excluded = {layout["id"]: original[inside_obstacles(xyz, layout["boxes"])].tolist()
                for layout in room["layouts"]}
    removed = sorted({i for values in excluded.values() for i in values})
    retained = original[~np.isin(original, removed)]
    return {"original_count": len(original), "retained_count": len(retained),
            "excluded_count": len(removed), "original_position_ids": original.tolist(),
            "retained_position_ids": retained.tolist(), "excluded_position_ids": removed,
            "excluded_by_layout": excluded,
            "coordinates_digest": digest(room["survey_xyz"][retained].tolist()),
            "occupancy": "Exclude if XY is covered by a box footprint, including its boundary, and 0 <= z <= box height; union across every layout; no extra clearance"}


def prepare_matched_obstacles(original_root, destination, config, qd_root):
    original_root, destination, qd_root = map(Path, (original_root, destination, qd_root))
    original = Observations(original_root, config)
    store = Store(destination / "store")
    masks = {room["id"]: common_survey_mask(room, config["survey"]) for room in original.rooms}
    simulator = {k: config["simulator"][k] for k in ("reflections", "diffuse", "materials")}
    simulator_hash = simulator_provenance(qd_root)["tree_hash"]
    identity = {"base_dataset": digest(original.manifest), "masks": masks,
                "simulator": simulator, "simulator_tree_hash": simulator_hash,
                "chunk_positions": config["simulator"]["chunk_positions"], "seed": config["seed"]}
    if (destination / "dataset.json").exists():
        existing = read_json(destination / "dataset.json")
        if existing["identity"] != identity:
            raise ValueError("Matched-survey acquisition inputs differ; preserve this dataset")
        return existing
    rooms = copy.deepcopy(original.rooms)
    for room in rooms:
        for layout in room["layouts"]:
            # Keep original dense position IDs; only the shared subset is acquired.
            layout["queries_xyz"] = room["survey_xyz"]
            layout["reconstruction_position_ids"] = np.asarray(masks[room["id"]]["retained_position_ids"])
    scene_value = store.put(rooms)
    truth_value = store.put(original.store.get(original.manifest["truth_value"]))
    jobs, reuse = [], []
    seeds = Seeds(config["seed"])
    size = config["simulator"]["chunk_positions"]
    for room in rooms:
        selected = np.asarray(masks[room["id"]]["retained_position_ids"])
        for layout in room["layouts"]:
            for ap in room["access_points_xyz"]:
                if layout["box_count"] == 0:
                    for old_job in original.manifest["jobs"]:
                        if ((old_job["room"], old_job["ap"], old_job["group"]) != (room["id"], ap, "survey")
                                or not set(selected).intersection(old_job["position_ids"])):
                            continue
                        receipt = read_json(original_root / "native" / old_job["key"] / "receipt.json")
                        if receipt["simulator_tree_hash"] != simulator_hash:
                            raise ValueError("Original empty survey used a different simulator")
                        for field in ("native_value", "visualization_value"):
                            assert store.put(original.store.get(receipt[field])) == receipt[field]
                        write_json(destination / "native" / old_job["key"] / "receipt.json", receipt)
                        jobs.append({**old_job, "group": layout["id"], "priority": 1, "reused": True})
                        reuse.append({"key": old_job["key"], "source_receipt": str(original_root / "native" / old_job["key"] / "receipt.json"),
                                      "native_value": receipt["native_value"], "visualization_value": receipt["visualization_value"],
                                      "group_alias": {"from": "survey", "to": layout["id"]}})
                    continue
                for chunk, start in enumerate(range(0, len(selected), size)):
                    ids = selected[start:start+size]
                    scenario = prepare_scenario(destination / "native", room, ap, layout, simulator,
                                                seeds, qd_root.resolve(), ids, chunk)
                    jobs.append({"key": scenario["key"], "room": room["id"], "ap": ap,
                                 "group": layout["id"], "priority": 1,
                                 "position_ids": ids.tolist(), "reused": False})
    manifest = {"schema": "vt.matched_obstacles.1", "identity": identity,
                "scene_value": scene_value, "truth_value": truth_value, "jobs": jobs,
                "base_dataset": digest(original.manifest), "base_data_root": str(original_root.resolve()),
                "common_survey_masks": masks, "reused_empty_receipts": reuse}
    write_json(destination / "dataset.json", manifest)
    write_json(destination / "common_survey_masks.json", masks)
    return manifest


def acquire_matched_obstacles(original_root, destination, config, qd_root, matlab):
    destination, qd_root = Path(destination), Path(qd_root)
    manifest = prepare_matched_obstacles(original_root, destination, config, qd_root)
    store = Store(destination / "store")
    pending = [job for job in manifest["jobs"] if not job["reused"]]
    status = {"state": "acquiring", "total_chunks": len(pending), "completed_chunks": 0,
              "reused_empty_chunks": len(manifest["reused_empty_receipts"]), "failed": []}
    write_json(destination / "acquisition_status.json", status)
    def work(job):
        receipt = run_scenario(destination / "native" / job["key"], qd_root.resolve(), matlab, store)
        return job["key"], receipt["frames"], receipt["elapsed_s"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(work, job): job for job in pending}
        for future in as_completed(futures):
            try:
                key, count, seconds = future.result()
                status["completed_chunks"] += 1
                print({"acquired": key, "positions": count, "seconds": seconds,
                       "completed": status["completed_chunks"], "total": len(pending)}, flush=True)
            except Exception as error:
                status["failed"].append({"key": futures[future]["key"], "error": str(error)})
                status["state"] = "failed"
                write_json(destination / "acquisition_status.json", status)
                for f in futures:
                    f.cancel()
                raise
            write_json(destination / "acquisition_status.json", status)
    status["state"] = "complete"
    write_json(destination / "acquisition_status.json", status)
    return manifest
