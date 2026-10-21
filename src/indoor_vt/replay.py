"""Independently rerun the five-box pilot acquisition with its recorded seed."""

from pathlib import Path
from .provenance import SeedBook, read_json, write_json
from .simulation import prepare_scenario, run_scenario


def replay_pilot(root, qd_root, matlab):
    root = Path(root)
    key = "T_calibration_00/A/layout_09"
    original = root / "data/pilot/scenarios" / key
    if not (original / "receipt.json").exists():
        return {"status": "not_available", "reason": "Five-box pilot has not been generated"}
    design = read_json(root / "data/pilot/design.json")
    acquisition = read_json(original / "input_manifest.json")
    room = next(r for r in design["rooms"] if r["id"] == acquisition["room_id"])
    layout = next(x for x in room["layouts"] if x["id"] == acquisition["layout"])
    destination = root / "data/pilot_replay/scenarios"
    prepare_scenario(
        destination,
        room,
        acquisition["ap"],
        layout,
        design["config"],
        SeedBook(design["seeds"]["root"], "pilot"),
        Path(qd_root),
        acquisition["position_indices"],
    )
    actual = run_scenario(destination / key, Path(qd_root), matlab)
    expected = read_json(original / "receipt.json")
    result = {
        "status": "exact"
        if actual["native_uncompressed_sha256"] == expected["native_uncompressed_sha256"]
        else "different",
        "original": str(original),
        "replay": str(destination / key),
        "original_native_hash": expected["native_uncompressed_sha256"],
        "replayed_native_hash": actual["native_uncompressed_sha256"],
        "frames": actual["frames"],
        "same_simulator": expected["simulator_tree_hash"] == actual["simulator_tree_hash"],
    }
    write_json(root / "runs/pilot/native_replay.json", result)
    if result["status"] != "exact" or not result["same_simulator"]:
        raise RuntimeError(
            "Independent native replay differs; inspect runs/pilot/native_replay.json"
        )
    return result
