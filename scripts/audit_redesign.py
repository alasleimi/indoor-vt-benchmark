"""Verify completed new-study receipts against saved predictions and frozen inputs."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from indoor_vt.acquisition.observations import Observations
from indoor_vt.acquisition.records import digest, read_json, write_json
from indoor_vt.execution.artifacts import ArtifactCache, file_lock, implementation_id, read_npz
from indoor_vt.execution.experiment import configurations
from indoor_vt.execution.trials import read_trials
from indoor_vt.metrics import source_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
parser.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
args = parser.parse_args()
root = args.run
config, status = read_json(root / "config.json"), read_json(root / "status.json")
assert status["state"] == "complete", status["state"]
version = read_json(root / "implementation.json")
assert version["id"] == implementation_id()
assert version["dataset"] == digest(read_json(args.data / "dataset.json"))
assert version["experiment"] == digest(
    (Path(__file__).parents[1] / "src/indoor_vt/execution/experiment.py").read_text())
trials = read_trials(root / "trials")
selected = {r["label"]: r for r in (read_json(p) for p in (root / "selected").glob("*.json"))}
assert len(selected) == 10
assert all(r["level"] == 0 and r["split"] == "calibration" for r in trials)
assert len({r["id"] for r in trials}) == len(trials) == status["target_trials"]
expected_counts = {}
for method in ("mpurge", "mpurge_map", "smart_lea_truth", "smart_lea", "joint"):
    space = "smart_lea" if method == "smart_lea_truth" else method
    count = (config["optimization"]["joint_trials"] if method == "joint"
             else len(configurations(config["optimization"][space])))
    for mode in ("unified", "ununified"):
        expected_counts[method + "_" + mode] = count
assert Counter(r["label"] for r in trials) == expected_counts
for label, choice in selected.items():
    rows = [r for r in trials if r["label"] == label]
    assert choice["loss"] == min(r["loss"] for r in rows)
    assert choice["loss"] == choice["metrics"]["ospa" if choice["method"] == "mpurge" else "mean_m"]
    mode = "unified" if choice["unify"] else "ununified"
    if choice["method"] in ("smart_lea", "smart_lea_truth"):
        assert choice["frozen"]["map"] == selected["mpurge_map_" + mode]["parameters"]
    if choice["method"] == "smart_lea":
        assert choice["frozen"]["mpurge"] == selected["mpurge_" + mode]["parameters"]
    if choice["method"] == "joint":
        baseline = next(r for r in rows if r["number"] == 0)
        assert baseline["parameters"] == {
            "mpurge": selected["mpurge_" + mode]["parameters"],
            "map": selected["mpurge_map_" + mode]["parameters"],
            "lea": selected["smart_lea_" + mode]["parameters"]}
        assert choice["loss"] <= baseline["loss"]
        assert choice["frozen"] == {}
observations = Observations(args.data, config)
rooms = {room["id"]: room for room in observations.rooms}
assert len(rooms) == 1 and config["room_families"] == ["rectangle"]
assert config["observations"]["track"] == "direct"
assert config["survey"]["spacing_m"] == .5 and config["scene"]["survey_spacing_m"] == .1
survey = observations.survey(next(iter(rooms.values())), 0., 0)
assert survey.xyz.shape == (280, 3)
for axis in (0, 1):
    assert_allclose(np.diff(np.unique(survey.xyz[:, axis])), .5, atol=1e-12)
results = read_json(root / "results.json")
expected_conditions = {(label, split, level) for label in selected
    for split, levels in (("calibration", [0.0]), ("evaluation", config["observations"]["noise_levels"]))
    for level in levels}
assert {(r["label"], r["split"], r["level"]) for r in results} == expected_conditions
assert len(results) == len(expected_conditions)
maximum = max(r["max_mpurge_seconds"] for r in trials + results)
assert maximum <= config["runtime"]["mpurge_limit_seconds"]
query_predictions = 0
for receipt in results:
    choice = selected[receipt["label"]]
    if receipt["split"] == "calibration":
        assert_allclose(receipt["loss"], choice["loss"], rtol=1e-10, atol=1e-12)
    for key in ("method", "parameters", "unify", "frozen"):
        assert receipt[key] == choice[key]
    detail, metadata = read_npz(root / receipt["outcome"])
    for key in ("method", "parameters", "unify", "level", "split", "frozen"):
        assert metadata[key] == receipt[key]
    repetitions = config["observations"][receipt["split"] + "_repetitions"]
    if receipt["level"] == 0:
        repetitions = repetitions[:1]
    expected_sources = ({(room, repetition) for room in rooms for repetition in repetitions}
                        if receipt["method"] in ("mpurge", "smart_lea", "joint") else set())
    expected_queries = ({(room["id"], layout["id"], repetition) for room in rooms.values()
                         for layout in room["layouts"] for repetition in repetitions}
                        if receipt["method"] != "mpurge" else set())
    assert {(r["room"], r["repetition"]) for r in detail["sources"]} == expected_sources
    assert len(detail["sources"]) == len(expected_sources)
    assert {(r["room"], r["layout"], r["repetition"]) for r in detail["queries"]} == expected_queries
    assert len(detail["queries"]) == len(expected_queries)
    if receipt["method"] == "mpurge":
        measures = []
        for record in detail["sources"]:
            truth = observations.truth(rooms[record["room"]])
            for aps in (("A",), ("B",), ("A", "B")):
                measures.append(source_metrics(np.concatenate([record["by_ap"][ap] for ap in aps]),
                    np.concatenate([truth.by_ap[ap] for ap in aps])))
        for name in ("ospa", "f1"):
            assert_allclose(receipt["metrics"][name], np.mean([m[name] for m in measures]), atol=1e-12)
    else:
        errors, failures = [], []
        for record in detail["queries"]:
            room = rooms[record["room"]]
            layout = next(item for item in room["layouts"] if item["id"] == record["layout"])
            boundary = int(len(layout["queries_xyz"]) * config["observations"]["query_calibration_fraction"])
            expected = (np.arange(boundary) if receipt["split"] == "calibration"
                        else np.arange(boundary, len(layout["queries_xyz"])))
            assert np.array_equal(record["query_ids"], expected)
            assert record["xyz"].shape == (len(expected), 3)
            assert np.isfinite(record["xyz"]).all()
            actual_errors = np.linalg.norm(record["xyz"] - layout["queries_xyz"][expected], axis=1)
            assert_allclose(actual_errors, record["errors_m"], atol=1e-12)
            errors.extend(actual_errors)
            failures.extend(record["status"] != "ok")
            query_predictions += len(expected)
        errors = np.asarray(errors)
        actual = {"mean_m": errors.mean(), "rmse_m": np.sqrt(np.mean(errors**2)),
                  "median_m": np.median(errors), "p90_m": np.quantile(errors, .9),
                  "fallback_fraction": np.mean(failures)}
        for name, value in actual.items():
            assert_allclose(receipt["metrics"][name], value, atol=1e-12)
cache = ArtifactCache(args.data / "numerical-cache-v2")
with file_lock(cache.directory / "locks" / "budget.lock"):
    cache.inventory.sync()
    cache_inventory = cache.inventory.usage()
    cache_bytes = sum(p.stat().st_size for p in cache.directory.glob("*/*.npz"))
assert cache_inventory["disk_bytes"] == cache_bytes
outcome_bytes = sum(p.stat().st_size for p in (root / "outcomes").glob("*.npz"))
audit_bytes = sum(p.stat().st_size for p in (root / "audit").glob("*/*.npz"))
trial_bytes = sum(p.stat().st_size for p in (root / "trials").iterdir() if p.is_file())
assert cache_bytes <= config["storage"]["cache_mib"] * 1024**2
assert outcome_bytes <= config["storage"]["outcomes_mib"] * 1024**2
assert audit_bytes <= config["storage"]["audit_mib"] * 1024**2
report = {"verified": True, "selection_trials": len(trials), "selected_entries": len(selected),
          "conditions": len(results), "query_predictions_checked": query_predictions,
          "max_mpurge_seconds": maximum, "cache_bytes": cache_bytes,
          "cache_inventory_bytes": cache_inventory["inventory_bytes"],
          "outcome_bytes": outcome_bytes, "audit_bytes": audit_bytes, "trial_log_bytes": trial_bytes,
          "implementation": version}
write_json(root / "completion-audit.json", report)
print(report)
