import importlib.util
from pathlib import Path

import numpy as np
import pytest

from indoor_vt import revision_campaign as revision
from indoor_vt.calibration import valid_candidate
from indoor_vt.evaluation import Evaluator
from indoor_vt.localization import SmartLEA, LEAConfig
from indoor_vt.contracts import Prediction
from indoor_vt.provenance import read_json, write_json, digest
from test_campaign import fixture_dataset


def scheduler():
    path = Path(__file__).parents[1] / "scripts/run_revision_campaign.py"
    spec = importlib.util.spec_from_file_location("revision_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scheduler_main_before_shadows_and_round_robin():
    runner = scheduler()
    jobs = runner.build_jobs()
    ids = {job["id"] for job in jobs}
    assert len(ids) == len(jobs)
    assert all(set(job["dependencies"]) <= ids for job in jobs)
    assert all(job["phase"] >= 2 for job in jobs if job["kind"].startswith("shadow"))
    assert all(valid_candidate(job["parameters"]) for job in jobs if job["kind"] == "preset")
    states = {job["id"]: {"state": "pending", "turns": 0} for job in jobs}
    assert runner.eligible_jobs(jobs, states)[0] == 0
    for job in jobs:
        if job["phase"] == 0:
            states[job["id"]]["state"] = "complete"
    phase, available = runner.eligible_jobs(jobs, states)
    assert phase == 1 and all(job["kind"] == "search" for job in available)
    first = available[0]
    states[first["id"]]["turns"] = 1
    assert runner.eligible_jobs(jobs, states)[1][0]["id"] != first["id"]
    states[first["id"]]["state"] = "running"
    for job in jobs:
        if job["phase"] == 1 and job != first:
            states[job["id"]]["state"] = "complete"
    assert runner.eligible_jobs(jobs, states) == (1, [])
    states[first["id"]]["state"] = "complete"
    assert runner.eligible_jobs(jobs, states)[0] == 2


@pytest.mark.parametrize("regime", ["component", "end_to_end"])
def test_revision_candidate_resume_freeze_and_test(tmp_path, monkeypatch, regime):
    fixture_dataset(tmp_path)
    monkeypatch.setattr(revision, "initial_space", lambda name: ({}, []))
    monkeypatch.setattr(revision, "lea_space", lambda: ({}, []))
    monkeypatch.setattr(
        revision, "vt_spec", lambda name: (revision.reconstruction_default(name), {}, [])
    )
    evaluators = {}
    name = "mpurge_map__mpurge_local"
    if regime == "component":
        specs = [("coarse", "mpurge_map"), ("vt", "mpurge_local"), ("lea", name)]
    else:
        specs = [("pipeline", name)]
    for stage, method in specs:
        job = {
            "kind": "search",
            "stage": stage,
            "method": method,
            "track": "direct",
            "regime": regime,
        }
        result = revision.run_job(tmp_path, "development", "test_revision", job, evaluators)
        assert result["state"] == "complete"
        assert revision.run_job(tmp_path, "development", "test_revision", job, evaluators) == result
    job = {"kind": "freeze", "method": name, "track": "direct", "regime": regime}
    frozen = revision.run_job(tmp_path, "development", "test_revision", job, evaluators)
    assert read_json(Path(frozen["artifact"]))["revision"] == "test_revision"
    result = revision.run_job(
        tmp_path, "development", "test_revision", {**job, "kind": "evaluate"}, evaluators
    )
    assert "revisions" in Path(result["artifact"]).parts


def test_refinement_cache_is_identical_and_reused(tmp_path, monkeypatch):
    fixture_dataset(tmp_path)
    e = Evaluator(tmp_path, "development", "calibration", "direct")
    room = e.rooms[0]
    cfg = revision.pipeline_config("mpurge_map", "mpurge_local")
    subset, level = ("A", "B"), "0.0"
    coarse = e.coarse_predictions(cfg["initial"], room, subset, level)
    sources = e.source_map("mpurge_local", cfg["vt"], room, subset, level)
    query = e.query_cases(room, subset, level)[0][2]
    initial = coarse["predictions"][0]
    expected = SmartLEA(LEAConfig(**cfg["lea"])).predict(
        query, Prediction(np.array(initial["xyz"]), initial["status"]), sources
    )
    actual = e.refined_predictions(coarse, sources, cfg["lea"], room, subset, level)
    np.testing.assert_array_equal(actual["predictions"][0]["xyz"], expected.xyz)

    def fail(*args, **kwargs):
        raise AssertionError("Refinement computed twice")

    monkeypatch.setattr(SmartLEA, "predict", fail)
    assert e.refined_predictions(coarse, sources, cfg["lea"], room, subset, level) == actual


def test_explicit_legacy_compatibility_reuses_predictions_and_sources(tmp_path, monkeypatch):
    import gzip
    import json
    from indoor_vt.evaluation import gzip_json
    from indoor_vt.localization import FingerprintLocalizer
    from indoor_vt.reconstruction import MPUrgeReconstructor

    fixture_dataset(tmp_path)
    e = Evaluator(tmp_path, "development", "calibration", "direct")
    room, subset, level = e.rooms[0], ("A",), "0.0"
    cfg = revision.pipeline_config("mpurge_map", "mpurge_local")
    coarse = e.coarse_predictions(cfg["initial"], room, subset, level)
    source = e.source_map("mpurge_local", cfg["vt"], room, subset, level)
    # Construct historical test artifacts with the same numerical algorithm.
    old_identity = {**coarse["identity"], "code": "legacy"}
    write_json(
        e.cache / "coarse" / (digest(old_identity) + ".json"), {**coarse, "identity": old_identity}
    )
    (e.cache / "coarse" / (coarse["cache_key"] + ".json")).unlink()
    source_path = e.cache / "sources" / (source.diagnostics["cache_key"] + ".json.gz")
    with gzip.open(source_path, "rt") as handle:
        payload = json.load(handle)
    old_source = {
        **payload["identity"],
        "code": "legacy",
        "parameters": {
            k: v for k, v in cfg["vt"].items() if k not in ("merger", "dbscan_min_samples")
        },
    }
    gzip_json(
        e.cache / "sources" / (digest(old_source) + ".json.gz"), {**payload, "identity": old_source}
    )
    source_path.unlink()
    e.compatible = [{"code_hash": "legacy", "coarse": True, "sources": True}]

    def fail(*args, **kwargs):
        raise AssertionError("Compatible historical work was recomputed")

    monkeypatch.setattr(FingerprintLocalizer, "score_pair", fail)
    monkeypatch.setattr(MPUrgeReconstructor, "fit", fail)
    actual = e.coarse_predictions(cfg["initial"], room, subset, level)
    assert actual["predictions"] == coarse["predictions"]
    assert actual["reuse"]["code_hash"] == "legacy"
    reused = e.source_map("mpurge_local", cfg["vt"], room, subset, level)
    np.testing.assert_array_equal(reused.by_ap["A"], source.by_ap["A"])
    assert reused.diagnostics["reuse"]["code_hash"] == "legacy"
