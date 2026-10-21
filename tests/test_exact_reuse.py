from dataclasses import replace

import numpy as np
import pytest

from indoor_vt import evaluation
from indoor_vt.contracts import DelayObservation, Prediction, SourceMap
from indoor_vt.evaluation import Evaluator
from indoor_vt.localization import FingerprintLocalizer, InitialConfig, SmartLEA, LEAConfig
from indoor_vt.nearest_scores import nearest_geometry, geometry_scores
from indoor_vt.reconstruction import MPUrgeReconstructor, ReconstructionConfig
from indoor_vt.revision_campaign import pipeline_config
from indoor_vt.stage_cache import StageCache
from test_campaign import fixture_dataset
from test_components import planted


@pytest.mark.parametrize("chunk_bytes", [64, 16 * 1024 * 1024])
def test_batched_nearest_scores_bit_exact_for_ragged_empty_and_duplicate_profiles(chunk_bytes):
    rng = np.random.default_rng(53)
    rows = [np.empty(0), np.array([0.0]), np.ones(9)]
    rows += [np.sort(rng.uniform(0, 30, count)) for count in (1, 2, 3, 7, 8, 9, 24, 65, 129)]
    geometry = nearest_geometry(rows, rows[::-1], maximum_work_bytes=chunk_bytes)
    for method in ("mca", "chamfer"):
        for eps in (0.01, 0.5, 1.6, 10.0, 100.0):
            model = FingerprintLocalizer(InitialConfig(method=method, epsilon_m=eps))
            expected = np.array([[model.score_pair(a, b) for b in rows[::-1]] for a in rows])
            np.testing.assert_array_equal(geometry_scores(geometry, method, eps), expected)


def test_nearest_geometry_reused_across_methods_and_tolerances(tmp_path, monkeypatch):
    fixture_dataset(tmp_path)
    e = Evaluator(tmp_path, "development", "calibration", "direct")
    room = e.rooms[0]
    config = pipeline_config("mca", "mpurge_local")["initial"]
    e.channel_scores(config, room, "A", "0.0")

    def fail(*args, **kwargs):
        raise AssertionError("Nearest-delay geometry computed again")

    monkeypatch.setattr(evaluation, "nearest_geometry", fail)
    for method, eps in (("mca", 0.1), ("mca", 5.0), ("chamfer", 1.0)):
        parameters = {**config, "method": method, "epsilon_m": eps}
        actual = e.channel_scores(parameters, room, "A", "0.0")[0]
        cases = e.query_cases(room, ("A",), "0.0")
        survey = e.survey(room, ("A",), "0.0")
        model = FingerprintLocalizer(InitialConfig(**parameters))
        expected = [
            [
                model.score_pair(query.channels["A"], reference.channels["A"])
                for reference in survey.observations
            ]
            for _, _, query, _ in cases
        ]
        np.testing.assert_array_equal(actual, expected)
    assert len(list((e.cache / "nearest_geometry").glob("*.npz"))) == 1


def test_nearest_index_storage_has_no_peak_count_cap():
    rows = [np.linspace(0, 30, 513), np.array([30.0, 0.0, 14.2])]
    geometry = nearest_geometry(rows, rows[::-1])
    assert geometry["nearest_indices"].dtype == np.uint16
    assert geometry["reference_values"].dtype == np.float64
    for method in ("mca", "chamfer"):
        model = FingerprintLocalizer(InitialConfig(method=method))
        expected = np.array([[model.score_pair(q, r) for r in rows[::-1]] for q in rows])
        np.testing.assert_array_equal(geometry_scores(geometry, method), expected)


@pytest.mark.parametrize("order", [(1, 2, 4), (4, 2, 1)])
def test_lea_iteration_prefixes_match_uncached_and_do_not_repeat_passes(
    tmp_path, monkeypatch, order
):
    fixture_dataset(tmp_path)
    e = Evaluator(tmp_path, "development", "calibration", "direct")
    room, subset, level = e.rooms[0], ("A", "B"), "0.0"
    cfg = pipeline_config("mpurge_map", "mpurge_local")
    coarse = e.coarse_predictions(cfg["initial"], room, subset, level)
    sources = SourceMap(
        {
            "A": np.array([[-2, 0, 2], [7, 1, 3], [2, -6, 4]], dtype=float),
            "B": np.array([[2, 6, 3], [4, -2, 1], [-3, -2, 5]], dtype=float),
        }
    )
    target = np.array(coarse["predictions"][0]["xyz"]) + np.array([0.1, -0.1, 0.1])
    query = DelayObservation(
        {ap: np.linalg.norm(anchors - target, axis=1) for ap, anchors in sources.by_ap.items()}
    )
    case = e.query_cases(room, subset, level)[0]
    monkeypatch.setattr(e, "query_cases", lambda *args: [(case[0], case[1], query, target)])
    initial = Prediction(np.array(coarse["predictions"][0]["xyz"]))
    expected = {
        n: SmartLEA(LEAConfig(**{**cfg["lea"], "iterations": n})).predict(query, initial, sources)
        for n in order
    }
    assert all(row.status == "ok" for row in expected.values())
    original, calls = SmartLEA.predict, []

    def tracked(self, *args, **kwargs):
        calls.append(self.config.iterations)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SmartLEA, "predict", tracked)
    for n in order:
        actual = e.refined_predictions(
            coarse, sources, {**cfg["lea"], "iterations": n}, room, subset, level
        )
        np.testing.assert_array_equal(actual["predictions"][0]["xyz"], expected[n].xyz)
        assert actual["predictions"][0]["status"] == expected[n].status
    assert len(calls) == 4 and all(n == 1 for n in calls)


def test_lea_later_failure_returns_original_coarse_and_remains_terminal(tmp_path, monkeypatch):
    fixture_dataset(tmp_path)
    e = Evaluator(tmp_path, "development", "calibration", "direct")
    room, subset, level = e.rooms[0], ("A",), "0.0"
    cfg = pipeline_config("mpurge_map", "mpurge_local")
    coarse = e.coarse_predictions(cfg["initial"], room, subset, level)
    sources = SourceMap({"A": np.array([[1, 2, 3]])})
    calls = []

    def fail_second(self, query, initial, sources):
        calls.append(initial.xyz.copy())
        if len(calls) == 1:
            return Prediction(initial.xyz + 1)
        return Prediction(initial.xyz.copy(), "fallback_lea_failure")

    monkeypatch.setattr(SmartLEA, "predict", fail_second)
    for n in (1, 2, 4):
        actual = e.refined_predictions(
            coarse, sources, {**cfg["lea"], "iterations": n}, room, subset, level
        )
    assert len(calls) == 2
    assert actual["predictions"][0]["xyz"] == coarse["predictions"][0]["xyz"]
    assert actual["predictions"][0]["status"] == "fallback_lea_failure"


def test_group_membership_reused_before_raw_fit_lookup(tmp_path, monkeypatch):
    from indoor_vt import reconstruction

    survey, _ = planted()
    cache = StageCache(tmp_path, "test")
    config = ReconstructionConfig(scope="full")
    model = MPUrgeReconstructor(config, stage_cache=cache)
    first = model.fit(survey)

    def fail(*args, **kwargs):
        raise AssertionError("Grouping recomputed for a merger change")

    # Keep identity comparison with the default grouping factory meaningful.
    monkeypatch.setattr(reconstruction, "graph_groups", fail)
    model.group_builder = fail
    model.config = replace(config, beta=0, merger="greedy_mean")
    second = model.fit(survey)
    np.testing.assert_array_equal(first.by_ap["A"], second.by_ap["A"])
    assert len(list((tmp_path / "groups").glob("*.json.gz"))) == 1


def test_source_summary_skips_full_audit_and_invalidates_when_replaced(tmp_path, monkeypatch):
    import gzip
    import json
    from indoor_vt.evaluation import gzip_json

    fixture_dataset(tmp_path)
    e = Evaluator(tmp_path, "development", "calibration", "direct")
    cfg = pipeline_config("mpurge_map", "mpurge_local")
    room = e.rooms[0]
    source = e.source_map("mpurge_local", cfg["vt"], room, ("A",), "0.0")
    path = e.cache / "sources" / (source.diagnostics["cache_key"] + ".json.gz")
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    summary = e.source_summary(path)
    original_open = gzip.open

    def fail(*args, **kwargs):
        raise AssertionError("Full audit decompressed again")

    monkeypatch.setattr(gzip, "open", fail)
    assert e.source_summary(path) == summary
    e.source_summaries.clear()
    assert e.source_summary(path) == summary
    monkeypatch.setattr(gzip, "open", original_open)
    payload["sources"]["A"][0][0] += 1
    gzip_json(path, payload)
    assert e.source_summary(path)["sources"] == payload["sources"]


def test_stage_compatibility_requires_explicit_implementation_identity(tmp_path):
    old = StageCache(tmp_path, "old")
    expected = old.get("raw_fits", {"bank": "x"}, lambda: [1, 2, 3])

    def fail():
        raise AssertionError("Compatible stage recomputed")

    new = StageCache(tmp_path, "new", previous_implementations=["old"])
    assert new.get("raw_fits", {"bank": "x"}, fail) == expected
    assert new.hits == 1
    assert StageCache(tmp_path, "different").get("raw_fits", {"bank": "x"}, lambda: [4]) == [4]
