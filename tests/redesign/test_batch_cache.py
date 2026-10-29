"""Scientific grouping contracts and bounded batch reuse."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import time

import numpy as np
import pytest
from numpy.testing import assert_allclose

from indoor_vt.execution.artifacts import ArtifactCache, read_npz
from indoor_vt.execution.localization import LEAExecution
from indoor_vt.execution.presets import smart_lea
from indoor_vt.execution.trials import TrialLog
from indoor_vt.science.grouping import PairMatches, ConnectedComponentsGrouper, group_within_scopes, nearest_neighborhoods
from indoor_vt.science.models import SourceMap, Survey, Ragged, Prediction
from indoor_vt.science.matching import _delta, _windows


def test_cache_ignores_unrelated_implementation_hash_but_tracks_numerical_inputs(tmp_path, monkeypatch):
    from indoor_vt.execution import artifacts
    cache = ArtifactCache(tmp_path, free_bytes=0)
    calls = []
    def calculate():
        calls.append(1)
        return np.array([len(calls)])
    first, key = cache.compute("example.1", {"weight": .35}, {"x": 1}, calculate)
    monkeypatch.setattr(artifacts, "implementation_id", lambda: "unrelated code changed")
    again, same = cache.compute("example.1", {"weight": .35}, {"x": 1}, calculate)
    assert key == same and len(calls) == 1
    assert_allclose(first, again)
    for operation, parameters, inputs in (
        ("example.2", {"weight": .35}, {"x": 1}),
        ("example.1", {"weight": .7}, {"x": 1}),
        ("example.1", {"weight": .35}, {"x": 2}),
    ):
        _, changed = cache.compute(operation, parameters, inputs, calculate)
        assert changed != key
    assert len(calls) == 4


@pytest.mark.parametrize("scale,floor,reuse", [(.1, .03, True), (.01, .03, False), (.01, 0., True)])
def test_old_cache_survives_inactive_floor_removal_only(tmp_path, scale, floor, reuse):
    from copy import deepcopy
    from indoor_vt.execution.artifacts import write_npz, identity
    from indoor_vt.execution.store import digest, component_spec
    from indoor_vt.execution.cache_compatibility import LEGACY_IMPLEMENTATION, ESTIMATOR_WITH_FLOOR
    from indoor_vt.science.estimation import Multilateration3D
    estimator = Multilateration3D(loss_scale_m=scale)
    parameters = {"estimator": estimator}
    old_parameters = deepcopy(component_spec(parameters))
    old_parameters["estimator"]["source"] = ESTIMATOR_WITH_FLOOR
    old_parameters["estimator"]["settings"]["loss_scale_floor_m"] = floor
    inputs = {"xyz": np.arange(12).reshape(4, 3)}
    old_spec = {"operation": "grouped-candidates.2", "parameters": old_parameters,
                "inputs": identity(inputs), "implementation": LEGACY_IMPLEMENTATION}
    old_key = digest(old_spec)
    cache = ArtifactCache(tmp_path, free_bytes=0)
    write_npz(cache.path(old_key), np.array([123]), metadata=old_spec)
    calls = []
    def calculate():
        calls.append(1)
        return np.array([456])
    value, key = cache.compute("grouped-candidates.2", parameters, inputs, calculate)
    assert (key == old_key) is reuse
    assert len(calls) == (0 if reuse else 1)
    assert value[0] == (123 if reuse else 456)


def test_pattern_gap_formula_matches_its_pairwise_definition():
    rng = np.random.default_rng(42)
    a, b = np.sort(rng.uniform(0, 20, 12)), np.sort(rng.uniform(0, 20, 17))
    for p in range(1, 8):
        wa, wb = _windows(a, p), _windows(b, p)
        width = 2 * p + 1
        pattern = np.zeros((len(a), len(b)))
        for left in range(width):
            for right in range(left + 1, width):
                pattern += np.abs(np.abs(wa[:, left] - wa[:, right])[:, None]
                                  - np.abs(wb[:, left] - wb[:, right])[None, :])
        for normalize in (False, True):
            divisor = width * (width - 1) / 2 if normalize else 1
            expected = .7 * np.abs(a[:, None] - b[None, :]) + .3 * pattern / divisor
            assert_allclose(_delta(a, b, p, .7, normalize, -1., 0.), expected, atol=1e-12)
    assert_allclose(_delta(a, b, 7, 1., False, -1., 0.), np.abs(a[:, None] - b[None, :]))


def test_zero_db_receiver_repetitions_remain_distinct(tmp_path, monkeypatch):
    from collections import OrderedDict
    from indoor_vt.acquisition import observations
    from indoor_vt.acquisition.records import Seeds, write_json
    obs = observations.Observations.__new__(observations.Observations)
    obs.root, obs._channels, obs.seeds = tmp_path, OrderedDict(), Seeds(10)
    obs.config = {"observations": {"receiver": {}}}
    obs.manifest = {"jobs": [{"key": "test", "room": "R", "ap": "A", "group": "survey",
                              "position_ids": [0]}]}
    write_json(tmp_path / "native/test/receipt.json", {"native_value": "test"})
    obs._native = lambda _: {"position_ids": [0], "offsets": [0, 1], "paths": np.ones((1, 7))}
    monkeypatch.setattr(observations, "NativeFrame", lambda *args: None)
    monkeypatch.setattr(observations, "receiver_observation",
                        lambda frame, level, rng, config: (np.array([rng.random()]), {}))
    first = obs.channel({"id": "R"}, "A", "survey", [0], 0, 0, "receiver")
    second = obs.channel({"id": "R"}, "A", "survey", [0], 0, 1, "receiver")
    assert not np.array_equal(first.values, second.values)
    clean = obs.channel({"id": "R"}, "A", "survey", [0], 0, 0)
    assert clean is obs.channel({"id": "R"}, "A", "survey", [0], 0, 1)


def graph():
    return PairMatches(np.array([[0, 1], [0, 2], [1, 2]]), np.arange(4),
        np.array([[0, 2], [1, 3], [2, 3]], dtype=np.int32),
        np.array([3., 1., 1.]), np.array([0, 2, 3, 4]))


def test_dsu_conflict_priority_and_local_scope_boundaries():
    matches, grouper = graph(), ConnectedComponentsGrouper()
    assert grouper.group(matches) == [((0, 1), (1, 0), (2, 0))]
    assert group_within_scopes(matches, [(0, 1), (1, 2)], grouper) == [
        ((0, 0), (1, 0)), ((1, 0), (2, 0))]
    assert group_within_scopes(matches, [(0, 1, 2)], grouper,
        allowed_pairs=np.array([[0, 1]])) == [((0, 0), (1, 0))]
    assert grouper.group(matches, np.array([], dtype=int)) == []


def test_neighborhoods_include_boundary_ties():
    xyz = np.array([[0., 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    assert tuple(range(5)) in nearest_neighborhoods(xyz, size=2)
    assert all(len(s) == 2 for s in nearest_neighborhoods(xyz, size=2, include_ties=False))
    # Lattice rounding should not choose a direction because 0.1 is inexact.
    shifted = xyz + np.array([.1, .2, .3])
    assert nearest_neighborhoods(xyz, size=2) == nearest_neighborhoods(shifted, size=2)


def test_batch_cache_single_flight_and_quota(tmp_path):
    cache = ArtifactCache(tmp_path, ram_bytes=8192, disk_bytes=65536, free_bytes=0)
    calls = []
    def calculate():
        calls.append(1)
        time.sleep(.02)
        return {"points": np.arange(1000), "offsets": np.array([0, 1000])}
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: cache.compute("example", {}, {"x": 1}, calculate), range(4)))
    assert len(calls) == 1 and len({key for _, key in results}) == 1
    value, metadata = read_npz(cache.path(results[0][1]))
    assert_allclose(value["points"], np.arange(1000))
    assert metadata["operation"] == "example"
    for i in range(8):
        cache.compute("example", {}, {"x": i + 2}, calculate)
    assert cache.usage()["disk_bytes"] <= 65536
    assert cache.memory_size <= 8192
    assert cache.stats["evicted"] > 0
    assert not list(tmp_path.glob("*/*.tmp"))


def test_inventory_tracks_other_writers_without_directory_rescans(tmp_path):
    first = ArtifactCache(tmp_path, disk_bytes=70000, ram_bytes=0, free_bytes=0)
    second = ArtifactCache(tmp_path, disk_bytes=70000, ram_bytes=0, free_bytes=0)
    for i in range(12):
        cache = first if i % 2 else second
        cache.compute("inventory-example", {}, {"i": i}, lambda: np.arange(2000))
        actual = sum(p.stat().st_size for p in tmp_path.glob("*/*.npz"))
        assert first.usage()["disk_bytes"] == second.usage()["disk_bytes"] == actual
        assert actual <= 70000
    assert first.inventory.scans + second.inventory.scans == 1


def test_inventory_recovers_interrupted_publication(tmp_path):
    from indoor_vt.execution.artifacts import write_npz
    cache = ArtifactCache(tmp_path, free_bytes=0)
    cache.compute("one", {}, {}, lambda: np.arange(20))
    cache.inventory.dirty.touch()
    write_npz(cache.path("f" * 64), np.arange(30))
    recovered = ArtifactCache(tmp_path, free_bytes=0)
    usage = recovered.usage()
    assert usage["files"] == 2
    assert usage["disk_bytes"] == sum(p.stat().st_size for p in tmp_path.glob("*/*.npz"))
    assert usage["inventory_scans"] == 1 and not recovered.inventory.dirty.exists()


def test_smart_lea_reuses_iteration_prefix_and_matches_science(tmp_path):
    sources = SourceMap({"A": np.array([[-2., 0, 0], [2., 0, 0], [0, -2., 0], [0, 2., 0], [0, 0, 3.]])})
    truth = np.array([[.2, .3, .8]])
    queries = Survey(truth, {"A": Ragged.pack([np.linalg.norm(sources.by_ap["A"] - truth[0], axis=1)])}, np.array([0]))
    initial = {"xyz": np.array([[.3, .4, .9]]), "status": ["ok"]}
    cache = ArtifactCache(tmp_path, free_bytes=0)
    lea = smart_lea({"half_window": 2, "alpha": .7, "iterations": 1})
    LEAExecution(lea, cache).predict(queries, initial, sources)
    extended = replace(lea, iterations=4)
    actual, refs = LEAExecution(extended, cache).predict(queries, initial, sources)
    expected = extended.predict({"A": queries.channels["A"][0]}, Prediction(initial["xyz"][0]), sources)
    assert_allclose(actual["xyz"][0], expected.xyz)
    assert actual["status"][0] == expected.status
    assert cache.stats["smart-lea-step.2:computed"] == 4 and len(refs["iterations"]) == 4


def test_trial_log_recovers_only_unfinished_tail(tmp_path):
    log = TrialLog(tmp_path, "stage", {"method": "mpurge"})
    log.append({"id": "first", "method": "mpurge", "loss": .2})
    with log.path.open("ab") as stream:
        stream.write(b'{"id": "interrupted"')
    resumed = TrialLog(tmp_path, "stage", {"method": "mpurge"})
    assert resumed.get("first")["loss"] == .2
    assert resumed.path.read_bytes().endswith(b"\n")


def test_status_publication_retries_transient_windows_lock(tmp_path, monkeypatch):
    from indoor_vt.acquisition import records
    original, calls = records.os.replace, []
    def replace_after_reader_closes(source, destination):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("reader has file open")
        return original(source, destination)
    monkeypatch.setattr(records.os, "replace", replace_after_reader_closes)
    records.write_json(tmp_path / "status.json", {"state": "selecting"})
    assert records.read_json(tmp_path / "status.json")["state"] == "selecting"
    assert len(calls) == 3 and not list(tmp_path.glob("*.tmp"))
