"""Scientific and storage contracts for the new implementation only."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import time

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from shapely.geometry import Point, Polygon

from indoor_vt.acquisition.dataset import survey_indices
from indoor_vt.acquisition.geometry import room_design
from indoor_vt.acquisition.records import Seeds, read_json
from indoor_vt.execution.presets import standard_mpurge
from indoor_vt.execution.reconstruction import MPUrgeExecution
from indoor_vt.execution.localization import LocalizerExecution
from indoor_vt.execution.store import Store, Ref
from indoor_vt.execution.artifacts import ArtifactCache
from indoor_vt.science.estimation import Multilateration3D
from indoor_vt.science.grouping import global_scope, NeighborhoodPairs
from indoor_vt.science.matching import MPUrgeMatcher
from indoor_vt.science.models import Ragged, Survey
from indoor_vt.science.localization import FingerprintLocalizer, MatchBasedScorer, DistanceWeightedKNN
from dataclasses import replace


def test_dense_lattice_subset_and_ap_clearance():
    config = read_json("configs/redesign.json")
    room = room_design("rectangle", "scene", 0, config["scene"], Seeds(config["seed"]))
    polygon = Polygon(room["polygon_xy"])
    for xyz in room["access_points_xyz"].values():
        assert polygon.boundary.distance(Point(xyz[:2])) >= .05
    indices = survey_indices(room, spacing_m=.5, phase_xy=(2, 2))
    lattice = room["survey_lattice"][indices]
    assert np.all(lattice[:, :2] % 5 == 2)
    assert set(lattice[:, 2]) == {0, 1}
    assert_array_equal(room["survey_position_ids"][indices], indices)
    for axis in (0, 1):
        assert_allclose(np.diff(np.unique(room["survey_xyz"][indices, axis])), .5)


def test_matching_reports_original_peak_indices():
    left, right = np.array([4., 1., 3.]), np.array([4.1, 1.1])
    result = MPUrgeMatcher().match(left, right)
    pairs = result["matches"][:, :2].astype(int)
    assert set(map(tuple, pairs)) == {(0, 0), (1, 1)}
    assert_allclose(abs(left[pairs[:, 0]] - right[pairs[:, 1]]), .1)
    assert len(MPUrgeMatcher(size_unification_norm=None).match(left, right)["left_indices"]) == 3


def test_signed_3d_estimate_and_degenerate_geometry():
    anchors = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
    source = np.array([-2., 3., -4.])
    point, detail = Multilateration3D().fit(anchors, np.linalg.norm(anchors - source, axis=1))
    assert detail["status"] == "ok"
    assert_allclose(point, source, atol=1e-8)
    point, detail = Multilateration3D().fit(anchors[:3], np.ones(3))
    assert point is None and detail["status"] == "insufficient_3d_geometry"


def test_store_shares_values_and_computations_across_runs(tmp_path):
    store = Store(tmp_path)
    values = np.arange(6).reshape(2, 3)
    child = store.put(values)
    parent = store.put({"input": Ref(child), "again": values.copy()})
    assert store.describe(parent)["value"] == {"input": {"$value": child}, "again": {"$array": child}}
    assert_array_equal(store.get(parent)["input"], values)
    calls, guard = [], threading.Lock()
    def calculate():
        with guard:
            calls.append(1)
        time.sleep(.05)
        return values * 2
    def worker(run):
        try:
            return store.compute("double.1", {}, {"x": values}, calculate, run=run)
        finally:
            store.close()
    with ThreadPoolExecutor(2) as pool:
        a, b = list(pool.map(worker, ["first", "second"]))
    assert len(calls) == 1 and a[1] == b[1]
    assert_array_equal(a[0], values * 2)
    store.close()


def test_instrumented_and_readable_new_pipeline_agree(tmp_path):
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.], [1, 1, 1]])
    true = np.array([[-2., 3., 4.], [3., -1., 2.]])
    rows = Ragged.pack([np.sort(np.linalg.norm(true - p, axis=1)) for p in xyz])
    survey = Survey(xyz, {"A": rows}, np.arange(len(xyz)))
    algorithm = standard_mpurge({"neighbors": 5, "half_window": 1, "alpha": .5,
                                "radius_m": .1, "beta": 0}, unify=True)
    scopes = partial(global_scope)
    algorithm = replace(algorithm, pair_selector=NeighborhoodPairs(scopes), group_scope_selector=scopes)
    expected = algorithm.fit(survey)
    store = ArtifactCache(tmp_path, free_bytes=0)
    actual = MPUrgeExecution(algorithm, store).fit(survey)
    assert_allclose(actual.by_ap["A"], expected.by_ap["A"])
    assert len(actual.by_ap["A"]) > 0
    count = store.stats["pair-matches.2:computed"]
    without_unification = replace(algorithm, matcher=replace(algorithm.matcher, size_unification_norm=None))
    reused = MPUrgeExecution(without_unification, store).fit(survey)
    assert_allclose(reused.by_ap["A"], actual.by_ap["A"])
    assert store.stats["pair-matches.2:computed"] == count
    store.close()


def test_coverage_and_knn_changes_reuse_match_statistics(tmp_path):
    xyz = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0]])
    reference = Ragged.pack([[1., 3.], [1.1, 3.2, 4.], [2., 4.]])
    query = Ragged.pack([[1.05, 3.1], []])
    survey = Survey(xyz, {"A": reference}, np.arange(3))
    queries = Survey(xyz[:2], {"A": query}, np.arange(2))
    store = ArtifactCache(tmp_path, free_bytes=0)
    for coverage, k in ((True, 1), (False, 2)):
        localizer = FingerprintLocalizer(MatchBasedScorer(MPUrgeMatcher(), coverage), DistanceWeightedKNN(k))
        result, _ = LocalizerExecution(localizer, store).predict(survey, queries)
        localizer.fit(survey)
        expected = [localizer.predict({"A": query[i]}).xyz for i in range(len(query))]
        assert_allclose(result["xyz"], expected)
    assert store.stats["fingerprint-match-statistics.2:computed"] == 1
    store.close()
