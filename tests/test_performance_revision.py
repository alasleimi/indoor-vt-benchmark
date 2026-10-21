from dataclasses import replace
import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage, fcluster

from indoor_vt.calibration import Search, SearchPending, random_candidates, valid_candidate
from indoor_vt.clustering import memberships, merge
from indoor_vt.evaluation import Evaluator
from indoor_vt.localization import FingerprintLocalizer, InitialConfig
from indoor_vt.reconstruction import MPUrgeReconstructor, ReconstructionConfig
from indoor_vt.stage_cache import StageCache
from test_campaign import fixture_dataset
from test_components import planted


def test_future_candidates_exclude_degenerate_matching_and_preserve_history(tmp_path):
    default = {"half_window": 1, "alpha": 0.5}
    space = {"half_window": {"values": [0, 1]}, "alpha": {"values": [0.0, 0.5]}}
    candidates = list(random_candidates(default, space, np.random.default_rng(9), 30))
    assert len(candidates) == 3
    assert all(valid_candidate(c) for c in candidates)
    assert not valid_candidate({"vt": {"baseline": {"half_window": 0, "alpha": 0}}})
    runner = Search(
        split="calibration",
        track="direct",
        regime="component",
        seed=9,
        budget={"broad": 4, "maximum_expansions": 0, "expansion": 0, "refine": 0},
        dataset_hash="data",
        code_hash="code",
    )
    history = [
        {
            "parameters": {"half_window": 0, "alpha": 0},
            "score": 0.0,
            "phase": "historical",
            "audit": {"original": "retained"},
        }
    ]
    calls = []

    def objective(p):
        calls.append(p)
        return 1.0, {"case_count": 1}

    for _ in range(5):
        try:
            result = runner.run(
                objective,
                default,
                space,
                [],
                tmp_path / "search.json",
                max_evaluations=1,
                history=history,
            )
            break
        except SearchPending:
            pass
    assert len(calls) == 3 and len(result["trials"]) == 4
    assert result["trials"][0]["eligible"] is False
    assert result["selected"]["score"] == 1.0


def test_nested_conditional_parameters_are_active_only_for_dbscan():
    default = {"vt": {"merger": "greedy_median", "dbscan_min_samples": 5}}
    space = {
        "vt.merger": {"values": ["greedy_median", "dbscan"], "categorical": True},
        "vt.dbscan_min_samples": {"values": [2, 10], "when": {"merger": "dbscan"}},
    }
    values = list(random_candidates(default, space, np.random.default_rng(9), 10))
    assert len(values) == 3
    assert all(x["vt"]["dbscan_min_samples"] == 5 for x in values if x["vt"]["merger"] != "dbscan")


def test_clustering_variants_preserve_greedy_memberships_and_exact_hierarchy(tmp_path):
    rng = np.random.default_rng(8)
    points = rng.normal(size=(80, 3))
    records = [{"xyz": x.tolist(), "members": [(i, 0)]} for i, x in enumerate(points)]
    cache = StageCache(tmp_path, "test")
    for radius in (0.5, 1.0, 2.0):
        expected = []
        for i in sorted(range(len(points)), key=lambda i: (*points[i], i)):
            for group in expected:
                if all(np.linalg.norm(points[i] - points[j]) <= radius for j in group):
                    group.append(i)
                    break
            else:
                expected.append([i])
        median = merge(records, radius, "greedy_median", cache=cache)
        mean = merge(records, radius, "greedy_mean", cache=cache)
        assert [x["candidate_ids"] for x in median] == expected
        assert [x["candidate_ids"] for x in mean] == expected
        for ids, row in zip(expected, mean):
            np.testing.assert_array_equal(row["xyz"], points[ids].mean(0))
        actual = memberships(points, radius, "agglomerative", cache=cache)
        labels = fcluster(linkage(points, method="complete"), t=radius, criterion="distance")
        assert {tuple(x) for x in actual} == {
            tuple(np.flatnonzero(labels == label)) for label in set(labels)
        }
    assert len(list((tmp_path / "hierarchies").glob("*.json.gz"))) == 1
    assert cache.hits >= 5
    line = np.array([[0, 0, 0], [0.4, 0, 0], [0.8, 0, 0]])
    assert len(memberships(line, 0.5, "greedy_median")) == 2
    assert len(memberships(line, 0.5, "agglomerative")) == 2
    assert memberships(np.empty((0, 3)), 0.5) == []
    assert memberships(line[:1], 0.5, "agglomerative") == [[0]]


def test_dbscan_rejects_isolated_fits_and_keeps_density_connected_clusters():
    points = np.array([[x, 0, 0] for x in (0, 0.125, 0.25, 2, 2.125, 2.25, 9)])
    assert memberships(points, 0.3, "dbscan", 3) == [[0, 1, 2], [3, 4, 5]]
    # eps bounds neighborhoods, not the diameter of the entire density-connected cluster.
    assert memberships(np.array([[x, 0, 0] for x in (0, 0.4, 0.8)]), 0.5, "dbscan", 2) == [
        [0, 1, 2]
    ]


@pytest.mark.parametrize("method", ["mpurge_map", "mpurge", "mca", "chamfer"])
def test_channel_cache_reuses_ab_and_neighbors_with_identical_predictions(
    tmp_path, monkeypatch, method
):
    fixture_dataset(tmp_path)
    evaluator = Evaluator(tmp_path, "development", "calibration", "direct")
    room = evaluator.rooms[0]
    config = InitialConfig(method=method)
    from dataclasses import asdict

    for subset in (("A",), ("B",), ("A", "B")):
        query = evaluator.query_cases(room, subset, "0.0")[0][2]
        survey = evaluator.survey(room, subset, "0.0")
        for k in (1, 3, 5):
            parameters = asdict(replace(config, neighbors=k))
            reference = FingerprintLocalizer(InitialConfig(**parameters)).fit(survey).predict(query)
            actual = evaluator.coarse_predictions(parameters, room, subset, "0.0")
            np.testing.assert_array_equal(actual["predictions"][0]["xyz"], reference.xyz)
            assert actual["predictions"][0]["status"] == reference.status
    assert len(list((tmp_path / "runs/cache/channel_scores").glob("*.npz"))) == 2

    def fail(*args, **kwargs):
        raise AssertionError("Scores were recomputed when only k changed")

    monkeypatch.setattr(FingerprintLocalizer, "score_pair", fail)
    evaluator.coarse_predictions(asdict(replace(config, neighbors=7)), room, ("A", "B"), "0.0")


def test_reconstruction_reuses_graph_and_raw_fits_across_downstream_settings(tmp_path):
    survey, source = planted()
    cache = StageCache(tmp_path, "test")
    cfg = ReconstructionConfig(scope="full")
    first = MPUrgeReconstructor(cfg, stage_cache=cache).fit(survey)
    second = MPUrgeReconstructor(replace(cfg, beta=0, merge_radius_m=1), stage_cache=cache).fit(
        survey
    )
    np.testing.assert_array_equal(first.by_ap["A"], second.by_ap["A"])
    assert cache.hits >= 2
    assert len(list((tmp_path / "graphs").glob("*.json.gz"))) == 1
    assert len(list((tmp_path / "raw_fits").glob("*.json.gz"))) == 1
