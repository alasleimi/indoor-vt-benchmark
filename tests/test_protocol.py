import numpy as np
import pytest
from indoor_vt.calibration import Search, promotion
from indoor_vt.truth import source_catalogue
from indoor_vt.zayets2019 import dynamic_cosine_clusters, ward_groups
from indoor_vt.consensus import ConsensusConfig, ConsensusReconstructor
from test_components import planted


def test_analytic_shoebox_sources_and_rotation():
    room = {
        "polygon_xy": [[0, 0], [6, 0], [6, 7], [0, 7]],
        "ceiling_m": 3.2,
        "access_points_xyz": {"A": [2, 3, 2.5]},
    }
    original = source_catalogue(room, "A")
    assert len(original) == 25
    assert any(e["surface_sequences"] == [[]] for e in original)
    angle = 0.7
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    room["polygon_xy"] = (np.array(room["polygon_xy"]) @ rot.T).tolist()
    room["access_points_xyz"]["A"] = [*(np.array([2.0, 3.0]) @ rot.T), 2.5]
    rotated = source_catalogue(room, "A")
    expected = np.array([e["xyz"] for e in original])
    expected[:, :2] = expected[:, :2] @ rot.T
    actual = np.array([e["xyz"] for e in rotated])
    from scipy.spatial.distance import cdist

    assert np.max(cdist(expected, actual).min(axis=1)) < 1e-8


def test_2019_embedding_is_nondegenerate_and_no_owner_conflicts():
    values = np.r_[np.linspace(1, 1.01, 6), np.linspace(10, 10.01, 6)]
    clusters = dynamic_cosine_clusters(values, 1.0)
    assert len(clusters) >= 2
    assert sorted(np.concatenate(clusters)) == list(range(len(values)))
    groups = ward_groups(
        np.array([[0.0, 1, 0], [0.1, 1, 0], [0.2, 1, 0], [0.3, 1, 0]]), [0, 1, 0, 2], 100
    )
    for group in groups:
        assert len({[0, 1, 0, 2][i] for i in group}) == len(group)


def test_consensus_recovers_signed_3d_source():
    survey, source = planted()
    for method in ("ransac", "zayets2018"):
        config = ConsensusConfig(
            method=method,
            trials=100,
            batch=20,
            neighborhood_radius_m=5,
            group_threshold=10,
            inlier_scale_m=0.1,
        )
        result = ConsensusReconstructor(config, seed=42).fit(survey)
        np.testing.assert_allclose(result.by_ap["A"], source[None], atol=1e-6)


def test_calibration_rejects_test_and_resumes_adaptive_search(tmp_path):
    kwargs = dict(
        split="calibration",
        track="direct",
        regime="component",
        seed=42,
        budget={"broad": 4, "maximum_expansions": 2, "expansion": 3, "refine": 4},
        dataset_hash="dataset",
        code_hash="source",
    )
    with pytest.raises(ValueError):
        Search(**{**kwargs, "split": "test"})
    space = {"x": {"values": [1.0, 2.0, 4.0], "expand": True, "bounds": [0.1, 100]}}

    def objective(p):
        return (p["x"] - 10) ** 2, {"case_count": 1}

    full = Search(**kwargs).run(objective, {"x": 1.0}, space, [], tmp_path / "full.json")
    calls = []

    def interrupt(p):
        calls.append(p)
        if len(calls) > 5:
            raise RuntimeError("simulated interruption")
        return objective(p)

    with pytest.raises(RuntimeError):
        Search(**kwargs).run(interrupt, {"x": 1.0}, space, [], tmp_path / "resume.json")
    resumed = Search(**kwargs).run(objective, {"x": 1.0}, space, [], tmp_path / "resume.json")
    assert full == resumed
    assert promotion(0.95, 1) and not promotion(0.951, 1) and not promotion(0, 0)
