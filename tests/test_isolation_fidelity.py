import ast
from pathlib import Path
import numpy as np
import pytest
from indoor_vt.contracts import DelayObservation
from indoor_vt.observations import NativeFrame, receiver_observation
from indoor_vt.provenance import read_json
from indoor_vt.reconstruction import merge_candidates


def test_receiver_matches_recorded_published_adapter_outputs():
    fixtures = read_json(Path(__file__).parent / "fixtures/receiver.json")
    config = read_json(Path(__file__).parents[1] / "configs/dataset.json")["receiver"]
    for case in fixtures["cases"]:
        frame = NativeFrame(**{k: np.array(v) for k, v in case["frame"].items()})
        actual, _ = receiver_observation(
            frame, case["snr"], np.random.default_rng(case["seed"]), config
        )
        np.testing.assert_allclose(actual, case["expected"], atol=1e-12)


def test_estimator_modules_have_no_privileged_imports_or_feature_fields():
    base = Path(__file__).parents[1] / "src/indoor_vt"
    for name in (
        "matching",
        "reconstruction",
        "fitting",
        "diffassign",
        "localization",
        "consensus",
        "zayets2019",
        "pipeline",
    ):
        tree = ast.parse((base / (name + ".py")).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in (
                    "truth",
                    "generation",
                    "simulation",
                    "evaluation",
                    "observations",
                    "dataset",
                )
            if isinstance(node, ast.Attribute):
                assert node.attr not in (
                    "access_points_xyz",
                    "polygon_xy",
                    "ceiling_m",
                    "gain_db",
                    "aoa_az_deg",
                    "aoa_el_deg",
                    "queries_xyz",
                )
    observation = DelayObservation({"A": [1.0, 2.0]})
    with pytest.raises((AttributeError, TypeError)):
        observation.true_ap = [0, 0, 0]
    with pytest.raises(ValueError):
        observation.channels["A"][0] = 999


def test_spatial_merge_matches_exhaustive_complete_link():
    rng = np.random.default_rng(42)
    points = rng.normal(size=(300, 3))
    records = [{"xyz": p.tolist(), "members": [(i, 0)]} for i, p in enumerate(points)]
    expected = []
    for i in sorted(range(len(points)), key=lambda i: (*points[i], i)):
        for group in expected:
            if all(np.linalg.norm(points[i] - points[j]) <= 0.5 for j in group):
                group.append(i)
                break
        else:
            expected.append([i])
    actual = merge_candidates(records, 0.5)
    assert [row["candidate_ids"] for row in actual] == expected
