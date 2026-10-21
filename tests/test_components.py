import json
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from indoor_vt.contracts import DelayObservation, Survey, SourceMap, Prediction
from indoor_vt.matching import PatternMatcher, MatcherConfig
from indoor_vt.reconstruction import MPUrgeReconstructor, ReconstructionConfig, graph_groups
from indoor_vt.diffassign import DiffAssign, DiffAssignConfig, objective
from indoor_vt.fitting import fit_source
from indoor_vt.localization import SmartLEA


def planted():
    xyz = np.array(
        [
            [0, 0, 1],
            [2, 0, 1],
            [0, 2, 1],
            [2, 2, 1],
            [0, 0, 1.5],
            [2, 0, 1.5],
            [0, 2, 1.5],
            [2, 2, 1.5],
        ]
    )
    source = np.array([-3.0, 4.0, -2.0])
    rows = np.linalg.norm(xyz - source, axis=1)
    return Survey(xyz, tuple(DelayObservation({"A": [r]}) for r in rows)), source


def test_signed_3d_fit_and_full_room_all_pairs():
    survey, source = planted()
    reconstructed = MPUrgeReconstructor(ReconstructionConfig(scope="full")).fit(survey)
    np.testing.assert_allclose(reconstructed.by_ap["A"], source[None], atol=1e-7)
    assert reconstructed.diagnostics["channels"]["A"]["pair_count"] == 28
    fit, audit = fit_source(survey.xyz[:4], [o.channels["A"][0] for o in survey.observations[:4]])
    assert fit is None and audit["status"] == "insufficient_3d_geometry"


def test_diffassign_gradient_and_fixed_cardinality():
    survey, source = planted()
    initial = np.array([source + [0.1, 0.1, 0.1], [1.0, 4.0, 2.0]])
    rows = [o.channels["A"] for o in survey.observations]
    config = DiffAssignConfig(unmatched_cost=100)
    args = (survey.xyz, rows, config)
    numeric = approx_derivative(lambda x: objective(x, *args)[0], initial.ravel()).ravel()
    analytic = objective(initial.ravel(), *args)[1]
    np.testing.assert_allclose(numeric, analytic, atol=2e-6)
    result = DiffAssign(config).fit(survey, SourceMap({"A": initial}))
    assert result.by_ap["A"].shape == initial.shape
    empty = DiffAssign().fit(survey, SourceMap({"A": []}))
    assert empty.bag().shape == (0, 3)


def test_triangle_gate_and_ap_safe_lea_fallback():
    matcher = PatternMatcher()
    assert len(matcher([1], [10], separation=1, tolerance=0)) == 0
    assert len(matcher([1], [1], separation=5, tolerance=0)) == 0
    initial = Prediction(np.array([1.0, 2.0, 1.0]))
    query = DelayObservation({"A": [1, 2, 3, 4]})
    sources = SourceMap({"B": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    result = SmartLEA().predict(query, initial, sources)
    assert result.status.startswith("fallback")
    np.testing.assert_array_equal(result.xyz, initial.xyz)


def test_matcher_published_adapter_fixtures():
    path = Path(__file__).parent / "fixtures/matching.json"
    fixtures = json.loads(path.read_text())
    matcher = PatternMatcher(MatcherConfig(normalized_pattern=True, alpha=0.7))
    for f in fixtures["cases"]:
        got = matcher.match_array(f["a"], f["b"])
        np.testing.assert_allclose(got, np.asarray(f["matches"]).reshape(-1, 3), atol=1e-12)


def test_group_conflict_resolution_has_one_peak_per_reference():
    edges = [
        ((0, 0), (1, 0), 1.0, 1.0),
        ((1, 0), (2, 0), 2.0, 1.0),
        ((2, 0), (0, 1), 3.0, 1.0),
        ((2, 0), (3, 0), 1.0, 1.0),
    ]
    for group in graph_groups(edges, "components"):
        assert len({node[0] for node in group}) == len(group)
