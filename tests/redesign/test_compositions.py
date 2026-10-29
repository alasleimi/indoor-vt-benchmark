import numpy as np
import pytest
from numpy.testing import assert_allclose

from indoor_vt.execution.artifacts import ArtifactCache
from indoor_vt.execution.compositions import CompositionExecution, composed_reconstructor
from indoor_vt.execution.composition_experiment import source_schedule
from indoor_vt.science.models import Survey, Ragged, SourceMap
from indoor_vt.science.reconstruction import match_pairs, estimate_sources
from indoor_vt.science.grouping import group_within_scopes


@pytest.mark.parametrize("method", ["zayets_robust", "ransac_consumption",
    "mpurge_consumption", "mpurge_robust", "mpurge_triangle", "zayets_naked", "ransac_naked",
    "mpurge_allow_crossing", "mpurge_allow_crossing_consumption", "mpurge_allow_crossing_robust"])
def test_compositions_execute_the_injected_scientific_components(tmp_path, method):
    rng = np.random.default_rng(23)
    xyz = rng.uniform(-1, 1, (12, 3))
    truth = np.array([[2., 3., 4.], [-3., 2., 1.]])
    rows = Ragged.pack([np.sort(np.linalg.norm(truth - x, axis=1)) for x in xyz])
    survey = Survey(xyz, {"A": rows}, np.arange(len(xyz)))
    p = {"half_window": 2, "alpha": .5, "neighbors": 10, "radius_m": .5, "beta": 1.,
         "triangle_tolerance_m": 0., "scope_radius_m": 2., "gap_m": .6, "group_cost": .5,
         "trials": 100, "tolerance_m": .18, "support_fraction": .08, "duplicate_m": .12,
         "smooth_weight": .7, "median_penalty_weight": .2, "median_clip_factor": 8.}
    algorithm = composed_reconstructor(method, p, unify=False, seed=4)
    if method in ("mpurge_triangle", "mpurge_allow_crossing") or method.endswith("_naked"):
        expected = algorithm.fit(survey)
    else:
        c = algorithm.proposals
        if method.startswith("mpurge"):
            pairs, scopes = c.pair_selector(xyz), c.group_scope_selector(xyz)
            edges = match_pairs(survey, pairs, c.matcher)
            groups = {ap: group_within_scopes(e, scopes, c.grouper) for ap, e in edges.items()}
            candidates, _ = estimate_sources(survey, groups, c.estimator)
            proposals = SourceMap({ap: np.asarray([r["xyz"] for r in records]).reshape(-1, 3)
                                   for ap, records in candidates.items()})
        else:
            proposals = c.fit(survey)
        expected = algorithm.selector.select(survey, proposals)
    store = ArtifactCache(tmp_path, disk_bytes=32 * 1024**2, ram_bytes=8 * 1024**2, free_bytes=0)
    actual = CompositionExecution(method, p, store, unify=False, seed=4, max_trials=100).fit(survey)
    assert_allclose(actual.by_ap["A"], expected.by_ap["A"], rtol=1e-10, atol=1e-10)
    if method.endswith("_naked"):
        assert "selection" not in actual.evidence["A"]


@pytest.mark.parametrize("method", ["zayets_naked", "ransac_naked"])
def test_naked_output_preserves_all_candidates_including_duplicates(tmp_path, monkeypatch, method):
    from indoor_vt.execution import compositions
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    survey = Survey(xyz, {"A": Ragged.pack([np.array([1., 2.])] * 4)}, np.arange(4))
    proposals = np.array([[2., 3., 4.], [2., 3., 4.], [100., 200., 300.]])
    def forbidden(*args, **kwargs):
        raise AssertionError("Naked runs must not construct or call a finisher")
    monkeypatch.setattr(compositions, "finisher", forbidden)
    monkeypatch.setattr(compositions, "finish_one", forbidden)
    monkeypatch.setattr(compositions, "zayets_one", lambda *a: {"xyz": proposals})
    monkeypatch.setattr(compositions, "ransac_pool", lambda *a: {})
    monkeypatch.setattr(compositions, "retain_ransac", lambda *a: {"xyz": proposals})
    p = {"scope_radius_m": 1., "gap_m": .6, "group_cost": .5, "trials": 10, "tolerance_m": .1}
    cache = ArtifactCache(tmp_path, free_bytes=0)
    actual = CompositionExecution(method, p, cache, unify=None, max_trials=10).fit(survey)
    assert_allclose(actual.by_ap["A"], proposals)


def test_sampled_search_is_reproducible_and_does_not_repeat_inactive_windows():
    space = {"alpha": [0., .5, 1.], "half_window": [1, 2, 3], "neighbors": [5, 10]}
    a = source_schedule(space, 10, 17)
    assert a == source_schedule(space, 10, 17)
    assert len({tuple(sorted(p.items())) for p in a}) == 10
    assert all(p["half_window"] == 1 for p in a if p["alpha"] == 1)
