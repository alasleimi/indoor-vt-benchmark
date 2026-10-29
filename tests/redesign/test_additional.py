from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from indoor_vt.execution.additional import (zayets_proposals, zayets_one, ransac_proposals,
    ransac_pool, retain_ransac, finisher, finish_one)
from indoor_vt.science.models import Ragged, SourceMap, Survey
from indoor_vt.execution.artifacts import identity
from indoor_vt.execution.additional import ROBUST_FINISHER_PARAMETERS, score_many


def example():
    xyz = np.array([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.], [0., 0., .5],
                    [.5, .5, .5], [.5, 0., .5], [0., .5, .5], [.5, .5, 0.]])
    sources = np.array([[2., 3., 4.], [-3., 2., 1.]])
    rows = Ragged.pack([np.sort(np.linalg.norm(sources - x, axis=1)) for x in xyz])
    return Survey(xyz, {"A": rows}, np.arange(len(xyz))), sources


def test_zayets_execution_preserves_reference_proposals():
    survey, _ = example()
    p = {"scope_radius_m": 1., "gap_m": .6, "group_cost": .5}
    algorithm = zayets_proposals(p)
    expected = algorithm.fit(survey).by_ap["A"]
    actual = zayets_one(survey, algorithm)["xyz"]
    assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_ransac_pool_and_retention_preserve_seed_order_and_prefix():
    survey, _ = example()
    algorithm = ransac_proposals({"trials": 500, "tolerance_m": .18}, seed=4)
    expected = algorithm.fit(survey)
    pool = ransac_pool(survey, algorithm)
    actual = retain_ransac(survey, pool, algorithm)
    assert_array_equal(pool["samples"], expected.evidence["A"]["sample_members"])
    assert_allclose(actual["xyz"], expected.by_ap["A"], atol=1e-10)
    assert_allclose(actual["scores"], expected.evidence["A"]["scores"], atol=1e-10)
    short = replace(algorithm, trials=200)
    assert_allclose(retain_ransac(survey, pool, short)["xyz"], short.fit(survey).by_ap["A"])


def test_consuming_finishers_match_reference_and_leave_observations_untouched():
    survey, sources = example()
    points = np.vstack([sources, sources + .01, [4., 2., 8.]])
    before = survey.channels["A"].values.copy()
    candidates = SourceMap({"A": points})
    for family in ("zayets", "ransac"):
        selector = finisher(family, {"tolerance_m": .18, "support_fraction": .08,
                                     "duplicate_m": .12})
        expected = selector.select(survey, candidates)
        actual = finish_one(survey, candidates, selector)
        assert_allclose(actual["xyz"], expected.by_ap["A"], atol=1e-10)
        assert [h["status"] for h in actual["selection"]["history"]] == [
            h["status"] for h in expected.evidence["selection"]["A"]["history"]]
    assert_array_equal(before, survey.channels["A"].values)


def test_finishers_preserve_noisy_support_consumption_and_refitting():
    rng = np.random.default_rng(16)
    xyz = rng.uniform(-2, 2, (20, 3))
    sources = rng.uniform(-4, 4, (5, 3))
    rows = Ragged.pack([np.sort(np.linalg.norm(sources - x, axis=1)
                               + rng.normal(0, .04, len(sources))) for x in xyz])
    survey = Survey(xyz, {"A": rows}, np.arange(len(xyz)))
    candidates = SourceMap({"A": np.vstack((sources + rng.normal(0, .06, sources.shape),
        sources + .01, rng.uniform(-5, 5, (8, 3))))})
    for family in ("zayets", "ransac"):
        selector = finisher(family, {"tolerance_m": .18, "support_fraction": .16,
                                     "duplicate_m": .12})
        expected = selector.select(survey, candidates)
        actual = finish_one(survey, candidates, selector)
        assert_allclose(actual["xyz"], expected.by_ap["A"], rtol=1e-10, atol=1e-10)
        actual_history = actual["selection"]["history"]
        expected_history = expected.evidence["selection"]["A"]["history"]
        assert len(actual_history) == len(expected_history)
        for a, b in zip(actual_history, expected_history):
            assert a["status"] == b["status"] and a["proposal"] == b["proposal"]
            assert_allclose(a["score"], b["score"], rtol=1e-12, atol=1e-12)
            if "support" in a:
                assert_array_equal(a["support"], b["support"])


def test_robust_finisher_parameters_change_ranking_and_cache_identity():
    p = {"tolerance_m": .1, "support_fraction": .04, "duplicate_m": .06}
    original = finisher("ransac", p)
    errors = np.array([[0., .2, .25, .5, 1., 2.], [.06, .06, .06, .06, .18, .18]])
    original_scores = score_many(errors, original.scorer)
    for key, value in zip(ROBUST_FINISHER_PARAMETERS, (.7, .2, 2.)):
        changed = finisher("ransac", {**p, key: value})
        scores = score_many(errors, changed.scorer)
        assert getattr(changed.scorer, key) == value
        assert np.any(scores != original_scores)
        assert identity(changed) != identity(original)
        assert_allclose(scores, [changed.scorer(row) for row in errors])
    assert [getattr(original.scorer, k) for k in ROBUST_FINISHER_PARAMETERS] == [.35, .1, 4.]
    no_terms = finisher("ransac", {**p, "smooth_weight": 0., "median_penalty_weight": 0.})
    assert_array_equal(score_many(errors, no_terms.scorer), (errors <= .1).sum(axis=1))
    # These parameters do not change RANSAC proposal retention or the other finisher.
    changed_p = {**p, "smooth_weight": .7, "median_penalty_weight": .2, "median_clip_factor": 8.}
    assert ransac_proposals({**p, "trials": 100}, 4) == ransac_proposals({**changed_p, "trials": 100}, 4)
    assert finisher("zayets", p) == finisher("zayets", changed_p)
