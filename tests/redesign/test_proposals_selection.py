from functools import partial
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from indoor_vt.science.costs import mean
from indoor_vt.science.estimation import Multilateration3D
from indoor_vt.science.grouping import global_scope
from indoor_vt.science.models import Ragged, Survey, SourceMap
from indoor_vt.science.ransac import (ThreeSphereIntersections, TopPerTrialBatch, GlobalTopK,
                                     RansacReconstructor, uniform_peak_sample)
from indoor_vt.science.selection import SupportFirstScorer, GreedySourceSelector
from indoor_vt.science.zayets import (split_at_gaps, LongestClusterTemplateAssigner,
    ConstrainedSingleLinkageGrouper, ZayetsReconstructor, cluster_feature_cost)


def test_gap_split_and_mean_template_assignment():
    assert [ids.tolist() for ids in split_at_gaps(np.array([1., 1.2, 3., 3.1]), .6)] == [[0, 1], [2, 3]]
    calls = []
    def reducer(values):
        result = mean(values)
        calls.append(result)
        return result
    tracks = LongestClusterTemplateAssigner(template_reducer=reducer).assign([
        np.array([1., 3., 6.]), np.array([2., 4., 7.]), np.array([3.4, 6.4])])
    assert_allclose(calls[0], [1.5, 3.5, 6.5])
    assert tracks == [[(0, 0), (1, 0)], [(0, 1), (1, 1), (2, 0)], [(0, 2), (1, 2), (2, 1)]]


def test_kruskal_ignores_cycles_and_enforces_fingerprint_constraint():
    owners = [0, 1, 2, 0]
    edges = [(.1, 0, 1), (.2, 1, 2), (.3, 0, 2), (.4, 2, 3)]
    assert ConstrainedSingleLinkageGrouper(.5).group(owners, edges) == [[0, 1, 2], [3]]


def test_three_spheres_preserve_both_signed_solutions():
    xyz = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    source = np.array([2., 3., 4.])
    points = ThreeSphereIntersections().solve(xyz, np.linalg.norm(xyz - source, axis=1))
    assert_allclose(points, [[2, 3, 4], [2, 3, -4]])
    assert ThreeSphereIntersections().solve(np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]]), np.ones(3)) == []


def test_retention_counts_rejected_trials_and_global_top_is_bounded():
    def p(score, order):
        return {"score": score, "order": order}
    trials = [[p(1., 0)], [], [p(5., 1), p(4., 2)], [p(3., 3)]]
    assert [r["order"] for r in TopPerTrialBatch(2, 1).retain(iter(trials))] == [0, 1]
    assert [r["order"] for r in GlobalTopK(2).retain(iter(trials))] == [1, 2]


def test_selector_consumes_only_its_private_evidence_and_keeps_positions():
    xyz = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    source = np.array([2., 3., 4.])
    nearby = source + np.array([.04, 0, 0])
    rows = Ragged.pack([[np.linalg.norm(p - source)] for p in xyz])
    before = rows.values.copy()
    survey = Survey(xyz, {"A": rows}, np.arange(len(xyz)))
    candidates = SourceMap({"A": np.array([source, nearby])})
    config = dict(scorer=SupportFirstScorer(.1), min_support_positions=4,
                  duplicate_tolerance_m=0, min_support_fraction=0)
    consumed = GreedySourceSelector(**config, consume_support=True).select(survey, candidates)
    retained = GreedySourceSelector(**config, consume_support=False).select(survey, candidates)
    assert_allclose(consumed.by_ap["A"], [source])
    assert_allclose(retained.by_ap["A"], candidates.by_ap["A"])
    assert_array_equal(before, rows.values)


def test_proposal_generators_return_sources_without_a_selector():
    xyz = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    source = np.array([2., 3., 4.])
    rows = Ragged.pack([[np.linalg.norm(p - source)] for p in xyz])
    survey = Survey(xyz, {"A": rows}, np.arange(len(xyz)))
    zayets = ZayetsReconstructor(global_scope, partial(split_at_gaps, threshold=.6),
        cluster_feature_cost, ConstrainedSingleLinkageGrouper(10),
        LongestClusterTemplateAssigner(), Multilateration3D())
    assert_allclose(zayets.fit(survey).by_ap["A"], [source], atol=1e-7)
    ransac = RansacReconstructor(uniform_peak_sample, ThreeSphereIntersections(),
        SupportFirstScorer(.01), GlobalTopK(3), trials=20, seed=1)
    a, b = ransac.fit(survey), ransac.fit(survey)
    assert_allclose(a.by_ap["A"], b.by_ap["A"])
    assert len(a.by_ap["A"]) == 3
    assert np.min(np.linalg.norm(a.by_ap["A"] - source, axis=1)) < 1e-7
