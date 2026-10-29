"""Spatial diagnostics must agree with the scientific pair inspector."""
import numpy as np
import pytest

from indoor_vt.execution.peak_explorer import PeakExplorer, pair_data, spatial_match_counts
from indoor_vt.science.matching import MPUrgeMatcher, CrossingAllowedMPUrgeMatcher


def profile(ranges, labels, xyz):
    return {"ranges": np.asarray(ranges, float), "source_ids": np.asarray(labels, int),
            "xyz": np.asarray(xyz, float)}


@pytest.mark.parametrize("cls", [MPUrgeMatcher, CrossingAllowedMPUrgeMatcher])
@pytest.mark.parametrize("unify", [False, True])
@pytest.mark.parametrize("triangle", [None, 0.])
def test_spatial_matches_pair_inspector_for_both_stages(cls, unify, triangle):
    options = {} if unify else {"size_unification_norm": None}
    matcher = cls(half_window=2, alpha=.7, triangle_tolerance_m=triangle, **options)
    a = profile([.1, 1.2, 2., 3., 4., 5.], [0, 1, 2, 3, 4, 5], [0., 0., 1.])
    b = profile([.2, 2.1, 3., 5.], [0, 2, 1, 6], [1., 0., 2.])
    spatial = spatial_match_counts(a, b, matcher)
    detail = pair_data(a, b, matcher, [])
    for row, key in zip(spatial, ["quality", "finalQuality"]):
        q = detail[key]
        assert row == [q["tp"], q["tp"] + q["fp"], q["available"], 24]
        assert row[2] == 3  # Includes true correspondences removed by unification.


def test_empty_profiles_do_not_fabricate_matches_or_true_negatives():
    a = profile([], [], [0., 0., 0.])
    b = profile([1., 2.], [1, 2], [0., 0., 1.])
    assert spatial_match_counts(a, b, MPUrgeMatcher()) == [[0, 0, 0, 0]] * 2


def test_all_pairs_include_far_pairs_height_and_local_membership():
    xyz = np.array([[0., 0., 0.], [0., 0., 3.], [4., 0., 0.]])
    matcher = MPUrgeMatcher(half_window=0, alpha=1., size_unification_norm=None)
    data = {"profiles": [profile([1., 2.], [0, 1], p) for p in xyz]}
    context = (xyz, np.array([[0, 1]]), matcher)
    explorer = PeakExplorer(None)
    result = explorer.spatial(context, data)
    np.testing.assert_array_equal(result["pairs"], [[0, 1], [0, 2], [1, 2]])
    np.testing.assert_allclose(result["distance_m"], [3., 4., 5.])
    assert result["neighborhood"] == [True, False, False]
    np.testing.assert_array_equal(result["final_counts"], [[2, 2, 2, 4]] * 3)
    assert explorer.spatial(context, data) is result
