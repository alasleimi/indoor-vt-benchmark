"""Ground-truth metric semantics, missing observations, and production matcher parity."""
import numpy as np
import pytest

from indoor_vt.execution.true_matching import (truth_counts, final_match_counts, pool_rows,
                                              profile_arrays, violation_classes, identity_accuracy,
                                              TOLERANCE_M)
from indoor_vt.execution.peak_explorer import pair_data
from indoor_vt.science.matching import MPUrgeMatcher


def profile(ranges, ids, x=0):
    return {"ranges": np.asarray(ranges, dtype=float), "source_ids": np.asarray(ids, dtype=int),
            "xyz": np.array([x, 0., 0.])}


def test_cross_any_counts_edges_not_inversions_and_preserves_full_observed_rank():
    # Three true edges: 0->1, 1->0, 3->3. The unshared peak at rank 2 stays in the rank order.
    profiles = [profile([1, 2, 3, 4], [0, 1, 3, 2]),
                profile([1, 2, 3, 4], [1, 0, 4, 2])]
    result = truth_counts(profiles, np.array([[0., 0, 0], [1., 0, 0]]), np.array([[0, 1]]), 5)
    assert result["available"] == 3
    assert result["same_rank"] == 1
    assert result["cross_any"] == 2  # One inversion involves two true edges.


def test_rank_is_not_recomputed_after_intersection_of_visible_sources():
    profiles = [profile([1, 2, 3], [0, 1, 2]), profile([2, 3], [1, 2])]
    result = truth_counts(profiles, np.zeros((2, 3)), np.array([[0, 1]]), 3)
    assert result["available"] == 2
    assert result["same_rank"] == 0
    assert result["cross_any"] == 0


def test_geometric_counts_and_exact_boundaries():
    profiles = [profile([.2, 1., 5.], [0, 1, 2]), profile([.3, 4., 7.], [0, 1, 2])]
    result = truth_counts(profiles, np.array([[0., 0, 0], [0., 0, 2.]]), np.array([[0, 1]]), 3)
    assert result["difference_satisfies"] == 2
    assert result["sum_satisfies"] == 2
    profiles[1]["ranges"][2] += 2 * TOLERANCE_M
    result = truth_counts(profiles, np.array([[0., 0, 0], [0., 0, 2.]]), np.array([[0, 1]]), 3)
    assert result["difference_satisfies"] == 1


def test_ties_are_stable_full_ranks_and_duplicate_source_ids_fail():
    profiles = [profile([1., 1., 2.], [1, 0, 2]), profile([1., 1., 2.], [0, 1, 2])]
    result = truth_counts(profiles, np.zeros((2, 3)), np.array([[0, 1]]), 3)
    assert result["same_rank"] == 1
    assert result["cross_any"] == 2
    with pytest.raises(ValueError, match="unique"):
        profile_arrays([profile([1, 2], [0, 0])], 1)


def test_final_match_counts_match_all_accepted_rounds_of_peak_diagnostic():
    profiles = [profile([1., 2., 3., 4., 8.], [0, 1, 2, 3, 4]),
                profile([1.1, 2.2, 3.1, 4.2], [0, 2, 1, 4], .5)]
    matcher = MPUrgeMatcher(half_window=2, alpha=.7, size_unification_norm=None)
    result = final_match_counts(profiles, np.array([p["xyz"] for p in profiles]),
                                np.array([[0, 1]]), matcher)
    diagnostic = pair_data(*profiles, matcher, [])
    edges = diagnostic["final"]["matches"]
    assert result["total"] == diagnostic["finalQuality"]["tp"] + diagnostic["finalQuality"]["fp"]
    assert result["correct"] == diagnostic["finalQuality"]["tp"]
    assert result["same_rank_correct"] + result["same_rank_incorrect"] == np.sum(edges[:, 0] == edges[:, 1])
    for bound in ("difference", "sum"):
        assert result[bound + "_violation_correct"] <= result["correct"]
        assert result[bound + "_violation_incorrect"] <= result["total"] - result["correct"]


def test_pooling_adds_counts_instead_of_averaging_percentages():
    rows = []
    for ap, total, same in (("A", 10, 8), ("B", 90, 9)):
        counts = {"available": total, "same_rank": same}
        rows.append({"ap": ap, "noise": 1., "repetition": 2, "global": counts,
                     "neighborhood": counts, "matches": {"total": total},
                     "observations": {"discarded_negative": 0}})
    combined, = pool_rows(rows)
    assert combined["global"] == {"available": 100, "same_rank": 17}
    assert combined["realization_records"] == 2
    assert pool_rows(rows, "A")[0]["global"]["available"] == 10


@pytest.mark.parametrize("a,b,bound", [([1., 4.], [1.2, 7.], "difference"),
                                      ([.1, .3], [.2, 2.], "sum")])
def test_wrong_match_can_be_expected_violation_for_only_one_endpoint(a, b, bound):
    pa, pb = profile(a, [0, 1]), profile(b, [0, 1])
    ia, ib = (np.array([0]), np.array([1])) if bound == "difference" else (np.array([1]), np.array([0]))
    result = violation_classes(pa, pb, ia, ib, 1., bound)
    assert result == {"violation_edges": 1, "counterpart_violates": 1,
                      "counterpart_satisfies": 1, "counterpart_unavailable": 0,
                      "endpoint_disagreement_edges": 1}
    reverse = violation_classes(pb, pa, ib, ia, 1., bound)
    assert reverse == result


def test_missing_true_partner_is_not_a_false_or_expected_violation():
    result = violation_classes(profile([1., 4.], [0, 1]), profile([7.], [1]),
                               np.array([0]), np.array([0]), 1., "difference")
    assert result["violation_edges"] == 1
    assert result["counterpart_unavailable"] == 1
    assert result["counterpart_violates"] == 1
    assert result["counterpart_satisfies"] == 0


def test_conditional_identity_accuracy_uses_violating_edge_denominator():
    result = identity_accuracy({"total": 100, "correct": 60,
                                "difference_violation_correct": 3,
                                "difference_violation_edges": 10}, "difference")
    assert result["violating"] == {"correct": 3, "incorrect": 7, "total": 10,
                                   "fraction_correct": .3}
    assert result["overall"]["fraction_correct"] == .6


def test_no_violating_edges_has_undefined_not_zero_accuracy():
    result = identity_accuracy({"total": 100, "correct": 60, "sum_violation_correct": 0,
                                "sum_violation_edges": 0}, "sum")
    assert result["violating"]["fraction_correct"] is None
    assert result["overall"]["fraction_correct"] == .6
