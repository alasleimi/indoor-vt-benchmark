"""Numerical and presentation contracts for the true-count baseline."""
from itertools import permutations

import numpy as np
import pytest

from indoor_vt.execution.explorer import method_entries
from indoor_vt.science.models import Ragged
from indoor_vt.science.range_rmse import RangeRMSE, assign_ranges, range_sse


def test_rectangular_assignment_is_one_to_one_and_minimizes_observed_squared_error():
    xyz = np.array([[0., 0., 0.], [1., 1., 1.]])
    sources = np.array([[2., 0., 0.], [4., 0., 0.], [7., 0., 0.]])
    ranges = Ragged.pack([[2.1, 6.8], [3.2, 1.6, 6.1]])
    assigned, indices = assign_ranges(xyz, ranges, sources)
    expected = 0.
    for s in range(len(xyz)):
        predicted = np.linalg.norm(sources - xyz[s], axis=1)
        expected += min(sum((ranges[s][i] - predicted[k])**2 for i, k in enumerate(p))
                        for p in permutations(range(3), len(ranges[s])))
        assert sorted(indices[s][indices[s] >= 0]) == list(range(len(ranges[s])))
    assert range_sse(xyz, assigned, sources) == pytest.approx(expected)
    assert np.isnan(assigned[0]).sum() == 1


def test_single_source_reconstructs_without_position_truth_and_records_monotone_objective():
    rng = np.random.default_rng(91)
    xyz = rng.uniform(-3, 3, (40, 3))
    truth = np.array([6., -4., 2.])
    ranges = Ragged.pack(np.linalg.norm(xyz - truth, axis=1)[:, None])
    solver = RangeRMSE(starts=3, max_iterations=30)
    predicted, audit = solver.fit(xyz, ranges, 1)
    assert np.linalg.norm(predicted[0] - truth) < 1e-6
    assert audit["range_rmse_m"] < 1e-8
    for run in audit["starts"]:
        assert np.all(np.diff(run["sse_history"]) <= 1e-8)
    assert audit["starts"][audit["selected_start"]]["range_rmse_m"] == min(
        r["range_rmse_m"] for r in audit["starts"])
    again, _ = solver.fit(xyz, ranges, 1)
    np.testing.assert_array_equal(predicted, again)


def test_rejects_more_observed_peaks_than_known_sources():
    xyz = np.zeros((4, 3))
    with pytest.raises(ValueError, match="at most one"):
        RangeRMSE(starts=1).fit(xyz, Ragged.pack([[1., 2.]] * 4), 1)


def test_catalog_discloses_true_count_without_claiming_calibration_tuning():
    entry, = method_entries({"method": "range_rmse_true_count", "label": "range_rmse_true_count"})
    assert "(cheating true count)" in entry["name"]
    assert entry["group"] == "vt" and entry["selection"] == "fixed"
    assert entry["unify"] is None and entry["family"] == "range_rmse"
    constructor, = entry["code"]["Composition constructors"]
    assert constructor["file"] == "science/range_rmse.py" and constructor["line"] > 1
