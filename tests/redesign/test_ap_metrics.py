import numpy as np
import pytest
from types import SimpleNamespace

from indoor_vt.metrics import ap_source_metrics, matched_source_metrics
from indoor_vt.execution.explorer import display_ap_source_metrics, summarize_sources
from indoor_vt.execution.obstacle_reconstruction import metrics_for_record


def test_swapped_ap_sources_cannot_receive_credit_or_map_lines():
    truth = {"A": [[0., 0., 0.]], "B": [[10., 0., 0.]]}
    predicted = {"A": truth["B"], "B": truth["A"]}
    pooled = ap_source_metrics(predicted, truth)
    assert pooled["ospa"] == 1 and pooled["f1"] == 0
    assert pooled["tp"] == 0 and pooled["fp"] == pooled["fn"] == 2
    assert pooled["matches"] == [] and pooled["matched_rmse_m"] is None
    _, by_ap = summarize_sources([{"room": "r", "by_ap": predicted}],
        {"r": SimpleNamespace(by_ap=truth)})
    assert by_ap["A+B"] == {"ospa": 1., "f1": 0., "f1_0_1m": 0., "f1_0_3m": 0.,
                            "matched_rmse_m": None}
    record = {"room": "r", "layout": "x", "by_ap": predicted,
              "truth_catalog": {ap: [{"xyz": p} for p in points] for ap, points in truth.items()}}
    assert metrics_for_record(record)[0]["A+B"] == pooled


def test_cardinality_errors_in_different_aps_do_not_cancel():
    truth = {"A": [[0., 0., 0.]], "B": [[10., 0., 0.]]}
    predicted = {"A": [[0., 0., 0.], [10., 0., 0.]], "B": []}
    result = ap_source_metrics(predicted, truth)
    assert result["ospa"] == pytest.approx(2/3)
    assert result["ospa_normalizer"] == 3
    assert result["f1"] == .5 and result["tp"] == 1
    assert result["matched_rmse_m"] == 0 and result["matches"] == [(0, 0)]


def test_rmse_pools_squared_errors_and_map_indices_keep_ap_offsets():
    truth = {"A": [[0., 0., 0.], [2., 0., 0.]], "B": [[10., 0., 0.]]}
    predicted = {"A": [[2., 0., .1], [0., 0., .1]], "B": [[10., 0., .4], [99., 0., 0.]]}
    result = ap_source_metrics(predicted, truth)
    assert result["matched_rmse_m"] == pytest.approx(np.sqrt((.01+.01+.16)/3))
    assert result["f1"] == 6/7
    assert result["matches"] == [(0, 1), (1, 0), (2, 2)]
    for ap in ("A", "B"):
        separate = ap_source_metrics(predicted, truth, (ap,))
        ordinary = matched_source_metrics(predicted[ap], truth[ap])
        assert all(separate[key] == value for key, value in ordinary.items())


def test_all_empty_ap_sets_are_well_defined():
    measured = ap_source_metrics({"A": [], "B": []}, {"A": [], "B": []})
    assert measured["ospa"] == 0 and measured["f1"] == 1
    assert measured["matched_rmse_m"] is None and measured["matches"] == []


@pytest.mark.parametrize("gate,key", [(.1, "f1_0_1m"), (.3, "f1_0_3m")])
def test_tighter_f1_gate_reassigns_instead_of_filtering_half_meter_matches(gate, key):
    # At 0.5 m the least-squares tie break chooses distances [0, 1.1 * gate].
    # At the tighter gate the maximum-cardinality solution instead uses
    # [0.9 * gate, 0.9 * gate], so simply filtering the old assignment is wrong.
    r, d = .9 * gate, 1.1 * gate
    x = (2 * r**2 - d**2) / (2 * r)
    predicted = {"A": [[0., 0., 0.], [r, 0., 0.]], "B": []}
    truth = {"A": [[0., 0., 0.], [x, np.sqrt(r**2 - x**2), 0.]], "B": []}
    measured = display_ap_source_metrics(predicted, truth)
    assert measured["matches"] == [(0, 0), (1, 1)]
    assert measured[key] == 1.
    original = ap_source_metrics(predicted, truth)
    assert all(measured[k] == v for k, v in original.items())


def test_tighter_f1_gates_pool_ap_counts_and_use_obstacle_truth():
    truth = {"A": [[0., 0., 0.]], "B": [[10., 0., 0.], [20., 0., 0.], [30., 0., 0.]]}
    predicted = {"A": [[0., 0., .1]], "B": [[10., 0., .1], [20., 0., .3], [30., 0., .3]]}
    measured = display_ap_source_metrics(predicted, truth)
    assert measured["f1_0_1m"] == .5
    assert measured["f1_0_3m"] == 1.
    record = {"room": "r", "layout": "x", "by_ap": predicted,
              "truth_catalog": {ap: [{"xyz": p} for p in points] for ap, points in truth.items()}}
    obstacle_metrics, _ = metrics_for_record(record, metric_fn=display_ap_source_metrics)
    assert obstacle_metrics["A+B"] == measured
    assert obstacle_metrics["A"]["f1_0_1m"] == 1.
    assert obstacle_metrics["B"]["f1_0_1m"] == 1/3


def test_tighter_f1_gates_preserve_macro_average_over_noise_realizations():
    truths = {"r": SimpleNamespace(by_ap={"A": [[0., 0., 0.]], "B": []})}
    records = [{"room": "r", "by_ap": {"A": predictions, "B": []}}
               for predictions in ([[0., 0., 0.]], [[0., 0., 0.], [5., 0., 0.], [9., 0., 0.]])]
    _, by_ap = summarize_sources(records, truths)
    for ap in ("A", "A+B"):
        assert by_ap[ap]["f1_0_1m"] == .75
        assert by_ap[ap]["f1_0_3m"] == .75
    assert by_ap["B"]["f1_0_1m"] == by_ap["B"]["f1_0_3m"] == 1.
