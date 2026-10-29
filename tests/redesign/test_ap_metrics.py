import numpy as np
import pytest
from types import SimpleNamespace

from indoor_vt.metrics import ap_source_metrics, matched_source_metrics
from indoor_vt.execution.explorer import summarize_sources
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
    assert by_ap["A+B"] == {"ospa": 1., "f1": 0., "matched_rmse_m": None}
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
