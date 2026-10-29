"""The explorer must not confuse pipeline selection losses with stage measurements."""
from types import SimpleNamespace

import numpy as np

from indoor_vt.execution.explorer import (build_catalog, saved_stage_metrics, inside_obstacles,
                                         display_source_metrics, summarize_sources)
from indoor_vt.metrics import source_metrics
from indoor_vt.execution.viewer import ExperimentView
from indoor_vt.science.matching import MPUrgeMatcher
from indoor_vt.execution.peak_explorer import PeakExplorer, check_matching_data, pair_data
from indoor_vt.acquisition.records import digest, write_json
from indoor_vt.execution.viewer import source_page, preview
from indoor_vt.execution.artifacts import write_npz
import pytest


def receipt(method="joint", **changes):
    return {"id": "a" * 64, "label": method + "_ununified", "method": method,
            "unify": False, "level": 0., "split": "calibration", "number": 0,
            "loss": .1, "metrics": {"mean_m": .1}, "parameters": {}, "frozen": {}, **changes}


def matching_dataset(view, root):
    view.data_root, view.run_root = root / "data", root / "run"
    manifest = {"schema": "vt.dataset.2", "scene_value": "saved-scene", "truth_value": "saved-truth"}
    write_json(view.data_root / "dataset.json", manifest)
    # These describe unrelated code changes, not a different matching dataset.
    write_json(view.run_root / "implementation.json", {"dataset": digest(manifest),
        "id": "old-whole-implementation", "experiment": "old-search-grid",
        "compositions": "old-finisher"})
    def fail_full_replay():
        raise AssertionError("Pair diagnostics must not replay the full scientific experiment")
    view.check_replay_version = fail_full_replay
    view.replay = fail_full_replay
    return manifest


def test_matching_diagnostic_uses_saved_parameters_despite_unrelated_code_revisions(tmp_path):
    view = SimpleNamespace()
    matching_dataset(view, tmp_path)
    p = {"alpha": .7, "half_window": 2, "neighbors": 2, "radius_m": .1, "beta": 2.}
    saved = receipt(method="mpurge", parameters=p)
    view.receipt = lambda key: saved
    view.config = {"name": "saved-study", "survey": {"spacing_m": .5, "phase_xy": [0, 0]},
                   "observations": {"track": "direct", "calibration_repetitions": [0], "noise_levels": [0., .1]}}
    room = {"id": "room", "survey_spacing_m": .5,
            "survey_lattice": np.array([[0, 0], [1, 0]]),
            "survey_xyz": np.array([[0., 0., 1.], [.5, 0., 1.]]),
            "polygon_xy": [], "access_points_xyz": {}}
    view.observations = SimpleNamespace(rooms=[room])
    study = SimpleNamespace(owner=lambda key: view, overview=lambda: {"methods": []})
    explorer = PeakExplorer(study)
    result = explorer.request("init", trial=saved["id"])
    assert result["meta"]["alpha"] == .7
    assert result["meta"]["half_window"] == 2
    assert result["scene"]["pairs"].tolist() == [[0, 1]]
    view.config["engine"] = "obstacle_reconstruction"
    with pytest.raises(ValueError, match="layout truth"):
        explorer.request("init", trial=saved["id"])


def test_matching_diagnostic_rejects_different_data_and_unsupported_schema(tmp_path):
    view = SimpleNamespace()
    manifest = matching_dataset(view, tmp_path)
    check_matching_data(view)
    write_json(view.data_root / "dataset.json", {**manifest, "scene_value": "different-room"})
    with pytest.raises(ValueError, match="dataset differs"):
        check_matching_data(view)
    write_json(view.data_root / "dataset.json", {**manifest, "schema": "unknown"})
    with pytest.raises(ValueError, match="Unsupported dataset schema"):
        check_matching_data(view)


def test_saved_results_remain_readable_but_recomputation_checks_versions(tmp_path):
    view = ExperimentView.__new__(ExperimentView)
    view.run_root = tmp_path
    view.receipt = lambda key: receipt(outcome="outcomes/saved.npz")
    def changed_version():
        raise ValueError("implementation changed")
    view.check_replay_version = changed_version
    saved = {"sources": [], "queries": []}
    write_npz(tmp_path / "outcomes/saved.npz", saved)
    assert view.replay("a" * 64) == saved
    with pytest.raises(ValueError, match="implementation changed"):
        view.replay("a" * 64, refresh=True)
    view.receipt = lambda key: receipt()
    with pytest.raises(ValueError, match="implementation changed"):
        view.replay("a" * 64)


def test_naked_catalog_identifies_absence_of_finisher_in_all_pipeline_stages():
    from indoor_vt.execution.explorer import method_entries
    for family in ("zayets", "ransac"):
        for suffix in ("", "_lea", "_joint"):
            entries = method_entries(receipt(method=family + "_naked" + suffix))
            assert entries
            for entry in entries:
                assert entry["finisher"] is None and entry["naked"]
                assert entry["source_full_name"].endswith("(no finisher)")
                assert "finisher:none" in entry["tags"]


def test_catalog_only_exposes_measured_stages_and_uses_calibration_winner():
    winner = receipt()
    loser = receipt(id="b" * 64, number=1, loss=.2)
    evaluation = receipt(id="c" * 64, split="evaluation", loss=.001,
                         metrics={"mean_m": .001})
    catalog = build_catalog([winner, loser], [evaluation], {})
    assert [m["id"] for m in catalog["methods"]] == ["joint_ununified"]
    assert catalog["methods"][0]["selected_id"] == winner["id"]
    assert all(r["selection_loss"] == .1 for r in catalog["conditions"])
    measured = {evaluation["id"]: {"sources": {"ospa": .8, "f1": .2},
                                    "initial": {"mean_m": 1.7}}}
    catalog = build_catalog([winner, loser], [evaluation], measured)
    assert len(catalog["methods"]) == 3
    source = next(r for r in catalog["conditions"] if r["method_id"].endswith("_sources"))
    assert source["metrics"]["ospa"] == .8  # Never substitute the .001 pipeline error.
    assert source["receipt_id"] == evaluation["id"]
    assert source["selection_loss"] == .1
    assert not any(r["split"] == "calibration" and r["method_id"].endswith("_sources")
                   for r in catalog["conditions"])
    assert "Joint with MPUrge-MAP + SMART-LEA · Unification off" in catalog["methods"][0]["title"]


def test_saved_initializer_uses_its_own_coordinates_and_query_ids():
    rooms = {"room": {"layouts": [{"id": 5, "queries_xyz": np.array(
        [[99., 99., 99.], [0., 0., 0.], [2., 0., 0.]])}]}}
    detail = {"sources": [], "queries": [{"room": "room", "layout": 5,
        "query_ids": np.array([1, 2]), "initial_xyz": np.array([[3., 4., 0.], [2., 0., 0.]]),
        "xyz": np.array([[0., 0., 0.], [2., 0., 0.]]), "errors_m": np.zeros(2)}]}
    metrics = saved_stage_metrics(detail, rooms, {})
    assert metrics["sources"] == {}
    assert metrics["initial"]["mean_m"] == 2.5
    assert metrics["initial"]["rmse_m"] == np.sqrt(12.5)


def test_ap_selection_keeps_counts_and_scores_for_the_selected_source_set():
    truth = {"room": SimpleNamespace(by_ap={"A": np.array([[0., 0., 0.]]),
                                           "B": np.array([[10., 0., 0.], [20., 0., 0.]])})}
    source = {"room": "room", "repetition": 2,
              "by_ap": {"A": np.array([[0., 0., 0.]]), "B": np.array([[10., 0., 0.]])}}
    measured = saved_stage_metrics({"sources": [source], "queries": []}, {}, truth)
    assert measured["source_metrics_by_ap"]["A"] == {"ospa": 0., "f1": 1., "matched_rmse_m": 0.}
    assert measured["source_metrics_by_ap"]["B"] == {"ospa": .5, "f1": 2/3, "matched_rmse_m": 0.}
    assert measured["source_metrics_by_ap"]["A+B"] == {"ospa": 1/3, "f1": .8, "matched_rmse_m": 0.}
    assert measured["source_counts"] == [{"room": "room", "repetition": 2,
        "predicted": {"A": 1, "B": 1, "A+B": 2}, "true": {"A": 1, "B": 2, "A+B": 3}}]
    assert "true_count" not in measured["sources"]


def test_matched_rmse_uses_3d_distances_and_excludes_unmatched_predictions():
    predicted = [[0., 0., .3], [10., 0., .5], [25., 0., 0.]]
    truth = [[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]]
    measured = display_source_metrics(predicted, truth)
    original = source_metrics(predicted, truth)
    assert measured["matched_rmse_m"] == pytest.approx(np.sqrt((.3**2 + .5**2) / 2))
    assert measured["matches"] == [(0, 0), (1, 1)]
    for key in ("ospa", "f1", "tp", "fp", "fn"):
        assert measured[key] == original[key]


def test_matched_rmse_minimizes_squared_distance_after_maximizing_match_count():
    # All four edges are in range; boolean matching can choose nonzero errors.
    measured = display_source_metrics([[.4, 0., 0.], [0., 0., 0.]],
                                      [[0., 0., 0.], [.4, 0., 0.]])
    assert measured["matches"] == [(0, 1), (1, 0)]
    assert measured["matched_rmse_m"] == 0.
    # Two valid matches take priority over retaining only the exact match.
    measured = display_source_metrics([[.4, 0., 0.], [0., 0., 0.]],
                                      [[0., 0., 0.], [.8, 0., 0.]])
    assert measured["tp"] == 2
    assert measured["matched_rmse_m"] == pytest.approx(.4 / np.sqrt(2))


@pytest.mark.parametrize("predicted,truth", [([], []), ([], [[0., 0., 0.]]),
    ([[0., 0., 0.]], []), ([[0., 0., 0.]], [[1., 0., 0.]])])
def test_matched_rmse_without_matches_is_missing_not_zero(predicted, truth):
    measured = display_source_metrics(predicted, truth)
    assert measured["matched_rmse_m"] is None
    assert measured["matched_squared_error_m2"] == 0.
    assert measured["matches"] == []


def test_matched_rmse_pools_matches_across_repetitions_for_each_ap():
    empty = np.empty((0, 3))
    truths = {"room": SimpleNamespace(by_ap={"A": np.array([[0., 0., 0.], [10., 0., 0.]]),
                                            "B": np.array([[30., 0., 0.]])})}
    records = [{"room": "room", "by_ap": {"A": np.array(a), "B": empty}}
               for a in ([[0., 0., .1], [10., 0., .1]], [[0., 0., .4]], [[50., 0., 0.]])]
    _, by_ap = summarize_sources(records, truths)
    expected = np.sqrt((.1**2 + .1**2 + .4**2) / 3)
    assert by_ap["A"]["matched_rmse_m"] == pytest.approx(expected)
    assert by_ap["B"]["matched_rmse_m"] is None
    assert by_ap["A+B"]["matched_rmse_m"] == pytest.approx(expected)


def test_obstacle_reporting_pools_query_errors_and_keeps_initial_stage_separate():
    rooms = {"room": {"layouts": [
        {"id": name, "box_count": boxes, "queries_xyz": np.zeros((3, 3))}
        for name, boxes in (("empty", 0), ("furnished", 3))]}}
    records = []
    for layout, errors in (("empty", [1.]), ("furnished", [2., 3., 4.]), ("furnished", [5.])):
        xyz = np.zeros((len(errors), 3))
        xyz[:, 0] = errors
        records.append({"room": "room", "layout": layout, "query_ids": np.arange(len(errors)),
                        "xyz": xyz, "initial_xyz": xyz * 10})
    measured = saved_stage_metrics({"sources": [], "queries": records}, rooms, {})
    final = measured["by_obstacles"]["localization"][3]
    assert final["query_count"] == 4
    assert final["mean_m"] == final["median_m"] == 3.5
    assert final["rmse_m"] == np.sqrt(13.5)
    assert measured["by_obstacles"]["initial"][3]["mean_m"] == 35.
    row = receipt()
    catalog = build_catalog([row], [row], {row["id"]: measured})
    initial = next(r for r in catalog["conditions"] if r["method_id"].endswith("_initial"))
    final = next(r for r in catalog["conditions"] if r["method_id"] == row["label"])
    assert initial["obstacle_metrics"][3]["mean_m"] == 35.
    assert final["obstacle_metrics"][3]["mean_m"] == 3.5


def test_box_membership_respects_rotated_footprint_height_and_boundaries():
    boxes = [{"footprint_xy": [[0., 1.], [1., 0.], [0., -1.], [-1., 0.]], "height_m": 1.2}]
    points = np.array([[0., 0., 1.], [0., 0., 1.5], [.8, .8, 1.], [1., 0., 1.2], [0., 0., -.1]])
    np.testing.assert_array_equal(inside_obstacles(points, boxes), [0, 3])
    assert len(inside_obstacles(points, [])) == 0


def test_joint_evidence_keeps_sources_and_both_query_stages():
    view = ExperimentView.__new__(ExperimentView)
    room = {"id": "room", "polygon_xy": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "ceiling_m": 3., "access_points_xyz": {"A": [0., 0., 2.], "B": [5., 5., 2.]},
            "survey_xyz": np.array([[0., 0., 1.]]), "survey_position_ids": np.array([0]),
            "layouts": [{"id": 0, "queries_xyz": np.array([[1., 1., 1.]])}]}
    view.rooms = {"room": room}
    points = {"A": np.array([[0., 0., 2.]]), "B": np.array([[5., 5., 2.]])}
    view.truths = {"room": SimpleNamespace(by_ap=points)}
    view.truth_catalog = {"room": {ap: [{"id": ap, "xyz": xyz[0], "surface_sequences": [[]]}]
                                     for ap, xyz in points.items()}}
    view.config = {"survey": {"spacing_m": .5, "phase_xy": [0, 0]},
                   "observations": {"evaluation_repetitions": [2, 3]}}
    source = {"room": "room", "repetition": 2, "by_ap": points, "audit": {"sources": "saved"}}
    query = {"room": "room", "layout": 0, "repetition": 2, "query_ids": np.array([0]),
             "xyz": np.array([[1., 1., 1.]]), "initial_xyz": np.array([[4., 5., 1.]]),
             "audit": {"initial": {"step": "map"}, "refinement": {"step": "lea"}}}
    view.receipt = lambda key: receipt(split="evaluation")
    view.replay = lambda key: {"sources": [source], "queries": [query]}
    # Geometry selection has its own tests; this fixture isolates result presentation.
    view.source_map = lambda record, ap: {"ap": ap}
    result = view.result("a" * 64)
    assert all(case["realization"] == 1 and case["repetition"] == 2 for case in result["cases"])
    assert [c["kind"] for c in result["cases"]] == ["sources", "localization", "initial"]
    assert set(result["cases"][0]["maps"]) == {"A", "B", "A+B"}
    assert result["cases"][1]["metrics"]["mean_m"] == 0.
    assert result["cases"][2]["metrics"]["mean_m"] == 5.
    assert result["cases"][2]["audit"] == {"step": "map"}


def test_peak_matrix_preserves_original_indices_after_size_unification(monkeypatch, tmp_path):
    view = ExperimentView.__new__(ExperimentView)
    view.receipt = lambda key: receipt(method="mpurge", parameters={})
    matching_dataset(view, tmp_path)
    view.config = {"observations": {"calibration_repetitions": [0]}}
    view.rooms = {"room": {"id": "room"}}
    survey = SimpleNamespace(xyz=np.array([[0., 0., 0.], [1., 0., 0.]]),
        channels={"A": [np.array([1., 2., 10.]), np.array([1., 9.])]})
    view.observations = SimpleNamespace(rooms=list(view.rooms.values()), survey=lambda *a: survey)
    monkeypatch.setattr("indoor_vt.execution.viewer.standard_mpurge",
        lambda *a, **k: SimpleNamespace(matcher=MPUrgeMatcher(half_window=1, alpha=1.)))
    value = view.match("a" * 64, "A", 0, 1)
    assert value["first_round_costs"].shape == (3, 2)
    assert np.isnan(value["first_round_costs"][1]).all()  # The removed middle peak stays visible.
    assert value["first_round_costs"][0, 0] == 0.
    assert value["first_round_costs"][2, 1] == 1.
    for left, right, cost, round_id, _accepted in value["trace"]:
        if round_id == 1:
            assert value["first_round_costs"][int(left), int(right)] == cost


def test_full_peak_explorer_keeps_unification_gaps_and_truth_out_of_matching():
    matcher = MPUrgeMatcher(half_window=1, alpha=1.)
    a = {"ranges": np.array([1., 2., 10.]), "source_ids": np.array([7, 8, 9])}
    b = {"ranges": np.array([1., 9.]), "source_ids": np.array([7, 9])}
    result = pair_data(a, b, matcher, [])
    matrix = result["matrix"]["values"].reshape(3, 2)
    assert np.isnan(matrix[1]).all()
    assert result["finalQuality"] == {"tp": 2, "fp": 0, "fn": 0,
        "available": 2, "precision": 1., "recall": 1., "f1": 1.}
    # Altering truth affects only quality and overlays, never scientific matches.
    swapped = pair_data(a, {**b, "source_ids": np.array([9, 7])}, matcher, [])
    np.testing.assert_array_equal(result["final"]["matches"], swapped["final"]["matches"])
    assert swapped["finalQuality"]["tp"] == 0


def test_code_view_only_serves_local_package_python():
    page = source_page("science/matching.py").decode()
    assert 'id="L1"' in page and "class MPUrgeMatcher" in page
    for name in ("../../../../pyproject.toml", "execution/web/app.js", "science/missing.py"):
        with pytest.raises(ValueError, match="Unknown source file"):
            source_page(name)


def test_triangle_peak_explorer_applies_both_geometric_inequalities():
    matcher = MPUrgeMatcher(half_window=1, alpha=1., size_unification_norm=None,
                            triangle_tolerance_m=0.)
    a = {"ranges": np.array([.1, 2., 10.]), "source_ids": np.array([1, 2, 3]), "xyz": [0., 0., 0.]}
    b = {"ranges": np.array([.1, 2., 10.]), "source_ids": np.array([1, 2, 3]), "xyz": [1., 0., 0.]}
    result = pair_data(a, b, matcher, [])
    matrix = result["matrix"]["values"].reshape(3, 3)
    assert not np.isfinite(matrix[0, 0])  # Sum too short to bridge the receivers.
    assert not np.isfinite(matrix[1, 2])  # Difference greater than receiver separation.
    assert matrix[1, 1] == 0.
    np.testing.assert_array_equal(result["final"]["matches"], [[1, 1], [2, 2]])


def test_variant_catalog_names_finisher_and_links_composition_constructor():
    from indoor_vt.execution.explorer import method_entries
    for method, name in (("zayets_robust", "Zayets + robust range-refinement and peak-consumption finisher"),
                         ("mpurge_consumption", "MPUrge + support-first peak-consumption finisher"),
                         ("mpurge_triangle", "MPUrge triangle")):
        entry = method_entries(receipt(method))[0]
        assert entry["name"] == name and entry["group"] == "vt"
        link = entry["code"]["Composition constructors"][0]
        page = source_page(link["file"]).decode()
        assert f'id="L{link["line"]}"' in page
        assert "def composed_reconstructor" in page
        assert "family:" + entry["family"] in entry["tags"]


def test_finisher_history_arrays_keep_pageable_paths_through_lists():
    support = np.arange(160).reshape(80, 2)
    artifact = {"selection": {"history": [{"support": support}]}}
    field = preview(artifact)["selection"]["history"][0]["support"]["field"]
    assert field == "/selection/history/0/support"
    view = ExperimentView.__new__(ExperimentView)
    view.store = SimpleNamespace(get=lambda key: artifact)
    page = view.artifact("a" * 64, "b" * 64, field, 30)["value"]
    assert page["start"] == 30 and page["shape"] == [80, 2]
    np.testing.assert_array_equal(page["preview"], support[30:60])
