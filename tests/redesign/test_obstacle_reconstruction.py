"""Privileged truth and frozen-parameter boundaries of obstacle evaluation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from indoor_vt.acquisition.records import write_json
from indoor_vt.execution.obstacle_reconstruction import visible_source_catalog, frozen_source_spec, layout_inputs


def test_new_frozen_methods_can_be_appended_without_changing_old_protocol():
    from indoor_vt.execution.obstacle_reconstruction import extends_frozen_protocol
    old = {"positions": "same 270", "methods": [{"id": "old", "source": {"alpha": .7}}]}
    new = {**old, "methods": [*old["methods"], {"id": "new", "source": {"alpha": .5}}]}
    assert extends_frozen_protocol(old,new)
    assert not extends_frozen_protocol(old,{**new,"positions":"different"})
    assert not extends_frozen_protocol(old,{**new,"methods":[new['methods'][1]]})
    assert not extends_frozen_protocol(old,{**new,"methods":[{"id":"old","source":{"alpha":.1}}]})


def test_archived_paths_recover_images_and_handle_reversed_reflection_polylines(tmp_path):
    ap = np.array([1., 0., 1.])
    xyz = np.array([[2., 1., 1.], [3., 1., 1.]])
    direct = [[*ap, *x] for x in xyz]
    reflected = [[[*x, 0., 1. / (x[0] + 1.), 1., *ap]] for x in xyz]
    records = [{"TX": 0, "RX": 1, "PAA_TX": 0, "PAA_RX": 0, "Rorder": order, "MPC": paths}
               for order, paths in ((0, direct), (1, reflected))]
    ranges = np.array([[np.linalg.norm(x-ap), np.linalg.norm(x-[-1., 0., 1.])] for x in xyz])
    native = {"paths": np.column_stack((ranges.ravel()/3e8, np.zeros((4, 6)))),
              "offsets": np.array([0, 2, 4]), "position_ids": np.array([0, 1])}
    data = {"native": native, "visual": {"Output/Visualizer/Mpc.json": np.frombuffer(
        "\n".join(json.dumps(r) for r in records).encode(), dtype=np.uint8)}}
    write_json(tmp_path / "native/job/receipt.json", {"native_value": "native", "visualization_value": "visual"})
    observations = SimpleNamespace(root=tmp_path, store=SimpleNamespace(get=data.__getitem__),
        manifest={"jobs": [{"room": "r", "ap": "A", "group": "layout", "key": "job"}]})
    room = {"id": "r", "access_points_xyz": {"A": ap}}
    catalog, audit = visible_source_catalog(observations, room, {"id": "layout", "queries_xyz": xyz})
    np.testing.assert_allclose([e["xyz"] for e in catalog["A"]], [ap, [-1., 0., 1.]], atol=1e-12)
    assert audit["A"]["count"] == 2 and audit["A"]["path_count"] == 4
    assert audit["A"]["maximum_native_range_discrepancy_m"] < 1e-12
    subset, subset_audit = visible_source_catalog(observations, room,
        {"id": "layout", "queries_xyz": xyz, "reconstruction_position_ids": np.array([1])})
    assert subset_audit["A"]["path_count"] == 2
    assert all(e["visible_position_ids"] == [1] for e in subset["A"])
    native["paths"][0, 0] += 1e-5 / 3e8
    with pytest.raises(ValueError, match="do not reproduce"):
        visible_source_catalog(observations, room, {"id": "layout", "queries_xyz": xyz})


def test_joint_source_parameters_are_reused_without_localizer_or_refiner():
    p = {"half_window": 2, "alpha": .7}
    spec = frozen_source_spec({"method": "joint", "parameters": {"mpurge": p, "map": {"x": 1}, "lea": {}}, "unify": False}, {})
    assert spec["method"] == "mpurge" and spec["parameters"] is p and spec["unify"] is False
    spec = frozen_source_spec({"method": "zayets_robust_joint", "parameters": {"source": p, "map": {}, "lea": {}}, "unify": True}, {})
    assert spec["method"] == "zayets_robust" and spec["parameters"] is p


def test_layout_inputs_use_actual_layout_group_all_positions_and_zero_noise():
    from indoor_vt.science.models import Ragged
    calls = []
    def channel(*args):
        calls.append(args)
        return Ragged.pack([[1.], [2.]])
    xyz = np.array([[0., 0., 1.], [1., 1., 1.]])
    room = {"access_points_xyz": {"A": [0., 0., 0.], "B": [1., 0., 0.]}}
    survey = layout_inputs(SimpleNamespace(channel=channel), room, {"id": "obstacle_group", "queries_xyz": xyz, "boxes": []})
    np.testing.assert_array_equal(survey.xyz, xyz)
    assert len(calls) == 2
    for call in calls:
        assert call[2] == "obstacle_group" and call[4:] == (0., 2, "direct")
        np.testing.assert_array_equal(call[3], [0, 1])


def test_obstacle_evidence_replay_never_calls_empty_room_evaluator(tmp_path, monkeypatch):
    import threading
    from indoor_vt.execution import obstacle_reconstruction as module
    from indoor_vt.execution.artifacts import write_npz
    from indoor_vt.execution.viewer import ExperimentView
    from indoor_vt.science.models import SourceMap
    view = ExperimentView.__new__(ExperimentView)
    view.config = {"engine": "obstacle_reconstruction"}
    view.run_root = tmp_path
    view.receipt = lambda key: {"view_method_id": "frozen", "outcome": "saved.npz"}
    layout = {"id": "layout_03"}
    view.rooms = {"room": {"id": "room", "layouts": [layout]}}
    view.observations, view.evaluator, view.lock = object(), object(), threading.RLock()
    write_json(tmp_path / "protocol.json", {"methods": [{"id": "frozen", "source": {"parameters": {"alpha": .7}}}]})
    write_npz(tmp_path / "saved.npz", {"sources": [{"room": "room", "layout": "layout_03",
        "truth_catalog": {"A": [{"xyz": [1., 2., 3.]}]}}], "queries": []})
    survey = object()
    def inputs(observations, room, selected_layout):
        assert observations is view.observations and selected_layout is layout
        return survey
    def reconstruct(spec, received, evaluator, count, room_id):
        assert received is survey and evaluator is view.evaluator
        assert spec == {"parameters": {"alpha": .7}} and count == {"A": 1} and room_id == "room"
        return SourceMap({"A": np.array([[4., 5., 6.]])}, {"replayed": True})
    monkeypatch.setattr(module, "layout_inputs", inputs)
    monkeypatch.setattr(module, "reconstruct", reconstruct)
    result = view.replay("test", refresh=True)
    np.testing.assert_array_equal(result["sources"][0]["by_ap"]["A"], [[4., 5., 6.]])
    assert result["sources"][0]["audit"] == {"replayed": True}


def test_common_survey_mask_uses_union_3d_boundaries_without_extra_clearance():
    from indoor_vt.acquisition.matched_obstacles import common_survey_mask
    xyz = np.array([[0., 0., 1.], [.5, .5, 1.], [.5, .5, 1.5],
                    [1.5, .5, 1.], [2.001, .5, 1.], [0., 0., 0.]])
    boxes = [{"footprint_xy": [[0, 0], [1, 0], [1, 1], [0, 1]], "height_m": 1.2},
             {"footprint_xy": [[1, 0], [2, 0], [2, 1], [1, 1]], "height_m": 1.1}]
    room = {"survey_xyz": xyz, "survey_lattice": np.zeros((6, 3), int), "survey_spacing_m": 1.,
            "layouts": [{"id": "empty", "boxes": []}, {"id": "one", "boxes": [boxes[0]]},
                        {"id": "two", "boxes": [boxes[1]]}]}
    mask = common_survey_mask(room, {"spacing_m": 1., "phase_xy": [0, 0]})
    assert mask["original_count"] == 6 and mask["retained_count"] == 2
    assert mask["retained_position_ids"] == [2, 4]  # above a box; 1 mm beyond another box
    assert mask["excluded_position_ids"] == [0, 1, 3, 5]
    assert mask["excluded_by_layout"] == {"empty": [], "one": [0, 1, 5], "two": [3]}
