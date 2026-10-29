"""Verify live viewer maps, AP metrics and actual obstacle matching inputs."""
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from indoor_vt.acquisition.observations import Observations
from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.execution.obstacle_reconstruction import layout_inputs


def request(path, *, post=False, **query):
    url = "http://127.0.0.1:8771" + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return urllib.request.urlopen(urllib.request.Request(url, method="POST" if post else "GET"))


def main():
    root = Path("runs/redesign/rectangle-matched-obstacle-reconstruction-v1")
    config, rows = read_json(root / "config.json"), read_json(root / "results.json")
    assert len(rows) == 140
    observed = Observations(config["data_root"], config)
    room = observed.rooms[0]
    overview = json.load(request("/api/overview"))
    current = [r for r in overview["conditions"] if "obstacle_count" in r]
    assert len(current) == 140
    assert {r["run"] for r in current} == {root.name}
    assert {r["input_count"] for r in current} == {270}
    maps = []
    for n in (0, 1, 3, 5):
        row = min((r for r in rows if r["obstacle_count"] == n), key=lambda r: r["metrics"]["ospa"])
        result = json.load(request("/api/result", post=True, id=row["id"]))
        case, = result["cases"]
        assert case["input_positions"] == 270 and len(case["obstacles"]) == n
        assert case["realization"] == 1
        survey = layout_inputs(observed, room, next(layout for layout in room["layouts"] if layout["box_count"] == n))
        for ap in ("A", "B", "A+B"):
            item = case["maps"][ap]
            np.testing.assert_array_equal(item["scene"]["survey_xyz"], survey.xyz)
            assert len(item["scene"]["obstacles"]) == n
            assert len(item["truth"]) == row["source_counts"][0]["true"][ap]
            assert len(item["predicted"]) // 3 == row["source_counts"][0]["predicted"][ap]
            for metric in ("ospa", "f1", "matched_rmse_m"):
                np.testing.assert_allclose(item["metrics"][metric], row["source_metrics_by_ap"][ap][metric], rtol=0, atol=1e-12)
        maps.append({"obstacles": n, "receipt": row["id"], "ap_metric_sets": 3, "receiver_count": 270})
    row = next(r for r in rows if r["obstacle_count"] == 5 and r["view_method_id"] == "mpurge_unified")
    survey = layout_inputs(observed, room, next(layout for layout in room["layouts"] if layout["box_count"] == 5))
    pair = json.load(request("/api/match", post=True, trial=row["id"], ap="A", left=100, right=150, kind="survey"))
    np.testing.assert_array_equal(pair["left_ranges_m"], survey.channels["A"][100])
    np.testing.assert_array_equal(pair["right_ranges_m"], survey.channels["A"][150])
    try:
        request("/api/match", post=True, trial=row["id"], ap="A", left=0, right=1, kind="query")
        raise AssertionError("Obstacle query matching must not substitute empty-room inputs")
    except urllib.error.HTTPError as error:
        assert error.code == 400
    files = ["acquisition/matched_obstacles.py", "execution/obstacle_reconstruction.py"]
    assert all(request("/source", file=file).status == 200 for file in files)
    report = {"overview_obstacle_conditions": 140, "all_obstacle_rows_use_matched_survey": True,
              "maps_verified": maps, "all_maps_have_identical_receiver_coordinates": True,
              "obstacle_pair_matching_uses_actual_acquired_ranges": True,
              "query_matching_rejected_for_obstacle_source_receipts": True,
              "scientific_provenance_links_http_status": 200}
    write_json(root / "viewer_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
