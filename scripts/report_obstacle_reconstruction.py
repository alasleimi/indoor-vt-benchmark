"""Audit saved obstacle results and export an AP-specific comparison table."""
import csv
import json
from pathlib import Path
import urllib.request

import numpy as np

from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.execution.artifacts import read_npz
from indoor_vt.execution.obstacle_reconstruction import metrics_for_record


def main(root=Path("runs/redesign/rectangle-obstacle-reconstruction-v1")):
    root = Path(root)
    rows, protocol = read_json(root / "results.json"), read_json(root / "protocol.json")
    input_count = protocol.get("common_survey_mask", {}).get("retained_count", 100)
    expected = {(s["id"], n) for s in protocol["methods"] for n in (0, 1, 3, 5)}
    assert len(rows) == len(expected) == 140
    assert {(r["view_method_id"], r["obstacle_count"]) for r in rows} == expected
    checks, data = 0, []
    for row in rows:
        saved, _ = read_npz(root / row["outcome"])
        record, = saved["sources"]
        assert len(record["survey_xyz"]) == input_count
        assert len(record["obstacles"]) == row["obstacle_count"]
        metrics, counts = metrics_for_record(record)
        assert counts == row["source_counts"][0]
        origin = row["frozen_origin"]
        original = read_json(root.parent / origin["run"] / "evaluations" / (origin["receipt_id"] + ".json"))
        for key in ("parameters", "frozen", "unify", "method", "label"):
            assert row[key] == original[key]
        for ap in ("A", "B", "A+B"):
            for key in ("ospa", "f1", "matched_rmse_m"):
                value = metrics[ap][key]
                assert value is None and row["source_metrics_by_ap"][ap][key] is None or np.isclose(
                    value, row["source_metrics_by_ap"][ap][key], rtol=0, atol=1e-12)
            checks += 1
            data.append({"method": row["view_method_id"], "obstacles": row["obstacle_count"],
                         "layout": row["layout"], "ap": ap, "sigma_m": 0,
                         **{k: metrics[ap][k] for k in ("ospa", "f1", "matched_rmse_m", "tp", "fp", "fn")},
                         "predicted_count": counts["predicted"][ap], "true_count": counts["true"][ap],
                         "calibration_run": origin["run"], "frozen_receipt": origin["receipt_id"],
                         "receipt": row["id"]})
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    old = read_json(root / "previous_overview.json")
    live = json.load(urllib.request.urlopen("http://127.0.0.1:8771/api/overview"))
    before = {(r["receipt_id"], r["method_id"]): r for r in old["conditions"] if "obstacle_count" not in r}
    after = {(r["receipt_id"], r["method_id"]): r for r in live["conditions"] if "obstacle_count" not in r}
    assert before == after
    assert len([r for r in live["conditions"] if "obstacle_count" in r]) == 140
    assert len(live["trials"]) == len(old["trials"])
    verification = {"completed_conditions": len(rows), "verified_ap_metric_sets": checks,
                    "original_condition_rows_unchanged": len(before), "selection_trials_unchanged": len(old["trials"]),
                    "all_frozen_parameters_identical": True, "input_positions_per_layout": input_count,
                    "omissions": [], "viewer_http_status": 200}
    write_json(root / "verification.json", verification)
    lines = ["Frozen obstacle reconstruction", "35 variants x 4 layouts = 140 completed conditions. No new tuning.",
             protocol["input"], protocol["positions"],
             "Truth: union of source images visible in archived NIST paths at these receivers, direct plus up to 2 reflections.",
             "No identity/coordinate truth enters solvers; only RMSE receives scene-specific source count.", "",
             "Lowest OSPA per layout (descriptive comparison; does not select or change parameters):"]
    for n in (0, 1, 3, 5):
        best = min((r for r in data if r["obstacles"] == n and r["ap"] == "A+B"), key=lambda r: r["ospa"])
        lines.append(f"{n} obstacles: {best['method']}; OSPA={best['ospa']:.9f}; F1={100*best['f1']:.3f}%; matched RMSE={best['matched_rmse_m']:.9f} m; truth={best['true_count']}")
    lines += ["", json.dumps(verification, indent=2), "", "Complete AP-specific results: metrics.csv"]
    (root / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-obstacle-reconstruction-v1"))
    main(parser.parse_args().run)
