import gzip
import importlib.util
import json
from pathlib import Path

from indoor_vt.provenance import file_hash, write_json


def test_report_aggregates_frozen_results_and_reuses_compact_cache(tmp_path):
    source = Path(__file__).parents[1] / "scripts/report_campaign.py"
    spec = importlib.util.spec_from_file_location("campaign_report_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    write_json(
        tmp_path / "runs/development/campaign/status.json",
        {
            "updated_utc": "2026-10-21T00:00:00+00:00",
            "phase": 2,
            "finished": False,
            "completed": 1,
            "total": 2,
            "running": [],
            "failed": [],
            "searches": [],
        },
    )
    destination = tmp_path / "runs/development/test/direct/component/example.json.gz"
    destination.parent.mkdir(parents=True)
    payload = {
        "freeze": {
            "track": "direct",
            "regime": "component",
            "freeze_hash": "frozen",
            "config": {"initial": {"method": "mpurge_map"}, "vt_method": "mpurge_local"},
        },
        "localization": {
            "initial_mean_m": 2.0,
            "median_m": 0.5,
            "p90_m": 2.0,
            "case_count": 4,
            "fallback_count": 1,
        },
        "vt": {"case_count": 2, "cases": [{"f1": 0.5}, {"f1": 1.0}]},
    }
    with gzip.open(destination, "wt") as handle:
        json.dump(payload, handle)
    write_json(
        destination.with_suffix(".summary.json"),
        {
            "result": str(destination),
            "result_sha256": file_hash(destination),
            "final_mean_error_m": 1.0,
            "vt_ospa": 0.25,
        },
    )
    first = module.report(tmp_path, "development", tmp_path / "shadow")
    assert first["test_results"][0]["vt_f1_at_0_5m"] == 0.75
    assert first["test_results"][0]["initial_mean_error_m"] == 2.0
    second = module.report(tmp_path, "development", tmp_path / "shadow")
    assert first == second
    assert "0.7500" in (tmp_path / "runs/development/campaign/report.md").read_text()
