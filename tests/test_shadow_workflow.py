import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

from indoor_vt import campaign
from indoor_vt.provenance import read_json
from test_campaign import fixture_dataset


@pytest.mark.parametrize(
    "regime,qualifies", [("component", True), ("end_to_end", True), ("component", False)]
)
def test_shadow_uses_canonical_baseline_and_freezes_before_test(
    tmp_path, monkeypatch, regime, qualifies
):
    source = Path(__file__).parents[2] / "experiments/indoor_vt_shadow/workflow.py"
    if not source.exists():
        pytest.skip("The unpromoted shadow workspace is intentionally outside the repository")
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("shadow_workflow_fixture", source)
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    fixture_dataset(tmp_path)
    for name in ("initial_space", "vt_space", "lea_space"):
        monkeypatch.setattr(campaign, name, lambda *args: ({}, []))
    campaign.calibrate(tmp_path, "development", "direct", regime, "mpurge_map", "mpurge_local")
    args = SimpleNamespace(
        benchmark=tmp_path,
        namespace="development",
        track="direct",
        regime=regime,
        scope="local",
        method="weighted_star",
    )
    destination = tmp_path / "shadow_results"
    monkeypatch.setattr(workflow, "output", lambda *_: destination)
    # Test orchestration on both sides of the gate; its numeric 5% rule has a separate test.
    monkeypatch.setattr(workflow, "promotion", lambda *_: qualifies)
    workflow.select(args)
    frozen = read_json(destination / "weighted_star_promotion.json")
    baseline_name = (
        "vt_mpurge_local" if regime == "component" else "pipeline_mpurge_map__mpurge_local"
    )
    canonical = read_json(
        tmp_path / "runs/development/calibration/direct" / regime / (baseline_name + ".json")
    )
    assert frozen["baseline_freeze"] == canonical["freeze_hash"]
    assert frozen["baseline_score"] == canonical["selected"]["score"]
    assert frozen["selection_split"] == "calibration"
    if not qualifies:

        def forbidden(*args):
            raise AssertionError("Unqualified methods must not load test data")

        monkeypatch.setattr(workflow, "evaluator", forbidden)
    result = workflow.evaluate(args)
    assert result["qualifies"] is qualifies
    if qualifies:
        assert result["status"] == "evaluated" and result["query_cases"] == 3
    else:
        assert result["status"].startswith("not_qualified")
