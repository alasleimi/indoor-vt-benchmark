import importlib.util
from pathlib import Path

from indoor_vt.provenance import write_json


def queue_module():
    source = Path(__file__).parents[1] / "scripts/run_campaign.py"
    spec = importlib.util.spec_from_file_location("campaign_queue", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_graph_covers_methods_tracks_and_regimes(tmp_path):
    module = queue_module()
    jobs = module.build_jobs(tmp_path, "development", tmp_path / "shadow")
    assert len(jobs) == 248 and len({j["id"] for j in jobs}) == 248
    assert [sum(j["phase"] == phase for j in jobs) for phase in (0, 1, 2)] == [60, 64, 124]
    pairs = module.pipeline_pairs()
    assert len(pairs) == len(set(pairs)) == 15
    assert {v for _, v in pairs} == set(module.VT_METHODS)
    assert {i for i, _ in pairs} == set(module.INITIAL_METHODS)
    assert all(j["phase"] == 2 for j in jobs if "evaluate" in j["kind"])


def test_queue_waits_for_selection_before_test_and_reuses_completion(tmp_path, monkeypatch):
    module = queue_module()
    write_json(
        tmp_path / "data/development/archive_audit.json",
        {
            "missing": [],
            "missing_observations": [],
            "completed_scenarios": 1,
            "expected_scenarios": 1,
            "completed_observation_conditions": 1,
            "expected_observation_conditions": 1,
        },
    )
    jobs = [
        {
            "id": str(i),
            "phase": i,
            "kind": "fake",
            "command": [str(i)],
            "locks": [],
            "artifact": str(tmp_path / f"result_{i}.json"),
        }
        for i in range(3)
    ]
    launched = []

    class Process:
        pid = 42
        returncode = 0

        def __init__(self, command, **kwargs):
            phase = int(command[-1])
            assert all(Path(jobs[i]["artifact"]).exists() for i in range(phase))
            launched.append(phase)
            write_json(Path(jobs[phase]["artifact"]), {})

        def poll(self):
            return 0

    monkeypatch.setattr(module, "build_jobs", lambda *_: jobs)
    monkeypatch.setattr(
        module, "code_provenance", lambda *_: {"source_hash": "fixed", "git_commit": "commit"}
    )
    monkeypatch.setattr(module, "process_signature", lambda *_: 123)
    monkeypatch.setattr(
        module,
        "completion",
        lambda job, _: {"artifact": job["artifact"]} if Path(job["artifact"]).exists() else None,
    )
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    result = module.run(tmp_path, "development", tmp_path / "shadow")
    assert result["finished"] and result["completed"] == 3 and launched == [0, 1, 2]
    module.run(tmp_path, "development", tmp_path / "shadow")
    assert launched == [0, 1, 2]
