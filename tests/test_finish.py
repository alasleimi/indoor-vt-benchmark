from pathlib import Path
import pytest
from indoor_vt import finish as module
from indoor_vt.provenance import write_json, read_json


def test_finisher_starts_missing_jobs_derives_and_audits(tmp_path, monkeypatch):
    write_json(
        tmp_path / "data/development/design.json",
        {"rooms": [{}], "config": {}, "seeds": {"root": 42}},
    )
    calls = {"launch": 0, "derive": [], "audit": 0}

    class Process:
        def __init__(self, command, **kwargs):
            calls["launch"] += 1
            assert "simulate" in command and command[-1] == "4"
            for ap in ("A", "B"):
                for index in range(11):
                    write_json(
                        tmp_path
                        / "data/development/scenarios/room"
                        / ap
                        / str(index)
                        / "receipt.json",
                        {},
                    )

        def poll(self):
            return 0

    def audit(root, namespace):
        calls["audit"] += 1
        assert len(calls["derive"]) == 22
        return {"frames": 100}

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "materialize", lambda path, *_: calls["derive"].append(path))
    monkeypatch.setattr(module, "audit_dataset", audit)
    monkeypatch.setattr(module, "replay_pilot", lambda *_: {"status": "exact"})
    result = module.finish(tmp_path, "development", Path("nist"), "matlab", 4)
    assert result["finished"] and result["native_frames"] == 100
    assert calls["launch"] == 1 and calls["audit"] == 1
    assert len(set(calls["derive"])) == 22
    assert read_json(tmp_path / "runs/development/generation/status.json") == result


def test_finisher_lock_rejects_overlapping_runner_and_releases(tmp_path):
    path = tmp_path / "runner.lock"
    with module.exclusive_lock(path):
        with pytest.raises(OSError):
            with module.exclusive_lock(path):
                pass
    with module.exclusive_lock(path):
        pass
