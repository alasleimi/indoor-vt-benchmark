import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from indoor_vt.evaluation import gzip_json
from indoor_vt.provenance import json_default, read_json, write_json
from test_campaign import fixture_dataset


def test_streamed_audit_keeps_exact_json_bytes(tmp_path):
    value = {
        "values": np.random.default_rng(4).normal(size=(20000, 3)),
        "unicode": "\u00e9\U0001f600",
        "small": [True, None, -0.0, 1e-200, 1e200],
    }
    expected = json.dumps(
        value, separators=(",", ":"), allow_nan=False, default=json_default
    ).encode()
    path = tmp_path / "audit.gz"
    gzip_json(path, value)
    with gzip.open(path, "rb") as handle:
        assert handle.read() == expected
    first = path.read_bytes()
    gzip_json(path, value)
    assert path.read_bytes() == first


@pytest.mark.skipif(os.name != "nt", reason="Windows quota enforcement")
def test_private_and_file_backed_memory_quotas_in_isolated_process(tmp_path):
    script = """
import json,tempfile,numpy as np
from indoor_vt.worker_memory import install_limits,memory_snapshot
install_limits(128*1024**2,512*1024**2)
blocked=False
try: x=np.empty(600*1024**2,dtype=np.uint8)
except MemoryError: blocked=True
with tempfile.TemporaryFile(dir=__import__('sys').argv[1]) as f:
    f.truncate(256*1024**2)
    a=np.memmap(f,mode='r+',dtype=np.uint8,shape=(256*1024**2,))
    a[:]=1
    result=memory_snapshot()
    a._mmap.close()
print(json.dumps(dict(blocked=blocked,**result)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 0, result.stderr
    memory = json.loads(result.stdout)
    assert memory["blocked"]
    assert memory["peak_working_set_bytes"] <= 129 * 1024**2
    assert memory["peak_private_bytes"] <= 512 * 1024**2
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="Windows campaign worker quotas")
def test_runner_replaces_worker_between_tasks_and_records_memory(tmp_path):
    fixture_dataset(tmp_path)
    write_json(
        tmp_path / "data/development/archive_audit.json",
        {"missing": [], "missing_observations": []},
    )
    helper = Path(__file__).parents[1] / "scripts/probe_worker_lifecycle.py"
    result = subprocess.run(
        [sys.executable, str(helper), str(tmp_path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    base = tmp_path / "runs/development/revisions/memory_fixture"
    status = read_json(base / "status.json")
    assert status["finished"] and status["completed"] == 2
    assert len({value["pid"] for value in status["jobs"].values()}) == 2
    assert all(value["worker_memory"]["private_bytes"] > 0 for value in status["jobs"].values())
    assert status["worker_pids"] == []
    assert status["memory_policy"]["tasks_per_worker"] == 1
    assert (base / "memory.jsonl").read_text()
