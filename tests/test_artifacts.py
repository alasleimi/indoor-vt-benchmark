import gzip
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from indoor_vt.dataset import materialize, delay_file, load_observations
from indoor_vt.provenance import SeedBook, file_hash, read_json, write_json
from indoor_vt.audit import audit_observations
from indoor_vt.observations import native_frames
import pytest


def test_frontend_versions_replay_and_keep_old_artifacts(tmp_path):
    scenario = tmp_path / "room/A/survey"
    scenario.mkdir(parents=True)
    native = scenario / "native/qdOutput.json.gz"
    native.parent.mkdir()
    record = {
        "TX": 0,
        "RX": 1,
        "PAA_TX": 0,
        "PAA_RX": 0,
        "Delay": [[1e-8, 2e-8], [2e-8]],
        "Gain": [[-70, -80], [-75]],
        "Phase": [[0, 0.1], [0.2]],
        "AOAAZ": [[0, 90], [180]],
        "AOAEL": [[90, 90], [90]],
        "AODAZ": [[0, 90], [180]],
        "AODEL": [[90, 90], [90]],
    }
    raw = json.dumps(record).encode() + b"\n"
    with gzip.open(native, "wb") as f:
        f.write(raw)
    write_json(
        scenario / "input_manifest.json",
        {"frames": 2, "position_indices": [0, 1], "key": "room/A/survey"},
    )
    write_json(
        scenario / "receipt.json",
        {
            "outputs": {"native/qdOutput.json.gz": file_hash(native)},
            "native_uncompressed_sha256": "test-native",
        },
    )
    config = read_json(Path(__file__).parents[1] / "configs/dataset.json")
    config["direct_sigma_m"] = [0.0, 0.1]
    config["receiver_snr_db"] = [25]
    first = materialize(scenario, config, SeedBook(42, "test"))
    assert materialize(scenario, config, SeedBook(42, "test")) == first
    base = scenario / "observations/direct/0.1"
    old_path = delay_file(base)
    old_hash = file_hash(old_path)
    changed = deepcopy(config)
    changed["receiver"]["dynamic_range_db"] = 40
    materialize(scenario, changed, SeedBook(42, "test"))
    new_path = delay_file(base)
    assert new_path != old_path and file_hash(old_path) == old_hash
    # The direct track's realized values are unchanged by an unrelated frontend setting.
    assert file_hash(new_path) == old_hash
    restored = materialize(scenario, config, SeedBook(42, "test"))
    assert restored == first and delay_file(base) == old_path
    rows = load_observations(tmp_path, "room", ("A",), "survey", "direct", "0.0")
    np.testing.assert_allclose(rows[0].channels["A"], [3.0, 6.0])
    audited, missing = audit_observations(
        scenario, config, SeedBook(42, "test"), list(native_frames(native))
    )
    assert len(audited) == 3 and not missing
    old_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        audit_observations(scenario, config, SeedBook(42, "test"), list(native_frames(native)))


def test_atomic_manifest_survives_temporary_windows_reader_lock(tmp_path, monkeypatch):
    from indoor_vt import provenance

    original = provenance.os.replace
    calls = []

    def replace(source, destination):
        calls.append(1)
        if len(calls) < 3:
            error = PermissionError("temporary reader lock")
            error.winerror = 5
            raise error
        return original(source, destination)

    monkeypatch.setattr(provenance.os, "replace", replace)
    monkeypatch.setattr(provenance.time, "sleep", lambda _: None)
    write_json(tmp_path / "status.json", {"complete": False})
    assert read_json(tmp_path / "status.json") == {"complete": False}
    assert len(calls) == 3
