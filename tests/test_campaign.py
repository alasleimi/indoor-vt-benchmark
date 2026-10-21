"""Exercise the real calibration/freeze/test path on a small labelled fixture."""

import gzip
import json
import numpy as np
import pytest

from indoor_vt import campaign
from indoor_vt.dataset import atomic_npz, pack
from indoor_vt.provenance import digest, file_hash, read_json, write_json
from test_components import planted


def fixture_dataset(root):
    survey, source = planted()
    rooms, truth = [], {}
    for split in ("calibration", "test"):
        room = {
            "id": split,
            "split": split,
            "survey_xyz": survey.xyz.tolist(),
            "layouts": [{"id": "layout_00", "queries_xyz": [[1.0, 1.0, 1.2]]}],
        }
        rooms.append(room)
        truth[split] = {}
        for ap, offset in (("A", np.zeros(3)), ("B", np.array([1.0, 0.0, 0.0]))):
            actual = source + offset
            truth[split][ap] = [{"xyz": actual.tolist()}]
            for label, points in (("survey", survey.xyz), ("layout_00", np.array([[1, 1, 1.2]]))):
                scenario = root / "data/development/scenarios" / split / ap / label
                write_json(
                    scenario / "input_manifest.json", {"position_indices": list(range(len(points)))}
                )
                directory = scenario / "observations/direct/0.0"
                rows = [[r] for r in np.linalg.norm(points - actual, axis=1)]
                values, offsets = pack(rows)
                path = directory / "delays.npz"
                atomic_npz(path, {"ranges_m": values, "offsets": offsets})
                write_json(directory / "manifest.json", {"files": {"delays.npz": file_hash(path)}})
    design = {
        "rooms": rooms,
        "config": {"direct_sigma_m": [0.0], "receiver_snr_db": []},
        "seeds": {"root": 42},
    }
    design["design_hash"] = digest(design)
    write_json(root / "data/development/design.json", design)
    write_json(
        root / "data/development/evaluation/source_truth.json",
        {"sources": truth, "truth_hash": digest(truth)},
    )
    write_json(
        root / "configs/calibration.json",
        {"budget": {"broad": 0, "maximum_expansions": 0, "expansion": 0, "refine": 0}},
    )


@pytest.mark.parametrize("regime", ["component", "end_to_end"])
def test_complete_campaign_freeze_and_reject_changed_data(tmp_path, monkeypatch, regime):
    fixture_dataset(tmp_path)
    # A one-candidate fixture tests orchestration without running a parameter study.
    for name in ("initial_space", "vt_space", "lea_space"):
        monkeypatch.setattr(campaign, name, lambda *args: ({}, []))
    freeze = campaign.calibrate(
        tmp_path, "development", "direct", regime, "mpurge_map", "mpurge_local"
    )
    frozen = read_json(freeze)
    assert frozen["selection_split"] == "calibration"
    assert (
        campaign.calibrate(tmp_path, "development", "direct", regime, "mpurge_map", "mpurge_local")
        == freeze
    )
    output = campaign.evaluate_frozen(tmp_path, freeze)
    with gzip.open(output, "rt") as handle:
        result = json.load(handle)
    assert result["localization"]["case_count"] == 3  # A, B and A+B
    assert all(c["room"] == "test" for c in result["localization"]["cases"])
    assert np.isfinite(result["final_mean_error_m"])
    assert result["vt_ospa"] < 1e-6
    manifest = (
        tmp_path / "data/development/scenarios/test/A/survey/observations/direct/0.0/manifest.json"
    )
    changed = read_json(manifest)
    changed["changed_after_freeze"] = True
    write_json(manifest, changed)
    with pytest.raises(RuntimeError, match="exactly the frozen"):
        campaign.evaluate_frozen(tmp_path, freeze)
