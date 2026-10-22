"""Checks for scientific aggregation and provenance in the read-only explorer."""

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "results_explorer", Path(__file__).parents[1] / "scripts/results_explorer.py"
)
explorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explorer)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def campaign(tmp_path):
    revision = tmp_path / "runs/development/revisions/v6"
    rooms = [
        {
            "id": f"room{i}",
            "family": family,
            "index": 0,
            "split": split,
            "width_m": 6,
            "depth_m": 7,
            "ceiling_m": 3,
            "layouts": [{"id": "layout_00", "box_count": 0}, {"id": "layout_01", "box_count": 1}],
        }
        for i, (family, split) in enumerate(
            [("rectangle", "calibration"), ("L", "calibration"), ("T", "test")]
        )
    ]
    save(
        tmp_path / "data/development/design.json",
        {"rooms": rooms, "config": {"direct_sigma_m": [0], "receiver_snr_db": [40, 25, 20]}},
    )
    cases = [
        {
            "room": f"room{room}",
            "subset": subset,
            "level": level,
            "layout": layout,
            "final_error_m": error,
            "initial_error_m": error,
            "final_status": "fallback_empty" if error == 100 else "ok",
        }
        for room, subset, level, layout, error in [
            (0, ["A"], "40", "layout_00", 0),
            (0, ["A"], "40", "layout_01", 100),
            (1, ["B"], "25", "layout_00", 0),
            (1, ["A", "B"], "20", "layout_01", 0),
        ]
    ]
    for track in ("receiver", "direct"):
        for stage in ("coarse", "pipeline", "vt"):
            params = (
                {"scope": "local", "merger": "greedy_median"}
                if stage == "vt"
                else (
                    {
                        "initial": {"method": "chamfer"},
                        "vt_method": "mpurge_local",
                        "vt": {"scope": "local"},
                        "lea": {},
                    }
                    if stage == "pipeline"
                    else {}
                )
            )
            method = (
                "mpurge_local"
                if stage == "vt"
                else ("chamfer__mpurge_local" if stage == "pipeline" else "chamfer")
            )
            identity = {"track": track, "stage": stage, "method": method, "parameters": params}
            if stage == "vt":
                selected = [
                    {
                        "room": f"room{i}",
                        "subset": ["A"],
                        "level": "40",
                        "ospa": ospa,
                        "f1": f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "ospa_p": 1,
                        "ospa_c_m": 1,
                        "f1_gate_m": 0.5,
                        "nested": [[], {"quote": 'escaped " quote · Unicode'}],
                    }
                    for i, (ospa, f1, tp, fp, fn) in enumerate(
                        [(0.1, 1, 1, 0, 0), (0.9, 0, 0, 99, 99)]
                    )
                ]
                audit, score = {"case_count": 2, "cases": selected}, 0.5
            else:
                audit, score = (
                    {
                        "case_count": 4,
                        "initial_mean_m": 25,
                        "median_m": 0,
                        "p90_m": 70,
                        "fallback_count": 1,
                        "cases": cases,
                    },
                    25,
                )
            artifact = revision / "objectives" / f"{track}_{stage}.json.gz"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(artifact, "wt", encoding="utf-8") as handle:
                json.dump({"identity": identity, "score": score, "audit": audit}, handle)
            save(
                revision / "presets" / f"{track}_{stage}.json",
                {
                    "job": identity | {"id": f"preset/{track}/{stage}/{method}"},
                    "score": score,
                    "selection_split": "calibration",
                    "status": "uncalibrated preset",
                    "code_hash": "current",
                    "audit": {
                        "artifact": str(artifact),
                        "sha256": explorer.sha256(artifact),
                        "case_count": audit["case_count"],
                    },
                },
            )
    save(revision / "status.json", {"running": [], "completed": 6})
    return tmp_path, revision


def test_streaming_across_tiny_chunks_and_truncation(campaign):
    root, revision = campaign
    path = revision / "objectives/receiver_vt.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        expected = json.load(handle)
    stream = explorer.ObjectiveCases(path, chunk_size=7)
    assert list(stream) == expected["audit"]["cases"]
    assert stream.header["identity"] == expected["identity"]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(expected)[:-30])
    with pytest.raises(ValueError, match="Truncated"):
        list(explorer.ObjectiveCases(path, chunk_size=7))


def test_exact_quantiles_macro_f1_filters_and_track_isolation(campaign):
    root, revision = campaign
    cache, index = explorer.build_index(root)
    app = explorer.Explorer(cache, index, revision / "status.json")
    assert len(index["rooms"]) == 2  # Held-out room is not exposed.
    local = app.rows({"view": "raw", "track": "receiver"})["rows"]
    assert len(local) == 1  # Direct NIST must never be pooled with receiver.
    assert local[0]["mean"] == 25
    assert local[0]["median"] == 0
    assert local[0]["p90"] == pytest.approx(70)
    assert local[0]["fallbacks"] == 1
    assert len(app.rows({"view": "all", "track": "receiver"})["rows"]) == 2
    assert len(app.rows({"view": "all", "stage": "pipeline"})["rows"]) == 1
    empty = app.rows({"view": "all", "family": "not_a_room"})
    assert empty["rows"] == []
    selected = app.rows(
        {"view": "raw", "room": "room0", "noise": "40", "ap": "A", "obstacles": "1", "layout": "1"}
    )["rows"][0]
    assert selected["count"] == 1 and selected["mean"] == 100
    vt = app.rows({"view": "vt", "obstacles": "1", "layout": "1"})["rows"][0]
    assert vt["f1"] == 0.5  # Macro-F1, not pooled TP/FP/FN (0.01).
    assert vt["predicted"] == vt["truth"] == 50
    detail = app.detail({"id": local[0]["id"]})
    assert detail["rooms"][0]["count"] == 2
    assert detail["rooms"][0]["mean"] == 50
    assert detail["ecdf"] == [[0, 0.75], [0, 0.75], [0, 0.75], [100, 1]]
    assert app.detail({"id": local[0]["id"], "room": "missing"})["metrics"] is None


def test_cache_reuse_and_checksum_failure(campaign, monkeypatch):
    root, revision = campaign
    cache, index = explorer.build_index(root)
    original_hash = explorer.sha256

    def unexpected_read(path):
        pytest.fail("Unchanged startup must reuse the verified index")

    monkeypatch.setattr(explorer, "sha256", unexpected_read)
    assert explorer.build_index(root)[1] == index
    monkeypatch.setattr(explorer, "sha256", original_hash)
    # Changing even one source invalidates the index and must fail verification.
    with (revision / "objectives/receiver_vt.json.gz").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        explorer.build_index(root)
    assert np.load(cache / index["jobs"][0]["columns"]).size > 0


def test_changed_receipt_is_checked_even_with_cached_columns(campaign):
    root, revision = campaign
    explorer.build_index(root)
    receipt_path = revision / "presets/receiver_vt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["score"] = 0.2
    save(receipt_path, receipt)
    with pytest.raises(ValueError, match="Cached objective score/count mismatch"):
        explorer.build_index(root)
