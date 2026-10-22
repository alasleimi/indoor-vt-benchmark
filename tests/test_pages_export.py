"""Check lossless publication columns and fail closed on missing/corrupt LFS data."""

import ast
import gzip
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


def test_metric_export_preserves_float_bits_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    export = importlib.import_module("export_results_site")
    (tmp_path / "columns").mkdir()
    values = np.array([0.0, -0.0, 0.1, np.nextafter(0.0, 1.0), 1.2345678901234567])
    first = export.export_column(tmp_path, values, "<f8")
    second = export.export_column(tmp_path, values.copy(), "<f8")
    assert first == second
    assert len(list((tmp_path / "columns").iterdir())) == 1
    compressed = (tmp_path / first["file"]).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == first["sha256"]
    assert gzip.decompress(compressed) == values.astype("<f8").tobytes()


def test_pages_build_verifies_data_and_keeps_archive_out_of_site(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    pages = importlib.import_module("build_pages")
    snapshot = tmp_path / "published/v6"
    snapshot.mkdir(parents=True)
    payloads = {"meta.json": b"{}", "calibration-inputs.zip": b"test archive"}
    records = []
    for name, payload in payloads.items():
        (snapshot / name).write_bytes(payload)
        records.append({"file": name, "sha256": hashlib.sha256(payload).hexdigest()})
    (snapshot / "manifest.json").write_text(json.dumps({"files": records}))
    assets = tmp_path / "web/results-explorer"
    assets.mkdir(parents=True)
    for name in (
        "index.html",
        "style.css",
        "app.js",
        "data-client.js",
        "static-worker.js",
        "method-catalog.json",
    ):
        (assets / name).write_text('<html lang="en">')
    output = pages.build(tmp_path)
    assert 'data-mode="static"' in (output / "index.html").read_text()
    assert (output / "meta.json").read_bytes() == payloads["meta.json"]
    assert not (output / "calibration-inputs.zip").exists()
    assert (snapshot / "calibration-inputs.zip").read_bytes() == payloads["calibration-inputs.zip"]
    (snapshot / "calibration-inputs.zip").write_bytes(b"version https://git-lfs.github.com/spec/v1")
    with pytest.raises(ValueError, match="checksum mismatch"):
        pages.build(tmp_path)
    with pytest.raises(ValueError, match="separate from generated data"):
        pages.build(tmp_path, tmp_path / "data")


def test_every_catalog_link_points_to_its_named_definition():
    catalog = json.loads(
        (ROOT / "web/results-explorer/method-catalog.json").read_text(encoding="utf8")
    )
    assert len(catalog["methods"]) == 18
    for entry in catalog["methods"].values():
        assert entry["code"]
        for link in entry["code"]:
            tree = ast.parse((ROOT / link["path"]).read_text(encoding="utf8"))
            assert any(
                isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name == link["symbol"]
                and node.lineno == link["line"]
                for node in ast.walk(tree)
            ), link
            assert link["url"].startswith("https://github.com/alasleimi/indoor-vt-benchmark/blob/")
    assert len(catalog["methods"]["vt/zayets2018"]["steps"]) == 10
