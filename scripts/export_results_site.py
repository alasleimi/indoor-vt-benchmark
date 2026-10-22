"""Copy a verified calibration snapshot for publication; never modify generated data.

Float64 metric columns are gzip-compressed and content-addressed. Identical
columns (including repeated initial errors) are stored once. Calibration delay
inputs are copied into a separate ZIP; native simulator archives remain local.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np

from results_explorer import ROOT, build_index, read_json, sha256, summarize, write_json


REPOSITORY = "https://github.com/alasleimi/indoor-vt-benchmark"


def sanitize(value, root):
    if isinstance(value, dict):
        return {k: sanitize(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, root) for v in value]
    if isinstance(value, str):
        for prefix in (str(root) + "\\", root.as_posix() + "/"):
            if value.startswith(prefix):
                return value[len(prefix) :].replace("\\", "/")
    return value


def export_column(output, values, dtype):
    values = np.asarray(values, dtype=dtype)
    raw = values.tobytes(order="C")
    raw_hash = hashlib.sha256(raw).hexdigest()
    path = output / "columns" / (raw_hash + ".bin.gz")
    if not path.exists():
        path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return {
        "file": path.relative_to(output).as_posix(),
        "dtype": values.dtype.str,
        "length": values.size,
        "bytes": path.stat().st_size,
        "raw_bytes": len(raw),
        "sha256": sha256(path),
        "raw_sha256": raw_hash,
    }


def export_inputs(root, output):
    base = root / "data/development"
    design = read_json(base / "design.json")
    design["rooms"] = [r for r in design["rooms"] if r["split"] == "calibration"]
    entries = []
    archive = output / "calibration-inputs.zip"
    temporary = archive.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("design.json", json.dumps(design, separators=(",", ":")))
        z.writestr("seeds.json", (root / "configs/seeds.json").read_bytes())
        truth = read_json(base / "evaluation/source_truth.json")
        # Keep the original truth schema but only calibration room entries.
        room_ids = {r["id"] for r in design["rooms"]}
        if "sources" in truth:
            truth["sources"] = {k: v for k, v in truth["sources"].items() if k in room_ids}
        elif "rooms" in truth:
            if isinstance(truth["rooms"], dict):
                truth["rooms"] = {k: v for k, v in truth["rooms"].items() if k in room_ids}
            else:
                truth["rooms"] = [
                    r for r in truth["rooms"] if r.get("id", r.get("room")) in room_ids
                ]
        else:
            raise ValueError("Inspect source truth schema before publishing")
        z.writestr("source_truth.json", json.dumps(truth, separators=(",", ":")))
        for room in design["rooms"]:
            for manifest_path in sorted(
                (base / "scenarios" / room["id"]).glob("*/*/observations/*/*/manifest.json")
            ):
                record = read_json(manifest_path)
                source = manifest_path.parent / record["delay_file"]
                expected = record["files"][record["delay_file"]]
                if sha256(source) != expected:
                    raise ValueError(f"Observation checksum mismatch: {source}")
                ap, layout, _, track, level, _ = manifest_path.relative_to(
                    base / "scenarios" / room["id"]
                ).parts
                name = f"observations/{room['id']}/{ap}/{layout}/{track}/{level}.npz"
                z.write(source, name, compress_type=zipfile.ZIP_STORED)
                entries.append(
                    {
                        "room": room["id"],
                        "ap": ap,
                        "layout": layout,
                        "track": track,
                        "level": level,
                        "file": name,
                        "sha256": expected,
                        "identity_hash": record["identity_hash"],
                        "native_hash": record["identity"]["native_hash"],
                        "frames": record["frames"],
                        "peak_count": record["peak_count"],
                        "measurement_seeds": [
                            {"frame_id": m["frame_id"], "seed": m["seed"]}
                            for m in record["measurements"]
                        ],
                    }
                )
        z.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema": "indoor-vt.calibration-inputs.v1",
                    "split": "calibration",
                    "observations": entries,
                    "omitted": [
                        "held-out test rooms",
                        "native power, phase, angles and ray geometry",
                        "receiver waveforms and simulator workspaces",
                    ],
                },
                separators=(",", ":"),
            ),
        )
    if temporary.stat().st_size > 95 * 1024**2:
        raise ValueError("Calibration ZIP exceeds the publication file budget; split it first")
    temporary.replace(archive)
    return {
        "file": archive.name,
        "sha256": sha256(archive),
        "bytes": archive.stat().st_size,
        "observations": len(entries),
    }


def export(root=ROOT, revision="v6", include_inputs=False):
    root = Path(root).resolve()
    output = root / "published" / revision
    (output / "columns").mkdir(parents=True, exist_ok=True)
    cache, index = build_index(root, revision)
    jobs = []
    for number, job in enumerate(index["jobs"], 1):
        array = np.load(cache / job["columns"], mmap_mode="r", allow_pickle=False)
        levels = index["levels"][job["track"]]
        conditions = np.column_stack(
            (
                array["room"],
                array["subset"],
                np.searchsorted(sorted(levels), array["level"]),
                array["layout"] + 1,
            )
        )
        columns = {"conditions": export_column(output, conditions, "u1")}
        fields = (
            ("ospa", "f1", "predicted", "truth")
            if job["stage"] == "vt"
            else ("final", "initial", "fallback")
        )
        for field in fields:
            columns[field] = export_column(
                output, array[field], "u1" if field == "fallback" else "<f8"
            )
        jobs.append(
            sanitize({k: v for k, v in job.items() if k != "columns"}, root)
            | {"metrics": summarize(array, job["stage"]), "columns": columns}
        )
        del array
        if number % 20 == 0:
            print(f"Exported {number}/{len(index['jobs'])} result tables", flush=True)
    status = read_json(root / "runs/development/revisions" / revision / "status.json")
    result = {k: v for k, v in index.items() if k not in ("jobs", "fingerprint")}
    result.update(
        {
            "export_schema": "indoor-vt.pages.v1",
            "mode": "static",
            "repository": REPOSITORY,
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "levels": {k: sorted(v) for k, v in index["levels"].items()},
            "jobs": jobs,
            "counts": dict(Counter(j["stage"] for j in jobs)),
            "status": {
                k: status[k]
                for k in (
                    "completed",
                    "total",
                    "running",
                    "failed",
                    "updated_utc",
                    "execution_state",
                )
            },
        }
    )
    if include_inputs:
        result["inputs"] = export_inputs(root, output)
    elif (output / "meta.json").exists():
        old = read_json(output / "meta.json")
        if "inputs" in old:
            result["inputs"] = old["inputs"]
    write_json(output / "meta.json", result)
    files = [
        {"file": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "manifest.json" and ".tmp" not in p.name
    ]
    total = sum(f["bytes"] for f in files)
    if total > 512 * 1024**2:
        raise ValueError(f"Publication budget exceeded: {total / 1024**2:.1f} MiB")
    write_json(
        output / "manifest.json",
        {
            "schema": "indoor-vt.publication-manifest.v1",
            "total_bytes": total,
            "files": files,
            "metric_precision": "IEEE-754 float64",
            "case_subsampling": False,
            "scientific_artifacts_modified": False,
        },
    )
    print(f"Publication snapshot: {total / 1024**2:.2f} MiB, {len(files)} files", flush=True)
    from build_explorer_assets import build as build_explorer_assets

    build_explorer_assets(root)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-inputs", action="store_true")
    parser.add_argument("--revision", default="v6")
    arguments = parser.parse_args()
    export(revision=arguments.revision, include_inputs=arguments.include_inputs)
