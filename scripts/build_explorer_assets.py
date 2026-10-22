"""Build lightweight UI summaries and 3D maps from existing, verified results only."""

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from results_explorer import (
    ROOT,
    Explorer,
    ObjectiveCases,
    build_index,
    read_json,
    sha256,
    write_json,
)


def blob(output, raw):
    checksum = hashlib.sha256(raw).hexdigest()
    path = output / "bundles" / f"{checksum}.bin.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    return {
        "file": path.relative_to(output).as_posix(),
        "bytes": path.stat().st_size,
        "raw_bytes": len(raw),
        "sha256": sha256(path),
        "raw_sha256": checksum,
    }


def json_blob(output, value):
    return blob(output, json.dumps(value, separators=(",", ":"), allow_nan=False).encode())


def column_bytes(output, descriptor):
    packed = (output / descriptor["file"]).read_bytes()
    if hashlib.sha256(packed).hexdigest() != descriptor["sha256"]:
        raise ValueError("Published column checksum mismatch")
    raw = gzip.decompress(packed)
    if hashlib.sha256(raw).hexdigest() != descriptor["raw_sha256"]:
        raise ValueError("Published raw column checksum mismatch")
    start = descriptor.get("byte_offset", 0)
    length = descriptor.get("byte_length", descriptor["raw_bytes"])
    return raw[start : start + length]


def build(root=ROOT):
    root = Path(root)
    output = root / "published/v6"
    meta = read_json(output / "meta.json")
    cache, index = build_index(root, "v6")
    explorer = Explorer(cache, index, root / "runs/development/revisions/v6/status.json")

    # One tiny request per VT track instead of dozens of independently compressed columns.
    for track in meta["levels"]:
        payload = bytearray()
        slices = {}
        assignments = []
        for job in meta["jobs"]:
            if job["stage"] != "vt" or job["track"] != track:
                continue
            for field, descriptor in job["columns"].items():
                raw = column_bytes(output, descriptor)
                key = hashlib.sha256(raw).hexdigest()
                if key not in slices:
                    payload.extend(b"\0" * (-len(payload) % 8))
                    slices[key] = (len(payload), len(raw))
                    payload.extend(raw)
                assignments.append((job, field, descriptor, slices[key]))
        packed = blob(output, bytes(payload))
        for job, field, original, (offset, length) in assignments:
            job["columns"][field] = {
                **packed,
                "dtype": original["dtype"],
                "length": original["length"],
                "byte_offset": offset,
                "byte_length": length,
            }

    # Room/family filters should fetch only those rooms, with shared columns stored once.
    local_jobs = {job["id"]: job for job in index["jobs"]}
    partitions = defaultdict(dict)
    for track in meta["levels"]:
        jobs = [job for job in meta["jobs"] if job["stage"] != "vt" and job["track"] == track]
        for room_index, room in enumerate(meta["rooms"]):
            payload, slices, assignments = bytearray(), {}, []
            for job in jobs:
                full = np.load(cache / local_jobs[job["id"]]["columns"], mmap_mode="r")
                selected = full[full["room"] == room_index]
                columns = {
                    "conditions": np.column_stack(
                        (
                            selected["room"],
                            selected["subset"],
                            np.searchsorted(meta["levels"][track], selected["level"]),
                            selected["layout"] + 1,
                        )
                    ).astype("u1")
                }
                columns.update(
                    {
                        name: selected[name].astype("u1" if name == "fallback" else "<f8")
                        for name in ("final", "initial", "fallback")
                    }
                )
                for field, values in columns.items():
                    raw = values.tobytes()
                    key = hashlib.sha256(raw).hexdigest()
                    if key not in slices:
                        payload.extend(b"\0" * (-len(payload) % 8))
                        slices[key] = (len(payload), len(raw))
                        payload.extend(raw)
                    assignments.append(
                        (job["id"], field, values.dtype.str, values.size, slices[key])
                    )
                del full, selected, columns
            packed = blob(output, bytes(payload))
            rows = {}
            for job_id, field, dtype, length, (offset, size) in assignments:
                rows.setdefault(job_id, {})[field] = {
                    **packed,
                    "dtype": dtype,
                    "length": length,
                    "byte_offset": offset,
                    "byte_length": size,
                }
            partitions[track][room["id"]] = rows
    meta["partitions"] = json_blob(output, dict(partitions))

    # Default tables already have exact metrics. Default detail panels also need no raw arrays.
    details = {}
    for job in meta["jobs"]:
        detail = explorer.detail({"id": job["id"]})
        details[job["id"]] = {k: detail[k] for k in ("metrics", "rooms", "ecdf")}
    meta["details"] = json_blob(output, details)

    with zipfile.ZipFile(output / "calibration-inputs.zip") as archive:
        design = json.loads(archive.read("design.json"))
        truth = json.loads(archive.read("source_truth.json"))["sources"]
    scenes = {
        room["id"]: {
            **{
                k: room[k]
                for k in (
                    "id",
                    "family",
                    "polygon_xy",
                    "ceiling_m",
                    "access_points_xyz",
                    "survey_xyz",
                )
            },
            "truth": truth[room["id"]],
        }
        for room in design["rooms"]
    }
    maps = defaultdict(lambda: {"points": {}, "methods": {}})
    for job in meta["jobs"]:
        if job["stage"] != "vt":
            continue
        artifact = root / job["artifact"]
        if sha256(artifact) != job["artifact_sha256"]:
            raise ValueError(f"Result checksum mismatch: {job['id']}")
        singles = {}
        for case in ObjectiveCases(artifact):
            if len(case["subset"]) != 1:
                continue
            pack = maps[(job["track"], case["room"])]
            points = case["predicted"]
            raw = np.asarray(points, dtype="<f8").reshape(-1, 3).tobytes()
            key = hashlib.sha256(raw).hexdigest()
            pack["points"][key] = {**blob(output, raw), "count": len(points)}
            singles[(case["room"], float(case["level"]), case["subset"][0])] = key
        for case in ObjectiveCases(artifact):
            pack = maps[(job["track"], case["room"])]
            channels = {
                ap: singles[(case["room"], float(case["level"]), ap)] for ap in case["subset"]
            }
            combined = b"".join(
                gzip.decompress((output / pack["points"][key]["file"]).read_bytes())
                for key in channels.values()
            )
            if combined != np.asarray(case["predicted"], dtype="<f8").reshape(-1, 3).tobytes():
                raise ValueError("A+B source map differs from its saved single-AP maps")
            record = {
                "points": channels,
                **{
                    k: case[k]
                    for k in (
                        "matches",
                        "ospa",
                        "f1",
                        "tp",
                        "fp",
                        "fn",
                        "f1_gate_m",
                    )
                },
            }
            pack["methods"].setdefault(job["id"], {}).setdefault("+".join(case["subset"]), {})[
                str(float(case["level"]))
            ] = record
    descriptors = defaultdict(dict)
    for (track, room), pack in maps.items():
        descriptors[track][room] = json_blob(output, pack)
    meta["visualizer"] = {"scenes": json_blob(output, scenes), "maps": dict(descriptors)}
    meta["export_schema"] = "indoor-vt.pages.v2"
    write_json(output / "meta.json", meta)
    files = [
        {"file": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "manifest.json" and ".tmp" not in p.name
    ]
    manifest = read_json(output / "manifest.json")
    manifest.update(files=files, total_bytes=sum(f["bytes"] for f in files))
    write_json(output / "manifest.json", manifest)
    print(
        f"Built summaries for {len(details)} results and 3D maps in {len(maps)} room/track bundles"
    )
    print(
        f"3D map indices: {sum(d['bytes'] for rooms in descriptors.values() for d in rooms.values()) / 1024**2:.2f} MiB"
    )
    return meta


if __name__ == "__main__":
    build()
