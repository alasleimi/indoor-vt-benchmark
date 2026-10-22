"""Read-only, local explorer of completed preset receipts and their exact case metrics.

Scientific artifacts are never changed. A streaming import writes compact NumPy
columns under runs/, then the HTTP server memory-maps those columns for filtering.
No evaluation, optimization, test selection, or campaign workers are started.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
BASE_FIELDS = [("room", "u1"), ("subset", "u1"), ("level", "f8"), ("layout", "i1")]
VT_DTYPE = np.dtype(BASE_FIELDS + [(k, "f8") for k in ("ospa", "f1", "predicted", "truth")])
LOCAL_DTYPE = np.dtype(BASE_FIELDS + [("final", "f8"), ("initial", "f8"), ("fallback", "?")])
SUBSETS = ["A", "B", "A+B"]
NAMES = {
    "mpurge_map": "MPUrge-MAP",
    "mpurge": "MPUrge",
    "mca": "MCA",
    "chamfer": "Chamfer-kNN",
    "mpurge_local": "MPUrge",
    "mpurge_full": "MPUrge",
    "mpurge_star_local": "MPUrge Star",
    "mpurge_star_full": "MPUrge Star",
    "mpurge_triangle_local": "MPUrge + triangle inequality",
    "mpurge_triangle_full": "MPUrge + triangle inequality",
    "mpurge_diffassign_local": "MPUrge + DiffAssign",
    "mpurge_diffassign_full": "MPUrge + DiffAssign",
    "pba_local": "PBA",
    "ransac": "RANSAC",
    "zayets2018": "Zayets–Steinbach 2018",
    "zayets2019_assumption": "Zayets et al. 2019 (assumed adapter)",
}
MERGERS = {
    "greedy_median": "Greedy complete-link · median",
    "greedy_mean": "Greedy complete-link · mean",
    "agglomerative": "Agglomerative complete-link",
    "dbscan": "DBSCAN",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class ObjectiveCases:
    """Stream the known objective schema, retaining only one input chunk/case.

    The header is parsed as JSON, so metadata is checked independently of receipts.
    Cases may span chunks and contain nested arrays, escaped quotes, or Unicode.
    """

    def __init__(self, path, chunk_size=262144):
        self.path, self.chunk_size, self.header = path, chunk_size, None

    def __iter__(self):
        decoder = json.JSONDecoder()
        with gzip.open(self.path, "rt", encoding="utf-8") as handle:
            buffer = ""
            while True:
                part = handle.read(self.chunk_size)
                if not part:
                    raise ValueError(f"Missing cases array: {self.path}")
                buffer += part
                match = re.search(r'"cases"\s*:\s*\[', buffer)
                if match:
                    self.header = json.loads(buffer[: match.end()] + "]}}")
                    buffer = buffer[match.end() :]
                    break
                if len(buffer) > 16 * 1024 * 1024:
                    raise ValueError("Unexpectedly large objective header")
            position = 0
            while True:
                while position < len(buffer) and buffer[position] in " \r\n\t,":
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    case, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    part = handle.read(self.chunk_size)
                    if not part:
                        raise ValueError(f"Truncated or invalid objective: {self.path}") from None
                    buffer = buffer[position:] + part
                    position = 0
                    continue
                yield case
                position = end


def summarize(array, stage):
    if not len(array):
        return None
    if stage == "vt":
        return {
            "count": len(array),
            **{k: float(np.mean(array[k])) for k in ("ospa", "f1", "predicted", "truth")},
        }
    final, initial = array["final"], array["initial"]
    mean, initial_mean = float(np.mean(final)), float(np.mean(initial))
    return {
        "count": len(array),
        "mean": mean,
        "rmse": float(np.sqrt(np.mean(np.square(final)))),
        "median": float(np.median(final)),
        "p90": float(np.quantile(final, 0.9)),
        "initial": initial_mean,
        "improvement": 100 * (initial_mean - mean) / initial_mean if initial_mean else None,
        "fallbacks": int(np.sum(array["fallback"])),
    }


def compact_room(room):
    return {
        "id": room["id"],
        "family": room["family"],
        "index": room["index"],
        "width_m": room["width_m"],
        "depth_m": room["depth_m"],
        "ceiling_m": room["ceiling_m"],
        "layouts": {str(int(x["id"].split("_")[-1])): x["box_count"] for x in room["layouts"]},
    }


def presentation(job):
    stage, params = job["stage"], job["parameters"]
    vt_method = params["vt_method"] if stage == "pipeline" else job["method"]
    vt_params = params["vt"] if stage == "pipeline" else params
    if stage == "pipeline":
        name = NAMES.get(params["initial"]["method"], params["initial"]["method"])
        reconstruction = NAMES.get(vt_method, vt_method)
    else:
        name = NAMES.get(job["method"], job["method"])
        reconstruction = ""
    return {
        "name": name,
        "reconstruction": reconstruction,
        "scope": vt_params.get("scope", "") if stage != "coarse" else "",
        "merger": vt_params.get("merger", "") if stage != "coarse" else "",
    }


def build_index(root, revision="v6", force=False):
    root = Path(root).resolve()
    revision_dir = root / "runs/development/revisions" / revision
    cache = root / "runs/development/explorer" / revision
    cache.mkdir(parents=True, exist_ok=True)
    design_path = root / "data/development/design.json"
    design = read_json(design_path)
    rooms = [compact_room(x) for x in design["rooms"] if x["split"] == "calibration"]
    room_ids = {r["id"]: i for i, r in enumerate(rooms)}
    paths = sorted((revision_dir / "presets").glob("*.json"))
    if not paths:
        raise ValueError(f"No completed preset receipts in {revision_dir}")
    signature = hashlib.sha256()
    signature.update(Path(__file__).read_bytes())
    signature.update(design_path.read_bytes())
    receipts = []
    for path in paths:
        raw = path.read_bytes()
        receipt = json.loads(raw)
        artifact = Path(receipt["audit"]["artifact"])
        stat = artifact.stat()
        signature.update(raw)
        signature.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        receipts.append((path, receipt, artifact))
    fingerprint = signature.hexdigest()
    manifest = cache / "index.json"
    old = read_json(manifest) if manifest.exists() else None
    previous_rooms = [r["id"] for r in old["rooms"]] if old else None
    current_rooms = [r["id"] for r in rooms]
    if not force and manifest.exists():
        if old.get("fingerprint") == fingerprint and all(
            (cache / x["columns"]).exists() for x in old["jobs"]
        ):
            return cache, old
    jobs = []
    for number, (path, receipt, artifact) in enumerate(receipts, 1):
        job = receipt["job"]
        if receipt["selection_split"] != "calibration":
            raise ValueError("Only calibration presets belong in this explorer")
        expected = receipt["audit"]["sha256"]
        if sha256(artifact) != expected:
            raise ValueError(f"Checksum mismatch: {artifact}")
        key = f"{SCHEMA}-{expected}"
        column_path, summary_path = cache / f"{key}.npy", cache / f"{key}.json"
        cached = read_json(summary_path) if summary_path.exists() else {}
        if (
            force
            or not column_path.exists()
            or not cached
            or cached.get("room_ids", previous_rooms) != current_rooms
        ):
            count = receipt["audit"]["case_count"]
            dtype = VT_DTYPE if job["stage"] == "vt" else LOCAL_DTYPE
            temp = column_path.with_suffix(".tmp.npy")
            array = np.lib.format.open_memmap(temp, mode="w+", dtype=dtype, shape=(count,))
            stream = ObjectiveCases(artifact)
            actual_count = 0
            for i, case in enumerate(stream):
                if i >= count:
                    raise ValueError(f"Too many cases: {job['id']}")
                base = (
                    room_ids[case["room"]],
                    SUBSETS.index("+".join(case["subset"])),
                    float(case["level"]),
                    int(case.get("layout", "layout_-1").split("_")[-1]),
                )
                if job["stage"] == "vt":
                    values = (
                        case["ospa"],
                        case["f1"],
                        case["tp"] + case["fp"],
                        case["tp"] + case["fn"],
                    )
                    if (case["ospa_p"], case["ospa_c_m"], case["f1_gate_m"]) != (1, 1, 0.5):
                        raise ValueError("Incompatible VT metric definition")
                else:
                    values = (
                        case["final_error_m"],
                        case["initial_error_m"],
                        case["final_status"].startswith("fallback"),
                    )
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(f"Non-finite metric: {job['id']}")
                array[i] = base + values
                actual_count += 1
            if actual_count != count:
                raise ValueError(f"Case count mismatch: {job['id']}")
            header = stream.header
            for field in ("stage", "method", "track", "parameters"):
                if header["identity"][field] != job[field]:
                    raise ValueError(f"Objective identity mismatch ({field}): {job['id']}")
            metrics = summarize(array, job["stage"])
            score = metrics["ospa" if job["stage"] == "vt" else "mean"]
            if not math.isclose(score, receipt["score"], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"Score mismatch: {job['id']}")
            if not math.isclose(score, header["score"], abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"Objective score mismatch: {job['id']}")
            if job["stage"] != "vt":
                for source, dest in (
                    ("median_m", "median"),
                    ("p90_m", "p90"),
                    ("initial_mean_m", "initial"),
                    ("fallback_count", "fallbacks"),
                ):
                    if not math.isclose(header["audit"][source], metrics[dest], abs_tol=1e-12):
                        raise ValueError(f"Summary mismatch ({source}): {job['id']}")
            array.flush()
            del array
            temp.replace(column_path)
            write_json(
                summary_path,
                {"metrics": metrics, "identity": header["identity"], "room_ids": current_rooms},
            )
        cached = read_json(summary_path)
        if job["stage"] != "vt" and "rmse" not in cached["metrics"]:
            array = np.load(column_path, mmap_mode="r", allow_pickle=False)
            cached["metrics"] = summarize(array, job["stage"])
            del array
            write_json(summary_path, cached)
        for field in ("stage", "method", "track", "parameters"):
            if cached["identity"][field] != job[field]:
                raise ValueError(f"Cached objective identity mismatch ({field}): {job['id']}")
        metric = "ospa" if job["stage"] == "vt" else "mean"
        if cached["metrics"]["count"] != receipt["audit"]["case_count"] or not math.isclose(
            cached["metrics"][metric], receipt["score"], abs_tol=1e-12, rel_tol=1e-12
        ):
            raise ValueError(f"Cached objective score/count mismatch: {job['id']}")
        jobs.append(
            {
                "id": job["id"],
                "stage": job["stage"],
                "method": job["method"],
                "track": job["track"],
                "parameters": job["parameters"],
                **presentation(job),
                "status": receipt["status"],
                "score": receipt["score"],
                "metrics": cached["metrics"],
                "columns": column_path.name,
                "artifact": str(artifact.relative_to(root)),
                "artifact_sha256": expected,
                "receipt": str(path.relative_to(root)),
                "code_hash": receipt["code_hash"],
                "objective_identity": cached["identity"],
                "compatibility_reuse": receipt.get("compatibility_reuse"),
            }
        )
        print(f"Indexed {number}/{len(receipts)} · {job['id']}", flush=True)
    config = design["config"]
    result = {
        "schema": SCHEMA,
        "fingerprint": fingerprint,
        "revision": revision,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "rooms": rooms,
        "levels": {"direct": config["direct_sigma_m"], "receiver": config["receiver_snr_db"]},
        "mergers": MERGERS,
        "jobs": jobs,
    }
    write_json(manifest, result)
    return cache, result


class Explorer:
    def __init__(self, cache, index, status_path):
        self.cache, self.index, self.status_path = Path(cache), index, Path(status_path)

    def metadata(self):
        return {k: v for k, v in self.index.items() if k != "jobs"} | {
            "counts": dict(Counter(j["stage"] for j in self.index["jobs"])),
            "status": {
                k: v
                for k, v in read_json(self.status_path).items()
                if k
                in ("completed", "total", "running", "failed", "updated_utc", "execution_state")
            },
        }

    def selected(self, job, filters):
        array = np.load(self.cache / job["columns"], mmap_mode="r", allow_pickle=False)
        mask = np.ones(len(array), dtype=bool)
        rooms = self.index["rooms"]
        family, room = filters.get("family", ""), filters.get("room", "")
        if family or room:
            indices = [
                i
                for i, r in enumerate(rooms)
                if (not family or r["family"] == family) and (not room or r["id"] == room)
            ]
            mask &= np.isin(array["room"], indices)
        if filters.get("ap"):
            mask &= array["subset"] == SUBSETS.index(filters["ap"])
        if filters.get("noise"):
            mask &= array["level"] == float(filters["noise"])
        if job["stage"] != "vt":
            if filters.get("layout"):
                mask &= array["layout"] == int(filters["layout"])
            if filters.get("obstacles"):
                allowed = np.array(
                    [
                        [
                            r["layouts"].get(str(i), r["layouts"].get(i, -99))
                            == int(filters["obstacles"])
                            for i in range(10)
                        ]
                        for r in rooms
                    ]
                )
                mask &= allowed[array["room"], array["layout"]]
        return array[mask]

    def rows(self, filters):
        view = filters.get("view", "vt")
        if view not in ("vt", "raw", "all"):
            raise ValueError("Unknown view")
        track = filters.get("track", "receiver")
        if track not in self.index["levels"]:
            raise ValueError("Unknown input track")
        stages = {"vt"} if view == "vt" else {"coarse"} if view == "raw" else {"coarse", "pipeline"}
        if filters.get("stage") and view == "all":
            stages &= {filters["stage"]}
        result = []
        for job in self.index["jobs"]:
            if job["stage"] not in stages or job["track"] != track:
                continue
            if filters.get("scope") and job["scope"] != filters["scope"]:
                continue
            if filters.get("merger") and job["merger"] != filters["merger"]:
                continue
            haystack = " ".join(str(job[k]) for k in ("id", "name", "reconstruction", "merger"))
            if filters.get("q", "").lower() not in haystack.lower():
                continue
            metrics = summarize(self.selected(job, filters), job["stage"])
            if metrics:
                result.append(
                    {
                        k: job[k]
                        for k in (
                            "id",
                            "name",
                            "stage",
                            "method",
                            "scope",
                            "merger",
                            "reconstruction",
                        )
                    }
                    | metrics
                )
        return {"rows": result, "filters": filters}

    def detail(self, filters):
        job = next((j for j in self.index["jobs"] if j["id"] == filters.get("id")), None)
        if job is None:
            raise ValueError("Unknown result")
        array = self.selected(job, filters)
        metric = "ospa" if job["stage"] == "vt" else "final"
        breakdown = []
        for i, room in enumerate(self.index["rooms"]):
            metrics = summarize(array[array["room"] == i], job["stage"])
            if metrics:
                breakdown.append({"room": room["id"], "family": room["family"], **metrics})
        # Exact ranks at up to 201 order statistics; this is a display sample only.
        ordered = np.sort(array[metric])
        indices = np.unique(np.linspace(0, len(ordered) - 1, min(201, len(ordered)), dtype=int))
        ecdf = (
            [
                [
                    float(ordered[i]),
                    float(np.searchsorted(ordered, ordered[i], side="right") / len(ordered)),
                ]
                for i in indices
            ]
            if len(ordered)
            else []
        )
        return {
            "job": {k: v for k, v in job.items() if k != "columns"},
            "metrics": summarize(array, job["stage"]),
            "rooms": breakdown,
            "ecdf": ecdf,
        }


def serve(explorer, assets, port):
    routes = {
        "/": ("index.html", "text/html"),
        "/index.html": ("index.html", "text/html"),
        "/app.js": ("app.js", "text/javascript"),
        "/style.css": ("style.css", "text/css"),
        "/data-client.js": ("data-client.js", "text/javascript"),
        "/static-worker.js": ("static-worker.js", "text/javascript"),
        "/vt-map.js": ("vt-map.js", "text/javascript"),
        "/matching.html": ("matching.html", "text/html"),
        "/peak-matching.css": ("peak-matching.css", "text/css"),
        "/peak-matching.js": ("peak-matching.js", "text/javascript"),
        "/peak-worker.js": ("peak-worker.js", "text/javascript"),
        "/peak-matcher.js": ("peak-matcher.js", "text/javascript"),
        "/method-catalog.json": ("method-catalog.json", "application/json"),
    }
    snapshot = assets.parents[1] / "published/v6"
    manifest = snapshot / "manifest.json"
    published_files = (
        {record["file"] for record in read_json(manifest)["files"]} if manifest.exists() else set()
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path in routes:
                    name, mime = routes[parsed.path]
                    payload = (assets / name).read_bytes()
                elif parsed.path.lstrip("/") in published_files:
                    name = parsed.path.lstrip("/")
                    source = (snapshot / name).resolve()
                    if not source.is_relative_to(snapshot.resolve()):
                        raise ValueError("Invalid published asset path")
                    payload = source.read_bytes()
                    mime = (
                        "application/json" if name.endswith(".json") else "application/octet-stream"
                    )
                elif parsed.path in ("/api/meta", "/api/results", "/api/detail"):
                    filters = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
                    value = (
                        explorer.metadata()
                        if parsed.path == "/api/meta"
                        else (
                            explorer.rows(filters)
                            if parsed.path == "/api/results"
                            else explorer.detail(filters)
                        )
                    )
                    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
                    mime = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime + "; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, KeyError, IndexError) as error:
                self.send_error(400, str(error))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Results explorer: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--revision", default="v6")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--rebuild", action="store_true", help="Reverify and rebuild every derived column"
    )
    args = parser.parse_args()
    cache, index = build_index(args.root, args.revision, args.rebuild)
    print(f"Ready: {len(index['jobs'])} verified calibration presets", flush=True)
    if not args.build_only:
        serve(
            Explorer(
                cache,
                index,
                args.root / "runs/development/revisions" / args.revision / "status.json",
            ),
            args.root / "web/results-explorer",
            args.port,
        )


if __name__ == "__main__":
    main()
