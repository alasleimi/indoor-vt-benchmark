"""Persistent local campaign queue. Run from the benchmark's pinned environment.

The runner is deliberately outside src/: scheduling changes do not change an
algorithm's freeze identity. Its own exact bytes and job graph are recorded.
"""

import argparse
import ctypes
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from indoor_vt.finish import exclusive_lock, existing_process
from indoor_vt.provenance import code_provenance, digest, file_hash, read_json, write_json
from indoor_vt.registry import INITIAL_METHODS, VT_METHODS


SHADOW_METHODS = (
    "four_cycle",
    "four_clique",
    "overlapping_cliques",
    "weighted_star",
    "weighted_components",
    "correlation",
    "min_cost_flow",
    "mcl",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def process_signature(pid):
    """Creation identity avoids attaching to a recycled PID after a restart."""
    if os.name != "nt":
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[19]
        except FileNotFoundError:
            return None
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    times = [wintypes.FILETIME() for _ in range(4)]
    try:
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            return None
        return (times[0].dwHighDateTime << 32) + times[0].dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def pipeline_pairs():
    return [("mpurge_map", vt) for vt in VT_METHODS] + [
        (initial, "mpurge_local") for initial in INITIAL_METHODS if initial != "mpurge_map"
    ]


def build_jobs(root, namespace, shadow):
    root, shadow = Path(root), Path(shadow)
    jobs = []
    # Receiver first gives the shorter three-condition track early progress.
    for regime in ("component", "end_to_end"):
        for initial, vt in pipeline_pairs():
            name = initial + "__" + vt
            for track in ("receiver", "direct"):
                freeze = root / "configs/frozen" / namespace / track / regime / (name + ".json")
                common = [
                    "--namespace",
                    namespace,
                    "--track",
                    track,
                    "--regime",
                    regime,
                    "--initial",
                    initial,
                    "--vt",
                    vt,
                ]
                locks = (
                    [f"coarse/{track}/{initial}", f"vt/{track}/{vt}"]
                    if regime == "component"
                    else []
                )
                jobs.append(
                    {
                        "id": f"calibrate/{track}/{regime}/{name}",
                        "phase": 0,
                        "kind": "calibrate",
                        "command": ["-m", "indoor_vt.cli", "calibrate", *common],
                        "artifact": str(freeze),
                        "locks": locks,
                    }
                )
                result = (
                    root
                    / "runs"
                    / namespace
                    / "test"
                    / track
                    / regime
                    / (name + ".json.summary.json")
                )
                jobs.append(
                    {
                        "id": f"evaluate/{track}/{regime}/{name}",
                        "phase": 2,
                        "kind": "evaluate",
                        "command": ["-m", "indoor_vt.cli", "evaluate", "--frozen", str(freeze)],
                        "artifact": str(result),
                        "locks": [],
                    }
                )
    for regime in ("component", "end_to_end"):
        for scope in ("local", "full"):
            for method in SHADOW_METHODS:
                for track in ("receiver", "direct"):
                    common = [
                        "--benchmark",
                        str(root),
                        "--namespace",
                        namespace,
                        "--track",
                        track,
                        "--regime",
                        regime,
                        "--scope",
                        scope,
                        "--method",
                        method,
                    ]
                    out = shadow / "results" / namespace / track / regime / scope
                    jobs.append(
                        {
                            "id": f"shadow/{track}/{regime}/{scope}/{method}",
                            "phase": 1,
                            "kind": "shadow",
                            "command": [str(shadow / "run.py"), *common],
                            "artifact": str(out / (method + "_promotion.json")),
                            "locks": [],
                        }
                    )
                    jobs.append(
                        {
                            "id": f"shadow_evaluate/{track}/{regime}/{scope}/{method}",
                            "phase": 2,
                            "kind": "shadow_evaluate",
                            "command": [str(shadow / "evaluate.py"), *common],
                            "artifact": str(out / (method + "_test.summary.json")),
                            "locks": [],
                        }
                    )
    return sorted(jobs, key=lambda j: j["phase"])


def completion(job, code_hash):
    path = Path(job["artifact"])
    if not path.exists():
        return None
    record = read_json(path)
    if job["kind"] == "calibrate":
        check = dict(record)
        claimed = check.pop("freeze_hash")
        if digest(check) != claimed or record["code_hash"] != code_hash:
            raise RuntimeError(f"Incompatible calibration artifact: {path}")
    elif job["kind"] == "evaluate":
        if file_hash(Path(record["result"])) != record["result_sha256"]:
            raise RuntimeError(f"Changed evaluation result: {path}")
    else:
        if record["core_code_hash"] != code_hash:
            raise RuntimeError(f"Incompatible shadow artifact: {path}")
    return {"artifact": str(path), "sha256": file_hash(path), "completed_utc": utc_now()}


def progress(root, namespace):
    records = []
    directory = root / "runs" / namespace / "calibration"
    for path in directory.glob("*/*/*.json"):
        data = read_json(path)
        if "trials" in data:
            records.append(
                {
                    "search": path.relative_to(directory).as_posix(),
                    "trials_completed": len(data["trials"]),
                    "frozen": data.get("complete", False),
                }
            )
    return records


def run(root, namespace, shadow, workers=2, attached_pid=None):
    root, shadow = Path(root).resolve(), Path(shadow).resolve()
    destination = root / "runs" / namespace / "campaign"
    destination.mkdir(parents=True, exist_ok=True)
    core = code_provenance(root)
    archive_path = root / "data" / namespace / "archive_audit.json"
    archive = read_json(archive_path)
    if (
        archive["missing"]
        or archive["missing_observations"]
        or archive["completed_scenarios"] != archive["expected_scenarios"]
        or archive["completed_observation_conditions"] != archive["expected_observation_conditions"]
    ):
        raise RuntimeError("The full verified dataset is required before calibration")
    jobs = build_jobs(root, namespace, shadow)
    specification = {
        "schema": "indoor-vt.campaign-queue.v1",
        "namespace": namespace,
        "core_code_hash": core["source_hash"],
        "core_commit": core["git_commit"],
        "archive_audit_sha256": file_hash(archive_path),
        "runner_hash": file_hash(Path(__file__)),
        "shadow_hashes": {p.name: file_hash(p) for p in sorted(shadow.glob("*.py"))},
        "pairs": [list(pair) for pair in pipeline_pairs()],
        "workers": workers,
        "jobs": jobs,
        "phase_rule": "All primary calibrations, then all shadow selection/final freezes, then test evaluation.",
        "failure_rule": "Retry a failed process once using its existing trial cache; unresolved failures block later phases.",
        "publication_namespace": "reserved; this command does not generate publication data",
    }
    specification_path = destination / "specification.json"
    if specification_path.exists():
        # Documentation/queue commits do not change the frozen estimator bytes.
        specification["core_commit"] = read_json(specification_path)["core_commit"]
    specification["identity"] = digest(specification)
    if specification_path.exists() and read_json(specification_path) != specification:
        raise RuntimeError("Campaign specification changed; preserve and diagnose before resuming")
    write_json(specification_path, specification)
    states = {j["id"]: {"state": "pending", "attempts": 0} for j in jobs}
    state_path = destination / "status.json"
    if state_path.exists():
        previous = read_json(state_path)
        if previous["specification_identity"] != specification["identity"]:
            raise RuntimeError("Campaign status has another specification")
        states.update(previous["jobs"])
    env = dict(os.environ)
    env.update(
        {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    active = {}
    with exclusive_lock(destination / "runner.lock"), ExitStack() as stack:
        for identifier, state in states.items():
            if state["state"] == "running":
                signature = process_signature(state["pid"])
                if signature is not None and signature == state.get("process_created"):
                    alive = stack.enter_context(existing_process(state["pid"]))
                    if alive():
                        active[identifier] = {"alive": alive, "process": None, "log": None}
                        continue
                state["state"] = "pending"
        if attached_pid is not None:
            alive = stack.enter_context(existing_process(attached_pid))
            first = next(
                j
                for j in jobs
                if j["id"] == "calibrate/receiver/component/mpurge_map__mpurge_local"
            )
            if alive():
                active[first["id"]] = {"alive": alive, "process": None, "log": None}
                states[first["id"]].update(
                    state="running",
                    pid=attached_pid,
                    started_utc=utc_now(),
                    process_created=process_signature(attached_pid),
                )
        while True:
            # Do not mix a newly edited estimator into an already frozen campaign.
            if code_provenance(root)["source_hash"] != core["source_hash"]:
                raise RuntimeError(
                    "Core code changed during campaign; existing workers remain auditable, no new jobs launched"
                )
            if {p.name: file_hash(p) for p in sorted(shadow.glob("*.py"))} != specification[
                "shadow_hashes"
            ]:
                raise RuntimeError(
                    "Shadow implementation changed during campaign; no new jobs launched"
                )
            for job in jobs:
                state = states[job["id"]]
                if state["state"] != "complete":
                    result = completion(job, core["source_hash"])
                    if result is not None and job["id"] not in active:
                        state.update(state="complete", **result)
            for identifier, running in list(active.items()):
                if running["alive"]():
                    continue
                if running["log"] is not None:
                    running["log"].close()
                job = next(j for j in jobs if j["id"] == identifier)
                result = completion(job, core["source_hash"])
                if result is not None:
                    states[identifier].update(state="complete", **result)
                else:
                    attempts = states[identifier]["attempts"]
                    states[identifier].update(
                        state="failed" if attempts >= 2 else "pending",
                        last_exit_code=running["process"].returncode
                        if running["process"]
                        else None,
                        failure_utc=utc_now(),
                    )
                del active[identifier]
            unfinished = [j for j in jobs if states[j["id"]]["state"] != "complete"]
            phase = min((j["phase"] for j in unfinished), default=None)
            held = {lock for j in jobs if j["id"] in active for lock in j["locks"]}
            for job in unfinished:
                if len(active) >= workers:
                    break
                if (
                    job["phase"] != phase
                    or states[job["id"]]["state"] != "pending"
                    or held.intersection(job["locks"])
                ):
                    continue
                state = states[job["id"]]
                state["attempts"] += 1
                log_path = (
                    destination
                    / "logs"
                    / (job["id"].replace("/", "__") + f"__{state['attempts']:02d}.log")
                )
                log_path.parent.mkdir(exist_ok=True)
                log = log_path.open("w", encoding="utf-8")
                child = subprocess.Popen(
                    [sys.executable, "-u", *job["command"]],
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                active[job["id"]] = {
                    "alive": lambda p=child: p.poll() is None,
                    "process": child,
                    "log": log,
                }
                held.update(job["locks"])
                state.update(
                    state="running",
                    pid=child.pid,
                    log=str(log_path),
                    started_utc=utc_now(),
                    process_created=process_signature(child.pid),
                )
            status = {
                "specification_identity": specification["identity"],
                "updated_utc": utc_now(),
                "phase": phase,
                "running": list(active),
                "completed": sum(s["state"] == "complete" for s in states.values()),
                "total": len(jobs),
                "failed": [k for k, v in states.items() if v["state"] == "failed"],
                "jobs": states,
                "searches": progress(root, namespace),
                "finished": not unfinished,
            }
            write_json(state_path, status)
            if not unfinished:
                print(json.dumps({"finished": True, "jobs": len(jobs)}), flush=True)
                return status
            if not active and all(
                states[j["id"]]["state"] == "failed" for j in unfinished if j["phase"] == phase
            ):
                raise RuntimeError(
                    "Campaign needs investigation; failed jobs are listed in status.json"
                )
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--namespace", choices=["development"], default="development")
    parser.add_argument("--shadow", type=Path, default=Path("../experiments/indoor_vt_shadow"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--attach-pid", type=int)
    args = parser.parse_args()
    try:
        run(args.root, args.namespace, args.shadow, args.workers, args.attach_pid)
    except Exception:
        error = args.root / "runs" / args.namespace / "campaign/runner_error.json"
        write_json(error, {"time_utc": utc_now(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
