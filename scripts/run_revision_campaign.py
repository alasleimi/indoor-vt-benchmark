"""Persistent preset-first campaign; shadows wait for all main hyperparameter searches."""

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import threading
import traceback
from types import SimpleNamespace

from indoor_vt.finish import exclusive_lock, existing_process
from indoor_vt.provenance import code_provenance, digest, file_hash, read_json, write_json
from indoor_vt.revision_campaign import build_primary_jobs, directory, run_job

SHADOWS = (
    "four_cycle",
    "four_clique",
    "overlapping_cliques",
    "weighted_star",
    "weighted_components",
    "correlation",
    "min_cost_flow",
    "mcl",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def build_jobs():
    jobs = build_primary_jobs()
    for track in ("receiver", "direct"):
        for regime in ("component", "end_to_end"):
            for scope in ("local", "full"):
                for method in SHADOWS:
                    identifier = f"shadow/{track}/{regime}/{scope}/{method}"
                    common = {"track": track, "regime": regime, "scope": scope, "method": method}
                    jobs.append(
                        {
                            **common,
                            "id": identifier,
                            "phase": 2,
                            "kind": "shadow",
                            "dependencies": [],
                        }
                    )
                    jobs.append(
                        {
                            **common,
                            "id": identifier.replace("shadow/", "shadow_evaluate/"),
                            "phase": 3,
                            "kind": "shadow_evaluate",
                            "dependencies": [identifier],
                        }
                    )
    return jobs


def worker(root, namespace, revision, shadow, expected_hash, tasks, results, parent_pid):
    def monitor_parent():
        with existing_process(parent_pid) as alive:
            while alive():
                time.sleep(5)
        os._exit(3)  # No competing cache writers after an abrupt runner exit.

    threading.Thread(target=monitor_parent, daemon=True).start()
    evaluators = {}
    sys.path.insert(0, str(shadow))
    while True:
        job = tasks.get()
        if job is None:
            return
        try:
            if code_provenance(root)["source_hash"] != expected_hash:
                raise RuntimeError("Frozen campaign code changed")
            if job["kind"].startswith("shadow"):
                import workflow

                args = SimpleNamespace(
                    benchmark=root,
                    namespace=namespace,
                    revision=revision,
                    track=job["track"],
                    regime=job["regime"],
                    scope=job["scope"],
                    method=job["method"],
                )
                if job["kind"] == "shadow":
                    workflow.select(args)
                    path = workflow.output(args) / (job["method"] + "_promotion.json")
                else:
                    workflow.evaluate(args)
                    path = workflow.output(args) / (job["method"] + "_test.summary.json")
                result = {"state": "complete", "artifact": str(path)}
            else:
                result = run_job(root, namespace, revision, job, evaluators)
            results.put({"id": job["id"], "pid": os.getpid(), **result})
        except Exception:
            results.put(
                {
                    "id": job["id"],
                    "pid": os.getpid(),
                    "state": "error",
                    "traceback": traceback.format_exc(),
                }
            )


def eligible_jobs(jobs, states):
    unfinished = [job for job in jobs if states[job["id"]]["state"] != "complete"]
    phase = min((job["phase"] for job in unfinished), default=None)
    eligible = [
        job
        for job in unfinished
        if job["phase"] == phase
        and states[job["id"]]["state"] == "pending"
        and all(states[dep]["state"] == "complete" for dep in job["dependencies"])
    ]
    return phase, sorted(
        eligible,
        key=lambda job: (
            job.get("order", 0),
            states[job["id"]].get("turns", 0),
            # Alternate input tracks without draining every task of one track first.
            job["id"].replace("receiver", "").replace("direct", ""),
            job["track"] != "receiver",
        ),
    )


def report(base, jobs, states, phase, processes, spec):
    completed = sum(s["state"] == "complete" for s in states.values())
    running = [key for key, state in states.items() if state["state"] == "running"]
    failed = [key for key, state in states.items() if state["state"] == "failed"]
    state = {
        "updated_utc": now(),
        "phase": phase,
        "completed": completed,
        "total": len(jobs),
        "running": running,
        "failed": failed,
        "jobs": states,
        "finished": completed == len(jobs),
        "specification_identity": spec["identity"],
        "worker_pids": [p.pid for p in processes],
    }
    write_json(base / "status.json", state)
    phase_name = {
        0: "Initial preset coverage",
        1: "Main hyperparameter calibration",
        2: "Shadow calibration",
        3: "Frozen held-out evaluation",
        None: "Complete",
    }[phase]
    lines = [
        "# Development benchmark execution",
        "",
        f"Updated UTC: {state['updated_utc']}",
        "",
        f"**{phase_name}.** {completed}/{len(jobs)} jobs complete.",
        "",
        "Shadow work starts only after all main calibration and freezes finish.",
        "Initial preset scores use calibration rooms; held-out tests remain behind selection.",
        "",
        "## Running",
        "",
        *[f"- `{name}`" for name in running],
        "",
        "## Initial preset results",
        "",
        "| Track | Stage | Method | Merger | Calibration score (m) |",
        "|---|---|---|---|---:|",
    ]
    for job in jobs:
        saved = states[job["id"]]
        if job["kind"] == "preset" and saved["state"] == "complete":
            merger = job["parameters"].get("merger", "")
            lines.append(
                f"| {job['track']} | {job['stage']} | {job['method']} | {merger} | {saved.get('score', float('nan')):.6f} |"
            )
    lines.extend(
        [
            "",
            "VT scores are OSPA; coarse/pipeline scores are mean 3D localization error.",
            "Preset results are not calibrated winners or held-out rankings.",
            "",
            "## Main searches",
            "",
            "| Search | Saved configurations | Complete |",
            "|---|---:|---|",
        ]
    )
    for job in jobs:
        saved = states[job["id"]]
        if job["kind"] == "search" and saved.get("trials"):
            lines.append(f"| {job['id']} | {saved['trials']} | {saved['state'] == 'complete'} |")
    lines.extend(
        [
            "",
            "Original trials and their provenance remain in runs/development/calibration;",
            "compatible evaluations are imported with links, not overwritten.",
            "",
        ]
    )
    if failed:
        lines.extend(
            ["## Failures requiring attention", "", *[f"- `{name}`" for name in failed], ""]
        )
    (base / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return state


def run(root, namespace, revision, shadow, workers=2):
    root, shadow = Path(root).resolve(), Path(shadow).resolve()
    base = directory(root, namespace, revision)
    base.mkdir(parents=True, exist_ok=True)
    archive = read_json(root / "data" / namespace / "archive_audit.json")
    if archive["missing"] or archive["missing_observations"]:
        raise RuntimeError("Complete native and observation archives required")
    core = code_provenance(root)
    jobs = build_jobs()
    compatibility = root / "runs" / namespace / "cache_compatibility.json"
    imports = {
        p.relative_to(base).as_posix(): file_hash(p)
        for p in sorted((base / "imports").rglob("*.json"))
    }
    spec = {
        "schema": "indoor-vt.campaign-queue.v2",
        "namespace": namespace,
        "revision": revision,
        "core_code_hash": core["source_hash"],
        "runner_hash": file_hash(Path(__file__)),
        "shadow_hashes": {p.name: file_hash(p) for p in sorted(shadow.glob("*.py"))},
        "jobs": jobs,
        "imports": imports,
        "compatibility_hash": file_hash(compatibility) if compatibility.exists() else None,
        "phase_rule": "Primary presets; all main calibration/freeze; shadow selection; held-out evaluation.",
        "workers": workers,
    }
    spec["identity"] = digest(spec)
    if (base / "specification.json").exists() and read_json(base / "specification.json") != spec:
        raise RuntimeError("Campaign revision changed; preserve it and use another revision")
    write_json(base / "specification.json", spec)
    states = {job["id"]: {"state": "pending", "turns": 0, "failures": 0} for job in jobs}
    if (base / "status.json").exists():
        previous = read_json(base / "status.json")
        if previous["specification_identity"] != spec["identity"]:
            raise RuntimeError("Status belongs to another campaign")
        states.update(previous["jobs"])
        for state in states.values():
            if state["state"] == "running":
                state["state"] = "pending"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    context = mp.get_context("spawn")
    tasks, results = context.Queue(), context.Queue()
    processes = []
    with exclusive_lock(base / "runner.lock"):
        try:
            for _ in range(workers):
                process = context.Process(
                    target=worker,
                    args=(
                        root,
                        namespace,
                        revision,
                        shadow,
                        core["source_hash"],
                        tasks,
                        results,
                        os.getpid(),
                    ),
                )
                process.start()
                processes.append(process)
            while True:
                if code_provenance(root)["source_hash"] != core["source_hash"]:
                    raise RuntimeError("Source changed during frozen revision")
                if {p.name: file_hash(p) for p in sorted(shadow.glob("*.py"))} != spec[
                    "shadow_hashes"
                ]:
                    raise RuntimeError("Shadow implementation changed during frozen revision")
                for process in processes:
                    if not process.is_alive():
                        raise RuntimeError(
                            f"Worker {process.pid} exited unexpectedly; saved objectives are resumable"
                        )
                while True:
                    try:
                        result = results.get_nowait()
                    except queue.Empty:
                        break
                    state = states[result["id"]]
                    if result["state"] == "error":
                        state["failures"] += 1
                        result["state"] = "pending" if state["failures"] < 2 else "failed"
                    state.update(result, finished_utc=now())
                    print(json.dumps(result), flush=True)
                phase, available = eligible_jobs(jobs, states)
                active = sum(s["state"] == "running" for s in states.values())
                for job in available[: max(0, workers - active)]:
                    state = states[job["id"]]
                    state.update(state="running", started_utc=now(), turns=state["turns"] + 1)
                    tasks.put(job)
                status = report(base, jobs, states, phase, processes, spec)
                if status["finished"]:
                    return status
                if not status["running"] and not available:
                    raise RuntimeError("Campaign cannot advance; inspect failures and dependencies")
                time.sleep(5)
        finally:
            for _ in processes:
                tasks.put(None)
            # Keep completed artifacts; a failed runner must not leave competing writers.
            for process in processes:
                process.join(timeout=2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--namespace", default="development", choices=("development",))
    parser.add_argument("--revision", default="v2")
    parser.add_argument("--shadow", type=Path, default=Path("../experiments/indoor_vt_shadow"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    try:
        run(args.root, args.namespace, args.revision, args.shadow, args.workers)
    except Exception:
        base = directory(args.root, args.namespace, args.revision)
        write_json(
            base / "runner_error.json", {"time_utc": now(), "traceback": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    main()
