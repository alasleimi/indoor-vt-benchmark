"""Run the two authorized v6 presets, then stop before scheduling main searches."""

from pathlib import Path
import traceback

import run_revision_campaign as runner
from indoor_vt.provenance import code_provenance, file_hash, read_json, write_json


ALLOWED = frozenset(
    {
        "preset/receiver/vt/mpurge_star_local/greedy_mean",
        "preset/direct/vt/mpurge_star_local/agglomerative",
    }
)


class PresetsComplete(Exception):
    pass


def scoped_scheduler(original, on_complete=None):
    def select(jobs, states):
        if all(states[name]["state"] == "complete" for name in ALLOWED):
            if on_complete is not None:
                on_complete(states)
            raise PresetsComplete
        phase, available = original(jobs, states)
        if phase != 0 or any(job["id"] not in ALLOWED for job in available):
            raise RuntimeError("Scheduling would exceed the two authorized presets")
        return phase, available

    return select


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "runs/development/revisions/v6"
    specification = read_json(base / "specification.json")
    before = read_json(base / "status.json")
    receipt = {
        "started_utc": runner.now(),
        "allowed_jobs": sorted(ALLOWED),
        "wrapper_sha256": file_hash(Path(__file__)),
        "source_hash": code_provenance(root)["source_hash"],
        "specification_identity": specification["identity"],
        "completed_before": before["completed"],
        "status": "running_presets_only",
    }
    record = base / "presets_only_execution.json"
    write_json(record, receipt)
    completed_states = {}
    runner.eligible_jobs = scoped_scheduler(runner.eligible_jobs, completed_states.update)
    error = None
    try:
        runner.run(root, "development", "v6", root.parent / "experiments/indoor_vt_shadow")
    except PresetsComplete:
        state = "stopped_after_requested_presets"
    except Exception:
        state = "stopped_on_error"
        error = traceback.format_exc()
    else:
        raise RuntimeError("Unexpected completion of unrestricted campaign")
    status = read_json(base / "status.json")
    for identifier in status.get("running", []):
        status["jobs"][identifier]["state"] = "pending"
    if error is None:
        # The gate runs after result collection, before the next status report.
        status["jobs"] = completed_states
    status = runner.report(
        base, specification["jobs"], status["jobs"],
        1 if error is None else 0, [], specification,
    )
    status.update(execution_state=state, running=[], worker_pids=[], worker_memory=[])
    write_json(base / "status.json", status)
    report = base / "report.md"
    report.write_text(
        f"**Execution state: {state}. No workers are running.**\n\n" + report.read_text(),
        encoding="utf-8",
    )
    receipt.update(
        status=state, finished_utc=runner.now(), completed_after=status["completed"], error=error
    )
    write_json(record, receipt)
    (root / "runs/development/ACTIVE_CAMPAIGN.md").write_text(
        f"# Campaign stopped\n\nState: {state}.\n\n"
        f"{status['completed']}/110 initial presets complete. No workers are running.\n"
        "The two-preset authorization does not start main searches.\n"
        "See revisions/v6/report.md and presets_only_execution.json.\n",
        encoding="utf-8",
    )
    if error:
        raise RuntimeError(error)
    print(f"{status['completed']}/110 presets complete; stopped before main searches.", flush=True)


if __name__ == "__main__":
    main()
