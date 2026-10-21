"""Read-only aggregation of campaign progress and frozen evaluation artifacts."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean
import time

from indoor_vt.provenance import read_json, write_json, replace_file


def report(root, namespace, shadow):
    root, shadow = Path(root).resolve(), Path(shadow).resolve()
    directory = root / "runs" / namespace / "campaign"
    status = read_json(directory / "status.json")
    cache_path = directory / "report_cache.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    summaries = list((root / "runs" / namespace / "test").glob("*/*/*.summary.json"))
    summaries += list((shadow / "results" / namespace).glob("*/*/*/*_test.summary.json"))
    rows = []
    for path in sorted(summaries):
        summary = read_json(path)
        if "result_sha256" not in summary:
            continue
        key = summary["result_sha256"]
        if key not in cache:
            with gzip.open(summary["result"], "rt", encoding="utf-8") as handle:
                data = json.load(handle)
            frozen = data["freeze"]
            local, vt = data["localization"], data["vt"]
            shadow_result = "qualifies" in frozen
            config = frozen["pipeline_config"] if shadow_result else frozen["config"]
            name = config["initial"]["method"] + "__" + config["vt_method"]
            if shadow_result:
                name += "_" + frozen["scope"]
            cache[key] = {
                "track": frozen["track"],
                "regime": frozen["regime"],
                "pipeline": name,
                "category": "qualified shadow" if shadow_result else "primary",
                "final_mean_error_m": summary["final_mean_error_m"],
                "initial_mean_error_m": local["initial_mean_m"],
                "median_error_m": local["median_m"],
                "p90_error_m": local["p90_m"],
                "vt_ospa_m": summary["vt_ospa"],
                "vt_f1_at_0_5m": mean(c["f1"] for c in vt["cases"]),
                "query_cases": local["case_count"],
                "vt_cases": vt["case_count"],
                "fallbacks": local["fallback_count"],
                "config": config,
                "selection_freeze": frozen["freeze_hash"],
                "result": summary["result"],
                "result_sha256": key,
            }
            del data
        rows.append(cache[key])
    promotions = []
    for path in sorted((shadow / "results" / namespace).glob("*/*/*/*_promotion.json")):
        record = read_json(path)
        promotions.append(
            {
                key: record[key]
                for key in (
                    "qualifies",
                    "track",
                    "regime",
                    "scope",
                    "candidate_score",
                    "baseline_score",
                    "relative_reduction",
                )
            }
            | {"method": path.name.removesuffix("_promotion.json"), "artifact": str(path)}
        )
    aggregate = {
        "namespace": namespace,
        "updated_utc": status["updated_utc"],
        "finished": status["finished"],
        "phase": status["phase"],
        "completed_jobs": status["completed"],
        "total_jobs": status["total"],
        "running": status["running"],
        "failed": status["failed"],
        "search_progress": status["searches"],
        "test_results": rows,
        "shadow_selection": promotions,
    }
    write_json(cache_path, cache)
    write_json(directory / "report.json", aggregate)
    phase = {
        0: "Primary calibration",
        1: "Shadow calibration and freeze",
        2: "Frozen test evaluation",
        None: "Finished",
    }[status["phase"]]
    lines = [
        "# Development benchmark execution",
        "",
        f"Updated (UTC): {status['updated_utc']}",
        "",
        f"**{phase}.** {status['completed']}/{status['total']} queue jobs complete.",
        "",
        "The complete 16-room development dataset contains eight calibration and eight held-out test rooms. "
        "The publication namespace remains unused.",
        "",
        "## Active jobs",
        "",
    ]
    lines += [f"- `{name}`" for name in status["running"]] or ["No active jobs."]
    if status["failed"]:
        lines += ["", "## Failures requiring investigation", ""]
        lines += [f"- `{name}`" for name in status["failed"]]
    lines += [
        "",
        "## Calibration progress",
        "",
        "| Search | Completed candidates | Frozen |",
        "|---|---:|---|",
    ]
    lines += [
        f"| `{s['search']}` | {s['trials_completed']} | {s['frozen']} |" for s in status["searches"]
    ]
    if not status["searches"]:
        lines += ["| No full candidate finished yet | 0 | False |"]
    lines += ["", "## Held-out results", ""]
    if not rows:
        lines += [
            "No test results yet. Test execution waits for all calibration and shadow selections to finish."
        ]
    else:
        lines += [
            "| Track | Regime | Pipeline | Initial mean (m) | Final mean (m) | VT OSPA (m) | F1 at 0.5 m | Fallbacks |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['track']} | {row['regime']} | `{row['pipeline']}` | "
                f"{row['initial_mean_error_m']:.4f} | {row['final_mean_error_m']:.4f} | "
                f"{row['vt_ospa_m']:.4f} | {row['vt_f1_at_0_5m']:.4f} | {row['fallbacks']} |"
            )
    lines += [
        "",
        "## Shadow selection",
        "",
        f"{len(promotions)} comparisons completed; {sum(p['qualifies'] for p in promotions)} qualify under the 5% calibration improvement rule.",
        "",
        "Qualifying methods retain their external implementation and frozen parameters through test evaluation. "
        "Importing eligible implementations into the main registry follows the completed campaign audit.",
        "",
        "`report.json` contains selected configurations, result hashes and artifact paths. "
        "Raw timings come from concurrent runs and are not isolated hardware speed measurements.",
        "",
    ]
    temporary = directory / "report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    replace_file(temporary, directory / "report.md")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--namespace", default="development")
    parser.add_argument("--shadow", type=Path, default=Path("../experiments/indoor_vt_shadow"))
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        result = report(args.root, args.namespace, args.shadow)
        if result["finished"] or not args.watch:
            print(
                {"finished": result["finished"], "completed": result["completed_jobs"]}, flush=True
            )
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
