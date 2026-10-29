"""Verify saved solver outputs and summarize the noise experiment."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from indoor_vt.acquisition.observations import Observations
from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.execution.artifacts import read_npz
from indoor_vt.execution.explorer import summarize_sources
from indoor_vt.science.range_rmse import assign_ranges, range_sse


root = Path("runs/redesign/rectangle-range-rmse-true-count-v1")
config = read_json(root / "config.json")
rows = sorted(read_json(root / "results.json"), key=lambda r: r["level"])
assert [r["level"] for r in rows] == config["observations"]["noise_levels"]
observations = Observations("data/redesign/rectangle-20261025", config)
rooms = {r["id"]: r for r in observations.rooms}
truths = {k: observations.truth(r) for k, r in rooms.items()}
per_case, table = [], []
for row in rows:
    detail = read_npz(root / row["outcome"])[0]
    _, metrics = summarize_sources(detail["sources"], truths)
    for ap in ("A", "B", "A+B"):
        for key in ("ospa", "f1", "matched_rmse_m"):
            actual, expected = metrics[ap][key], row["source_metrics_by_ap"][ap][key]
            assert actual == expected or abs(actual - expected) < 1e-12
        table.append({"sigma_m": row["level"], "AP": ap, **metrics[ap]})
    for record in detail["sources"]:
        survey = observations.survey(rooms[record["room"]], row["level"], record["repetition"])
        for ap, sources in record["by_ap"].items():
            audit = record["audit"][ap]
            assert len(sources) == audit["known_source_count"] == len(truths[record["room"]].by_ap[ap])
            assigned, indices = assign_ranges(survey.xyz, survey.channels[ap], sources)
            np.testing.assert_array_equal(indices, audit["assignments"])
            rms = np.sqrt(range_sse(survey.xyz, assigned, sources) / audit["observation_count"])
            assert abs(rms - audit["range_rmse_m"]) < 1e-12
            assert len(audit["starts"]) == config["range_rmse"]["starts"]
            selected = audit["starts"][audit["selected_start"]]
            assert selected["range_rmse_m"] == min(s["range_rmse_m"] for s in audit["starts"])
            for start in audit["starts"]:
                assert np.all(np.diff(start["sse_history"]) <= 1e-7)
            per_case.append({"sigma_m": row["level"], "AP": ap,
                             "repetition": record["repetition"], "range_rmse_m": rms,
                             "selected_start": audit["selected_start"],
                             "missing_observations": audit["missing_observations"],
                             "selected_stop": selected["stop"], "selected_iterations": selected["iterations"],
                             "starts_at_limit": sum(s["stop"] == "iteration_limit" for s in audit["starts"]),
                             "seconds": audit["seconds"]})
write_json(root / "verification.json", {"conditions": len(rows), "AP_fits": len(per_case),
    "starts": len(per_case) * config["range_rmse"]["starts"],
    "metric_reproduction": True, "assignment_optimality_at_saved_sources": True,
    "monotone_objective": True, "best_start_by_measured_residual": True,
    "known_count_retained": True, "per_case": per_case})
with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
summary = ["Range RMSE (cheating true count)",
    "25 sources per AP, including the physical AP; no positions or identities supplied.",
    "280 survey positions; 16 starts per AP and realization; 150 alternating iterations maximum.",
    "Fixed assignments: nonlinear least squares. Fixed positions: Hungarian squared range assignment.",
    "Minimum measured-range residual selects the start; reconstruction truth does not select it.",
    "One realization at sigma=0, two for each positive level, using existing benchmark observations.",
    "Negative noisy ranges are removed by the existing observation pipeline.",
    "A+B results (OSPA p=1,c=1m; F1 gate=.5m; matched coordinate RMSE gate=.5m):",
    "sigma_m   OSPA_m     F1_percent   matched_RMSE_m"]
for r in table:
    if r["AP"] == "A+B":
        matched = f"{r['matched_rmse_m']:.6f}" if r["matched_rmse_m"] is not None else "no matches"
        summary.append(f"{r['sigma_m']:7g}   {r['ospa']:.6f}   {100*r['f1']:10.3f}   {matched}")
summary += [f"Total evaluation wall time: {sum(r['seconds'] for r in rows):.2f} s.",
    "A finite multistart local optimizer; these are not certified globally optimal reconstructions.",
    "Differences between noise levels can reflect different local minima, not only noise sensitivity.",
    "Matched RMSE excludes unmatched sources; always inspect alongside F1 and OSPA.",
    "Individual AP results are in metrics.csv; starts and assignments are in the saved NPZ outcomes."]
(root / "report.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
selected = [r for r in table if r["AP"] == "A+B"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for ax, (key, title, factor) in zip(axes, [("ospa", "OSPA (c = 1 m)", 1),
                                        ("f1", "F1 @ 0.5 m (%)", 100),
                                        ("matched_rmse_m", "Matched RMSE @ 0.5 m (m)", 1)]):
    ax.plot([r["sigma_m"] for r in selected],
            [r[key] * factor if r[key] is not None else np.nan for r in selected],
            "o-", color="#16836e", markersize=5, linewidth=1.8)
    ax.set_xscale("symlog", linthresh=.01)
    ax.set_xticks([0, .01, .05, .2, 1, 5], ["0", ".01", ".05", ".2", "1", "5"])
    ax.set_xlabel("Range noise σ (m)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    for r in selected:
        if r[key] is None:
            ax.annotate("no matches", (r["sigma_m"], .05), xycoords=("data", "axes fraction"),
                        rotation=90, ha="center", va="bottom", fontsize=8, color="#777777")
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_ylim(0, 100)
fig.suptitle("Range RMSE (cheating true count) · rectangle · AP A+B", fontsize=15, y=.99)
fig.text(.5, .01, "16 starts per AP / realization · local optimization · true count includes direct AP · matched RMSE excludes missed sources",
         ha="center", fontsize=9, color="#555555")
fig.tight_layout(rect=[0, .05, 1, .93])
fig.savefig(root / "noise_results.png", dpi=160)
fig.savefig(root / "noise_results.pdf")
print("\n".join(summary))
