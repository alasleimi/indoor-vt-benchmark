"""Summarize retained trial logs and fixed-parameter noise evaluations."""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.execution.trials import read_trials

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=Path("runs/redesign/rectangle-clean-selection-v4"))
args = parser.parse_args()
root = args.run
config, status = read_json(root / "config.json"), read_json(root / "status.json")
trials = read_trials(root / "trials")
selected = [read_json(p) for p in sorted((root / "selected").glob("*.json"))]
results_path = root / "results.json"
evaluated = [r for r in read_json(results_path) if r["split"] == "evaluation"] if results_path.exists() else []
by_stage = defaultdict(list)
for row in trials:
    by_stage[row["label"]].append(row)
summary = {
    "state": status["state"], "selection_trials": len(trials), "evaluation_conditions": len(evaluated),
    "max_mpurge_seconds": max([r.get("max_mpurge_seconds", 0) for r in trials + evaluated], default=0),
    "stages": {label: {"trials": len(rows), "total_measured_seconds": sum(r["seconds"] for r in rows),
                      "best_loss": min(r["loss"] for r in rows)} for label, rows in by_stage.items()},
    "selected": [{k: r[k] for k in ("label", "parameters", "frozen", "loss", "metrics")} for r in selected],
    "evaluation": evaluated,
}
write_json(root / "analysis.json", summary)
lines = ["# Rectangle experiment", "",
         f"State: **{status['state']}**. {len(trials):,} selection trials; "
         f"{len(evaluated)} evaluated method/noise conditions.", "",
         "Parameters are selected at zero noise, then frozen. Localization evaluation uses held-out "
         "query positions in the same room. Independent grids are exhaustive; joint tuning is sampled.", "",
         "Noise perturbs both survey and query ranges. Source maps and initial positions are recomputed "
         "at each noise level using the frozen parameters; their numerical outputs are not frozen.", "",
         f"Maximum recorded complete MPUrge fit: **{summary['max_mpurge_seconds']:.3f} s**.", "",
         "| Selected entry | Calibration loss | Parameters |",
         "|---|---:|---|"]
for row in selected:
    lines.append(f"| {row['label']} | {row['loss']:.6f} | {row['parameters']} |")
lines += ["", "Loss is OSPA in meters for MPUrge and mean 3D localization error in meters for the other entries.",
          "VT OSPA uses order 1 and a 1 m cutoff; F1 uses a 0.5 m matching gate. "
          "The reconstruction objective averages AP A, AP B and A+B equally.",
          "Frozen upstream configurations and all metrics are in analysis.json and the selected receipts.", "",
          "## Fixed-parameter evaluation", "",
          "| Entry | Noise (m) | Mean error / OSPA (m) | RMSE (m) / F1 |",
          "|---|---:|---:|---:|"]
for row in sorted(evaluated, key=lambda r: (r["label"], r["level"])):
    metrics = row["metrics"]
    value = metrics.get("ospa", metrics.get("mean_m"))
    extra = metrics.get("f1", metrics.get("rmse_m"))
    lines.append(f"| {row['label']} | {row['level']:g} | {value:.6f} | {extra:.6f} |")
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

names = {"mpurge": "MPUrge / OSPA", "mpurge_map": "MPUrge-MAP", "smart_lea_truth": "SMART-LEA / true VTs",
         "smart_lea": "SMART-LEA / frozen upstream", "joint": "Joint pipeline"}
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for ax, method in zip(axes.flat, ("mpurge", "mpurge_map", "smart_lea_truth", "smart_lea")):
    for unify, color in ((True, "#167a68"), (False, "#c47a34")):
        rows = [r for r in trials if r["method"] == method and r["unify"] == unify]
        windows, losses = [], []
        for p in range(1, 8):
            active = [r["loss"] for r in rows if r["parameters"]["half_window"] == p
                      and r["parameters"]["alpha"] != 1]
            if active:
                windows.append(p)
                losses.append(min(active))
        if losses:
            ax.plot(windows, losses, "o-", color=color, label="unification " + ("on" if unify else "off"))
        inactive = [r["loss"] for r in rows if r["parameters"]["alpha"] == 1]
        if inactive:
            ax.axhline(min(inactive), color=color, linestyle=":", linewidth=1)
    ax.set(title=names[method], xlabel="Half-window", ylabel="Best calibration loss (m)", xticks=range(1, 8))
    ax.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
fig.suptitle("Best active-pattern setting at each window; dotted = alpha 1 (window inactive)")
fig.savefig(root / "window-selection.png", dpi=180)
plt.close(fig)
if evaluated:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    palette = {"mpurge": "#167a68", "mpurge_map": "#5174b5", "smart_lea_truth": "#167a68",
               "smart_lea": "#9c5a93", "joint": "#a65a33"}
    for label in sorted({r["label"] for r in evaluated}):
        rows = sorted([r for r in evaluated if r["label"] == label], key=lambda r: r["level"])
        method, unify = rows[0]["method"], rows[0]["unify"]
        ax = axes[0 if method == "mpurge" else 1]
        ax.plot([r["level"] for r in rows],
                [r["metrics"].get("ospa", r["metrics"].get("mean_m")) for r in rows],
                marker="o", markersize=3, linestyle="-" if unify else "--", color=palette[method],
                label=names[method] + (" / on" if unify else " / off"))
    for ax, title, ylabel in zip(axes, ("Reconstruction", "Localization"), ("OSPA (m)", "Mean 3D error (m)")):
        ax.set(title=title, xlabel="Range noise sigma (m)", ylabel=ylabel)
        ax.set_xscale("symlog", linthresh=.01)
        ax.set_xticks(config["observations"]["noise_levels"],
                      [f"{v:g}" for v in config["observations"]["noise_levels"]], rotation=45)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Noise-free selection; fixed configurations on held-out queries")
    fig.savefig(root / "noise-evaluation.png", dpi=180)
    plt.close(fig)
print({k: summary[k] for k in ("state", "selection_trials", "evaluation_conditions", "max_mpurge_seconds")})
