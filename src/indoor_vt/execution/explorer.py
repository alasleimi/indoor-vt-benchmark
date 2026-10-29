"""Presentation of completed redesign experiments, including saved pipeline stages.

This module only summarizes existing outcomes. It never schedules an experiment
or uses evaluation measurements to select a configuration.
"""
import numpy as np
import ast
from pathlib import Path
from functools import lru_cache
from shapely.geometry import Point, Polygon

from ..metrics import matched_source_metrics, ap_source_metrics


FINISHER_DISPLAY_NAMES = {
    "consumption": "support-first peak-consumption finisher",
    "robust": "robust range-refinement and peak-consumption finisher",
}


def method_entries(receipt):
    method, label = receipt["method"], receipt["label"]
    if method == "range_rmse_true_count":
        name = "Range RMSE (cheating true count)"
        return [{"id": label, "label": label, "name": name, "group": "vt",
                 "component": "sources", "selection": "fixed", "unify": None,
                 "title": name, "subtitle": "Fixed solver settings; lowest measured range residual",
                 "tuning": "Fixed settings", "selection_label": "Fixed settings",
                 "vt_source": None, "tuning_pipeline": None, "selection_description": None,
                 "unification_scope": "Not applicable", "unification_label": "Unification N/A",
                 "source_full_name": name, "search_text": name + " Hungarian least squares known count",
                 "objective": "Measured range RMSE (m)", "derived": False,
                 "family": "range_rmse", "finisher": None, "triangle": False, "naked": True,
                 "tags": ["family:range_rmse", "unification:n/a", "finisher:none", "matching:hungarian"],
                 "code": {"Composition constructors": [constructor_link("science/range_rmse.py", "range_rmse")],
                          "Scientific algorithms": ["science/range_rmse.py"],
                          "Execution and experiment configuration": ["execution/range_rmse_experiment.py"],
                          "Caching and saved evidence": ["execution/artifacts.py"],
                          "Observation conversion and metrics": ["acquisition/observations.py", "metrics.py",
                                                                 "execution/explorer.py"]}}]
    family = method.split("_")[0] if method.startswith(("zayets", "ransac")) else "mpurge"
    from .compositions import VARIANTS
    variant = method.removesuffix("_joint").removesuffix("_lea")
    finishing = VARIANTS[variant][1] if variant in VARIANTS else {
        "mpurge": None, "zayets": "consumption", "ransac": "robust"}[family]
    source = {"mpurge": "MPUrge", "zayets": "Zayets", "ransac": "RANSAC"}[family]
    allow_crossing = variant.startswith("mpurge_allow_crossing")
    if allow_crossing:
        source += " (crossings allowed)"
    if variant == "mpurge_triangle":
        source += " triangle"
    elif variant.endswith("_naked"):
        source += " (no finisher)"
    elif finishing:
        source += " + " + FINISHER_DISPLAY_NAMES[finishing]
    full_source = source
    definitions = {
        "mpurge": [("", "vt", "sources", "ospa")],
        "mpurge_map": [("", "raw", "localization", "localization")],
        "smart_lea": [("", "pipeline", "localization", "stagewise")],
        "smart_lea_truth": [("", "pipeline", "localization", "truth")],
        "joint": [("_sources", "vt", "sources", "joint"),
                  ("_initial", "raw", "initial", "joint"),
                  ("", "pipeline", "localization", "joint")],
    }
    for extra in ("zayets", "ransac"):
        definitions[extra] = [("", "vt", "sources", "ospa")]
        definitions[extra + "_lea"] = [("", "pipeline", "localization", "stagewise")]
        definitions[extra + "_joint"] = definitions["joint"]
    for extra in VARIANTS:
        definitions[extra] = [("", "vt", "sources", "ospa")]
        definitions[extra + "_lea"] = [("", "pipeline", "localization", "stagewise")]
        definitions[extra + "_joint"] = definitions["joint"]
    entries = []
    for suffix, group, component, selection in definitions.get(method, []):
        name = {"vt": source, "raw": "MPUrge-MAP", "pipeline": "MPUrge-MAP → SMART-LEA"}[group]
        short_name = {"vt": source, "raw": "MAP", "pipeline": "MAP → LEA"}[group]
        tuning = {"ospa": "OSPA", "localization": "Independent", "stagewise": "Stagewise",
                  "joint": {"vt": "Joint with MPUrge-MAP + SMART-LEA",
                            "raw": f"Joint with {source} + SMART-LEA",
                            "pipeline": f"Joint with {source}"}[group], "truth": "LEA only"}[selection]
        vt_source = ("Ground truth" if selection == "truth" else source) if group == "pipeline" else None
        tuning_pipeline = source if group == "raw" and selection == "joint" else None
        selection_label = tuning
        selection_description = (
            f"{source}, MPUrge-MAP and SMART-LEA parameters were optimized together to minimize "
            "mean 3D query error after SMART-LEA refinement. Selection used noise-free calibration "
            "queries across all obstacle layouts, not each obstacle count separately."
            if selection == "joint" else None)
        context = ([f"VTs: {vt_source}"] if vt_source else []) + [selection_label]
        mode = "N/A" if receipt["unify"] is None else "on" if receipt["unify"] else "off"
        scope = ("Not applicable" if receipt["unify"] is None else
                 "MPUrge-MAP" if group == "raw" else
                 "MPUrge" if group == "vt" and family == "mpurge" else
                 "MPUrge + MPUrge-MAP + SMART-LEA" if family == "mpurge" and selection != "truth" else
                 "MPUrge-MAP + SMART-LEA")
        mode_label = f"{'MAP/LEA unification' if group == 'vt' and family != 'mpurge' and receipt['unify'] is not None else 'Unification'} {mode}"
        legend_context = ([f"VTs: {vt_source}"] if vt_source else []) + [tuning]
        title = " · ".join([short_name, *legend_context, mode_label])
        entries.append({"id": label + suffix, "label": label, "name": name, "group": group,
            "component": component, "selection": selection, "unify": receipt["unify"],
            "title": title, "subtitle": " · ".join([*context, mode_label]),
            "tuning": tuning, "selection_label": selection_label, "vt_source": vt_source,
            "selection_description": selection_description,
            "tuning_pipeline": tuning_pipeline, "unification_scope": scope, "unification_label": mode_label,
            "source_full_name": full_source,
            "search_text": " ".join([name, title, full_source, "true VTs" if selection == "truth" else ""]),
            "objective": "OSPA (c = 1 m)" if selection == "ospa" else "Mean 3D error (m)",
            "derived": (method == "joint" or method.endswith("_joint")) and bool(suffix),
            "family": family, "finisher": finishing, "triangle": variant == "mpurge_triangle",
            "allow_crossing": allow_crossing,
            "naked": variant.endswith("_naked"),
            "tags": ["family:" + family, "unification:" + mode.lower(),
                     "finisher:" + ("none" if variant.endswith("_naked") else finishing or "cluster"),
                     "matching:" + ("crossing_allowed" if allow_crossing else
                                    "triangle" if variant == "mpurge_triangle" else "standard")],
            "code": code_links(method, component)})
    return entries


@lru_cache(maxsize=256)
def code_links(method, component):
    from .compositions import VARIANTS
    variant = method.removesuffix("_joint").removesuffix("_lea")
    composition = variant in VARIANTS
    extra = method.startswith(("zayets", "ransac"))
    family = method.split("_")[0] if extra else "mpurge"
    initial_only = component == "initial" or method == "mpurge_map"
    scientific = []
    execution = ["execution/additional_experiment.py" if extra else "execution/experiment.py"]
    if not initial_only and method != "smart_lea_truth":
        scientific += (["science/" + family + ".py", "science/selection.py"] if extra else
                       ["science/reconstruction.py", "science/grouping.py", "science/clustering.py"])
        execution += ["execution/additional.py" if extra else "execution/reconstruction.py"]
    if component != "sources":
        scientific += ["science/localization.py"]
        execution += ["execution/localization.py"]
    if not extra or component != "sources":
        scientific += ["science/matching.py"]
        execution += ["execution/presets.py"]
    if not initial_only:
        scientific += ["science/estimation.py"]
    scientific += ["science/costs.py"]
    constructors = []
    if not initial_only and method != "smart_lea_truth":
        if composition:
            constructors.append(constructor_link("execution/compositions.py", "composed_reconstructor"))
            execution += ["execution/compositions.py", "execution/composition_experiment.py"]
        elif extra:
            constructors += [constructor_link("execution/additional.py", family + "_proposals"),
                             constructor_link("execution/additional.py", "finisher")]
        else:
            constructors.append(constructor_link("execution/presets.py", "standard_mpurge"))
    if component != "sources":
        constructors.append(constructor_link("execution/presets.py", "mpurge_map"))
        if not initial_only:
            constructors.append(constructor_link("execution/presets.py", "smart_lea"))
    if composition:
        execution = [p for p in execution if p != "execution/additional_experiment.py"]
        if variant.startswith("mpurge") and variant != "mpurge_triangle":
            scientific.append("science/selection.py")
    finishing = VARIANTS[variant][1] if composition else {
        "zayets": "consumption", "ransac": "robust"}.get(family)
    finisher_links = []
    if finishing and not initial_only:
        if composition:
            constructors.append(constructor_link("execution/additional.py", "finisher"))
        finisher_links = [constructor_link("execution/additional.py", "finisher"),
            constructor_link("science/selection.py", "SupportFirstScorer" if finishing == "consumption"
                             else "SupportAndResidualScorer"),
            constructor_link("science/selection.py", "GreedySourceSelector"),
            constructor_link("execution/additional.py", "finish_one")]
        if finishing == "robust":
            finisher_links += [constructor_link("science/selection.py", "NearestPeakRefitter"),
                constructor_link("science/estimation.py", "Multilateration3D"),
                constructor_link("science/costs.py", "gaussian_weights"),
                constructor_link("science/costs.py", "soft_l1_loss")]
    step_links = []
    if not initial_only and method != "smart_lea_truth":
        definitions = {
            "mpurge": [("science/grouping.py", "nearest_neighborhoods"),
                       ("science/matching.py", "_subset"),
                       ("science/matching.py", "_delta"),
                       ("science/matching.py", "_merge_purge_urge"),
                       ("science/grouping.py", "ConnectedComponentsGrouper"),
                       ("science/grouping.py", "group_within_scopes")],
            "zayets": [("science/grouping.py", "radius_neighborhoods"),
                       ("science/zayets.py", "split_at_gaps"),
                       ("science/zayets.py", "cluster_feature_cost"),
                       ("science/zayets.py", "ConstrainedSingleLinkageGrouper"),
                       ("science/zayets.py", "LongestClusterTemplateAssigner")],
            "ransac": [("science/ransac.py", "uniform_peak_sample"),
                       ("science/ransac.py", "ThreeSphereIntersections"),
                       ("science/ransac.py", "TopPerTrialBatch")],
        }
        step_links = [constructor_link(file, symbol) for file, symbol in definitions[family]]
        if variant.startswith("mpurge_allow_crossing"):
            step_links.append(constructor_link("science/matching.py", "CrossingAllowedMPUrgeMatcher"))
        step_links += [constructor_link("science/estimation.py", "Multilateration3D"),
                       constructor_link("science/selection.py", "nearest_peaks"),
                       constructor_link("science/selection.py", "range_curve_rms")]
        if component == "localization":
            step_links += [constructor_link("science/localization.py", name) for name in
                           ("MatchBasedScorer", "DistanceWeightedKNN", "SmartLEA")]
    return {"Composition constructors": constructors,
            **({"Finisher implementation": finisher_links} if finisher_links else {}),
            **({"Algorithm steps": step_links} if step_links else {}),
            "Scientific algorithms": list(dict.fromkeys(scientific)),
            "Execution and experiment configuration": list(dict.fromkeys(execution)),
            "Caching and saved evidence": ["execution/artifacts.py", "execution/inventory.py",
                                           "execution/trials.py", "execution/store.py"],
            "Observation conversion and metrics": ["acquisition/observations.py", "metrics.py"]}


@lru_cache(maxsize=32)
def constructor_link(file, function):
    tree = ast.parse((Path(__file__).parents[1] / file).read_text(encoding="utf-8"))
    line = next(node.lineno for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function)
    return {"file": file, "line": line, "label": function + "()"}


def localization_metrics(errors):
    errors = np.asarray(errors, dtype=float)
    if not len(errors):
        return {}
    return {"mean_m": float(errors.mean()), "rmse_m": float(np.sqrt(np.mean(errors**2))),
            "median_m": float(np.median(errors)), "p90_m": float(np.quantile(errors, .9))}


def query_truth(record, rooms):
    room = rooms[record["room"]]
    layout = next(item for item in room["layouts"] if item["id"] == record["layout"])
    return layout["queries_xyz"][record["query_ids"]]


def inside_obstacles(points, boxes):
    """Indices inside the actual box volume, including its boundary, not just its footprint."""
    footprints = [(Polygon(box["footprint_xy"]), box["height_m"]) for box in boxes]
    return np.array([i for i, (x, y, z) in enumerate(points)
                     if any(0 <= z <= height and polygon.covers(Point(x, y))
                            for polygon, height in footprints)], dtype=np.int64)


def display_source_metrics(predicted, truth):
    """Add gated matched RMSE to saved coordinates without changing selection metrics.

    Maximize F1's match count first, then minimize squared 3D distance among
    those assignments. The returned matches also supply the viewer's map lines.
    """
    return matched_source_metrics(predicted, truth)


def summarize_sources(records, truths):
    cases = {"A": [], "B": [], "A+B": []}
    for record in records:
        for aps in (("A",), ("B",), ("A", "B")):
            metrics = ap_source_metrics(record["by_ap"], truths[record["room"]].by_ap, aps)
            cases["+".join(aps)].append(metrics)
    by_ap = {ap: {key: float(np.mean([case[key] for case in values])) for key in ("ospa", "f1")}
             for ap, values in cases.items() if values}
    for ap, metrics in by_ap.items():
        matched_count = sum(case["tp"] for case in cases[ap])
        squared_error = sum(case["matched_squared_error_m2"] for case in cases[ap])
        metrics["matched_rmse_m"] = float(np.sqrt(squared_error / matched_count)) if matched_count else None
    overall = {key: float(np.mean([cases[ap][i][key] for i in range(len(records))
                                  for ap in ("A", "B", "A+B")]))
               for key in ("ospa", "f1")} if records else {}
    return overall, by_ap


def saved_stage_metrics(detail, rooms, truths):
    # Preserve the recorded search objective separately from the AP-specific display scores.
    sources, by_ap = summarize_sources(detail["sources"], truths)
    source_counts = []
    for record in detail["sources"]:
        predicted = {ap: len(record["by_ap"][ap]) for ap in ("A", "B")}
        true = {ap: len(truths[record["room"]].by_ap[ap]) for ap in ("A", "B")}
        predicted["A+B"], true["A+B"] = sum(predicted.values()), sum(true.values())
        source_counts.append({"room": record["room"], "repetition": record["repetition"],
                              "predicted": predicted, "true": true})
    errors = [np.linalg.norm(record["initial_xyz"] - query_truth(record, rooms), axis=1)
              for record in detail["queries"]]
    grouped = {"initial": {}, "localization": {}}
    for record in detail["queries"]:
        layout = next(item for item in rooms[record["room"]]["layouts"]
                      if item["id"] == record["layout"])
        obstacle_count = layout.get("box_count", len(layout.get("boxes", [])))
        truth = query_truth(record, rooms)
        for component, field in (("initial", "initial_xyz"), ("localization", "xyz")):
            grouped[component].setdefault(obstacle_count, []).append(
                np.linalg.norm(record[field] - truth, axis=1))
    by_obstacles = {component: {count: {**localization_metrics(np.concatenate(batches)),
                                      "query_count": sum(len(batch) for batch in batches)}
                               for count, batches in groups.items()}
                    for component, groups in grouped.items()}
    return {"sources": sources, "source_metrics_by_ap": by_ap,
            "source_counts": source_counts, "by_obstacles": by_obstacles,
            "initial": localization_metrics(np.concatenate(errors)) if errors else {}}


def build_catalog(trials, results, stage_metrics):
    """Expose only observed methods and measured conditions, preserving receipt IDs."""
    best, counts = {}, {}
    for trial in trials:
        label = trial["label"]
        counts[label] = counts.get(label, 0) + 1
        if label not in best or trial["loss"] < best[label]["loss"]:
            best[label] = trial
    conditions = {(r["label"], r["split"], r["level"]): r for r in results}
    for label, trial in best.items():
        conditions.setdefault((label, "calibration", 0.), trial)
    methods, rows = {}, []
    for receipt in conditions.values():
        for entry in method_entries(receipt):
            metrics = receipt["metrics"]
            if entry["derived"]:
                metrics = stage_metrics.get(receipt["id"], {}).get(entry["component"], {})
                if not metrics:  # A component without a saved output is not a measured result.
                    continue
            elif entry["group"] == "vt":
                metrics = {**metrics, **stage_metrics.get(receipt["id"], {}).get("sources", {})}
            if entry["group"] == "vt":
                metrics = {k: v for k, v in metrics.items() if k not in ("predicted_count", "true_count")}
            methods[entry["id"]] = {**entry, "trial_count": counts.get(entry["label"], 0),
                                     "selected_id": best.get(entry["label"], receipt)["id"]}
            rows.append({"method_id": entry["id"], "receipt_id": receipt["id"],
                         "split": receipt["split"], "level": receipt["level"],
                         "metrics": metrics, "selection_loss": best.get(entry["label"], receipt)["loss"],
                         "source_counts": (stage_metrics.get(receipt["id"], {}).get("source_counts", [])
                                           if entry["group"] == "vt" else []),
                         "source_metrics_by_ap": (stage_metrics.get(receipt["id"], {}).get("source_metrics_by_ap", {})
                                                  if entry["group"] == "vt" else {}),
                         "obstacle_metrics": stage_metrics.get(receipt["id"], {}).get(
                             "by_obstacles", {}).get(entry["component"], {})})
    return {"methods": list(methods.values()), "conditions": rows}
