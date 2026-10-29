"""Frozen reconstruction on already acquired obstacle-layout fingerprints.

Only this evaluation module reads privileged simulator paths. Reconstructors
receive receiver coordinates and delay-only observations, plus count for RMSE.
"""
import json
from pathlib import Path
import time

import numpy as np

from .artifacts import implementation_id, read_npz, write_npz
from .experiment import Evaluator
from .explorer import inside_obstacles
from ..metrics import ap_source_metrics
from .presets import standard_mpurge
from .reconstruction import MPUrgeExecution
from .additional import AdditionalExecution
from .compositions import CompositionExecution, VARIANTS
from ..acquisition.records import read_json, write_json, digest, Seeds
from ..science.models import Survey, SourceMap
from ..science.range_rmse import range_rmse
from ..science.grouping import all_pairs_within, nearest_neighborhoods

STUDY = "rectangle-obstacle-reconstruction-v1"


def extends_frozen_protocol(previous, current):
    """Allow extra frozen methods without changing existing methods or inputs."""
    old, new = dict(previous), dict(current)
    old_methods, new_methods = old.pop("methods", []), new.pop("methods", [])
    by_id = {method["id"]: method for method in new_methods}
    return (old == new and len(by_id) == len(new_methods)
            and all(by_id.get(method["id"]) == method for method in old_methods))


def layout_inputs(observations, room, layout):
    """Use the explicit layout receiver set and its actual acquired channel."""
    ids = np.asarray(layout.get("reconstruction_position_ids", np.arange(len(layout["queries_xyz"]))))
    xyz = np.asarray(layout["queries_xyz"])[ids]
    if len(inside_obstacles(xyz, layout["boxes"])):
        raise ValueError("An acquired reconstruction receiver is inside an obstacle")
    channels = {ap: observations.channel(room, ap, layout["id"], ids, 0., 2, "direct")
                for ap in room["access_points_xyz"]}
    return Survey(xyz, channels, ids)


def visible_source_catalog(observations, room, layout):
    """Union of distinct images in archived specular paths at the input receivers.

    This is an observed-scene target, not a global catalogue of every source
    visible anywhere in free space. Native full-precision polylines avoid
    inferring truth from estimator outputs or rounded AoA fields.
    """
    result, audit = {}, {}
    wanted = set(map(int, layout.get("reconstruction_position_ids", np.arange(len(layout["queries_xyz"])))))
    for ap, physical in room["access_points_xyz"].items():
        entries, by_position, native_ranges = [], {}, {}
        path_count = 0
        receipt_keys = []
        for job in observations.manifest["jobs"]:
            if (job["room"], job["ap"], job["group"]) != (room["id"], ap, layout["id"]):
                continue
            receipt = read_json(observations.root / "native" / job["key"] / "receipt.json")
            receipt_keys.append({"job": job["key"], "native": receipt["native_value"],
                                 "visualization": receipt["visualization_value"]})
            native = observations.store.get(receipt["native_value"])
            visual = observations.store.get(receipt["visualization_value"])
            for i, position in enumerate(native["position_ids"]):
                if int(position) not in wanted:
                    continue
                native_ranges[int(position)] = native["paths"][native["offsets"][i]:native["offsets"][i+1], 0] * 3e8
                by_position.setdefault(int(position), [])
            content = bytes(visual["Output/Visualizer/Mpc.json"]).decode()
            for line in content.splitlines():
                record = json.loads(line)
                if (record["TX"], record["RX"], record["PAA_TX"], record["PAA_RX"]) != (0, 1, 0, 0):
                    continue
                order = record["Rorder"]
                if order > 2:
                    raise ValueError("Unexpected reflection order")
                for i, flat in enumerate(record["MPC"]):
                    if flat is None:
                        continue
                    position = int(native["position_ids"][i])
                    if position not in wanted:
                        continue
                    for path in np.asarray(flat, float).reshape(-1, order+2, 3):
                        if not np.isfinite(path).all():
                            continue  # NIST uses null rows for an absent path.
                        # NIST direct polylines are Tx->Rx; reflected ones Rx->Tx.
                        if np.linalg.norm(path[-1] - physical) < 1e-7:
                            path = path[::-1]
                        if (np.linalg.norm(path[0] - physical) > 1e-7 or
                                np.linalg.norm(path[-1] - layout["queries_xyz"][position]) > 1e-7):
                            raise ValueError("Path endpoint differs from the saved receiver/AP")
                        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                        direction = path[-2] - path[-1]
                        image = path[-1] + length * direction / np.linalg.norm(direction)
                        entry = next((e for e in entries if np.linalg.norm(image - e["xyz"]) < 1e-6), None)
                        if entry is None:
                            entry = {"xyz": image.tolist(), "surface_sequences": [["reflection"] * order],
                                     "reflection_order": order, "visible_position_ids": [], "path_count": 0,
                                     "witness": {"type": "archived_NIST_specular_path", "position_id": position,
                                                 "polyline": path.tolist()}}
                            entries.append(entry)
                        entry["path_count"] += 1
                        if position not in entry["visible_position_ids"]:
                            entry["visible_position_ids"].append(position)
                        by_position[position].append(length)
                        path_count += 1
        errors = []
        for position, values in by_position.items():
            expected = native_ranges[position]
            if len(values) != len(expected):
                raise ValueError("Native and archived path counts differ")
            errors.extend(abs(np.sort(values) - np.sort(expected)))
        max_error = float(max(errors, default=0))
        if max_error > 2e-6:
            raise ValueError("Archived path lengths do not reproduce native observations")
        entries.sort(key=lambda e: (e["reflection_order"], *e["xyz"]))
        for i, entry in enumerate(entries):
            entry["id"] = f"{ap}_{layout['id']}_visible_source_{i:03d}"
        result[ap] = entries
        audit[ap] = {"count": len(entries), "path_count": path_count,
                     "maximum_native_range_discrepancy_m": max_error, "receipts": receipt_keys,
                     "visible_positions_per_source": [len(e["visible_position_ids"]) for e in entries]}
    return result, audit


def frozen_source_spec(receipt, config):
    method, p = receipt["method"], receipt["parameters"]
    if method == "joint":
        method, p = "mpurge", p["mpurge"]
    elif method.endswith("_joint"):
        method, p = method.removesuffix("_joint"), p["source"]
    if method not in ("mpurge", "range_rmse_true_count", "zayets", "ransac", *VARIANTS):
        raise ValueError(f"Unsupported source method {method}")
    return {"method": method, "parameters": p, "unify": receipt["unify"],
            "proposal_seed": config.get("additional", {}).get("proposal_seed", 9173),
            "max_trials": max(config.get("additional", {}).get("ransac", {}).get("trials", [30000]))}


def reconstruct(spec, survey, evaluator, truth_count, room_id):
    method, p, unify = spec["method"], spec["parameters"], spec["unify"]
    bank = all_pairs_within(nearest_neighborhoods(survey.xyz, size=40))
    if method == "mpurge":
        return MPUrgeExecution(standard_mpurge(p, unify=unify), evaluator.store,
                               STUDY, pair_bank=bank).fit(survey)
    if method in ("zayets", "ransac"):
        return AdditionalExecution(method, p, evaluator.store, seed=spec["proposal_seed"],
                                   max_trials=spec["max_trials"]).fit(survey)
    if method in VARIANTS:
        return CompositionExecution(method, p, evaluator.store, unify=unify, pair_bank=bank,
                                    seed=spec["proposal_seed"], max_trials=spec["max_trials"]).fit(survey)
    by_ap, audit = {}, {}
    for ap, ranges in survey.channels.items():
        settings = {**p, "seed": Seeds(p["seed"]).seed(room_id, ap)}
        by_ap[ap], audit[ap] = range_rmse(settings).fit(survey.xyz, ranges, truth_count[ap])
    return SourceMap(by_ap, audit)


def metrics_for_record(record):
    result, counts = {}, {"predicted": {}, "true": {}}
    for aps in (("A",), ("B",), ("A", "B")):
        name = "+".join(aps)
        truth = {ap: np.asarray([entry["xyz"] for entry in record["truth_catalog"][ap]]).reshape(-1, 3)
                 for ap in aps}
        result[name] = ap_source_metrics(record["by_ap"], truth, aps)
        counts["predicted"][name] = sum(len(record["by_ap"][ap]) for ap in aps)
        counts["true"][name] = sum(len(truth[ap]) for ap in aps)
    return result, {"room": record["room"], "layout": record["layout"], "repetition": 2, **counts}


def run_obstacle_reconstruction(data_root, base_run, run_root, *, selection_data_root=None,
                                study=STUDY, protocol_overrides=None):
    from .viewer import MultiStudyView
    view = MultiStudyView(selection_data_root or data_root, base_run)
    overview = view.overview()
    methods = [m for m in overview["methods"] if m["group"] == "vt"]
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    config = {**view.base.config, "name": study, "engine": "obstacle_reconstruction"}
    if selection_data_root is not None:
        config["data_root"] = str(Path(data_root).resolve())
        config["input_protocol"] = "matched_survey"
    evaluator = Evaluator(data_root, config, study)
    specifications = []
    for method in methods:
        row = next(r for r in overview["conditions"] if r["method_id"] == method["id"]
                   and r["split"] == "evaluation" and r["level"] == 0 and "obstacle_count" not in r)
        owner = view.owner(row["receipt_id"])
        receipt = owner.receipt(row["receipt_id"])
        specifications.append({"entry": method, "receipt": receipt,
                               "source": frozen_source_spec(receipt, owner.config)})
    protocol = {"input": "All100 already acquired query fingerprints per saved layout; known receiver coordinates; sigma=0",
                "calibration": "None: exact existing frozen OSPA-selected or localization-joint-selected parameters",
                "positions": "All100 saved layout positions (60 formerly used for localization calibration,40 held out there); these positions are reconstruction inputs, not a new held-out localization test",
                "truth": "Union of distinct image sources visible in archived NIST paths at these100 receivers, including direct, room and obstacle reflections up to order2; not a global free-space visibility catalogue",
                "truth_access": "Only evaluator reads paths/identities/coordinates; Range RMSE additionally receives integer count for each layout/AP",
                "obstacles": "Existing0,1,3,5-box layouts; no new geometry or simulation",
                "inside_obstacles": "Saved layout receivers are outside boxes; empty-room survey references, including those now inside boxes, are not used",
                "control": "0-box layout uses100 random receivers; existing280-survey results remain separate and unchanged",
                "noise_repetitions": "One deterministic zero-noise observation, stored repetition ID2",
                "methods": [{"id": s["entry"]["id"], "calibration_run": s["entry"]["run"],
                             "frozen_receipt": s["receipt"]["id"], "source": s["source"]} for s in specifications]}
    protocol.update(protocol_overrides or {})
    version = {"id": implementation_id(), "dataset": digest(evaluator.observations.manifest),
               "experiment": digest(Path(__file__).read_text())}
    if selection_data_root is not None:
        version["base_dataset"] = digest(view.base.observations.manifest)
    for name, value in (("config", config), ("protocol", protocol), ("implementation", version)):
        existing = run_root / (name + ".json")
        if existing.exists() and name != "implementation" and read_json(existing) != value:
            if name != "protocol" or not extends_frozen_protocol(read_json(existing), value):
                raise ValueError(f"Existing {name} differs")
        write_json(existing, value)
    results, reused = [], {}
    published = {r["id"]: r for r in read_json(run_root / "results.json")} if (run_root / "results.json").exists() else {}
    target = len(methods) * sum(len(r["layouts"]) for r in evaluator.observations.rooms)
    started = time.perf_counter()
    for room in evaluator.observations.rooms:
        for layout in room["layouts"]:
            survey = layout_inputs(evaluator.observations, room, layout)
            catalog, truth_audit = visible_source_catalog(evaluator.observations, room, layout)
            write_json(run_root / "truth" / (room["id"] + "_" + layout["id"] + ".json"),
                       {"catalog": catalog, "audit": truth_audit})
            for spec in specifications:
                entry, receipt, source = spec["entry"], spec["receipt"], spec["source"]
                numerical_inputs = {"xyz": survey.xyz.tolist(),
                                    "ranges": {ap: r.values.tolist() for ap, r in survey.channels.items()}}
                identity = {"method_id": entry["id"], "frozen_receipt": receipt["id"], "source": source,
                            "room": room["id"], "layout": layout["id"], "input": digest(numerical_inputs),
                            "truth": digest(catalog), "protocol": "obstacle_reconstruction.1"}
                key = digest(identity)
                destination = run_root / "evaluations" / (key + ".json")
                outcome = run_root / "outcomes" / (key + ".npz")
                numerical_key = digest({"source": source, "input": identity["input"], "truth_count": {ap: len(v) for ap, v in catalog.items()} if source["method"] == "range_rmse_true_count" else None})
                write_json(run_root / "status.json", {"state": "running", "current": {"method": entry["id"], "layout": layout["id"]},
                           "completed_conditions": len(results), "target_conditions": target,
                           "completed_trials": 0, "target_trials": 0})
                if destination.exists() and outcome.exists():
                    row = read_json(destination)
                else:
                    start = time.perf_counter()
                    if numerical_key in reused:
                        old = read_npz(reused[numerical_key])[0]["sources"][0]
                        sources = SourceMap(old["by_ap"], old["audit"])
                    else:
                        sources = reconstruct(source, survey, evaluator, {ap: len(v) for ap, v in catalog.items()}, room["id"])
                    record = {"room": room["id"], "layout": layout["id"], "obstacle_count": layout["box_count"],
                              "repetition": 2, "by_ap": sources.by_ap, "audit": sources.evidence,
                              "truth_catalog": catalog, "survey_xyz": survey.xyz,
                              "survey_position_ids": survey.position_ids, "obstacles": layout["boxes"]}
                    metrics, counts = metrics_for_record(record)
                    write_npz(outcome, {"sources": [record], "queries": []}, metadata=identity)
                    row = {**{k: receipt[k] for k in ("label", "method", "parameters", "unify", "frozen")},
                           "id": key, "view_method_id": entry["id"], "split": "evaluation", "level": 0.,
                           "obstacle_count": layout["box_count"], "layout": layout["id"],
                           "input_count": len(survey.xyz), "input_protocol": config.get("input_protocol", "query_positions"),
                           "source_metrics_by_ap": metrics, "source_counts": [counts],
                           "metrics": metrics["A+B"], "loss": metrics["A+B"]["ospa"],
                           "selection_loss": receipt["loss"], "frozen_origin": {"run": entry["run"], "receipt_id": receipt["id"]},
                           "outcome": str(outcome.relative_to(run_root)), "seconds": time.perf_counter()-start}
                    write_json(destination, row)
                reused[numerical_key] = outcome
                results.append(row)
                published[row["id"]] = row
                write_json(run_root / "results.json", list(published.values()))
                print({"completed": len(results), "target": target, "method": entry["id"], "obstacles": layout["box_count"],
                       "ospa": row["metrics"]["ospa"], "f1": row["metrics"]["f1"], "seconds": row["seconds"]}, flush=True)
    write_json(run_root / "status.json", {"state": "complete", "completed_conditions": len(results),
               "target_conditions": target, "completed_trials": 0, "target_trials": 0, "seconds": time.perf_counter()-started})
    return results
