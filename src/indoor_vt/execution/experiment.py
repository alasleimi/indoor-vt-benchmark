"""Noise-free selection, frozen stage-wise tuning, and joint pipeline tuning."""
from itertools import product
from pathlib import Path
import time
import shutil

import numpy as np

from .artifacts import ArtifactCache, implementation_id, write_npz, memory_bytes, file_lock
from .trials import TrialLog
from .presets import standard_mpurge, mpurge_map, smart_lea
from .reconstruction import MPUrgeExecution
from .localization import LocalizerExecution, LEAExecution
from ..science.grouping import all_pairs_within, nearest_neighborhoods
from ..acquisition.observations import Observations
from ..acquisition.records import Seeds, digest, read_json, write_json
from ..metrics import ap_source_metrics


class Evaluator:
    def __init__(self, data_root, config, run):
        self.observations = Observations(data_root, config, run=run)
        storage = config["storage"]
        self.store = ArtifactCache(Path(data_root) / "numerical-cache-v2",
            disk_bytes=storage["cache_mib"] * 1024**2, ram_bytes=storage["ram_mib"] * 1024**2,
            free_bytes=storage["minimum_free_mib"] * 1024**2)
        self.config, self.run, self.banks = config, run, {}

    def evaluate(self, method, parameters, *, unify, level, split, frozen=None, retain=False):
        started = time.perf_counter()
        frozen = frozen or {}
        settings = self.config["observations"]
        repetitions = settings[split + "_repetitions"]
        # Clean direct observations do not depend on the noise seed.
        if level == 0 and settings["track"] == "direct":
            repetitions = repetitions[:1]
        cases, source_records, query_records, timings = [], [], [], []
        for room in self.observations.rooms:
            truth = self.observations.truth(room)
            for repetition in repetitions:
                survey = self.observations.survey(room, level, repetition)
                sources = None
                if method in ("mpurge", "smart_lea", "joint"):
                    p = parameters if method == "mpurge" else (
                        parameters["mpurge"] if method == "joint" else frozen["mpurge"])
                    if room["id"] not in self.banks:
                        self.banks[room["id"]] = all_pairs_within(nearest_neighborhoods(survey.xyz,
                            size=max(self.config["optimization"]["mpurge"]["neighbors"])))
                    algorithm = standard_mpurge(p, unify=unify)
                    sources = MPUrgeExecution(algorithm, self.store, self.run,
                                              pair_bank=self.banks[room["id"]]).fit(survey)
                    timings.append(sources.evidence["seconds"])
                    if sources.evidence["seconds"] > self.config["runtime"]["mpurge_limit_seconds"]:
                        raise RuntimeError("Standard MPUrge exceeded its runtime budget: "
                                           f"{sources.evidence['seconds']:.2f}s; {p}")
                    if retain:
                        source_records.append({"room": room["id"], "repetition": repetition,
                                               "by_ap": sources.by_ap, "audit": sources.evidence})
                if method == "mpurge":
                    for aps in (("A",), ("B",), ("A", "B")):
                        metrics = ap_source_metrics(sources.by_ap, truth.by_ap, aps)
                        cases.append({"room": room["id"], "repetition": repetition,
                                      "aps": "+".join(aps), "ospa": metrics["ospa"], "f1": metrics["f1"]})
                    continue
                map_parameters = parameters if method == "mpurge_map" else (
                    parameters["map"] if method == "joint" else frozen["map"])
                localizer = LocalizerExecution(mpurge_map(map_parameters, unify=unify), self.store, self.run)
                for layout in room["layouts"]:
                    queries = self.observations.queries(room, layout, level, repetition, split)
                    initial, initial_audit = localizer.predict(survey, queries)
                    audit = {"initial": initial_audit}
                    if method in ("smart_lea", "smart_lea_truth", "joint"):
                        lea_parameters = parameters["lea"] if method == "joint" else parameters
                        result, lea_audit = LEAExecution(smart_lea(lea_parameters, unify=unify),
                            self.store, self.run).predict(queries, initial,
                                truth if method == "smart_lea_truth" else sources)
                        audit["refinement"] = lea_audit
                    else:
                        result = initial
                    errors = np.linalg.norm(result["xyz"] - queries.xyz, axis=1)
                    cases.extend({"error_m": float(e), "fallback": status != "ok"}
                                 for e, status in zip(errors, result["status"]))
                    if retain:
                        query_records.append({"room": room["id"], "layout": layout["id"],
                            "repetition": repetition, "query_ids": queries.position_ids,
                            "xyz": result["xyz"], "initial_xyz": initial["xyz"],
                            "errors_m": errors, "status": result["status"], "audit": audit})
        if method == "mpurge":
            metrics = {name: float(np.mean([c[name] for c in cases])) for name in ("ospa", "f1")}
            loss = metrics["ospa"]
        else:
            errors = np.array([c["error_m"] for c in cases])
            metrics = {"mean_m": float(errors.mean()), "rmse_m": float(np.sqrt(np.mean(errors**2))),
                       "median_m": float(np.median(errors)), "p90_m": float(np.quantile(errors, .9)),
                       "fallback_fraction": float(np.mean([c["fallback"] for c in cases]))}
            loss = metrics["mean_m"]
        result = {"loss": loss, "metrics": metrics, "seconds": time.perf_counter() - started,
                  "max_mpurge_seconds": max(timings, default=0)}
        if retain:
            result["detail"] = {"sources": source_records, "queries": query_records,
                                "source_metrics": cases if method == "mpurge" else []}
        return result


def configurations(space):
    """Exhaust the finite stage grid, collapsing inactive window and clipping choices."""
    keys, result, seen = list(space), [], set()
    for values in product(*(space[k] for k in keys)):
        point = dict(zip(keys, values))
        if point.get("alpha") == 1:
            point["half_window"] = min(space["half_window"])
        if point.get("median_penalty_weight") == 0 and "median_clip_factor" in point:
            point["median_clip_factor"] = min(space["median_clip_factor"])
        key = digest(point)
        if key not in seen:
            seen.add(key)
            result.append(point)
    return result


def joint_configurations(spaces, count, seed, baseline):
    rng = np.random.default_rng(seed)
    pools = {k: configurations(v) for k, v in spaces.items()}
    result, seen = [baseline], {digest(baseline)}
    maximum = int(np.prod([len(v) for v in pools.values()]))
    while len(result) < min(count, maximum):
        point = {k: choices[int(rng.integers(len(choices)))] for k, choices in pools.items()}
        if digest(point) not in seen:
            seen.add(digest(point))
            result.append(point)
    return result


def run_experiment(data_root, run_root, config):
    with file_lock(Path(run_root) / "run.lock"):
        return _run_experiment(data_root, run_root, config)


def _run_experiment(data_root, run_root, config):
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot = run_root / "config.json"
    if snapshot.exists() and read_json(snapshot) != config:
        raise ValueError("Run configuration changed; use a new run directory")
    write_json(snapshot, config)
    version_path = run_root / "implementation.json"
    version = {"id": implementation_id(), "experiment": digest(Path(__file__).read_text()),
               "dataset": digest(read_json(Path(data_root) / "dataset.json"))}
    if version_path.exists() and read_json(version_path) != version:
        raise ValueError("Implementation or data changed; use a new run directory")
    write_json(version_path, version)
    evaluator = Evaluator(data_root, config, config["name"])
    status_path = run_root / "status.json"
    status = {"state": "waiting_for_acquisition", "completed_trials": 0,
              "selection_noise_m": 0.0, "updated": time.time()}
    write_json(status_path, status)
    while not evaluator.observations.available():
        state_path = Path(data_root) / "acquisition_status.json"
        if state_path.exists() and read_json(state_path)["state"] == "failed":
            raise RuntimeError("Acquisition failed; inspect acquisition_status.json")
        time.sleep(5)
    optimization = config["optimization"]
    seeds = Seeds(optimization["seed"])
    modes = optimization["unification"]
    independent = {m: configurations(optimization[m]) for m in ("mpurge", "mpurge_map", "smart_lea")}
    per_mode = len(independent["mpurge"]) + len(independent["mpurge_map"])
    per_mode += 2 * len(independent["smart_lea"]) + optimization["joint_trials"]
    status["target_trials"] = len(modes) * per_mode
    selected = {}
    last_update = 0.0

    def update(current, *, force=False):
        nonlocal last_update
        if force or time.monotonic() - last_update >= 2:
            status.update(current=current, updated=time.time(), cache=evaluator.store.usage())
            write_json(status_path, status)
            last_update = time.monotonic()

    def select(method, unify, schedule, frozen):
        mode = "unified" if unify else "ununified"
        label = method + "_" + mode
        best = None
        context = {"method": method, "unify": unify, "level": 0.0,
                   "split": "calibration", "frozen": frozen, "label": label}
        log = TrialLog(run_root / "trials", label, context)
        status["state"] = "selecting"
        for number, parameters in enumerate(schedule):
            identity = {"method": method, "parameters": parameters, "unify": unify,
                        "level": 0.0, "split": "calibration", "frozen": frozen}
            key = digest(identity)
            trial = log.get(key)
            if trial is None:
                evaluated = evaluator.evaluate(method, parameters, unify=unify, level=0.0,
                                                split="calibration", frozen=frozen)
                trial = {"id": key, "number": number, "label": label, **identity, **evaluated}
                log.append(trial)
            status["completed_trials"] += 1
            if best is None or trial["loss"] < best["loss"]:
                best = trial
            update({"method": label, "trial": number + 1, "total": len(schedule),
                    "best_loss": best["loss"]})
        selected[(method, unify)] = best
        write_json(run_root / "selected" / (label + ".json"), best)
        update({"method": label, "selected": best["parameters"], "loss": best["loss"]}, force=True)

    # Finish the priority independent stages before either pipeline experiment.
    for method in ("mpurge", "mpurge_map", "smart_lea_truth", "smart_lea", "joint"):
        for unify in modes:
            frozen = {}
            if method in ("smart_lea_truth", "smart_lea", "joint"):
                frozen = {"map": selected[("mpurge_map", unify)]["parameters"]}
                if method != "smart_lea_truth":
                    frozen["mpurge"] = selected[("mpurge", unify)]["parameters"]
            if method == "joint":
                baseline = {**frozen, "lea": selected[("smart_lea", unify)]["parameters"]}
                schedule = joint_configurations(
                    {"mpurge": optimization["mpurge"], "map": optimization["mpurge_map"],
                     "lea": optimization["smart_lea"]}, optimization["joint_trials"],
                    seeds.seed("joint", unify), baseline)
                frozen = {}
            else:
                schedule = independent["smart_lea" if method == "smart_lea_truth" else method]
            select(method, unify, schedule, frozen)

    # Choices are now frozen. Neither noisy results nor held-out query positions select parameters.
    status["state"] = "evaluating"
    results = []
    for (method, unify), choice in selected.items():
        for split, levels in (("calibration", [0.0]), ("evaluation", config["observations"]["noise_levels"])):
            for level in levels:
                identity = {k: choice[k] for k in ("method", "parameters", "unify", "frozen")}
                identity.update(level=level, split=split)
                key = digest(identity)
                receipt = run_root / "evaluations" / (key + ".json")
                detail_path = run_root / "outcomes" / (key + ".npz")
                if receipt.exists() and detail_path.exists():
                    row = read_json(receipt)
                else:
                    evaluated = evaluator.evaluate(method, choice["parameters"], unify=unify,
                        level=level, split=split, frozen=choice["frozen"], retain=True)
                    detail = evaluated.pop("detail")
                    used = sum(p.stat().st_size for p in (run_root / "outcomes").glob("*.npz"))
                    estimate = memory_bytes(detail) + 65536
                    if used + estimate > config["storage"]["outcomes_mib"] * 1024**2:
                        raise RuntimeError("Selected outcomes reached their configured storage budget")
                    if shutil.disk_usage(run_root).free - estimate < config["storage"]["minimum_free_mib"] * 1024**2:
                        raise RuntimeError("Insufficient free disk for selected outcomes")
                    write_npz(detail_path, detail, metadata=identity)
                    row = {"id": key, "label": choice["label"], **identity, **evaluated,
                           "outcome": str(detail_path.relative_to(run_root))}
                    write_json(receipt, row)
                results.append(row)
                write_json(run_root / "results.json", results)
                update({"method": choice["label"], "noise": level, "split": split}, force=True)
    status["state"] = "complete"
    update(None, force=True)
    return results
