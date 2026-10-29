"""Finisher swaps and triangle matching: clean selection and frozen evaluation."""
from pathlib import Path
import time
import numpy as np

from .compositions import CompositionExecution, VARIANTS, composition_id
from .additional import extension_id, ROBUST_FINISHER_PARAMETERS
from .additional_experiment import additional_config, robust_finisher_grid
from .artifacts import file_lock, implementation_id, write_npz, memory_bytes
from .experiment import Evaluator, configurations, joint_configurations
from .localization import LocalizerExecution, LEAExecution
from .presets import mpurge_map, smart_lea
from .trials import TrialLog
from ..acquisition.records import read_json, write_json, digest, Seeds
from ..science.grouping import all_pairs_within, nearest_neighborhoods
from ..metrics import ap_source_metrics


def composition_config(base):
    config = additional_config(base)
    config["name"] = "rectangle-compositions-v1"
    config["engine"] = "compositions"
    mpurge = config["optimization"]["mpurge"]
    finisher_grid = {k: config["additional"]["zayets"][k]
                     for k in ("tolerance_m", "support_fraction", "duplicate_m")}
    proposal_grid = {k: mpurge[k] for k in ("half_window", "alpha", "neighbors")}
    config["compositions"] = {"source_trials": 512, "joint_trials": 512,
        "search": "Exhaustive up to 512; seeded sampling without replacement above 512",
        "spaces": {"zayets_robust": {**config["additional"]["zayets"], **robust_finisher_grid()},
                   "ransac_consumption": {k: v for k, v in config["additional"]["ransac"].items()
                                          if k not in ROBUST_FINISHER_PARAMETERS},
                   "mpurge_consumption": {**proposal_grid, **finisher_grid},
                   "mpurge_robust": {**proposal_grid, **finisher_grid, **robust_finisher_grid()},
                   "mpurge_triangle": {**mpurge, "triangle_tolerance_m": [0., .05, .2]},
                   "mpurge_allow_crossing": dict(mpurge),
                   "mpurge_allow_crossing_consumption": {**proposal_grid, **finisher_grid},
                   "mpurge_allow_crossing_robust": {**proposal_grid, **finisher_grid, **robust_finisher_grid()},
                   "zayets_naked": {k: config["additional"]["zayets"][k]
                                    for k in ("scope_radius_m", "gap_m", "group_cost")},
                   "ransac_naked": {k: config["additional"]["ransac"][k]
                                    for k in ("trials", "tolerance_m")}}}
    return config


def source_schedule(space, budget, seed):
    pool = configurations(space)
    if len(pool) <= budget:
        return pool
    ids = np.random.default_rng(seed).choice(len(pool), budget, replace=False)
    # Group nearby settings for exact stage-cache reuse, without changing the sampled set.
    return [pool[i] for i in sorted(ids)]


class CompositionEvaluator(Evaluator):
    def evaluate(self, method, parameters, *, unify, level, split, frozen=None, retain=False):
        started = time.perf_counter()
        family = method.removesuffix("_joint").removesuffix("_lea")
        if family not in VARIANTS:
            raise ValueError("Unknown additional reconstructor")
        frozen = frozen or {}
        source_only, joint = method == family, method.endswith("_joint")
        source_p = parameters if source_only else parameters["source"] if joint else frozen["source"]
        settings = self.config["observations"]
        repetitions = settings[split + "_repetitions"]
        if level == 0 and settings["track"] == "direct":
            repetitions = repetitions[:1]
        source_records, query_records, cases, timings = [], [], [], []
        for room in self.observations.rooms:
            truth = self.observations.truth(room)
            for repetition in repetitions:
                survey = self.observations.survey(room, level, repetition)
                if room["id"] not in self.banks:
                    self.banks[room["id"]] = all_pairs_within(nearest_neighborhoods(survey.xyz, size=40))
                sources = CompositionExecution(family, source_p, self.store, unify=unify,
                    pair_bank=self.banks.get(room["id"]),
                    seed=self.config["additional"]["proposal_seed"],
                    max_trials=max(self.config["additional"]["ransac"]["trials"])).fit(survey)
                timings.append(sources.evidence["seconds"])
                if retain:
                    source_records.append({"room": room["id"], "repetition": repetition,
                                           "by_ap": sources.by_ap, "audit": sources.evidence})
                if source_only:
                    for aps in (("A",), ("B",), ("A", "B")):
                        metrics = ap_source_metrics(sources.by_ap, truth.by_ap, aps)
                        cases.append({"ospa": metrics["ospa"], "f1": metrics["f1"]})
                    continue
                map_p = parameters["map"] if joint else frozen["map"]
                lea_p = parameters["lea"] if joint else parameters
                localizer = LocalizerExecution(mpurge_map(map_p, unify=unify), self.store, self.run)
                refiner = LEAExecution(smart_lea(lea_p, unify=unify), self.store, self.run)
                for layout in room["layouts"]:
                    queries = self.observations.queries(room, layout, level, repetition, split)
                    initial, initial_audit = localizer.predict(survey, queries)
                    result, lea_audit = refiner.predict(queries, initial, sources)
                    errors = np.linalg.norm(result["xyz"] - queries.xyz, axis=1)
                    cases.extend({"error_m": float(e), "fallback": status != "ok"}
                                 for e, status in zip(errors, result["status"]))
                    if retain:
                        query_records.append({"room": room["id"], "layout": layout["id"],
                            "repetition": repetition, "query_ids": queries.position_ids,
                            "xyz": result["xyz"], "initial_xyz": initial["xyz"], "errors_m": errors,
                            "status": result["status"],
                            "audit": {"initial": initial_audit, "refinement": lea_audit}})
        if source_only:
            metrics = {key: float(np.mean([c[key] for c in cases])) for key in ("ospa", "f1")}
            loss = metrics["ospa"]
        else:
            errors = np.array([c["error_m"] for c in cases])
            metrics = {"mean_m": float(errors.mean()), "rmse_m": float(np.sqrt(np.mean(errors**2))),
                       "median_m": float(np.median(errors)), "p90_m": float(np.quantile(errors, .9)),
                       "fallback_fraction": float(np.mean([c["fallback"] for c in cases]))}
            loss = metrics["mean_m"]
        result = {"loss": loss, "metrics": metrics, "seconds": time.perf_counter() - started,
                  "max_reconstruction_seconds": max(timings, default=0)}
        if retain:
            result["detail"] = {"sources": source_records, "queries": query_records,
                                "source_metrics": cases if source_only else []}
        return result


def run_compositions(data_root, run_root, base_root, *, methods=None):
    data_root, run_root, base_root = map(Path, (data_root, run_root, base_root))
    config = composition_config(read_json(base_root / "config.json"))
    selected_methods = methods is not None
    methods = list(VARIANTS) if methods is None else list(methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in VARIANTS for m in methods):
        raise ValueError("Choose distinct known composition methods")
    if selected_methods:
        config["name"] = run_root.name
        config["compositions"]["spaces"] = {m: config["compositions"]["spaces"][m] for m in methods}
    version = {"id": implementation_id(),
        "experiment": digest(Path(__file__).with_name("experiment.py").read_text()),
        "dataset": digest(read_json(data_root / "dataset.json")),
        "additional": extension_id(), "compositions": composition_id()}
    original = read_json(base_root / "implementation.json")
    if version["dataset"] != original["dataset"]:
        raise ValueError("Initializer study uses another dataset")
    # Selected MAP settings are immutable upstream inputs, not computations that
    # must be repeated whenever unrelated source methods or search code change.
    version["initializer_study"] = {"run": str(base_root), "implementation": original,
        "selected": {mode: read_json(base_root / "selected" / ("mpurge_map_" + mode + ".json"))["id"]
                     for mode in ("unified" if u else "ununified"
                                  for u in config["optimization"]["unification"])}}
    run_root.mkdir(parents=True, exist_ok=True)
    with file_lock(run_root / "run.lock"):
        for name, value in (("config.json", config), ("implementation.json", version)):
            path = run_root / name
            if path.exists() and read_json(path) != value:
                raise ValueError("Study changed; use a fresh run directory")
            write_json(path, value)
        evaluator = CompositionEvaluator(data_root, config, config["name"])
        if not evaluator.observations.available():
            raise ValueError("Native observations are not ready")
        seeds = Seeds(config["optimization"]["seed"])
        schedules = {m: source_schedule(space, config["compositions"]["source_trials"],
            seeds.seed("composition-source", m)) for m, space in config["compositions"]["spaces"].items()}
        lea_grid = configurations(config["optimization"]["smart_lea"])
        target = sum(len(schedules[m]) * (len(config["optimization"]["unification"])
                     if m.startswith("mpurge") else 1) for m in methods)
        target += len(methods) * len(config["optimization"]["unification"]) * (
            len(lea_grid) + config["compositions"]["joint_trials"])
        status = {"state": "selecting", "completed_trials": 0, "target_trials": target,
                  "selection_noise_m": 0., "current": None}
        def update(current):
            status.update(current=current, updated=time.time(), cache=evaluator.store.usage())
            write_json(run_root / "status.json", status)
        choices, results = {}, []

        def evaluate_choice(choice):
            status["state"] = "evaluating"
            for split, levels in (("calibration", [0.]),
                                  ("evaluation", config["observations"]["noise_levels"])):
                for level in levels:
                    identity = {k: choice[k] for k in ("method", "parameters", "unify", "frozen")}
                    identity.update(level=level, split=split)
                    key = digest(identity)
                    receipt = run_root / "evaluations" / (key + ".json")
                    outcome = run_root / "outcomes" / (key + ".npz")
                    if receipt.exists() and outcome.exists():
                        row = read_json(receipt)
                    else:
                        evaluated = evaluator.evaluate(**identity, retain=True)
                        detail = evaluated.pop("detail")
                        used = sum(p.stat().st_size for p in (run_root / "outcomes").glob("*.npz"))
                        if used + memory_bytes(detail) + 65536 > config["storage"]["outcomes_mib"] * 1024**2:
                            raise RuntimeError("Outcomes exceed storage budget")
                        write_npz(outcome, detail, metadata=identity)
                        row = {"id": key, "label": choice["label"], **identity, **evaluated,
                               "outcome": str(outcome.relative_to(run_root))}
                        write_json(receipt, row)
                    results.append(row)
                    write_json(run_root / "results.json", results)
                    update({"method": choice["label"], "noise": level, "split": split})

        def select(method, unify, schedule, frozen):
            status["state"] = "selecting"
            label = method + (("" if method.endswith("_naked") else "_finished")
                              if unify is None else "_unified" if unify else "_ununified")
            context = {"method": method, "unify": unify, "level": 0., "split": "calibration",
                       "frozen": frozen, "label": label}
            log, best = TrialLog(run_root / "trials", label, context), None
            for number, parameters in enumerate(schedule):
                identity = {"method": method, "parameters": parameters, "unify": unify,
                            "level": 0., "split": "calibration", "frozen": frozen}
                key = digest(identity)
                trial = log.get(key)
                if trial is None:
                    trial = {"id": key, "number": number, "label": label, **identity,
                             **evaluator.evaluate(**identity)}
                    log.append(trial)
                status["completed_trials"] += 1
                if best is None or trial["loss"] < best["loss"]:
                    best = trial
                update({"method": label, "trial": number + 1, "total": len(schedule),
                        "best_loss": best["loss"]})
            choices[(method, unify)] = best
            write_json(run_root / "selected" / (label + ".json"), best)
            print(f"Selected {label}: {best['loss']:.6f}", flush=True)
            evaluate_choice(best)

        try:
            # Finish all requested reconstruction comparisons before pipeline searches.
            for method in methods:
                for unify in config["optimization"]["unification"] if method.startswith("mpurge") else [None]:
                    select(method, unify, schedules[method], {})
            for method in methods:
                for unify in config["optimization"]["unification"]:
                    mode = "unified" if unify else "ununified"
                    map_choice = read_json(base_root / "selected" / ("mpurge_map_" + mode + ".json"))
                    source_mode = unify if method.startswith("mpurge") else None
                    frozen = {"source": choices[(method, source_mode)]["parameters"],
                              "map": map_choice["parameters"]}
                    select(method + "_lea", unify, lea_grid, frozen)
                    baseline = {**frozen, "lea": choices[(method + "_lea", unify)]["parameters"]}
                    schedule = joint_configurations({
                        "source": config["compositions"]["spaces"][method],
                        "map": config["optimization"]["mpurge_map"],
                        "lea": config["optimization"]["smart_lea"]},
                        config["compositions"]["joint_trials"],
                        seeds.seed("composition-joint", method, unify), baseline)
                    select(method + "_joint", unify, schedule, {})
            status["state"] = "complete"
            update(None)
            return results
        except Exception as error:
            status.update(state="failed", error=str(error))
            update(status["current"])
            raise

