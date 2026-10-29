"""Additional finished reconstructors: clean selection, frozen stages and joint tuning."""
from pathlib import Path
import time
import numpy as np

from .additional import AdditionalExecution, extension_id
from .artifacts import file_lock, implementation_id, write_npz, memory_bytes
from .experiment import Evaluator, configurations, joint_configurations
from .localization import LocalizerExecution, LEAExecution
from .presets import mpurge_map, smart_lea
from .trials import TrialLog
from ..acquisition.records import read_json, write_json, digest, Seeds
from ..metrics import ap_source_metrics


def robust_finisher_grid():
    return {"smooth_weight": [0., .35, .7],
            "median_penalty_weight": [0., .1, .2],
            "median_clip_factor": [2., 4., 8.]}


def additional_config(base):
    import copy
    config = copy.deepcopy(base)
    config["name"] = "rectangle-additional-selection-v1"
    config["engine"] = "additional"
    config["additional"] = {
        "zayets": {"scope_radius_m": [.75, 1.0], "gap_m": [.4, .6, .8],
                   "group_cost": [.25, .5, 1.], "tolerance_m": [.1, .18, .3],
                   "support_fraction": [.04, .08, .16], "duplicate_m": [.06, .12, .24]},
        "ransac": {"trials": [10000, 30000], "tolerance_m": [.1, .18, .3],
                   "support_fraction": [.04, .08, .16], "duplicate_m": [.06, .12, .24],
                   **robust_finisher_grid()},
        "joint_trials": 512, "proposal_seed": 9173}
    return config


class AdditionalEvaluator(Evaluator):
    def evaluate(self, method, parameters, *, unify, level, split, frozen=None, retain=False):
        started = time.perf_counter()
        family = method.split("_")[0]
        if family not in ("zayets", "ransac"):
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
                sources = AdditionalExecution(family, source_p, self.store,
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


def run_additional(data_root, run_root, base_root):
    data_root, run_root, base_root = map(Path, (data_root, run_root, base_root))
    base = read_json(base_root / "config.json")
    config = additional_config(base)
    base_version = read_json(base_root / "implementation.json")
    version = {"id": implementation_id(),
               "experiment": digest(Path(__file__).with_name("experiment.py").read_text()),
               "dataset": digest(read_json(data_root / "dataset.json")), "additional": extension_id()}
    if any(base_version[k] != version[k] for k in ("id", "experiment", "dataset")):
        raise ValueError("The initializer study must match this scientific implementation and dataset")
    run_root.mkdir(parents=True, exist_ok=True)
    with file_lock(run_root / "run.lock"):
        for name, value in (("config.json", config), ("implementation.json", version)):
            path = run_root / name
            if path.exists() and read_json(path) != value:
                raise ValueError("Additional study changed; use a fresh run directory")
            write_json(path, value)
        evaluator = AdditionalEvaluator(data_root, config, config["name"])
        if not evaluator.observations.available():
            raise ValueError("Required native observations are not ready")
        grids = {family: configurations(config["additional"][family]) for family in ("zayets", "ransac")}
        lea_grid = configurations(config["optimization"]["smart_lea"])
        target = sum(map(len, grids.values())) + 4 * len(lea_grid) + 4 * config["additional"]["joint_trials"]
        status = {"state": "selecting", "completed_trials": 0, "target_trials": target,
                  "selection_noise_m": 0., "current": None}
        def update(current):
            status.update(current=current, updated=time.time(), cache=evaluator.store.usage())
            write_json(run_root / "status.json", status)
        choices = {}
        def select(method, unify, schedule, frozen):
            label = method + ("_finished" if unify is None else "_unified" if unify else "_ununified")
            context = {"method": method, "unify": unify, "level": 0., "split": "calibration",
                       "frozen": frozen, "label": label}
            log = TrialLog(run_root / "trials", label, context)
            best = None
            for number, parameters in enumerate(schedule):
                identity = {"method": method, "parameters": parameters, "unify": unify,
                            "level": 0., "split": "calibration", "frozen": frozen}
                key = digest(identity)
                trial = log.get(key)
                if trial is None:
                    value = evaluator.evaluate(method, parameters, unify=unify, level=0.,
                                               split="calibration", frozen=frozen)
                    trial = {"id": key, "number": number, "label": label, **identity, **value}
                    log.append(trial)
                status["completed_trials"] += 1
                if best is None or trial["loss"] < best["loss"]:
                    best = trial
                update({"method": label, "trial": number+1, "total": len(schedule),
                        "best_loss": best["loss"]})
            choices[(method, unify)] = best
            write_json(run_root / "selected" / (label + ".json"), best)
            print(f"Selected {label}: {best['loss']:.6f}", flush=True)
        try:
            for family in ("zayets", "ransac"):
                select(family, None, grids[family], {})
                for unify in config["optimization"]["unification"]:
                    mode = "unified" if unify else "ununified"
                    map_choice = read_json(base_root / "selected" / ("mpurge_map_" + mode + ".json"))
                    frozen = {"source": choices[(family, None)]["parameters"],
                              "map": map_choice["parameters"]}
                    select(family + "_lea", unify, lea_grid, frozen)
                    baseline = {**frozen, "lea": choices[(family + "_lea", unify)]["parameters"]}
                    schedule = joint_configurations({"source": config["additional"][family],
                        "map": config["optimization"]["mpurge_map"],
                        "lea": config["optimization"]["smart_lea"]},
                        config["additional"]["joint_trials"],
                        Seeds(config["optimization"]["seed"]).seed("joint", family, unify), baseline)
                    select(family + "_joint", unify, schedule, {})
            status["state"] = "evaluating"
            results = []
            for (method, unify), choice in choices.items():
                for split, levels in (("calibration", [0.]),
                                      ("evaluation", config["observations"]["noise_levels"])):
                    for level in levels:
                        identity = {k: choice[k] for k in ("method", "parameters", "unify", "frozen")}
                        identity.update(level=level, split=split)
                        key = digest(identity)
                        receipt, outcome = run_root / "evaluations" / (key+".json"), run_root / "outcomes" / (key+".npz")
                        if receipt.exists() and outcome.exists():
                            row = read_json(receipt)
                        else:
                            evaluated = evaluator.evaluate(method, choice["parameters"], unify=unify,
                                level=level, split=split, frozen=choice["frozen"], retain=True)
                            detail = evaluated.pop("detail")
                            used = sum(p.stat().st_size for p in (run_root / "outcomes").glob("*.npz"))
                            if used + memory_bytes(detail) + 65536 > config["storage"]["outcomes_mib"] * 1024**2:
                                raise RuntimeError("Additional outcomes exceed their storage budget")
                            write_npz(outcome, detail, metadata=identity)
                            row = {"id": key, "label": choice["label"], **identity, **evaluated,
                                   "outcome": str(outcome.relative_to(run_root))}
                            write_json(receipt, row)
                        results.append(row)
                        write_json(run_root / "results.json", results)
                        update({"method": choice["label"], "noise": level, "split": split})
            status["state"] = "complete"
            update(None)
            return results
        except Exception as error:
            status.update(state="failed", error=str(error))
            update(status["current"])
            raise
