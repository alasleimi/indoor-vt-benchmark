"""A fixed-compute true-count baseline; truth never enters numerical fitting."""
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from .experiment import Evaluator
from .artifacts import implementation_id, write_npz
from .explorer import summarize_sources
from ..acquisition.records import Seeds, digest, read_json, write_json
from ..science.range_rmse import RangeRMSE, range_rmse


METHOD = "range_rmse_true_count"
STUDY = "rectangle-range-rmse-true-count-v1"


class RangeRMSEEvaluator(Evaluator):
    def evaluate(self, method, parameters, *, unify, level, split, frozen=None, retain=False):
        if method != METHOD or unify is not None:
            raise ValueError("Unknown true-count baseline configuration")
        started = time.perf_counter()
        records, timings = [], []
        truths = {r["id"]: self.observations.truth(r) for r in self.observations.rooms}
        repetitions = self.config["observations"][split + "_repetitions"]
        if level == 0:
            repetitions = repetitions[:1]
        for room in self.observations.rooms:
            # Only the integer count crosses the evaluator/solver boundary.
            counts = {ap: len(points) for ap, points in truths[room["id"]].by_ap.items()}
            for repetition in repetitions:
                survey = self.observations.survey(room, level, repetition)
                by_ap, audit = {}, {}
                for ap, count in counts.items():
                    settings = {**parameters, "seed": Seeds(parameters["seed"]).seed(room["id"], ap)}
                    source, evidence = range_rmse(settings).fit(survey.xyz, survey.channels[ap], count)
                    by_ap[ap], audit[ap] = source, evidence
                    timings.append(evidence["seconds"])
                records.append({"room": room["id"], "repetition": repetition,
                                "by_ap": by_ap, "audit": audit})
        overall, by_ap = summarize_sources(records, truths)
        result = {"loss": float(np.mean([a["range_rmse_m"] for r in records for a in r["audit"].values()])),
                  "metrics": overall, "source_metrics_by_ap": by_ap,
                  "seconds": time.perf_counter() - started, "source_seconds": timings}
        if retain:
            result["detail"] = {"sources": records, "queries": []}
        return result


def run_range_rmse(data_root, base_run, run_root):
    data_root, base_run, run_root = map(Path, (data_root, base_run, run_root))
    config = read_json(base_run / "config.json")
    config = {**config, "name": STUDY, "engine": "range_rmse"}
    parameters = asdict(RangeRMSE())
    config["range_rmse"] = parameters
    version = {"id": implementation_id(),
               "experiment": digest(Path(__file__).read_text()),
               "solver": digest(Path(__file__).parents[1].joinpath("science/range_rmse.py").read_text()),
               "dataset": digest(read_json(data_root / "dataset.json"))}
    protocol = {"name": "Range RMSE (cheating true count)",
                "allowed_truth": "Number of sources per AP only; includes the physical AP/direct source",
                "prohibited_truth": "Coordinates, identities, reflection order, clean ranges, and room walls",
                "objective": "sqrt(sum((observed_range - distance(receiver, assigned_source))^2) / observed_peak_count)",
                "assignment": "Hungarian squared range residual, one-to-one within each fingerprint",
                "missing": "Benchmark removes negative noisy ranges; unmatched source slots receive no residual",
                "selection": "Lowest observed-range SSE among 16 starts; no OSPA/F1 tuning or ground-truth initialization",
                "initialization": "One equal-rank algebraic start when all fingerprints are complete; remaining starts use random directions on measured-range shells centered at a sampled receiver",
                "stopping": "150 alternating iterations or relative SSE improvement <= 1e-8; inner LM at most 60 evaluations",
                "bounds": "Unconstrained 3D source coordinates; no direct-AP anchor",
                "limitations": "Local numerical optimizer, not a certified global optimum; fixed starts and budget at every noise level",
                "observations": config["observations"], "parameters": parameters}
    run_root.mkdir(parents=True, exist_ok=True)
    for name, value in (("config", config), ("implementation", version), ("protocol", protocol)):
        path = run_root / (name + ".json")
        if path.exists() and read_json(path) != value:
            raise ValueError(f"Existing {name} differs; preserve that run and use another output directory")
        write_json(path, value)
    evaluator = RangeRMSEEvaluator(data_root, config, STUDY)
    levels = config["observations"]["noise_levels"]
    results = []
    started = time.perf_counter()
    for index, level in enumerate(levels):
        identity = {"method": METHOD, "parameters": parameters, "level": level,
                    "split": "evaluation", "solver": version["solver"],
                    "dataset": version["dataset"], "survey": config["survey"],
                    "seed": config["seed"], "observations": config["observations"]}
        key = digest(identity)
        receipt = run_root / "evaluations" / (key + ".json")
        outcome = run_root / "outcomes" / (key + ".npz")
        write_json(run_root / "status.json", {"state": "running", "phase": "range RMSE evaluation",
                   "level": level, "completed_conditions": index, "target_conditions": len(levels),
                   "completed_trials": 0, "target_trials": 0})
        if receipt.exists() and outcome.exists():
            row = read_json(receipt)
        else:
            value = evaluator.evaluate(METHOD, parameters, unify=None, level=level,
                                       split="evaluation", retain=True)
            detail = value.pop("detail")
            write_npz(outcome, detail, metadata=identity)
            row = {"id": key, "label": METHOD, "method": METHOD, "parameters": parameters,
                   "unify": None, "level": level, "split": "evaluation", "frozen": {},
                   "number": index, "outcome": str(outcome.relative_to(run_root)), **value}
            write_json(receipt, row)
        results.append(row)
        write_json(run_root / "results.json", results)
        print({"level": level, "seconds": row["seconds"], **row["source_metrics_by_ap"]["A+B"]}, flush=True)
    write_json(run_root / "status.json", {"state": "complete", "completed_conditions": len(levels),
               "target_conditions": len(levels), "completed_trials": 0, "target_trials": 0,
               "seconds": time.perf_counter() - started})
    return results
