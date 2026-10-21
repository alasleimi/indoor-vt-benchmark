"""Calibration-only broad search, boundary expansion, refinement and freeze receipts."""

from copy import deepcopy
import math
from pathlib import Path
import numpy as np
from .provenance import canonical, digest, read_json, write_json


def uniform_mean(scores):
    values = np.asarray(scores, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("All declared cases must yield finite scores; do not omit failures")
    return float(values.mean())


def set_nested(config, key, value):
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def get_nested(config, key):
    for part in key.split("."):
        config = config[part]
    return config


def random_candidates(default, space, rng, budget):
    seen, attempts = set(), 0
    count = min(budget, math.prod(len(spec["values"]) for spec in space.values()))
    while len(seen) < count and attempts < max(100, count * 100):
        attempts += 1
        candidate = deepcopy(default)
        for key, spec in space.items():
            values = spec["values"]
            set_nested(candidate, key, values[int(rng.integers(len(values)))])
        identity = digest(candidate)
        if identity not in seen:
            seen.add(identity)
            yield candidate


def expanded_space(space, best):
    expanded, changes = deepcopy(space), []
    for key, spec in expanded.items():
        if spec.get("categorical") or not spec.get("expand", False):
            continue
        values = sorted(spec["values"])
        selected = get_nested(best, key)
        lower, upper = spec.get("bounds", [None, None])
        new = None
        if selected == values[0] and (lower is None or values[0] > lower):
            new = values[0] / 2
            if lower is not None:
                new = max(lower, new)
        elif selected == values[-1] and (upper is None or values[-1] < upper):
            new = values[-1] * 2
            if upper is not None:
                new = min(upper, new)
        if new is not None:
            if spec.get("integer"):
                new = int(round(new))
            if new not in values:
                spec["values"] = sorted([*values, new])
                changes.append({"parameter": key, "added": new})
    return expanded, changes


def refinement_space(space, best):
    refined = deepcopy(space)
    for key, spec in refined.items():
        if spec.get("categorical"):
            continue
        values = sorted(spec["values"])
        selected = get_nested(best, key)
        index = min(range(len(values)), key=lambda i: abs(values[i] - selected))
        neighbors = values[max(0, index - 1) : index + 2]
        values = [selected, *[(x + selected) / 2 for x in neighbors]]
        if spec.get("integer"):
            values = [int(round(x)) for x in values]
        spec["values"] = sorted(set(values))
    return refined


class Search:
    def __init__(self, *, split, track, regime, seed, budget, dataset_hash, code_hash):
        if split != "calibration":
            raise ValueError("Hyperparameter selection is forbidden on test rooms")
        if track not in ("direct", "receiver") or regime not in ("component", "end_to_end"):
            raise ValueError("Unknown evaluation track/regime")
        self.metadata = {
            "split": split,
            "track": track,
            "regime": regime,
            "seed": int(seed),
            "budget": budget,
            "dataset_hash": dataset_hash,
            "code_hash": code_hash,
        }

    def run(self, objective, default, space, published, output):
        output = Path(output)
        specification = {
            **self.metadata,
            "default": default,
            "space": space,
            "published": published,
        }
        identity = digest(specification)
        trials, expansions, cached = [], [], {}
        if output.exists():
            old = read_json(output)
            if old["search_hash"] != identity:
                raise RuntimeError("Search resume does not match the frozen search specification")
            if old.get("complete"):
                check = dict(old)
                claimed = check.pop("freeze_hash")
                if digest(check) != claimed:
                    raise RuntimeError("Calibration freeze checksum mismatch")
                return old
            cached = {digest(t["parameters"]): t for t in old["trials"]}
        by_hash = {}
        rng = np.random.Generator(np.random.PCG64(self.metadata["seed"]))

        def evaluate(candidate, phase, citation=None):
            key = digest(candidate)
            if key in by_hash:
                return
            if key in cached:
                score, audit = cached[key]["score"], cached[key]["audit"]
            else:
                score, audit = objective(candidate)
            if not np.isfinite(score):
                raise RuntimeError("Nonfinite objective must be diagnosed, not dropped")
            trial = {
                "parameters": candidate,
                "score": float(score),
                "phase": phase,
                "citation": citation,
                "audit": audit,
            }
            by_hash[key] = trial
            trials.append(trial)
            write_json(
                output,
                {
                    "search_hash": identity,
                    "specification": specification,
                    "trials": trials,
                    "expansions": expansions,
                    "complete": False,
                },
            )

        def best():
            return min(trials, key=lambda t: (t["score"], canonical(t["parameters"])))

        evaluate(default, "default")
        for candidate in published:
            evaluate(candidate["parameters"], "published", candidate["citation"])
        for candidate in random_candidates(default, space, rng, self.metadata["budget"]["broad"]):
            evaluate(candidate, "broad")
        current = deepcopy(space)
        for i in range(self.metadata["budget"]["maximum_expansions"]):
            current, changes = expanded_space(current, best()["parameters"])
            if not changes:
                break
            expansions.append({"round": i, "changes": changes})
            # Explicitly visit each newly exposed boundary as well as random combinations.
            for change in changes:
                candidate = deepcopy(best()["parameters"])
                set_nested(candidate, change["parameter"], change["added"])
                evaluate(candidate, f"boundary_{i}")
            for candidate in random_candidates(
                default, current, rng, self.metadata["budget"]["expansion"]
            ):
                evaluate(candidate, f"expansion_{i}")
        local = refinement_space(current, best()["parameters"])
        for candidate in random_candidates(
            best()["parameters"], local, rng, self.metadata["budget"]["refine"]
        ):
            evaluate(candidate, "refinement")
        winner = best()
        result = {
            "search_hash": identity,
            "specification": specification,
            "trials": trials,
            "expansions": expansions,
            "complete": True,
            "selected": winner,
            "remaining_boundary": expanded_space(current, winner["parameters"])[1],
        }
        result["freeze_hash"] = digest(result)
        write_json(output, result)
        return result


def promotion(candidate, baseline):
    return bool(baseline > 0 and candidate <= 0.95 * baseline)
