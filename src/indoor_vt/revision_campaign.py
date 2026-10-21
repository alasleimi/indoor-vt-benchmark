"""Preset coverage and candidate-level scheduling units for versioned campaigns."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .calibration import Search, SearchPending
from .evaluation import Evaluator, gzip_json
from .localization import LEAConfig
from .provenance import digest, file_hash, read_json, write_json
from .registry import INITIAL_METHODS, VT_METHODS, initial_default, reconstruction_default
from .search_spaces import initial_space, vt_space, lea_space

MERGERS = ("greedy_median", "greedy_mean", "agglomerative", "dbscan")


def directory(root, namespace, revision):
    return Path(root) / "runs" / namespace / "revisions" / revision


def pairs():
    return [("mpurge_map", vt) for vt in VT_METHODS] + [
        (initial, "mpurge_local") for initial in INITIAL_METHODS if initial != "mpurge_map"
    ]


def preset_initial(method):
    # The last explicit published candidate is from the MAP paper where available.
    return deepcopy(initial_space(method)[1][-1]["parameters"])


def pipeline_config(initial, vt, *, preset=False, merger="greedy_median"):
    parameters = reconstruction_default(vt)
    if vt.startswith("mpurge"):
        parameters["merger"] = merger
    return {
        "initial": preset_initial(initial) if preset else initial_default(initial),
        "vt_method": vt,
        "vt": parameters,
        "lea": asdict(LEAConfig()),
    }


def build_primary_jobs():
    jobs = []
    for track in ("receiver", "direct"):
        for method in INITIAL_METHODS:
            jobs.append(
                {
                    "id": f"preset/{track}/coarse/{method}",
                    "phase": 0,
                    "order": 0,
                    "kind": "preset",
                    "stage": "coarse",
                    "method": method,
                    "track": track,
                    "parameters": preset_initial(method),
                    "dependencies": [],
                }
            )
        for method in VT_METHODS:
            for merger in MERGERS if method.startswith("mpurge") else ("greedy_median",):
                config = reconstruction_default(method)
                if method.startswith("mpurge"):
                    config["merger"] = merger
                jobs.append(
                    {
                        "id": f"preset/{track}/vt/{method}/{merger}",
                        "phase": 0,
                        "order": 0 if merger == "greedy_median" else 2,
                        "kind": "preset",
                        "stage": "vt",
                        "method": method,
                        "track": track,
                        "parameters": config,
                        "dependencies": [],
                    }
                )
        for initial, vt in pairs():
            name = initial + "__" + vt
            jobs.append(
                {
                    "id": f"preset/{track}/pipeline/{name}",
                    "phase": 0,
                    "order": 1,
                    "kind": "preset",
                    "stage": "pipeline",
                    "method": name,
                    "track": track,
                    "parameters": pipeline_config(initial, vt, preset=True),
                    "dependencies": [
                        f"preset/{track}/coarse/{initial}",
                        f"preset/{track}/vt/{vt}/greedy_median",
                    ],
                }
            )
        for stage, methods in (("coarse", INITIAL_METHODS), ("vt", VT_METHODS)):
            for method in methods:
                jobs.append(
                    {
                        "id": f"search/{track}/component/{stage}_{method}",
                        "phase": 1,
                        "kind": "search",
                        "stage": stage,
                        "method": method,
                        "track": track,
                        "regime": "component",
                        "dependencies": [],
                    }
                )
        for initial, vt in pairs():
            name = initial + "__" + vt
            upstream = [
                f"search/{track}/component/coarse_{initial}",
                f"search/{track}/component/vt_{vt}",
            ]
            jobs.append(
                {
                    "id": f"search/{track}/component/lea_{name}",
                    "phase": 1,
                    "kind": "search",
                    "stage": "lea",
                    "method": name,
                    "track": track,
                    "regime": "component",
                    "dependencies": upstream,
                }
            )
            jobs.append(
                {
                    "id": f"search/{track}/end_to_end/pipeline_{name}",
                    "phase": 1,
                    "kind": "search",
                    "stage": "pipeline",
                    "method": name,
                    "track": track,
                    "regime": "end_to_end",
                    "dependencies": [],
                }
            )
            for regime in ("component", "end_to_end"):
                dependencies = (
                    upstream + [f"search/{track}/component/lea_{name}"]
                    if regime == "component"
                    else [f"search/{track}/end_to_end/pipeline_{name}"]
                )
                jobs.append(
                    {
                        "id": f"freeze/{track}/{regime}/{name}",
                        "phase": 1,
                        "kind": "freeze",
                        "method": name,
                        "track": track,
                        "regime": regime,
                        "dependencies": dependencies,
                    }
                )
                jobs.append(
                    {
                        "id": f"evaluate/{track}/{regime}/{name}",
                        "phase": 3,
                        "kind": "evaluate",
                        "method": name,
                        "track": track,
                        "regime": regime,
                        "dependencies": [f"freeze/{track}/{regime}/{name}"],
                    }
                )
    return jobs


def search_path(base, track, regime, name):
    return base / "calibration" / track / regime / (name + ".json")


def checked_search(path):
    record = read_json(path)
    check = dict(record)
    claimed = check.pop("freeze_hash")
    if not record["complete"] or digest(check) != claimed:
        raise RuntimeError("Upstream search is not frozen")
    return record


def measured(evaluator, base, stage, method, parameters):
    """One objective result is shared by presets and all compatible searches."""
    identity = {
        "stage": stage,
        "method": method,
        "parameters": parameters,
        "code_hash": evaluator.provenance["source_hash"],
        "track": evaluator.track,
        "data_hash": digest(
            [
                evaluator.design["design_hash"],
                evaluator.observation_hash,
                evaluator.truth["truth_hash"],
            ]
        ),
    }
    key = digest(identity)
    path = base / "objectives" / evaluator.track / (key + ".json.gz")
    receipt = path.with_suffix(".summary.json")
    if receipt.exists():
        record = read_json(receipt)
        path = Path(record.get("artifact", path))
        if record["identity"] != identity or file_hash(path) != record["sha256"]:
            raise RuntimeError("Objective artifact changed")
        return record["score"], {
            "artifact": str(path),
            "sha256": record["sha256"],
            "case_count": record["case_count"],
        }
    if stage == "coarse":
        score, audit = evaluator.localization_objective(parameters)
    elif stage == "vt":
        score, audit = evaluator.vt_objective(method, parameters)
    else:
        score, audit = evaluator.localization_objective(
            parameters["initial"],
            vt_method=parameters["vt_method"],
            vt_parameters=parameters["vt"],
            lea_parameters=parameters["lea"],
        )
    gzip_json(path, {"identity": identity, "score": score, "audit": audit})
    write_json(
        receipt,
        {
            "identity": identity,
            "score": score,
            "case_count": audit["case_count"],
            "sha256": file_hash(path),
            "artifact": str(path),
        },
    )
    return score, {
        "artifact": str(path),
        "sha256": file_hash(path),
        "case_count": audit["case_count"],
    }


def vt_spec(method):
    default = reconstruction_default(method)
    space, published = vt_space(method)
    if method.startswith("mpurge"):
        for merger in MERGERS[1:]:
            published.append(
                {
                    "parameters": {**default, "merger": merger},
                    "phase": "preset",
                    "citation": "Predeclared merger ablation with disclosed defaults; not a published setting",
                }
            )
    return default, space, published


def run_job(root, namespace, revision, job, evaluators):
    root = Path(root)
    base = directory(root, namespace, revision)
    track, kind = job["track"], job["kind"]
    if track not in evaluators:
        evaluators[track] = Evaluator(root, namespace, "calibration", track)
    e = evaluators[track]
    if kind == "preset":
        score, audit = measured(e, base, job["stage"], job["method"], job["parameters"])
        path = base / "presets" / (job["id"].replace("/", "__") + ".json")
        write_json(
            path,
            {
                "job": job,
                "score": score,
                "audit": audit,
                "selection_split": "calibration",
                "code_hash": e.provenance["source_hash"],
                "status": "uncalibrated preset: published values plus disclosed adapter defaults",
            },
        )
        return {"state": "complete", "artifact": str(path), "score": score}
    regime, name = job["regime"], job["method"]
    if kind == "evaluate":
        from .campaign import evaluate_frozen

        path = evaluate_frozen(root, base / "frozen" / track / regime / (name + ".json"))
        return {"state": "complete", "artifact": str(path)}
    if kind == "freeze":
        initial, vt = name.split("__")
        if regime == "component":
            records = [
                checked_search(search_path(base, track, regime, stage + "_" + method))
                for stage, method in (("coarse", initial), ("vt", vt), ("lea", name))
            ]
            config = {
                "initial": records[0]["selected"]["parameters"],
                "vt_method": vt,
                "vt": records[1]["selected"]["parameters"],
                "lea": records[2]["selected"]["parameters"],
            }
        else:
            records = [checked_search(search_path(base, track, regime, "pipeline_" + name))]
            config = {**records[0]["selected"]["parameters"], "vt_method": vt}
        freeze = {
            "schema": "indoor-vt.frozen_pipeline.v2",
            "namespace": namespace,
            "revision": revision,
            "track": track,
            "regime": regime,
            "name": name,
            "design_hash": e.design["design_hash"],
            "truth_hash": e.truth["truth_hash"],
            "observation_hash": e.observation_hash,
            "code_hash": e.provenance["source_hash"],
            "calibration_searches": [r["freeze_hash"] for r in records],
            "selection_split": "calibration",
            "config": config,
        }
        freeze["freeze_hash"] = digest(freeze)
        path = base / "frozen" / track / regime / (name + ".json")
        write_json(path, freeze)
        return {"state": "complete", "artifact": str(path)}

    stage = job["stage"]
    if stage == "coarse":
        default, (space, published) = initial_default(name), initial_space(name)

        def objective(p):
            return measured(e, base, "coarse", name, p)
    elif stage == "vt":
        default, space, published = vt_spec(name)

        def objective(p):
            return measured(e, base, "vt", name, p)
    else:
        initial, vt = name.split("__")
        if stage == "lea":
            coarse = checked_search(search_path(base, track, regime, "coarse_" + initial))
            sources = checked_search(search_path(base, track, regime, "vt_" + vt))
            default, (space, published) = asdict(LEAConfig()), lea_space()

            def objective(p):
                return measured(
                    e,
                    base,
                    "pipeline",
                    name,
                    {
                        "initial": coarse["selected"]["parameters"],
                        "vt": sources["selected"]["parameters"],
                        "vt_method": vt,
                        "lea": p,
                    },
                )
        else:
            default = pipeline_config(initial, vt)
            default.pop("vt_method")
            space, published = {}, []
            for prefix, (component_space, presets) in zip(
                ("initial", "vt", "lea"), (initial_space(initial), vt_spec(vt)[1:], lea_space())
            ):
                space.update({prefix + "." + key: value for key, value in component_space.items()})
                for preset in presets:
                    candidate = deepcopy(default)
                    candidate[prefix] = preset["parameters"]
                    published.append({**preset, "parameters": candidate})

            def objective(p):
                return measured(e, base, "pipeline", name, {**p, "vt_method": vt})

    search_name = stage + "_" + name
    path = search_path(base, track, regime, search_name)
    import_path = base / "imports" / track / regime / (search_name + ".json")
    history = read_json(import_path)["trials"] if import_path.exists() else None
    runner = Search(
        split="calibration",
        track=track,
        regime=regime,
        seed=e.seeds.seed("calibration", track, regime, search_name),
        budget=read_json(root / "configs/calibration.json")["budget"],
        dataset_hash=digest([e.design["design_hash"], e.observation_hash, e.truth["truth_hash"]]),
        code_hash=e.provenance["source_hash"],
    )
    try:
        result = runner.run(
            objective, default, space, published, path, max_evaluations=1, history=history
        )
        return {"state": "complete", "artifact": str(path), "trials": len(result["trials"])}
    except SearchPending:
        return {"state": "pending", "artifact": str(path), "trials": len(read_json(path)["trials"])}
