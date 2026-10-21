"""Reproducible two-regime orchestration; labels remain in the evaluator."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from .calibration import Search
from .evaluation import Evaluator, gzip_json
from .localization import LEAConfig
from .provenance import digest, file_hash, read_json, write_json
from .registry import initial_default, reconstruction_default
from .search_spaces import initial_space, vt_space, lea_space


def calibrate(root, namespace, track, regime, initial, vt_method):
    root = Path(root)
    evaluator = Evaluator(root, namespace, "calibration", track)
    specification = read_json(root / "configs/calibration.json")
    out = root / "runs" / namespace / "calibration" / track / regime
    code_hash = evaluator.provenance["source_hash"]

    def run_search(name, objective, default, space, published):
        def recorded(parameters):
            score, audit = objective(parameters)
            artifact = out / "trials" / name / (digest(parameters) + ".json.gz")
            gzip_json(artifact, {"parameters": parameters, "score": score, "audit": audit})
            return score, {
                "artifact": str(artifact.relative_to(root)),
                "sha256": file_hash(artifact),
                "case_count": audit["case_count"],
            }

        search = Search(
            split="calibration",
            track=track,
            regime=regime,
            seed=evaluator.seeds.seed("calibration", track, regime, name),
            budget=specification["budget"],
            dataset_hash=digest(
                [
                    evaluator.design["design_hash"],
                    evaluator.observation_hash,
                    evaluator.truth["truth_hash"],
                ]
            ),
            code_hash=code_hash,
        )
        return search.run(recorded, default, space, published, out / (name + ".json"))

    initial_cfg, vt_cfg = initial_default(initial), reconstruction_default(vt_method)
    lea_cfg = asdict(LEAConfig())
    pipeline_name = f"{initial}__{vt_method}"
    if regime == "component":
        # These identities do not contain the other upstream method's name.
        space, published = initial_space(initial)
        coarse_search = run_search(
            "coarse_" + initial,
            lambda p: evaluator.localization_objective(p),
            initial_cfg,
            space,
            published,
        )
        initial_cfg = coarse_search["selected"]["parameters"]
        space, published = vt_space(vt_method)
        vt_search = run_search(
            "vt_" + vt_method,
            lambda p: evaluator.vt_objective(vt_method, p),
            vt_cfg,
            space,
            published,
        )
        vt_cfg = vt_search["selected"]["parameters"]
        space, published = lea_space()
        final_search = run_search(
            "lea_" + pipeline_name,
            lambda p: evaluator.localization_objective(
                initial_cfg, vt_method=vt_method, vt_parameters=vt_cfg, lea_parameters=p
            ),
            lea_cfg,
            space,
            published,
        )
        lea_cfg = final_search["selected"]["parameters"]
        search_hashes = [
            coarse_search["freeze_hash"],
            vt_search["freeze_hash"],
            final_search["freeze_hash"],
        ]
    elif regime == "end_to_end":
        default = {"initial": initial_cfg, "vt": vt_cfg, "lea": lea_cfg}
        space, published = (
            {},
            [
                {
                    "parameters": deepcopy(default),
                    "citation": "Published component defaults with declared adapters",
                }
            ],
        )
        for prefix, (component_space, component_published) in zip(
            ("initial", "vt", "lea"), (initial_space(initial), vt_space(vt_method), lea_space())
        ):
            space.update({prefix + "." + key: value for key, value in component_space.items()})
            for item in component_published:
                candidate = deepcopy(default)
                candidate[prefix] = item["parameters"]
                published.append({"parameters": candidate, "citation": item["citation"]})
        final_search = run_search(
            "pipeline_" + pipeline_name,
            lambda p: evaluator.localization_objective(
                p["initial"], vt_method=vt_method, vt_parameters=p["vt"], lea_parameters=p["lea"]
            ),
            default,
            space,
            published,
        )
        selected = final_search["selected"]["parameters"]
        initial_cfg, vt_cfg, lea_cfg = selected["initial"], selected["vt"], selected["lea"]
        search_hashes = [final_search["freeze_hash"]]
    else:
        raise ValueError(regime)
    frozen = {
        "schema": "indoor-vt.frozen_pipeline.v1",
        "namespace": namespace,
        "track": track,
        "regime": regime,
        "name": pipeline_name,
        "design_hash": evaluator.design["design_hash"],
        "truth_hash": evaluator.truth["truth_hash"],
        "observation_hash": evaluator.observation_hash,
        "code_hash": code_hash,
        "calibration_searches": search_hashes,
        "selection_split": "calibration",
        "config": {"initial": initial_cfg, "vt_method": vt_method, "vt": vt_cfg, "lea": lea_cfg},
    }
    frozen["freeze_hash"] = digest(frozen)
    destination = root / "configs/frozen" / namespace / track / regime / (pipeline_name + ".json")
    if destination.exists() and read_json(destination) != frozen:
        raise RuntimeError("Refusing to replace an existing frozen pipeline")
    write_json(destination, frozen)
    return destination


def evaluate_frozen(root, frozen_path):
    root = Path(root)
    frozen = read_json(Path(frozen_path))
    check = dict(frozen)
    claimed = check.pop("freeze_hash")
    if digest(check) != claimed or frozen["selection_split"] != "calibration":
        raise RuntimeError("Invalid calibration freeze receipt")
    evaluator = Evaluator(root, frozen["namespace"], "test", frozen["track"])
    if (
        frozen["code_hash"] != evaluator.provenance["source_hash"]
        or frozen["design_hash"] != evaluator.design["design_hash"]
        or frozen["truth_hash"] != evaluator.truth["truth_hash"]
        or frozen["observation_hash"] != evaluator.observation_hash
    ):
        raise RuntimeError(
            "Test evaluation requires exactly the frozen code, dataset and reference bag"
        )
    cfg = frozen["config"]
    score, audit = evaluator.localization_objective(
        cfg["initial"],
        vt_method=cfg["vt_method"],
        vt_parameters=cfg["vt"],
        lea_parameters=cfg["lea"],
    )
    vt_score, vt_audit = evaluator.vt_objective(cfg["vt_method"], cfg["vt"])
    result = {
        "freeze": frozen,
        "final_mean_error_m": score,
        "vt_ospa": vt_score,
        "localization": audit,
        "vt": vt_audit,
        "provenance": evaluator.provenance,
    }
    run_base = root / "runs" / frozen["namespace"]
    if frozen.get("revision"):
        run_base = run_base / "revisions" / frozen["revision"]
    destination = (
        run_base / "test" / frozen["track"] / frozen["regime"] / (frozen["name"] + ".json.gz")
    )
    gzip_json(destination, result)
    write_json(
        destination.with_suffix(".summary.json"),
        {
            "final_mean_error_m": score,
            "vt_ospa": vt_score,
            "query_cases": audit["case_count"],
            "fallback_count": audit["fallback_count"],
            "freeze_hash": claimed,
            "result_sha256": file_hash(destination),
            "result": str(destination),
        },
    )
    return destination
