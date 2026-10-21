"""Link compatible completed work into a revision without changing historical artifacts."""

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

from indoor_vt.evaluation import Evaluator, gzip_json
from indoor_vt.provenance import code_provenance, digest, file_hash, read_json, write_json
from indoor_vt.registry import reconstruction_default
from indoor_vt.revision_campaign import directory


def normalized(stage, method, parameters):
    result = deepcopy(parameters)
    if stage == "vt":
        defaults = reconstruction_default(method)
        for key in ("merger", "dbscan_min_samples"):
            if key in defaults:
                result.setdefault(key, defaults[key])
    elif stage == "pipeline":
        result["vt"] = normalized("vt", result["vt_method"], result["vt"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--revision", default="v2")
    args = parser.parse_args()
    root = args.root.resolve()
    base = directory(root, "development", args.revision)
    if (base / "specification.json").exists():
        raise RuntimeError("Prepare imports before starting the frozen revision")
    old = read_json(args.snapshot / "provenance.json")
    for relative, expected in old["files"].items():
        if file_hash(args.snapshot / "source" / relative) != expected:
            raise RuntimeError("Historical source snapshot changed")
    current = code_provenance(root)
    if current["dependencies"] != old["dependencies"]:
        raise RuntimeError("Dependency versions changed; compatibility needs another audit")
    diagnostics = root / "runs/development/diagnostics"
    before = read_json(diagnostics / "equivalence_legacy.json")
    after = read_json(diagnostics / "equivalence_revised.json")
    if (
        before["source_hash"] != old["source_hash"]
        or after["source_hash"] != current["source_hash"]
        or before["coarse"] != after["coarse"]
        or before["sources"] != after["sources"]
    ):
        raise RuntimeError("Algorithms differ; do not reuse historical evaluations")
    evidence = {
        "legacy": file_hash(diagnostics / "equivalence_legacy.json"),
        "revised": file_hash(diagnostics / "equivalence_revised.json"),
        "exact_coarse_predictions": len(after["coarse"]),
        "exact_source_methods": len(after["sources"]),
        "coverage": "Calibration observations for coarse checks; synthetic multipath sources; no held-out evaluation",
    }
    receipt = {
        "target_code_hash": current["source_hash"],
        "accepted": [
            {
                "code_hash": old["source_hash"],
                "coarse": True,
                "sources": True,
                "source_rule": "Unchanged greedy-median default, or unchanged published comparator",
                "evidence": evidence,
            }
        ],
        "original_snapshot": str(args.snapshot.resolve()),
    }
    write_json(root / "runs/development/cache_compatibility.json", receipt)
    imported, presets = {}, 0
    for track in ("receiver", "direct"):
        evaluator = Evaluator(root, "development", "calibration", track)
        data_hash = digest(
            [
                evaluator.design["design_hash"],
                evaluator.observation_hash,
                evaluator.truth["truth_hash"],
            ]
        )
        for path in (root / "runs/development/calibration" / track / "component").glob("*.json"):
            previous = read_json(path)
            specification = previous["specification"]
            if (
                specification["code_hash"] != old["source_hash"]
                or specification["dataset_hash"] != data_hash
            ):
                raise RuntimeError("Historical search has incompatible code or observations")
            if not path.stem.startswith("coarse_"):
                raise RuntimeError("Additional historical search type requires explicit migration")
            trials = []
            for trial in previous["trials"]:
                artifact = Path(trial["audit"]["artifact"])
                if not artifact.is_absolute():
                    artifact = root / artifact
                if file_hash(artifact) != trial["audit"]["sha256"]:
                    raise RuntimeError("Historical trial artifact changed")
                trials.append(
                    {
                        **trial,
                        "phase": "historical_" + trial["phase"],
                        "origin": {
                            "search": str(path),
                            "search_hash": previous["search_hash"],
                            "search_file_sha256": file_hash(path),
                            "code_hash": old["source_hash"],
                        },
                    }
                )
            destination = base / "imports" / track / "component" / path.name
            write_json(destination, {"trials": trials, "source": str(path)})
            imported[f"{track}/{path.stem}"] = len(trials)
        for summary in (root / "runs/development/presets/legacy_v1" / track).glob("*.summary.json"):
            record = read_json(summary)
            artifact = Path(record["artifact"])
            if not artifact.is_absolute():
                artifact = root / artifact
            if (
                record["data_hash"] != data_hash
                or record["code_hash"] != old["source_hash"]
                or file_hash(artifact) != record["sha256"]
            ):
                raise RuntimeError("Historical preset identity mismatch")
            with gzip.open(artifact, "rt") as handle:
                payload = json.load(handle)
            identity = {
                "stage": record["kind"],
                "method": record["method"],
                "parameters": normalized(record["kind"], record["method"], record["parameters"]),
                "code_hash": current["source_hash"],
                "track": track,
                "data_hash": data_hash,
            }
            path = base / "objectives" / track / (digest(identity) + ".json.gz")
            reuse = {
                "artifact": str(artifact),
                "sha256": record["sha256"],
                "code_hash": old["source_hash"],
            }
            gzip_json(
                path,
                {
                    "identity": identity,
                    "score": record["score"],
                    "audit": payload["audit"],
                    "reuse": reuse,
                },
            )
            write_json(
                path.with_suffix(".summary.json"),
                {
                    "identity": identity,
                    "score": record["score"],
                    "case_count": record["case_count"],
                    "sha256": file_hash(path),
                    "artifact": str(path),
                    "reuse": reuse,
                },
            )
            presets += 1
    write_json(
        base / "migration.json",
        {
            "compatibility": receipt,
            "imported_trials": imported,
            "imported_presets": presets,
            "historical_policy": "Retain all old artifacts, including degenerate trials; no reruns or deletion",
        },
    )
    print(
        json.dumps({"imported_trials": imported, "imported_presets": presets, "evidence": evidence})
    )


if __name__ == "__main__":
    main()
