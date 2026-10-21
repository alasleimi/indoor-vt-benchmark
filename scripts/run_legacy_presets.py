"""Useful preset evaluations on an immutable source snapshot during migration."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--track", choices=("direct", "receiver"), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.snapshot / "source/src"))
    from indoor_vt import evaluation
    from indoor_vt.localization import LEAConfig
    from indoor_vt.provenance import digest, file_hash, read_json, write_json
    from indoor_vt.registry import VT_METHODS, initial_default, reconstruction_default

    provenance = read_json(args.snapshot / "provenance.json")
    for relative, expected in provenance["files"].items():
        if file_hash(args.snapshot / "source" / relative) != expected:
            raise RuntimeError("Legacy snapshot changed")
    evaluation.code_provenance = lambda root: dict(provenance)
    e = evaluation.Evaluator(args.root, "development", "calibration", args.track)
    out = args.root / "runs/development/presets/legacy_v1" / args.track
    jobs = [("coarse", x) for x in ("mca", "chamfer")]
    jobs += [("vt", x) for x in VT_METHODS]
    jobs += [("coarse", x) for x in ("mpurge", "mpurge_map")]
    pairs = [("mpurge_map", x) for x in VT_METHODS]
    pairs += [(x, "mpurge_local") for x in ("mca", "chamfer", "mpurge")]
    jobs += [("pipeline", x + "__" + y) for x, y in pairs]
    for kind, name in jobs:
        if (out.parent / "stop_requested").exists():
            write_json(out / "worker.json", {"state": "checkpointed", "next": [kind, name]})
            return
        destination = out / f"{kind}_{name}.summary.json"
        if destination.exists():
            continue
        write_json(out / "worker.json", {"state": "running", "current": [kind, name]})
        if kind == "coarse":
            config = initial_default(name)
            if name == "mpurge_map":
                config["half_window"] = 6
            score, audit = e.localization_objective(config)
        elif kind == "vt":
            config = reconstruction_default(name)
            score, audit = e.vt_objective(name, config)
        else:
            initial, vt = name.split("__")
            initial_config = initial_default(initial)
            if initial == "mpurge_map":
                initial_config["half_window"] = 6
            config = {
                "initial": initial_config,
                "vt": reconstruction_default(vt),
                "vt_method": vt,
                "lea": asdict(LEAConfig()),
            }
            score, audit = e.localization_objective(
                config["initial"],
                vt_method=vt,
                vt_parameters=config["vt"],
                lea_parameters=config["lea"],
            )
        artifact = out / f"{kind}_{name}.json.gz"
        record = {
            "kind": kind,
            "method": name,
            "parameters": config,
            "score": score,
            "audit": audit,
            "provenance": provenance,
            "selection_split": "calibration",
            "track": args.track,
            "preset_status": "published settings where specified plus disclosed adapter defaults",
            "data_hash": digest(
                [e.design["design_hash"], e.observation_hash, e.truth["truth_hash"]]
            ),
        }
        evaluation.gzip_json(artifact, record)
        write_json(
            destination,
            {key: value for key, value in record.items() if key not in ("audit", "provenance")}
            | {
                "case_count": audit["case_count"],
                "code_hash": provenance["source_hash"],
                "artifact": str(artifact),
                "sha256": file_hash(artifact),
            },
        )
        print(f"Completed {args.track} {kind}/{name}: {score}", flush=True)
    write_json(out / "worker.json", {"state": "finished"})


if __name__ == "__main__":
    main()
