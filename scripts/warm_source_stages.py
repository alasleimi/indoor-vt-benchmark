"""Recover graph/group/raw-fit caches from preserved, compatible full VT audits."""

import argparse
import gzip
import json
from pathlib import Path

from indoor_vt.evaluation import Evaluator
from indoor_vt.provenance import digest, write_json
from indoor_vt.registry import VT_METHODS, reconstruction_default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default="v3")
    args = parser.parse_args()
    root = args.root.resolve()
    report = root / "runs/development/revisions" / args.revision / "recovered_stages.json"
    records, written = [], 0
    for track in ("receiver", "direct"):
        e = Evaluator(root, "development", "calibration", track)
        versions = [
            {
                "code_hash": e.provenance["source_hash"],
                "sources": True,
                "legacy_merger_fields": False,
            },
            *e.compatible,
        ]
        for method in VT_METHODS:
            if not method.startswith(("mpurge", "pba")):
                continue
            parameters = reconstruction_default(method)
            for room, subset, level in e.cases:
                if len(subset) != 1:
                    continue
                ap = subset[0]
                survey = e.survey(room, subset, level)
                identity = {
                    "method": method,
                    "parameters": parameters,
                    "seed": e.seeds.seed("method", method, room["id"], ap),
                    "input": e._input_identity(room, subset, level, ["survey"]),
                    "survey_xyz": survey.xyz.tolist(),
                    "code": e.provenance["source_hash"],
                }
                path = None
                for version in versions:
                    if not version.get("sources"):
                        continue
                    params = parameters
                    if version.get("legacy_merger_fields", True):
                        params = {
                            k: v
                            for k, v in params.items()
                            if k not in ("merger", "dbscan_min_samples")
                        }
                    previous = {**identity, "parameters": params, "code": version["code_hash"]}
                    candidate = e.cache / "sources" / (digest(previous) + ".json.gz")
                    if candidate.exists():
                        path = candidate
                        break
                if path is None:
                    continue
                with gzip.open(path, "rt") as handle:
                    payload = json.load(handle)
                if payload["identity"] != previous:
                    raise RuntimeError("Source audit identity mismatch")
                audit = payload["diagnostics"]
                if "initial" in audit:
                    audit = audit["initial"]
                config, channel = audit["config"], audit["channels"][ap]
                bank = digest(
                    {"xyz": survey.xyz, "ranges": [o.channels[ap] for o in survey.observations]}
                )
                graph = {
                    "bank": bank,
                    "scopes": channel["scopes"],
                    "matcher": audit["matcher"],
                    "triangle": config["triangle"],
                    "tolerance": config["triangle_tolerance_m"] if config["triangle"] else 0,
                }
                stages = [
                    ("graphs", graph, [channel["edges"], channel["pair_count"]]),
                    ("groups", {"graph": graph, "grouping": config["grouping"]}, channel["groups"]),
                    (
                        "raw_fits",
                        {
                            "bank": bank,
                            "groups": channel["groups"],
                            "solver": "unweighted_signed_3d_lm",
                        },
                        [channel["raw_candidates"], channel["rejected"]],
                    ),
                ]
                created = []
                for stage, cache_identity, value in stages:
                    implementations = [
                        e.stage_cache.implementation,
                        *e.stage_cache.previous_implementations,
                    ]
                    if any(
                        (
                            e.stage_cache.directory
                            / stage
                            / (digest([implementation, stage, cache_identity]) + ".json.gz")
                        ).exists()
                        for implementation in implementations
                    ):
                        continue
                    e.stage_cache.get(stage, cache_identity, lambda: value)
                    created.append(stage)
                    written += 1
                records.append(
                    {
                        "source": str(path),
                        "track": track,
                        "method": method,
                        "room": room["id"],
                        "ap": ap,
                        "level": level,
                        "created": created,
                    }
                )
                write_json(
                    report, {"finished": False, "stages_written": written, "sources": records}
                )
    write_json(report, {"finished": True, "stages_written": written, "sources": records})
    print({"source_audits": len(records), "stages_written": written})


if __name__ == "__main__":
    main()
