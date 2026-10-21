"""Carry a numerically equivalent preset-phase campaign into a new code revision."""

import argparse
from pathlib import Path

from indoor_vt.provenance import code_provenance, digest, file_hash, read_json, write_json


def component_hash(provenance, names):
    return digest(
        {
            key: value
            for key, value in provenance["files"].items()
            if key.rsplit("/", 1)[-1] in names
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--previous", default="v2")
    parser.add_argument("--revision", default="v3")
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    before_base = root / "runs/development/revisions" / args.previous
    base = root / "runs/development/revisions" / args.revision
    if (base / "specification.json").exists():
        raise RuntimeError("Cannot change an active revision")
    if any((before_base / "calibration").rglob("*.json")):
        raise RuntimeError(
            "Use a search-history migration after hyperparameter trials have started"
        )
    before = code_provenance(root)
    after = code_provenance(args.code_root.resolve())
    specification = read_json(before_base / "specification.json")
    if before["source_hash"] != specification["core_code_hash"]:
        raise RuntimeError("Previous implementation changed before preparation")
    if before["dependencies"] != after["dependencies"]:
        raise RuntimeError("Dependency changes require a separate compatibility review")
    evidence = read_json(args.evidence)
    if (
        evidence["old_code_hash"] != before["source_hash"]
        or evidence["new_code_hash"] != after["source_hash"]
        or not evidence["exact"]
    ):
        raise RuntimeError("Equivalence evidence does not cover these implementations")
    old_compatibility = read_json(root / "runs/development/cache_compatibility.json")
    if old_compatibility["target_code_hash"] != before["source_hash"]:
        raise RuntimeError("Previous cache compatibility receipt changed")
    current = {
        "code_hash": before["source_hash"],
        "coarse": True,
        "sources": True,
        "legacy_merger_fields": False,
        "score_implementation": component_hash(
            before, ("localization.py", "matching.py", "contracts.py", "uv.lock")
        ),
        "stage_implementation": component_hash(
            before,
            (
                "matching.py",
                "reconstruction.py",
                "fitting.py",
                "clustering.py",
                "contracts.py",
                "stage_cache.py",
                "provenance.py",
                "uv.lock",
            ),
        ),
        "evidence": {"artifact": str(args.evidence.resolve()), "sha256": file_hash(args.evidence)},
    }
    compatibility = {
        "target_code_hash": after["source_hash"],
        "accepted": [current, *old_compatibility["accepted"]],
    }
    write_json(base / "prepared_cache_compatibility.json", compatibility)
    imported = []
    for path in (before_base / "imports").rglob("*.json"):
        target = base / path.relative_to(before_base)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for receipt in (before_base / "objectives").rglob("*.summary.json"):
        record = read_json(receipt)
        if record["identity"]["code_hash"] != before["source_hash"]:
            raise RuntimeError("Unexpected objective implementation")
        artifact = Path(record["artifact"])
        if file_hash(artifact) != record["sha256"]:
            raise RuntimeError("Preserved objective changed")
        identity = {**record["identity"], "code_hash": after["source_hash"]}
        target = base / "objectives" / identity["track"] / (digest(identity) + ".json.summary.json")
        write_json(
            target,
            {
                **record,
                "identity": identity,
                "reuse": {
                    "receipt": str(receipt),
                    "code_hash": before["source_hash"],
                    "receipt_sha256": file_hash(receipt),
                },
            },
        )
        imported.append(str(receipt))
    write_json(
        base / "migration.json",
        {
            "previous_revision": args.previous,
            "old_code_hash": before["source_hash"],
            "new_code_hash": after["source_hash"],
            "objective_receipts": imported,
            "imported_objectives": len(imported),
            "preserved_original_trials": 27,
            "evidence": current["evidence"],
            "policy": "Link unchanged numerical results; preserve originals and timings; no search-domain or budget changes",
        },
    )
    print({"imported_objectives": len(imported), "new_code_hash": after["source_hash"]})


if __name__ == "__main__":
    main()
