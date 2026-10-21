"""Compare frozen and revised algorithms without using held-out observations."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot:
        sys.path.insert(0, str(args.snapshot / "source/src"))
    import numpy as np
    from indoor_vt.contracts import Survey, DelayObservation
    from indoor_vt.dataset import load_observations, conditions
    from indoor_vt.localization import FingerprintLocalizer, InitialConfig
    from indoor_vt.provenance import read_json, write_json, code_provenance
    from indoor_vt.registry import VT_METHODS, make_reconstructor, reconstruction_default

    code_root = args.snapshot / "source" if args.snapshot else args.root
    result = {"source_hash": code_provenance(code_root)["source_hash"], "coarse": [], "sources": {}}
    design = read_json(args.root / "data/development/design.json")
    room = next(r for r in design["rooms"] if r["split"] == "calibration")
    data = args.root / "data/development/scenarios"
    for track in ("receiver", "direct"):
        level = next(label for tr, label, _ in conditions(design["config"]) if tr == track)
        for subset in (("A",), ("B",), ("A", "B")):
            observations = load_observations(data, room["id"], subset, "survey", track, level)
            indices = np.linspace(0, len(observations) - 1, 16, dtype=int)
            survey = Survey(
                np.asarray(room["survey_xyz"])[indices], [observations[i] for i in indices]
            )
            queries = load_observations(
                data, room["id"], subset, room["layouts"][0]["id"], track, level
            )[:4]
            for method in ("mpurge_map", "mpurge", "mca", "chamfer"):
                for k in (1, 3, 10):
                    config = InitialConfig(method=method, neighbors=k)
                    model = FingerprintLocalizer(config).fit(survey)
                    for query in queries:
                        predicted = model.predict(query)
                        result["coarse"].append(
                            {
                                "track": track,
                                "subset": subset,
                                "config": asdict(config),
                                "xyz": predicted.xyz.tolist(),
                                "status": predicted.status,
                            }
                        )
    rng = np.random.default_rng(812)
    xyz = rng.uniform([0, 0, 1], [5, 6, 1.5], (20, 3))
    anchors = np.array([[-2, 2, 2.5], [7, 4, 2.5], [2, -6, 2.5]])
    ranges = np.linalg.norm(xyz[:, None] - anchors[None], axis=2)
    survey = Survey(xyz, [DelayObservation({"A": row}) for row in ranges])
    for method in VT_METHODS:
        config = reconstruction_default(method)
        if "trials" in config:
            config["trials"] = 100  # Bounded verification fixture, not a calibration trial.
        fitted = make_reconstructor(method, config, 19).fit(survey)
        result["sources"][method] = {ap: sources.tolist() for ap, sources in fitted.by_ap.items()}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
