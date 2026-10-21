"""Calibration-input timing and exact-equivalence checks, not a parameter study."""

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np

from indoor_vt.contracts import Survey
from indoor_vt.dataset import load_observations, conditions
from indoor_vt.localization import FingerprintLocalizer, InitialConfig
from indoor_vt.matching import PatternMatcher, MatcherConfig
from indoor_vt.nearest_scores import nearest_geometry, geometry_scores
from indoor_vt.provenance import code_provenance, read_json, write_json


def timed(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "baseline_matching", args.baseline / "src/indoor_vt/matching.py"
    )
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    FingerprintLocalizer(InitialConfig(method="mpurge_map"), matcher=original.PatternMatcher())
    design = read_json(args.benchmark / "data/development/design.json")
    room = next(r for r in design["rooms"] if r["split"] == "calibration")
    data = args.benchmark / "data/development/scenarios"
    evidence = {
        "baseline_source_hash": code_provenance(args.baseline)["source_hash"],
        "revised_source_hash": code_provenance(Path.cwd())["source_hash"],
        "selection": "First calibration room, A; 100 queries, full reference survey",
        "tracks": {},
        "matcher": [],
    }
    pair_bank = []
    for track in ("receiver", "direct"):
        level = next(label for tr, label, _ in conditions(design["config"]) if tr == track)
        survey_observations = load_observations(data, room["id"], ("A",), "survey", track, level)
        queries_observations = load_observations(
            data, room["id"], ("A",), room["layouts"][0]["id"], track, level
        )
        survey = Survey(np.asarray(room["survey_xyz"]), survey_observations)
        references = [o.channels["A"] for o in survey_observations]
        queries = [o.channels["A"] for o in queries_observations]
        pair_bank.extend(zip(queries[:30], references[:30]))
        geometry, geometry_s = timed(lambda: nearest_geometry(queries, references))
        result = {
            "query_count": len(queries),
            "references": len(references),
            "geometry_s": geometry_s,
            "methods": [],
        }
        for method, eps in (("chamfer", 1.0), ("mca", 0.1), ("mca", 1.6), ("mca", 10.0)):
            model = FingerprintLocalizer(InitialConfig(method=method, epsilon_m=eps)).fit(survey)
            expected, old_s = timed(
                lambda: np.array([[model.score_pair(q, r) for r in references] for q in queries])
            )
            actual, cached_s = timed(lambda: geometry_scores(geometry, method, eps))
            np.testing.assert_array_equal(actual, expected)
            for row, query in zip(actual, queries_observations):
                np.testing.assert_array_equal(
                    model.predict_from_scores(row).xyz, model.predict(query).xyz
                )
            result["methods"].append(
                {
                    "method": method,
                    "epsilon": eps,
                    "scalar_s": old_s,
                    "geometry_score_s": cached_s,
                    "exact_scores": actual.size,
                    "exact_predictions": len(queries),
                }
            )
        evidence["tracks"][track] = result
    rng = np.random.default_rng(204)
    for _ in range(80):
        pair_bank.append(
            (
                rng.integers(0, 80, rng.integers(0, 30)).astype(float),
                rng.integers(0, 80, rng.integers(0, 30)).astype(float),
            )
        )
    for p in (0, 1, 2, 4, 6, 10):
        for alpha in (0.0, 0.5, 0.7, 1.0):
            for normalize in (False, True):
                for squared, printed in ((False, False), (True, False), (False, True)):
                    kwargs = dict(
                        half_window=p,
                        alpha=alpha,
                        normalized_pattern=normalize,
                        size_norm="l2" if squared else "l1",
                        cross_mode="printed" if printed else "order",
                    )
                    old = original.PatternMatcher(original.MatcherConfig(**kwargs))
                    new = PatternMatcher(MatcherConfig(**kwargs))
                    old.match_array(np.arange(4.0), np.arange(4.0))
                    new.match_array(np.arange(4.0), np.arange(4.0))
                    for separation in (None, 3.0):
                        expected, before_s = timed(
                            lambda: [
                                old.match_array(a, b, separation=separation, tolerance=0.5)
                                for a, b in pair_bank
                            ]
                        )
                        actual, after_s = timed(
                            lambda: [
                                new.match_array(a, b, separation=separation, tolerance=0.5)
                                for a, b in pair_bank
                            ]
                        )
                        for a, b in zip(expected, actual):
                            np.testing.assert_array_equal(a, b)
                        evidence["matcher"].append(
                            {
                                **kwargs,
                                "separation": separation,
                                "pairs": len(pair_bank),
                                "before_s": before_s,
                                "after_s": after_s,
                            }
                        )
    write_json(args.output, evidence)
    print(
        {
            "exact_match_cases": sum(r["pairs"] for r in evidence["matcher"]),
            "track_timings": evidence["tracks"],
        }
    )


if __name__ == "__main__":
    main()
