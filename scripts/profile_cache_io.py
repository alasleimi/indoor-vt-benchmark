"""Measure a complete calibration case, including cold/warm cache I/O."""

import argparse
from pathlib import Path
import time
import numpy as np

from indoor_vt import evaluation
from indoor_vt.provenance import code_provenance, digest, read_json, write_json
from indoor_vt.registry import initial_default
from indoor_vt.localization import FingerprintLocalizer, InitialConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = code_provenance(args.benchmark.resolve())
    revised = code_provenance(Path.cwd())
    evaluation.code_provenance = lambda _: dict(revised)
    old_score_hash = digest(
        {
            key: value
            for key, value in baseline["files"].items()
            if key.rsplit("/", 1)[-1]
            in ("localization.py", "matching.py", "contracts.py", "uv.lock")
        }
    )
    records = []
    for track in ("receiver", "direct"):
        e = evaluation.Evaluator(args.benchmark, "development", "calibration", track)
        room, level = e.rooms[0], e.levels[0]
        cases = e.query_cases(room, ("A",), level)
        inputs = e._input_identity(room, ("A",), level, ["survey", *sorted({x[0] for x in cases})])
        for method in ("mca", "chamfer"):
            parameters = initial_default(method)
            parameters["epsilon_m"] = 1.6 if method == "mca" else 1.0
            scoring = {"method": method}
            if method == "mca":
                scoring["epsilon_m"] = 1.6
            previous_identity = {"parameters": scoring, "code": old_score_hash, "input": inputs}
            previous = e.cache / "channel_scores" / (digest(previous_identity) + ".npz")
            if previous.exists():
                original = read_json(previous.with_suffix(".json"))
                with np.load(previous, allow_pickle=False) as packed:
                    expected = packed["scores"]
            else:
                model = FingerprintLocalizer(InitialConfig(**parameters))
                survey = e.survey(room, ("A",), level)
                started = time.perf_counter()
                expected = np.array(
                    [
                        [
                            model.score_pair(query.channels["A"], reference.channels["A"])
                            for reference in survey.observations
                        ]
                        for _, _, query, _ in cases
                    ]
                )
                original = {"compute_s": time.perf_counter() - started}
            started = time.perf_counter()
            scores, key, hit = e.channel_scores(parameters, room, "A", level)
            elapsed = time.perf_counter() - started
            np.testing.assert_array_equal(scores, expected)
            started = time.perf_counter()
            reloaded, _, warm_hit = e.channel_scores(parameters, room, "A", level)
            warm_s = time.perf_counter() - started
            np.testing.assert_array_equal(reloaded, scores)
            records.append(
                {
                    "track": track,
                    "method": method,
                    "queries": len(cases),
                    "scores": scores.size,
                    "original_score_compute_s": original["compute_s"],
                    "new_score_and_cache_io_s": elapsed,
                    "initial_score_cache_hit": hit,
                    "warm_cache_io_s": warm_s,
                    "warm_hit": warm_hit,
                    "score_cache": key,
                    "exact": True,
                }
            )
    write_json(
        args.output,
        {
            "old_code_hash": baseline["source_hash"],
            "new_code_hash": revised["source_hash"],
            "records": records,
            "exact": True,
            "limitation": "One AP/level/room per track; original timing from historical receipt, not a controlled same-time runtime comparison",
        },
    )
    print(records)


if __name__ == "__main__":
    main()
