"""Measure complete standard MPUrge runs on the active rectangle; no legacy comparisons."""
import argparse
from pathlib import Path
import time

from indoor_vt.acquisition.observations import Observations
from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.execution.artifacts import ArtifactCache
from indoor_vt.execution.presets import standard_mpurge
from indoor_vt.execution.reconstruction import MPUrgeExecution

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/redesign.json"))
parser.add_argument("--data", type=Path, default=Path("data/redesign/rectangle-20261025"))
parser.add_argument("--output", type=Path, default=Path("runs/redesign/runtime-investigation/final-runtime.json"))
args = parser.parse_args()
config = read_json(args.config)
observations = Observations(args.data, config)
cache = ArtifactCache(args.data / "numerical-cache-v2",
    disk_bytes=config["storage"]["cache_mib"] * 1024**2,
    ram_bytes=config["storage"]["ram_mib"] * 1024**2,
    free_bytes=config["storage"]["minimum_free_mib"] * 1024**2)
results = []
for noise in (0., 5.):
    survey = observations.survey(observations.rooms[0], noise, 2)
    for neighbors in (5, 40):
        for alpha in (0., .7):
            for unify in (True, False):
                parameters = dict(half_window=7, alpha=alpha, neighbors=neighbors,
                                  radius_m=.1, beta=0.)
                before = dict(cache.stats)
                started = time.perf_counter()
                result = MPUrgeExecution(standard_mpurge(parameters, unify=unify), cache).fit(survey)
                elapsed = time.perf_counter() - started
                row = {"noise": noise, "unify": unify, "parameters": parameters, "seconds": elapsed,
                       "sources": {ap: len(xyz) for ap, xyz in result.by_ap.items()},
                       "cache_events": {k: v - before.get(k, 0) for k, v in cache.stats.items()
                                        if not k.endswith(":seconds") and v != before.get(k, 0)}}
                results.append(row)
                write_json(args.output, {"runs": results, "cache": cache.usage(),
                    "limit_seconds": config["runtime"]["mpurge_limit_seconds"],
                    "scope": "280-position rectangle, complete fit including cache reads/writes",
                    "first_run_includes_jit_loading": True})
                print(row, flush=True)
                if elapsed > config["runtime"]["mpurge_limit_seconds"]:
                    raise RuntimeError(f"Runtime budget exceeded: {elapsed:.3f}s")
print(f"Maximum: {max(r['seconds'] for r in results):.3f}s", flush=True)
