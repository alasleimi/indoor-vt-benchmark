"""Evaluator owns labels. Estimators receive only Survey/DelayObservation/SourceMap."""

from pathlib import Path
import gzip
import json
import os
import time
from collections import OrderedDict
import numpy as np
from .calibration import uniform_mean
from .contracts import SourceMap, Survey, Prediction
from .dataset import load_observations, conditions, delay_file, observation_signature
from .localization import FingerprintLocalizer, InitialConfig, SmartLEA, LEAConfig
from .metrics import source_metrics
from .provenance import (
    digest,
    file_hash,
    read_json,
    write_json,
    SeedBook,
    code_provenance,
    json_default,
    replace_file,
)
from .registry import make_reconstructor
from .dataset import atomic_npz
from .stage_cache import StageCache


def gzip_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as z:
            z.write(
                json.dumps(
                    value, separators=(",", ":"), allow_nan=False, default=json_default
                ).encode()
            )
    replace_file(temporary, path)


class Evaluator:
    def __init__(
        self,
        root,
        namespace,
        split,
        track,
        *,
        pilot_room=None,
        pilot_subset=None,
        pilot_level=None,
        method_factory=make_reconstructor,
        implementation_hash=None,
    ):
        if split not in ("calibration", "test"):
            raise ValueError(split)
        if namespace != "pilot" and any(
            x is not None for x in (pilot_room, pilot_subset, pilot_level)
        ):
            raise ValueError("Subset debugging is restricted to the separately seeded pilot")
        self.root, self.namespace, self.split, self.track = Path(root), namespace, split, track
        self.design = read_json(self.root / "data" / namespace / "design.json")
        self.truth = read_json(self.root / "data" / namespace / "evaluation/source_truth.json")
        self.data = self.root / "data" / namespace / "scenarios"
        self.observation_hash = observation_signature(self.data, track)
        self.cache = self.root / "runs/cache"
        self.provenance = code_provenance(self.root)
        self.method_factory = method_factory
        if implementation_hash is not None:
            self.provenance["external_implementation_hash"] = implementation_hash
            self.provenance["source_hash"] = digest(
                [self.provenance["source_hash"], implementation_hash]
            )
        self.seeds = SeedBook(self.design["seeds"]["root"], namespace)
        self.rooms = [
            r
            for r in self.design["rooms"]
            if r["split"] == split and (not pilot_room or r["id"] == pilot_room)
        ]
        self.subsets = [tuple(pilot_subset)] if pilot_subset else [("A",), ("B",), ("A", "B")]
        self.levels = [
            label
            for tr, label, value in conditions(self.design["config"])
            if tr == track and (pilot_level is None or label == str(pilot_level))
        ]
        self.surveys, self.queries = {}, OrderedDict()
        self.input_hashes = {}
        self.stage_cache = StageCache(
            self.cache / "stages",
            digest(
                {
                    key: value
                    for key, value in self.provenance["files"].items()
                    if key.startswith("src/indoor_vt/")
                    and key.rsplit("/", 1)[-1]
                    in (
                        "matching.py",
                        "reconstruction.py",
                        "fitting.py",
                        "clustering.py",
                        "contracts.py",
                        "stage_cache.py",
                        "provenance.py",
                    )
                    or key == "uv.lock"
                }
            ),
        )
        self.score_implementation = digest(
            {
                key: value
                for key, value in self.provenance["files"].items()
                if key.rsplit("/", 1)[-1]
                in ("localization.py", "matching.py", "contracts.py", "uv.lock")
            }
        )
        compatibility = self.root / "runs" / namespace / "cache_compatibility.json"
        self.compatible = []
        if compatibility.exists():
            receipt = read_json(compatibility)
            if receipt["target_code_hash"] == self.provenance["source_hash"]:
                self.compatible = receipt["accepted"]
        self.partial_pilot = bool(pilot_room or pilot_subset or pilot_level)
        self.cases = [
            (room, subset, level)
            for room in self.rooms
            for subset in self.subsets
            for level in self.levels
        ]
        if not self.cases:
            raise ValueError("No cases selected")
        # Fail before scoring anything if a declared survey/query acquisition is absent.
        for room, subset, level in self.cases:
            for ap in subset:
                labels = ["survey", *[x["id"] for x in room["layouts"]]]
                for label in labels:
                    p = (
                        self.data
                        / room["id"]
                        / ap
                        / label
                        / "observations"
                        / track
                        / level
                        / "manifest.json"
                    )
                    if not p.exists() and not (self.partial_pilot and label != "survey"):
                        raise FileNotFoundError(f"Incomplete declared dataset: {p}")

    def survey(self, room, subset, level):
        key = (room["id"], subset, level)
        if key not in self.surveys:
            observations = load_observations(
                self.data, room["id"], subset, "survey", self.track, level
            )
            manifests = [
                read_json(self.data / room["id"] / ap / "survey/input_manifest.json")
                for ap in subset
            ]
            indices = manifests[0]["position_indices"]
            if any(m["position_indices"] != indices for m in manifests):
                raise ValueError("AP surveys have mismatched sample identities")
            self.surveys[key] = Survey(np.asarray(room["survey_xyz"])[indices], observations)
        return self.surveys[key]

    def query_cases(self, room, subset, level):
        key = (room["id"], subset, level)
        if key not in self.queries:
            rows = []
            for layout in room["layouts"]:
                if self.partial_pilot and any(
                    not (self.data / room["id"] / ap / layout["id"] / "receipt.json").exists()
                    for ap in subset
                ):
                    continue
                observations = load_observations(
                    self.data, room["id"], subset, layout["id"], self.track, level
                )
                manifests = [
                    read_json(self.data / room["id"] / ap / layout["id"] / "input_manifest.json")
                    for ap in subset
                ]
                indices = manifests[0]["position_indices"]
                if any(m["position_indices"] != indices for m in manifests):
                    raise ValueError("AP queries have mismatched sample identities")
                positions = np.asarray(layout["queries_xyz"])[indices]
                rows.extend(
                    (layout["id"], index, query, truth)
                    for index, query, truth in zip(indices, observations, positions)
                )
            if not rows:
                raise ValueError("No query observations")
            self.queries[key] = rows
            while len(self.queries) > 6:
                self.queries.popitem(last=False)
        self.queries.move_to_end(key)
        return self.queries[key]

    def _input_identity(self, room, subset, level, layouts):
        result = {}
        for ap in subset:
            for layout in layouts:
                directory = (
                    self.data / room["id"] / ap / layout / "observations" / self.track / level
                )
                manifest = directory / "manifest.json"
                previous = self.input_hashes.get(str(directory))
                if (
                    previous
                    and previous[0] == manifest.stat().st_mtime_ns
                    and previous[1] == previous[2].stat().st_mtime_ns
                ):
                    checksum = previous[3]
                else:
                    path = delay_file(directory)  # Validates bytes against the immutable manifest.
                    record = read_json(manifest)
                    checksum = record["files"][record.get("delay_file", "delays.npz")]
                    self.input_hashes[str(directory)] = (
                        manifest.stat().st_mtime_ns,
                        path.stat().st_mtime_ns,
                        path,
                        checksum,
                    )
                result[f"{ap}/{layout}"] = checksum
        return result

    def source_map(self, method, parameters, room, subset, level):
        # A+B literally combines independently fitted A and B maps, retaining labels.
        if len(subset) > 1:
            maps = [self.source_map(method, parameters, room, (ap,), level) for ap in subset]
            return SourceMap(
                {ap: m.by_ap[ap] for ap, m in zip(subset, maps)},
                {"channel_cache_keys": [m.diagnostics["cache_key"] for m in maps]},
            )
        seed = self.seeds.seed("method", method, room["id"], subset[0])
        identity = {
            "method": method,
            "parameters": parameters,
            "seed": seed,
            "input": self._input_identity(room, subset, level, ["survey"]),
            "survey_xyz": self.survey(room, subset, level).xyz.tolist(),
            "code": self.provenance["source_hash"],
        }
        key = digest(identity)
        path = self.cache / "sources" / f"{key}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as f:
                payload = json.load(f)
            return SourceMap(
                payload["sources"], {"cache_key": key, "elapsed_s": payload["elapsed_s"]}
            )
        for compatible in self.compatible:
            if (
                not compatible.get("sources")
                or parameters.get("merger", "greedy_median") != "greedy_median"
            ):
                continue
            legacy_parameters = {
                k: v for k, v in parameters.items() if k not in ("merger", "dbscan_min_samples")
            }
            previous_identity = {
                **identity,
                "code": compatible["code_hash"],
                "parameters": legacy_parameters,
            }
            previous_path = self.cache / "sources" / (digest(previous_identity) + ".json.gz")
            if previous_path.exists():
                with gzip.open(previous_path, "rt") as handle:
                    previous = json.load(handle)
                if previous["identity"] != previous_identity:
                    raise RuntimeError("Historical source cache identity mismatch")
                reuse = {
                    "source": str(previous_path),
                    "sha256": file_hash(previous_path),
                    "code_hash": compatible["code_hash"],
                }
                gzip_json(path, {**previous, "identity": identity, "reuse": reuse})
                return SourceMap(
                    previous["sources"],
                    {"cache_key": key, "elapsed_s": previous["elapsed_s"], "reuse": reuse},
                )
        started = time.perf_counter()
        if self.method_factory is make_reconstructor:
            model = make_reconstructor(method, parameters, seed, stage_cache=self.stage_cache)
        else:
            model = self.method_factory(method, parameters, seed)
        result = model.fit(self.survey(room, subset, level))
        elapsed = time.perf_counter() - started
        gzip_json(
            path,
            {
                "identity": identity,
                "sources": {a: v.tolist() for a, v in result.by_ap.items()},
                "diagnostics": result.diagnostics,
                "elapsed_s": elapsed,
            },
        )
        return SourceMap(result.by_ap, {"cache_key": key, "elapsed_s": elapsed})

    def channel_scores(self, parameters, room, ap, level):
        cases = self.query_cases(room, (ap,), level)
        survey = self.survey(room, (ap,), level)
        scoring = {"method": parameters["method"]}
        if scoring["method"] in ("mpurge", "mpurge_map"):
            scoring.update({key: parameters[key] for key in ("half_window", "alpha")})
        elif scoring["method"] == "mca":
            scoring["epsilon_m"] = parameters["epsilon_m"]
        identity = {
            "parameters": scoring,
            "code": self.score_implementation,
            "input": self._input_identity(
                room, (ap,), level, ["survey", *sorted({x[0] for x in cases})]
            ),
        }
        key = digest(identity)
        path = self.cache / "channel_scores" / (key + ".npz")
        manifest = path.with_suffix(".json")
        if path.exists() and manifest.exists():
            record = read_json(manifest)
            if record["identity"] != identity or file_hash(path) != record["sha256"]:
                raise RuntimeError("Channel score cache changed")
            with np.load(path, allow_pickle=False) as packed:
                return packed["scores"], key, True
        model = FingerprintLocalizer(InitialConfig(**parameters))
        scores = np.empty((len(cases), len(survey.xyz)))
        started = time.perf_counter()
        for index, (_, _, query, _) in enumerate(cases):
            values = query.channels[ap]
            for reference_index, reference in enumerate(survey.observations):
                scores[index, reference_index] = model.score_pair(values, reference.channels[ap])
        elapsed = time.perf_counter() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_npz(path, {"scores": scores})
        write_json(
            manifest,
            {
                "identity": identity,
                "sha256": file_hash(path),
                "compute_s": elapsed,
                "shape": list(scores.shape),
            },
        )
        return scores, key, False

    def coarse_predictions(self, parameters, room, subset, level):
        cases = self.query_cases(room, subset, level)
        labels = sorted({x[0] for x in cases})
        identity = {
            "parameters": parameters,
            "input": self._input_identity(room, subset, level, ["survey", *labels]),
            "survey_xyz": self.survey(room, subset, level).xyz.tolist(),
            "code": self.provenance["source_hash"],
        }
        key = digest(identity)
        path = self.cache / "coarse" / f"{key}.json"
        if path.exists():
            return read_json(path)
        for compatible in self.compatible:
            if not compatible.get("coarse"):
                continue
            previous_identity = {**identity, "code": compatible["code_hash"]}
            previous_path = self.cache / "coarse" / (digest(previous_identity) + ".json")
            if previous_path.exists():
                previous = read_json(previous_path)
                if previous["identity"] != previous_identity:
                    raise RuntimeError("Historical cache identity mismatch")
                result = {
                    **previous,
                    "cache_key": key,
                    "identity": identity,
                    "reuse": {
                        "source": str(previous_path),
                        "sha256": file_hash(previous_path),
                        "code_hash": compatible["code_hash"],
                    },
                    "timing_kind": "historical_compute_time",
                }
                write_json(path, result)
                return result
        model = FingerprintLocalizer(InitialConfig(**parameters))
        started = time.perf_counter()
        model.fit(self.survey(room, subset, level))
        fit_s = time.perf_counter() - started
        score_started = time.perf_counter()
        channels = [self.channel_scores(parameters, room, ap, level) for ap in sorted(subset)]
        score_wall = time.perf_counter() - score_started
        # Element-wise two-channel averaging has exactly the previous arithmetic order.
        scores = np.mean(np.stack([entry[0] for entry in channels]), axis=0)
        predictions, timings = [], []
        for row in scores:
            started = time.perf_counter()
            result = model.predict_from_scores(row)
            timings.append(time.perf_counter() - started)
            predictions.append({"xyz": result.xyz.tolist(), "status": result.status})
        result = {
            "cache_key": key,
            "predictions": predictions,
            "fit_s": fit_s,
            "predict_s": timings,
            "identity": identity,
            "score_cache_keys": [entry[1] for entry in channels],
            "score_cache_hits": [entry[2] for entry in channels],
            "score_wall_s": score_wall,
            "timing_kind": "ranking_per_query_plus_separate_channel_score_wall_time",
        }
        write_json(path, result)
        return result

    def refined_predictions(self, coarse, sources, parameters, room, subset, level):
        cases = self.query_cases(room, subset, level)
        identity = {
            "code": self.provenance["source_hash"],
            "parameters": parameters,
            "initial": coarse["predictions"],
            "sources": {ap: xyz.tolist() for ap, xyz in sources.by_ap.items()},
            "input": self._input_identity(room, subset, level, sorted({x[0] for x in cases})),
        }
        path = self.cache / "refinement" / (digest(identity) + ".json.gz")
        if path.exists():
            with gzip.open(path, "rt") as handle:
                return json.load(handle)
        lea = SmartLEA(LEAConfig(**parameters))
        predictions, timings = [], []
        for (_, _, query, _), saved in zip(cases, coarse["predictions"]):
            initial = Prediction(np.array(saved["xyz"]), saved["status"])
            started = time.perf_counter()
            final = lea.predict(query, initial, sources)
            timings.append(time.perf_counter() - started)
            if not np.isfinite(final.xyz).all() and np.isfinite(initial.xyz).all():
                final = Prediction(initial.xyz.copy(), "fallback_invalid_refinement")
            predictions.append({"xyz": final.xyz.tolist(), "status": final.status})
        result = {"identity": identity, "predictions": predictions, "predict_s": timings}
        gzip_json(path, result)
        return result

    def vt_objective(self, method, parameters):
        scores, records = [], []
        for room, subset, level in self.cases:
            sources = self.source_map(method, parameters, room, subset, level)
            truth = np.array(
                [s["xyz"] for ap in subset for s in self.truth["sources"][room["id"]][ap]]
            )
            metrics = source_metrics(sources.bag(), truth)
            value = metrics["ospa"]
            scores.append(value)
            records.append(
                {
                    "room": room["id"],
                    "subset": list(subset),
                    "level": level,
                    **metrics,
                    "predicted": sources.bag().tolist(),
                    "source_cache": sources.diagnostics,
                }
            )
        return uniform_mean(scores), {"case_count": len(scores), "cases": records}

    def localization_objective(
        self, initial_parameters, *, vt_method=None, vt_parameters=None, lea_parameters=None
    ):
        errors, records, initial_errors = [], [], []
        for room, subset, level in self.cases:
            coarse = self.coarse_predictions(initial_parameters, room, subset, level)
            sources = (
                self.source_map(vt_method, vt_parameters, room, subset, level)
                if vt_method
                else None
            )
            refined = (
                self.refined_predictions(coarse, sources, lea_parameters, room, subset, level)
                if vt_method
                else None
            )
            cases = self.query_cases(room, subset, level)
            for index, ((layout, qid, query, truth), saved) in enumerate(
                zip(cases, coarse["predictions"])
            ):
                initial = Prediction(np.array(saved["xyz"]), saved["status"])
                saved_final = refined["predictions"][index] if refined else saved
                final = Prediction(np.array(saved_final["xyz"]), saved_final["status"])
                elapsed = refined["predict_s"][index] if refined else 0.0
                error = float(np.linalg.norm(final.xyz - truth))
                initial_error = float(np.linalg.norm(initial.xyz - truth))
                errors.append(error)
                initial_errors.append(initial_error)
                records.append(
                    {
                        "room": room["id"],
                        "subset": list(subset),
                        "level": level,
                        "layout": layout,
                        "query_id": int(qid),
                        "initial_xyz": initial.xyz.tolist(),
                        "final_xyz": final.xyz.tolist(),
                        "truth_xyz": truth.tolist(),
                        "initial_error_m": initial_error,
                        "final_error_m": error,
                        "initial_status": initial.status,
                        "final_status": final.status,
                        "coarse_predict_s": coarse["predict_s"][index],
                        "coarse_timing_kind": coarse.get("timing_kind", "original_full_prediction"),
                        "lea_s": elapsed,
                        "coarse_cache": coarse["cache_key"],
                        "source_cache": sources.diagnostics if sources else None,
                    }
                )
        return uniform_mean(errors), {
            "case_count": len(errors),
            "initial_mean_m": uniform_mean(initial_errors),
            "median_m": float(np.median(errors)),
            "p90_m": float(np.quantile(errors, 0.9)),
            "fallback_count": sum(r["final_status"].startswith("fallback") for r in records),
            "cases": records,
        }
