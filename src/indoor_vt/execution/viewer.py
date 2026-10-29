"""One-room comparison and bounded, reproducible evidence replay."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import html
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np

from .artifacts import ArtifactCache, implementation_id, read_npz
from .experiment import Evaluator, configurations
from .presets import standard_mpurge, mpurge_map
from .trials import read_trials
from .explorer import (build_catalog, saved_stage_metrics, query_truth, localization_metrics,
                       inside_obstacles, display_ap_source_metrics)
from ..acquisition.records import digest, read_json
from ..acquisition.dataset import survey_indices
from ..science.matching import _delta
from ..metrics import ap_source_metrics


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def preview(value, count=30, field="", start=0):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes,
                "field": field, "start": start,
                "preview": value[start:start + count] if value.ndim else value}
    if isinstance(value, dict):
        return {k: preview(v, count, field + "/" + k) for k, v in value.items()}
    if isinstance(value, list):
        return [preview(v, count, field + "/" + str(i)) for i, v in enumerate(value[:count])]
    return value


class ExperimentView:
    def __init__(self, data_root, run_root):
        self.data_root, self.run_root = Path(data_root), Path(run_root)
        self.config = read_json(self.run_root / "config.json")
        if self.config.get("engine") == "range_rmse":
            from .range_rmse_experiment import RangeRMSEEvaluator
            self.evaluator = RangeRMSEEvaluator(data_root, self.config, "viewer")
        elif self.config.get("engine") == "compositions":
            from .composition_experiment import CompositionEvaluator
            self.evaluator = CompositionEvaluator(data_root, self.config, "viewer")
        elif self.config.get("engine") == "additional":
            from .additional_experiment import AdditionalEvaluator
            self.evaluator = AdditionalEvaluator(data_root, self.config, "viewer")
        else:
            self.evaluator = Evaluator(data_root, self.config, "viewer")
        self.store = self.evaluator.store
        self.observations = self.evaluator.observations
        self.audit = ArtifactCache(self.run_root / "audit",
            disk_bytes=self.config["storage"]["audit_mib"] * 1024**2,
            ram_bytes=32 * 1024**2,
            free_bytes=self.config["storage"]["minimum_free_mib"] * 1024**2)
        self.lock = threading.RLock()
        self.stage_summaries = {}
        self.rooms = {r["id"]: r for r in self.observations.rooms}
        self.truths = {key: self.observations.truth(room) for key, room in self.rooms.items()}
        self.truth_catalog = self.observations.store.get(self.observations.manifest["truth_value"])

    def read_optional(self, path, default):
        return read_json(path) if path.exists() else default

    def overview(self):
        trials = read_trials(self.run_root / "trials")
        results = self.read_optional(self.run_root / "results.json", [])
        if self.config.get("engine") == "obstacle_reconstruction":
            from .obstacle_reconstruction import metrics_for_record
            corrected = []
            for row in results:
                if row["id"] not in self.stage_summaries:
                    detail = read_npz(self.run_root / row["outcome"])[0]
                    measured, counts = metrics_for_record(detail["sources"][0],
                                                          metric_fn=display_ap_source_metrics)
                    self.stage_summaries[row["id"]] = {"source_metrics_by_ap": measured,
                        "source_counts": [counts], "metrics": measured["A+B"], "loss": measured["A+B"]["ospa"]}
                corrected.append({**row, **self.stage_summaries[row["id"]]})
            results = corrected
            return {"methods": [], "trials": [], "calibration": [], "evaluation": results,
                    "conditions": [{"method_id": r["view_method_id"], "receipt_id": r["id"],
                        "input_count": r.get("input_count", 100), "input_protocol": r.get("input_protocol", "query_positions"),
                        **{k: r[k] for k in ("split", "level", "metrics", "source_metrics_by_ap",
                            "source_counts", "obstacle_count", "layout", "selection_loss")}}
                        for r in results],
                    "status": self.read_optional(self.run_root / "status.json", {"state": "not_started"}),
                    "obstacle_reconstruction_protocol": self.read_optional(self.run_root / "protocol.json", {})}
        # Small scalar summaries are cached in memory; outcomes and experiment logs stay unchanged.
        with self.lock:
            for row in results:
                if row["id"] not in self.stage_summaries and row.get("outcome"):
                    path = self.run_root / row["outcome"]
                    if path.exists():
                        detail = read_npz(path)[0]
                        self.stage_summaries[row["id"]] = saved_stage_metrics(
                            detail, self.rooms, self.truths)
            catalog = build_catalog(trials, results, self.stage_summaries)
        from .presentation import search_spec
        for method in catalog['methods']:
            method['search_spec'] = search_spec(method, self.config)
            if method.get('exploratory'):
                method['selected_id'] = next(r['id'] for r in results
                                             if r['label'] == method['label'] and r['level'] == 0)
        if self.config.get("engine") == "compositions":
            settings = self.config["compositions"]
            source_counts = {}
            for method in catalog["methods"]:
                if method["selection"] == "joint":
                    note = (f"{settings['joint_trials']} reproducible joint configurations, "
                            "including the independently selected baseline.")
                elif method["selection"] == "stagewise":
                    count = len(configurations(self.config["optimization"]["smart_lea"]))
                    note = (f"Exhaustive search over {count} effective SMART-LEA settings; "
                            "upstream reconstruction and MAP settings stay frozen.")
                else:
                    variant = max((k for k in settings["spaces"]
                                   if method["label"] == k or method["label"].startswith(k + "_")), key=len)
                    if variant not in source_counts:
                        source_counts[variant] = len(configurations(settings["spaces"][variant]))
                    count, budget = source_counts[variant], settings["source_trials"]
                    note = (f"Exhaustive search over all {count} effective settings in the recorded grid."
                            if count <= budget else
                            f"{budget} distinct effective settings sampled reproducibly from "
                            f"{count} settings in the recorded grid. This is a budgeted search, "
                            "not an exhaustive grid search.")
                method["search_description"] = note
        best = {}
        for trial in trials:
            if trial["label"] not in best or trial["loss"] < best[trial["label"]]["loss"]:
                best[trial["label"]] = trial
        saved = {row["id"]: row for row in results}
        return {"config": self.config, **catalog,
            "status": self.read_optional(self.run_root / "status.json", {"state": "not_started"}),
            "acquisition": self.read_optional(self.data_root / "acquisition_status.json", {}),
            "rooms": [{"id": r["id"], "polygon_xy": r["polygon_xy"], "ceiling_m": r["ceiling_m"],
                "width_m": r["width_m"], "depth_m": r["depth_m"],
                "aps": r["access_points_xyz"],
                "layouts": [layout["id"] for layout in r["layouts"]],
                "query_layouts": [{"id": layout["id"], "box_count": layout["box_count"],
                    "total": len(layout["queries_xyz"]),
                    "calibration": int(len(layout["queries_xyz"]) *
                                       self.config["observations"]["query_calibration_fraction"])}
                    for layout in r["layouts"]],
                "survey_xyz": r["survey_xyz"][survey_indices(r, **self.config["survey"])],
                "dense_positions": len(r["survey_xyz"])} for r in self.observations.rooms],
            "trials": trials, "calibration": [saved.get(row["id"], row) for row in best.values()],
            "evaluation": [r for r in results if r["split"] == "evaluation"],
            "cache": self.store.usage()}

    def receipt(self, key):
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("Invalid result ID")
        path = self.run_root / "evaluations" / (key + ".json")
        if path.exists():
            return read_json(path)
        for trial in read_trials(self.run_root / "trials"):
            if trial["id"] == key:
                return trial
        raise KeyError("Unknown trial")

    def check_replay_version(self):
        version = read_json(self.run_root / "implementation.json")
        if self.config.get("engine") == "range_rmse":
            root = Path(__file__).parents[1]
            if (version["solver"] != digest((root / "science/range_rmse.py").read_text())
                    or version["experiment"] != digest((root / "execution/range_rmse_experiment.py").read_text())
                    or version["dataset"] != digest(read_json(self.data_root / "dataset.json"))):
                raise ValueError("Range-RMSE solver or dataset differs from this saved run")
            return
        if "compositions" in version:
            from .compositions import composition_id
            if version["compositions"] != composition_id():
                raise ValueError("Composition implementation changed; use its recorded version")
        if "additional" in version:
            from .additional import extension_id
            if version["additional"] != extension_id():
                raise ValueError("Additional experiment implementation changed; use its recorded version")
        if (version["id"] != implementation_id()
                or version["experiment"] != digest(Path(__file__).with_name("experiment.py").read_text())
                or version["dataset"] != digest(read_json(self.data_root / "dataset.json"))):
            raise ValueError("This run used another implementation or dataset; exact replay requires that version")

    def replay(self, key, *, refresh=False):
        receipt = self.receipt(key)
        # Saved outcomes need no numerical replay and remain readable after code edits.
        if not refresh and receipt.get("outcome"):
            return read_npz(self.run_root / receipt["outcome"])[0]
        if "view_method_id" in receipt and self.config.get("engine") == "obstacle_reconstruction":
            # Recreate only the requested frozen source stage on the actual
            # obstacle inputs; the ordinary evaluator uses empty-room surveys.
            from .obstacle_reconstruction import layout_inputs, reconstruct
            detail = read_npz(self.run_root / receipt["outcome"])[0]
            record, = detail["sources"]
            room = self.rooms[record["room"]]
            layout = next(x for x in room["layouts"] if x["id"] == record["layout"])
            source = next(x["source"] for x in read_json(self.run_root / "protocol.json")["methods"]
                          if x["id"] == receipt["view_method_id"])
            with self.lock:
                fitted = reconstruct(source, layout_inputs(self.observations, room, layout), self.evaluator,
                    {ap: len(values) for ap, values in record["truth_catalog"].items()}, room["id"])
            record["by_ap"], record["audit"] = fitted.by_ap, fitted.evidence
            return detail
        self.check_replay_version()
        def calculate():
            return self.evaluator.evaluate(receipt["method"], receipt["parameters"],
                unify=receipt["unify"], level=receipt["level"], split=receipt["split"],
                frozen=receipt["frozen"], retain=True)
        with self.lock:
            if refresh:
                return calculate()["detail"]
            value, _ = self.audit.compute("trial-replay.1", {},
                {"receipt": {k: receipt[k] for k in ("method", "parameters", "unify", "level", "split", "frozen")},
                 "dataset": self.observations.manifest}, calculate)
            return value["detail"]

    def result(self, key):
        receipt, detail = self.receipt(key), self.replay(key)
        repetitions = self.config["observations"][receipt["split"] + "_repetitions"]
        def realization(record):
            return repetitions.index(record["repetition"]) + 1
        cases = []
        for record in detail["sources"]:
            truth = ({ap: np.asarray([s["xyz"] for s in values]) for ap, values in record["truth_catalog"].items()}
                     if "truth_catalog" in record else self.truths[record["room"]].by_ap)
            cases.append({"title": ((str(record["obstacle_count"]) + " obstacles / " + record["layout"] + " / ")
                                    if "layout" in record else "Sources / ") + "noise realization " + str(realization(record)),
                "realization": realization(record),
                "kind": "sources", "room": record["room"], "predicted": record["by_ap"],
                "repetition": record["repetition"], "truth": truth,
                **({"layout": record["layout"], "obstacles": record["obstacles"],
                    "input_positions": len(record["survey_xyz"])} if "layout" in record else {}),
                "audit": preview(record["audit"]) if receipt["method"] == "range_rmse_true_count" else record["audit"],
                "maps": {ap: self.source_map(record, ap) for ap in ("A", "B", "A+B")}})
        for record in detail["queries"]:
            truth = query_truth(record, self.rooms)
            room = self.rooms[record["room"]]
            layout = next(item for item in room["layouts"] if item["id"] == record["layout"])
            boxes = layout.get("boxes", [])
            survey = room["survey_xyz"][survey_indices(room, **self.config["survey"])] if boxes else []
            covered = inside_obstacles(survey, boxes)
            for kind, field in (("localization", "xyz"), ("initial", "initial_xyz")):
                errors = np.linalg.norm(record[field] - truth, axis=1)
                cases.append({"title": f"{len(boxes)} obstacles / {record['layout']} / noise realization {realization(record)}",
                    "realization": realization(record),
                    "kind": kind, "room": record["room"], "layout": record["layout"],
                    "obstacles": boxes, "covered_survey_indices": covered,
                    "inside_obstacle_estimates": inside_obstacles(record[field], boxes),
                    "query_ids": record["query_ids"],
                    "repetition": record["repetition"], "predicted": record[field],
                    "truth": truth, "errors": errors, "metrics": localization_metrics(errors),
                    "audit": record["audit"] if kind == "localization" else record["audit"]["initial"]})
        return {"receipt": receipt, "cases": cases, "sources": preview(detail["sources"]),
                "seconds": receipt.get("seconds")}

    def source_map(self, record, ap):
        room = self.rooms[record["room"]]
        predicted, truth, channels = [], [], []
        for name in ap.split("+"):
            points = record["by_ap"][name]
            channels.append({"ap": name, "start": len(predicted), "count": len(points)})
            predicted.extend(points)
            truth.extend(record.get("truth_catalog", self.truth_catalog[room["id"]])[name])
        predicted = np.asarray(predicted).reshape(-1, 3)
        catalogue = record.get("truth_catalog", self.truth_catalog[room["id"]])
        metrics = ap_source_metrics(record["by_ap"],
            {name: np.asarray([s["xyz"] for s in catalogue[name]]).reshape(-1, 3)
             for name in ap.split("+")}, ap.split("+"))
        return {"ap": ap, "predicted": predicted.ravel(), "truth": truth, "channels": channels,
                "matches": metrics["matches"], "metrics": metrics,
                "scene": {"polygon_xy": room["polygon_xy"], "ceiling_m": room["ceiling_m"],
                    "access_points_xyz": room["access_points_xyz"],
                    "survey_xyz": record.get("survey_xyz", room["survey_xyz"][survey_indices(room, **self.config["survey"])]),
                    "obstacles": record.get("obstacles", [])}}

    def artifact(self, key, trial, field="", start=0):
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("Invalid artifact ID")
        try:
            value = self.store.get(key)
        except FileNotFoundError:
            self.replay(trial, refresh=True)
            try:
                value = self.store.get(key)
            except FileNotFoundError:
                raise ValueError("Artifact was not retained within the cache budget; inspect a smaller stage") from None
        if field:
            for part in field.strip("/").split("/"):
                value = value[int(part)] if isinstance(value, list) else value[part]
            if not isinstance(value, np.ndarray):
                raise ValueError("Requested field is not an array")
        return {"id": key, "value": preview(value, field=field, start=max(0, start))}

    def match(self, trial, ap, left, right, kind="survey", layout=None, room_id=None, repetition=None):
        receipt = self.receipt(trial)
        obstacle = self.config.get("engine") == "obstacle_reconstruction"
        if not obstacle:
            from .peak_explorer import check_matching_data
            check_matching_data(self)
        room = self.rooms[room_id] if room_id else self.observations.rooms[0]
        repetitions = self.config["observations"][receipt["split"] + "_repetitions"]
        repetition = repetitions[0] if repetition is None else int(repetition)
        if repetition not in repetitions:
            raise ValueError("Noise repetition is outside this result's split")
        if obstacle:
            if kind != "survey":
                raise ValueError("Obstacle reconstruction contains survey/survey inputs only")
            from .obstacle_reconstruction import layout_inputs
            selected = next(x for x in room["layouts"] if x["id"] == receipt["layout"])
            survey = layout_inputs(self.observations, room, selected)
        else:
            survey = self.observations.survey(room, receipt["level"], repetition)
        if not 0 <= right < len(survey.xyz) or left < 0:
            raise ValueError("Fingerprint row is outside the selected survey/query set")
        separation = None
        if kind == "survey":
            if left >= len(survey.xyz):
                raise ValueError("Survey row is out of bounds")
            from .compositions import VARIANTS, composed_reconstructor
            variant = receipt["method"].removesuffix("_joint").removesuffix("_lea")
            if variant in VARIANTS:
                if not variant.startswith("mpurge"):
                    raise ValueError("This proposer does not use MPUrge survey peak matching")
                parameters = (receipt["parameters"] if receipt["method"] == variant else
                    receipt["parameters"]["source"] if receipt["method"].endswith("_joint") else
                    receipt["frozen"]["source"])
                algorithm = composed_reconstructor(variant, parameters, unify=receipt["unify"])
                matcher = (algorithm if VARIANTS[variant][1] is None else algorithm.proposals).matcher
            else:
                parameters = receipt["parameters"] if receipt["method"] == "mpurge" else (
                    receipt["parameters"]["mpurge"] if receipt["method"] == "joint" else receipt["frozen"]["mpurge"])
                matcher = standard_mpurge(parameters, unify=receipt["unify"]).matcher
            a, b = survey.channels[ap][left], survey.channels[ap][right]
            separation = float(np.linalg.norm(survey.xyz[left] - survey.xyz[right]))
            result = matcher.match(a, b, separation=separation)
        elif kind == "query":
            parameters = receipt["parameters"] if receipt["method"] == "mpurge_map" else (
                receipt["parameters"]["map"] if (receipt["method"] == "joint" or receipt["method"].endswith("_joint")) else receipt["frozen"]["map"])
            selected_layout = next(item for item in room["layouts"] if str(item["id"]) == str(layout))
            queries = self.observations.queries(room, selected_layout, receipt["level"], repetition, receipt["split"])
            if left >= len(queries.xyz):
                raise ValueError("Query row is out of bounds")
            a, b = queries.channels[ap][left], survey.channels[ap][right]
            matcher = mpurge_map(parameters, unify=receipt["unify"]).scorer.matcher
            result = matcher.match(a, b)
        else:
            raise ValueError("Match kind must be survey or query")
        # Reuse the scientific scorer on exactly the matcher's unified first-round inputs.
        # Excluded peaks stay in their original positions, represented by null matrix cells.
        first_round = np.full((len(a), len(b)), np.nan)
        ia, ib = result["left_indices"], result["right_indices"]
        if len(ia) and len(ib):
            distance = separation if matcher.triangle_tolerance_m is not None else -1.
            first_round[np.ix_(ia, ib)] = _delta(a[ia], b[ib], matcher.half_window,
                matcher.alpha, matcher.normalized_pattern, distance,
                matcher.triangle_tolerance_m or 0.)
        return {"left_ranges_m": a, "right_ranges_m": b, **result,
                "first_round_costs": first_round, "parameters": parameters, "unify": receipt["unify"],
                "match_columns": ["left_peak", "right_peak", "cost"],
                "trace_columns": ["left_peak", "right_peak", "cost", "round", "accepted"]}


class MultiStudyView:
    """Read compatible sibling studies without copying their logs or outcomes."""
    def __init__(self, data_root, run_root):
        self.data_root, self.run_root = Path(data_root), Path(run_root)
        self.base = ExperimentView(data_root, run_root)
        self.views = [self.base]

    def discover(self):
        if self.base.config["name"] == "rectangle-clean-selection-v4":
            matched = self.run_root.parent / "rectangle-matched-obstacle-reconstruction-v1"
            obstacle_name = (matched.name if (matched / "implementation.json").exists()
                             else "rectangle-obstacle-reconstruction-v1")
            if obstacle_name == matched.name:
                self.views = [v for v in self.views if v.run_root.name != "rectangle-obstacle-reconstruction-v1"]
            for name in ("rectangle-additional-selection-v1", "rectangle-compositions-v1",
                         "rectangle-naked-selection-v1", "rectangle-range-rmse-true-count-v1",
                         "rectangle-allow-crossing-solo-v1", "rectangle-allow-crossing-consumption-v1",
                         "rectangle-allow-crossing-robust-v1",
                         "rectangle-stars-v1",
                         obstacle_name):
                extra = self.run_root.parent / name
                if not (extra / "implementation.json").exists() or any(v.run_root == extra for v in self.views):
                    continue
                version = read_json(extra / "implementation.json")
                original = read_json(self.run_root / "implementation.json")
                if version.get("base_dataset", version["dataset"]) != original["dataset"]:
                    raise ValueError("Additional study uses another dataset")
                config = read_json(extra / "config.json")
                if any(config[k] != self.base.config[k] for k in ("seed", "survey", "observations")):
                    raise ValueError("Additional study uses a different observation protocol")
                study_data = Path(config.get("data_root", self.data_root))
                if digest(read_json(study_data / "dataset.json")) != version["dataset"]:
                    raise ValueError("Additional study dataset differs from its recorded inputs")
                self.views.append(ExperimentView(study_data, extra))

    def overview(self):
        self.discover()
        snapshots = [v.overview() for v in self.views]
        result = dict(snapshots[0])
        for key in ("trials", "calibration", "evaluation", "methods", "conditions"):
            result[key] = [{**r, "run": v.config["name"]}
                           for v, s in zip(self.views, snapshots) for r in s[key]]
        result["studies"] = [{"name": v.config["name"], **s["status"]}
                             for v, s in zip(self.views, snapshots)]
        result["obstacle_reconstruction_protocol"] = next((s["obstacle_reconstruction_protocol"]
            for s in snapshots if "obstacle_reconstruction_protocol" in s), None)
        statuses = [s["status"] for s in snapshots]
        active = next((s for s in statuses if s["state"] != "complete"), statuses[0])
        result["status"] = {**active,
            "completed_trials": sum(s.get("completed_trials", 0) for s in statuses),
            "target_trials": sum(s.get("target_trials", 0) for s in statuses)}
        return result

    def owner(self, key):
        self.discover()
        for view in self.views:
            try:
                view.receipt(key)
                return view
            except KeyError:
                pass
        raise KeyError("Unknown trial")

    def result(self, key):
        return self.owner(key).result(key)

    def artifact(self, key, trial, field="", start=0):
        return self.owner(trial).artifact(key, trial, field, start)

    def match(self, trial, *args):
        return self.owner(trial).match(trial, *args)


def source_page(relative):
    root = Path(__file__).parents[1].resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.suffix != ".py" or not path.is_file():
        raise ValueError("Unknown source file")
    lines = path.read_text(encoding="utf-8").splitlines()
    code = "\n".join(f'<span id="L{i}"><a href="#L{i}">{i:4}</a> {html.escape(line)}</span>'
                     for i, line in enumerate(lines, 1))
    title = html.escape(relative)
    return f'''<!doctype html><html><meta charset="utf-8"><title>{title} · Indoor VT source</title>
    <style>body{{margin:0;font:14px Segoe UI,sans-serif;background:#f4f6f7;color:#20333b}}
    header{{padding:20px 28px;background:#142b32;color:white;position:sticky;top:0}}
    header a{{color:#9cdec9}}pre{{padding:20px 28px;line-height:1.7;font-size:12px}}
    pre span{{scroll-margin-top:150px}}span:target{{background:#d9efdf}}pre a{{color:#80988d;text-decoration:none}}</style>
    <header><a href="/">← Experiment explorer</a><h2>{title}</h2>Current local implementation</header>
    <pre>{code}</pre></html>'''.encode()


def serve(data_root, run_root, port=8770):
    view = MultiStudyView(data_root, run_root)
    from .peak_explorer import PeakExplorer
    peaks = PeakExplorer(view)
    true_matching = None
    asset_root = Path(__file__).with_name("web")
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(False)

        def do_POST(self):
            self.respond(True)

        def respond(self, post):
            nonlocal true_matching
            request = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(request.query).items()}
            try:
                if request.path == "/api/overview" and not post:
                    payload = view.overview()
                elif request.path == "/api/result" and post:
                    payload = view.result(query["id"])
                elif request.path == "/api/artifact" and post:
                    payload = view.artifact(query["id"], query["trial"], query.get("field", ""),
                                            int(query.get("start", 0)))
                elif request.path == "/api/match" and post:
                    payload = view.match(query["trial"], query.get("ap", "A"),
                        int(query["left"]), int(query["right"]), query.get("kind", "survey"),
                        query.get("layout"), query.get("room"), query.get("repetition"))
                elif request.path == "/api/peaks" and post:
                    payload = peaks.request(**query)
                elif request.path == "/api/true-matching" and not post:
                    if true_matching is None:
                        from .true_matching import TrueMatchingAnalysis
                        true_matching = TrueMatchingAnalysis(view)
                    payload = true_matching.request()
                elif request.path in ('/construction', '/api/construction') and not post:
                    from .presentation import construction, construction_page
                    owner = view.owner(query['trial'])
                    payload = construction(owner.receipt(query['trial']), owner.config,
                                           query.get('component', 'sources'))
                    if request.path == '/construction':
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/html; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(construction_page(payload))
                        return
                elif request.path == "/source" and not post:
                    content = source_page(query["file"])
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(content)
                    return
                elif request.path.startswith("/vendor/katex/") and not post:
                    vendor_root = (asset_root / "vendor" / "katex").resolve()
                    path = (asset_root / request.path[1:]).resolve()
                    mime = {".mjs": "text/javascript", ".css": "text/css", ".woff2": "font/woff2",
                            ".woff": "font/woff", ".ttf": "font/ttf", ".txt": "text/plain"}
                    if not path.is_relative_to(vendor_root) or path.suffix not in mime or not path.is_file():
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", mime[path.suffix])
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(path.read_bytes())
                    return
                elif request.path in ("/", "/app.js", "/pareto.mjs", "/style.css", "/explorer.css", "/vt-map.js",
                        "/site-runtime.js", "/finisher-explanations.js", "/finisher-explanations.css",
                        "/matching.html", "/peak-matching.js", "/peak-matching.css", "/peak-spatial.js", "/peak-distributions.js",
                        "/true-matching.html", "/true-matching.js", "/true-matching.css") and not post:
                    name = "index.html" if request.path == "/" else request.path[1:]
                    content = (asset_root / name).read_bytes()
                    mime = {"html": "text/html", "js": "text/javascript", "mjs": "text/javascript",
                            "css": "text/css"}[name.rsplit(".", 1)[1]]
                    self.send_response(200)
                    self.send_header("Content-Type", mime + "; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(content)
                    return
                else:
                    self.send_error(404)
                    return
                body = json.dumps(json_value(payload), allow_nan=False).encode()
                self.send_response(200)
            except (ValueError, KeyError, FileNotFoundError, IndexError, StopIteration) as error:
                body = json.dumps({"error": str(error)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            if args and str(args[1]) != "200":
                super().log_message(format, *args)
    print(f"Redesign viewer: http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
