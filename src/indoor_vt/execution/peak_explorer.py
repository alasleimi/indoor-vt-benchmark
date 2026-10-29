"""Old peak explorer data contract, backed by current selected scientific matchers."""
from collections import OrderedDict
import threading
import numpy as np

from .presets import standard_mpurge
from ..acquisition.dataset import survey_indices
from ..acquisition.records import digest, read_json
from ..acquisition.observations import direct_ranges
from ..acquisition.channel import angle_to_unit
from ..science.grouping import all_pairs_within, nearest_neighborhoods
from ..science.matching import _delta


def check_matching_data(view):
    """A pair diagnostic uses saved inputs/settings, not a full experiment replay.

    Source revisions describe the original run; unrelated reconstruction,
    finisher, optimization or metric edits cannot invalidate this diagnostic.
    Traces and scores are calculated together by the current matcher, never
    spliced into saved trial outcomes. The original dataset must still agree.
    """
    manifest = read_json(view.data_root / "dataset.json")
    if manifest.get("schema") != "vt.dataset.2":
        raise ValueError("Unsupported dataset schema for the matching diagnostic")
    version = read_json(view.run_root / "implementation.json")
    if version["dataset"] != digest(manifest):
        raise ValueError("Matching dataset differs from this saved run")


def quality(a, b, matches):
    available = int(np.sum(a["source_ids"][:, None] == b["source_ids"][None, :]))
    correct = sum(a["source_ids"][int(i)] == b["source_ids"][int(j)] for i, j in matches)
    predicted = len(matches)
    return {"tp": int(correct), "fp": int(predicted-correct), "fn": int(available-correct),
            "available": available, "precision": correct/predicted if predicted else 0.,
            "recall": correct/available if available else 0.,
            "f1": 2*correct/(predicted+available) if predicted+available else 0.}


def pair_data(a, b, matcher, sources):
    separation = (float(np.linalg.norm(np.asarray(a["xyz"]) - b["xyz"]))
                  if matcher.triangle_tolerance_m is not None else -1.)
    result = matcher.match(a["ranges"], b["ranges"], separation=separation)
    ia, ib = result["left_indices"], result["right_indices"]
    values = np.full((len(a["ranges"]), len(b["ranges"])), np.nan)
    pattern = values.copy()
    if len(ia) and len(ib):
        args = (a["ranges"][ia], b["ranges"][ib], matcher.half_window)
        values[np.ix_(ia, ib)] = _delta(*args, matcher.alpha, matcher.normalized_pattern,
                                      separation, matcher.triangle_tolerance_m or 0.)
        pattern[np.ix_(ia, ib)] = _delta(*args, 0., matcher.normalized_pattern,
                                       separation, matcher.triangle_tolerance_m or 0.)
    trace = result["trace"]
    first = trace[(trace[:, 3] == 1) & (trace[:, 4] == 1), :2].astype(int)
    final = trace[trace[:, 4] == 1, :2].astype(int)
    return {"a": a, "b": b, "sources": sources,
        "matrix": {"ia": ia, "ib": ib, "values": values.ravel(), "pattern": pattern.ravel(),
                   "matches": first, "rejected": trace[(trace[:, 3] == 1) & (trace[:, 4] == 0), :2].astype(int)},
        "final": {"matches": final, "rounds": trace[trace[:, 4] == 1, 3].astype(int),
                  "rejected": trace[trace[:, 4] == 0][:, [0, 1, 3]].astype(int)},
        "quality": quality(a, b, first), "finalQuality": quality(a, b, final)}


def moments(x, y):
    return np.array([x.size, y.sum(), x.sum(), np.square(x).sum(), x[y].sum()])


def correlation(m):
    n, pos, sx, sxx, sxy = m
    denominator = (n*sxx-sx*sx)*(n*pos-pos*pos)
    return float((n*sxy-sx*pos)/np.sqrt(denominator)) if denominator > 0 else None


def spatial_match_counts(a, b, matcher):
    """Compact counts for both stages, including truth lost to size unification.

    Count columns are correct matches, selected matches, available true pairs,
    and all candidate cells. No truth labels enter the scientific matcher.
    """
    separation = float(np.linalg.norm(np.asarray(a["xyz"]) - b["xyz"]))
    trace = matcher.match(a["ranges"], b["ranges"], separation=separation)["trace"]
    truth = a["source_ids"][:, None] == b["source_ids"][None, :]
    accepted = trace[trace[:, 4] == 1]
    counts = []
    for edges in (accepted[accepted[:, 3] == 1], accepted):
        left, right = edges[:, :2].astype(int).T
        counts.append([int(truth[left, right].sum()), len(edges), int(truth.sum()), truth.size])
    return counts


class PeakExplorer:
    def __init__(self, studies):
        self.studies, self.cache = studies, OrderedDict()
        self.lock = threading.RLock()

    def context(self, trial):
        if not trial:
            trial = read_json(self.studies.base.run_root / "selected/mpurge_ununified.json")["id"]
        view = self.studies.owner(trial)
        if view.config.get("engine") == "obstacle_reconstruction":
            raise ValueError("Use the obstacle result's pair diagnostic; room-wide source labels require its layout truth")
        check_matching_data(view)
        receipt = view.receipt(trial)
        if view.config["observations"]["track"] != "direct":
            raise ValueError("The native-source matching diagnostic requires direct observations")
        from .compositions import VARIANTS, composed_reconstructor
        variant = receipt["method"].removesuffix("_joint").removesuffix("_lea")
        if variant.startswith('mpurge_stars_'):
            from .star_experiment import star_reconstructor
            p = receipt['parameters']
            matcher = star_reconstructor(p, variant.removeprefix('mpurge_stars_')).proposals.matcher
        elif variant in VARIANTS and variant.startswith("mpurge"):
            p = (receipt["parameters"] if receipt["method"] == variant else
                 receipt["parameters"]["source"] if receipt["method"].endswith("_joint") else
                 receipt["frozen"]["source"])
            algorithm = composed_reconstructor(variant, p, unify=receipt["unify"])
            matcher = (algorithm if VARIANTS[variant][1] is None else algorithm.proposals).matcher
        else:
            if receipt["method"] not in ("mpurge", "joint", "smart_lea"):
                raise ValueError("The room-wide diagnostic requires an MPUrge reconstruction configuration")
            p = (receipt["parameters"] if receipt["method"] == "mpurge" else
                 receipt["parameters"]["mpurge"] if receipt["method"] == "joint" else receipt["frozen"]["mpurge"])
            matcher = standard_mpurge(p, unify=receipt["unify"]).matcher
        room = view.observations.rooms[0]
        indices = survey_indices(room, **view.config["survey"])
        xyz = room["survey_xyz"][indices]
        pairs = all_pairs_within(nearest_neighborhoods(xyz, size=p["neighbors"]))
        return view, receipt, p, room, indices, xyz, pairs, matcher

    def profiles(self, context, ap, noise, repetition=None):
        view, receipt, p, room, indices, xyz, pairs, matcher = context
        if repetition is None:
            repetition = view.config["observations"][receipt["split"] + "_repetitions"][0]
        else:
            allowed = (view.config["observations"].get("calibration_repetitions", []) +
                       view.config["observations"].get("evaluation_repetitions", []))
            if repetition not in allowed:
                raise ValueError("Unknown saved noise realization")
        if noise not in view.config["observations"]["noise_levels"] or ap not in ("A", "B"):
            raise ValueError("Unknown matching condition")
        key = (receipt["id"], ap, noise, repetition)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        sources = view.truth_catalog[room["id"]][ap]
        truth = np.array([s["xyz"] for s in sources])
        wanted = {int(position): i for i, position in enumerate(indices)}
        profiles = [None] * len(indices)
        for job in view.observations.manifest["jobs"]:
            if (job["room"], job["ap"], job["group"]) != (room["id"], ap, "survey"):
                continue
            if not wanted.keys() & set(job["position_ids"]):
                continue
            acquisition = read_json(view.data_root / "native" / job["key"] / "receipt.json")
            native = view.observations._native(acquisition["native_value"])
            for j, position in enumerate(native["position_ids"]):
                if int(position) not in wanted:
                    continue
                i = wanted[int(position)]
                paths = native["paths"][native["offsets"][j]:native["offsets"][j+1]]
                reconstructed = xyz[i] + paths[:, 0, None] * 3e8 * angle_to_unit(paths[:, 3], paths[:, 4])
                distance = np.linalg.norm(reconstructed[:, None, :] - truth[None, :, :], axis=2)
                if not np.all((distance < .01).sum(axis=1) == 1):
                    raise ValueError("Native path source identity is ambiguous")
                labels = distance.argmin(axis=1)
                if len(set(labels)) != len(labels):
                    raise ValueError("Duplicate native source identities need explicit matching semantics")
                seed = view.observations.seeds.seed("measurement", room["id"], ap, "survey",
                                                    int(position), "direct", noise, repetition)
                observed = direct_ranges(paths[:, 0], noise, seed)
                order = observed["native_path_indices"]
                profiles[i] = {"reference": i, "ranges": observed["ranges"], "source_ids": labels[order],
                               "gains": paths[order, 1], "xyz": xyz[i]}
        if any(p is None for p in profiles):
            raise ValueError("Survey acquisition is incomplete")
        actual = view.observations.survey(room, noise, repetition).channels[ap]
        if not all(np.array_equal(profile["ranges"], actual[i]) for i, profile in enumerate(profiles)):
            raise ValueError("Diagnostic does not reproduce the saved observation conversion")
        value = {"profiles": profiles, "sources": sources, "summaries": {}}
        self.cache[key] = value
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return value

    def summary(self, context, data, scope):
        if scope in data["summaries"]:
            return data["summaries"][scope]
        pairs, matcher = context[-2:]
        totals, candidate, accepted = np.zeros(3, dtype=int), np.zeros(5), np.zeros(5)
        counts, bins = [], {}
        for i, j in pairs:
            pair = pair_data(data["profiles"][i], data["profiles"][j], matcher, data["sources"])
            q = pair["finalQuality"] if scope == "final" else pair["quality"]
            row = [q["tp"], q["tp"]+q["fp"], q["available"]]
            totals += row
            counts.append(row)
            x = pair["matrix"]["values"]
            y = (pair["a"]["source_ids"][:, None] == pair["b"]["source_ids"][None, :]).ravel()
            finite = np.isfinite(x)
            candidate += moments(x[finite], y[finite])
            matches = pair["final"]["matches"] if scope == "final" else pair["matrix"]["matches"]
            ids = matches[:, 0]*len(pair["b"]["ranges"]) + matches[:, 1]
            accepted += moments(x[ids], y[ids])
            bin_ids = np.floor(x[finite]/.25).astype(int)
            for bin_id in np.unique(bin_ids):
                mask = bin_ids == bin_id
                record = bins.setdefault(int(bin_id), np.zeros(3))
                record += [mask.sum(), y[finite][mask].sum(), x[finite][mask].sum()]
        tp, n, available = map(int, totals)
        summary = {"tp": tp, "fp": n-tp, "fn": available-tp, "precision": tp/n if n else 0.,
                   "recall": tp/available if available else 0., "f1": 2*tp/(n+available) if n+available else 0.,
                   "pairs": len(pairs), "candidate_cells": int(candidate[0]),
                   "candidate_same_source": int(candidate[1]), "candidate_correlation": correlation(candidate),
                   "accepted_correlation": correlation(accepted)}
        result = {"summary": summary, "pair_counts": counts,
                  "bins": [[bid*.25, float(v[2]/v[0]), int(v[0]), int(v[1])] for bid, v in sorted(bins.items())]}
        data["summaries"][scope] = result
        return result

    def spatial(self, context, data):
        """Evaluate all unordered survey pairs once, for spatial diagnostics only."""
        if "spatial" not in data:
            xyz, local_pairs, matcher = context[-3:]
            local = {tuple(map(int, pair)) for pair in local_pairs}
            pairs = np.column_stack(np.triu_indices(len(xyz), 1))
            counts = np.array([spatial_match_counts(data["profiles"][i], data["profiles"][j], matcher)
                               for i, j in pairs], dtype=np.int64).reshape(-1, 2, 4)
            data["spatial"] = {
                "pairs": pairs,
                "distance_m": np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1),
                "neighborhood": [tuple(map(int, pair)) in local for pair in pairs],
                "count_columns": ["correct", "selected", "true", "candidate_cells"],
                "first_counts": counts[:, 0], "final_counts": counts[:, 1],
            }
        return data["spatial"]

    def request(self, action, trial="", ap="A", noise="0", left="0", right="1", matches="first"):
        with self.lock:
            context = self.context(trial)
            view, receipt, p, room, indices, xyz, pairs, matcher = context
            if action == "init":
                overview = self.studies.overview()
                choices = [{"id": m["selected_id"], "title": m["title"]} for m in overview["methods"]
                           if m["group"] == "vt" and m.get("family") == "mpurge"]
                return {"meta": {"room": room["id"], "room_label": room["id"], "noise_levels": view.config["observations"]["noise_levels"],
                    "trial": receipt["id"], "run": view.config["name"], "split": receipt["split"],
                    "repetition": view.config["observations"][receipt["split"] + "_repetitions"][0],
                    "choices": choices, **p, "parameters": p, "unify": receipt["unify"],
                    "matcher": type(matcher).__name__,
                    "triangle_tolerance_m": matcher.triangle_tolerance_m},
                    "scene": {"polygon_xy": room["polygon_xy"], "ceiling_m": room.get("ceiling_m"), "aps": room["access_points_xyz"],
                              "survey_xyz": xyz, "pairs": pairs}}
            noise = float(noise)
            scope = "final" if matches == "final" else "first"
            if action == "summaries":
                return [{"ap": a, "noise": n, "summary": self.summary(context, self.profiles(context, a, n), scope)["summary"]}
                        for a in ("A", "B") for n in view.config["observations"]["noise_levels"]]
            data = self.profiles(context, ap, noise)
            if action == "condition":
                return self.summary(context, data, scope)
            if action == "spatial":
                return self.spatial(context, data)
            if action == "distributions":
                from .peak_distributions import dissimilarity_distributions
                if "distributions" not in data:
                    data["distributions"] = dissimilarity_distributions(data["profiles"], pairs, matcher)
                return data["distributions"]
            if action != "pair":
                raise ValueError("Unknown peak diagnostic action")
            left, right = sorted((int(left), int(right)))
            if left < 0 or right >= len(xyz) or left == right:
                raise ValueError("Choose two different valid survey positions")
            return pair_data(data["profiles"][left], data["profiles"][right], matcher, data["sources"])
