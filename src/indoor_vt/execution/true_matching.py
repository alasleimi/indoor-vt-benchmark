"""Count truth correspondence properties on frozen empty-room observations.

Source identities are used only here, never passed to the production matcher.
The analysis cache follows the data/settings/semantics, not a repository hash.
"""
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import time

import numpy as np
from numba import njit

from .peak_explorer import PeakExplorer
from ..acquisition.records import digest, file_hash, read_json, write_json
from ..science.models import Ragged, Survey
from ..science.reconstruction import match_pairs


SCHEMA = "true-matching-analysis.2"
TOLERANCE_M = 1e-10  # Roundoff only, not a noise-dependent geometric allowance.
TRUTH_FIELDS = ("available", "same_rank", "cross_any", "difference_satisfies", "sum_satisfies")
MATCH_FIELDS = ("total", "correct", "same_rank_correct", "same_rank_incorrect",
                "difference_violation_correct", "difference_violation_incorrect",
                "sum_violation_correct", "sum_violation_incorrect") + tuple(
    bound + "_" + field for bound in ("difference", "sum") for field in (
        "violation_edges", "counterpart_violates", "counterpart_satisfies", "counterpart_unavailable",
        "endpoint_disagreement_edges"))


def profile_arrays(profiles, source_count):
    """Full observed-fingerprint rank, with -1 for unobserved source identities."""
    ranks = np.full((len(profiles), source_count), -1, dtype=np.int64)
    ranges = np.full((len(profiles), source_count), np.nan)
    for i, profile in enumerate(profiles):
        values, labels = np.asarray(profile["ranges"]), np.asarray(profile["source_ids"])
        if len(np.unique(labels)) != len(labels):
            raise ValueError("True correspondence requires unique source IDs per fingerprint")
        if np.any(np.diff(values) < 0):
            raise ValueError("Profiles must retain the stable observed-range ordering")
        ranks[i, labels] = np.arange(len(values))
        ranges[i, labels] = values
    return ranks, ranges


@njit(cache=True)
def _truth_counts(ranks, ranges, xyz, pairs, tolerance):
    counts = np.zeros(5, np.int64)
    for a, b in pairs:
        distance = np.sqrt(np.sum((xyz[a] - xyz[b]) ** 2))
        for source in range(ranks.shape[1]):
            ia, ib = ranks[a, source], ranks[b, source]
            if ia < 0 or ib < 0:
                continue
            counts[0] += 1
            counts[1] += ia == ib
            for other in range(ranks.shape[1]):
                ja, jb = ranks[a, other], ranks[b, other]
                if ja >= 0 and jb >= 0 and (ia - ja) * (ib - jb) < 0:
                    counts[2] += 1
                    break
            x, y = ranges[a, source], ranges[b, source]
            counts[3] += abs(x - y) <= distance + tolerance
            counts[4] += x + y + tolerance >= distance
    return counts


def truth_counts(profiles, xyz, pairs, source_count, tolerance=TOLERANCE_M):
    ranks, ranges = profile_arrays(profiles, source_count)
    counts = _truth_counts(ranks, ranges, np.asarray(xyz), np.asarray(pairs), tolerance)
    return {**dict(zip(TRUTH_FIELDS, map(int, counts))), "pairs": len(pairs)}


def violation_classes(pa, pb, ia, ib, distance, bound, tolerance=TOLERANCE_M):
    """Classify both anchors of each violating output edge against their true partners.

    The two endpoint incidences give an orientation-independent partition. An
    edge can have different classes at its two ends. Missing truth is explicit.
    """
    def violates(x, y):
        if bound == "difference":
            return np.abs(x - y) > distance + tolerance
        if bound == "sum":
            return x + y + tolerance < distance
        raise ValueError("Unknown geometric bound")
    mask = violates(pa["ranges"][ia], pb["ranges"][ib])
    output = {"violation_edges": int(mask.sum()), "counterpart_violates": 0,
              "counterpart_satisfies": 0, "counterpart_unavailable": 0,
              "endpoint_disagreement_edges": 0}
    if not np.any(mask):
        return output
    classes = []
    for anchor, other, ids in ((pa, pb, ia), (pb, pa, ib)):
        labels = anchor["source_ids"][ids[mask]]
        comparisons = labels[:, None] == other["source_ids"][None, :]
        present = np.any(comparisons, axis=1)
        partner = np.argmax(comparisons[present], axis=1)
        bad = violates(anchor["ranges"][ids[mask]][present], other["ranges"][partner])
        category = np.full(len(labels), -1, dtype=int)
        category[present] = bad.astype(int)
        classes.append(category)
        output["counterpart_violates"] += int(bad.sum())
        output["counterpart_satisfies"] += int((~bad).sum())
        output["counterpart_unavailable"] += int((~present).sum())
    output["endpoint_disagreement_edges"] = int(np.sum(
        (classes[0] >= 0) & (classes[1] >= 0) & (classes[0] != classes[1])))
    return output


def final_match_counts(profiles, xyz, pairs, matcher, tolerance=TOLERANCE_M):
    """Call the exact match bank used by standard MPUrge reconstruction."""
    rows = Ragged.pack([p["ranges"] for p in profiles])
    graph = match_pairs(Survey(np.asarray(xyz), {"A": rows}, np.arange(len(xyz))),
                        pairs, matcher)["A"]
    counts = dict.fromkeys(MATCH_FIELDS, 0)
    for block, (a, b) in enumerate(pairs):
        edges = graph.endpoints[graph.offsets[block]:graph.offsets[block + 1]]
        ia, ib = (edges - graph.node_offsets[[a, b]]).T.astype(int)
        pa, pb = profiles[a], profiles[b]
        correct = pa["source_ids"][ia] == pb["source_ids"][ib]
        same_rank = ia == ib  # Full, sorted, currently observed fingerprints.
        x, y = pa["ranges"][ia], pb["ranges"][ib]
        distance = float(np.linalg.norm(xyz[a] - xyz[b]))
        diff_bad = np.abs(x - y) > distance + tolerance
        sum_bad = x + y + tolerance < distance
        counts["total"] += len(edges)
        counts["correct"] += int(correct.sum())
        for name, mask in (("same_rank", same_rank), ("difference_violation", diff_bad),
                           ("sum_violation", sum_bad)):
            counts[name + "_correct"] += int(np.sum(mask & correct))
            counts[name + "_incorrect"] += int(np.sum(mask & ~correct))
        for bound in ("difference", "sum"):
            for name, count in violation_classes(pa, pb, ia, ib, distance, bound, tolerance).items():
                counts[bound + "_" + name] += count
    return {**counts, "pairs": len(pairs)}


def pool_rows(rows, ap="A+B"):
    """Pool integer numerators/denominators, never average percentages."""
    if ap not in ("A", "B", "A+B"):
        raise ValueError("Unknown AP selection")
    result = []
    selected = [r for r in rows if r["ap"] in ap.split("+")]
    for noise in sorted({r["noise"] for r in selected}):
        group = [r for r in selected if r["noise"] == noise]
        row = {"noise": noise, "ap": ap, "realization_records": len(group),
               "repetitions": sorted({r["repetition"] for r in group})}
        for field in ("global", "neighborhood", "matches", "observations"):
            row[field] = {k: sum(r[field][k] for r in group) for k in group[0][field]}
        result.append(row)
    return result


def identity_accuracy(matches, bound):
    """Correctness among violating edges versus overall; empty groups are undefined."""
    def fraction(correct, total):
        return {"correct": correct, "incorrect": total - correct, "total": total,
                "fraction_correct": correct / total if total else None}
    return {"violating": fraction(matches[bound + "_violation_correct"],
                                   matches[bound + "_violation_edges"]),
            "overall": fraction(matches["correct"], matches["total"])}


def add_accuracy_comparisons(report):
    """Derived presentation statistics reuse saved counts; no matcher rerun."""
    if report.get("status", {}).get("state") == "complete":
        for rows in report["pooled"].values():
            for row in rows:
                row["identity_accuracy"] = {bound: identity_accuracy(row["matches"], bound)
                                            for bound in ("difference", "sum")}
        report["metric_definitions"] = {
            "violation_frequency": "violating final output edges / all final output edges",
            "violation_endpoint_partition": "compare each violating edge endpoint's true observed partner; denominator 2 * all final output edges",
            "violating_identity_accuracy": "correct violating output edges / violating output edges; null when no violating outputs",
            "overall_identity_accuracy": "correct final output edges / all final output edges",
            "correct": "both observed peaks have the exact same physical or virtual source ID"}
    return report


class TrueMatchingAnalysis:
    def __init__(self, studies):
        self.studies = studies
        self.peaks = PeakExplorer(studies)
        self.context = self.peaks.context("")
        view, receipt, parameters, room, indices, xyz, pairs, matcher = self.context
        if (receipt["method"] != "mpurge" or receipt["unify"] or receipt["level"] != 0
                or receipt["split"] != "calibration" or len(xyz) != 280):
            raise ValueError("Analysis requires the selected ununified empty-room MPUrge and 280 survey positions")
        self.global_pairs = np.asarray(list(combinations(range(len(xyz)), 2)), dtype=np.int64)
        native_inputs = []
        for job in view.observations.manifest["jobs"]:
            if (job["room"] == room["id"] and job["group"] == "survey"
                    and set(map(int, indices)).intersection(job["position_ids"])):
                record = read_json(view.data_root / "native" / job["key"] / "receipt.json")
                native_inputs.append({"job": job["key"], "native_value": record["native_value"]})
        self.signature = {"schema": SCHEMA, "dataset": digest(view.observations.manifest),
            "native_inputs": native_inputs, "room": room["id"], "survey_indices": indices.tolist(),
            "seed": view.config["seed"], "survey": view.config["survey"],
            "noise_levels": view.config["observations"]["noise_levels"],
            "repetitions": view.config["observations"]["evaluation_repetitions"],
            "track": "direct", "method": receipt["method"], "parameters": parameters,
            "unify": False, "tolerance_m": TOLERANCE_M,
            "truth": "shared observed exact source ID, including physical AP/direct source",
            "rank": "full observed range order; stable ties by original native path order",
            "crossing": "edge crosses at least one other true edge in the same fingerprint pair",
            "geometric_violation_classes": "both output-edge endpoints; compare each anchor's true partner predicate; missing partner explicit; denominator twice final output edges",
            "pair_orientation": "lower survey index first; unordered pairs, deduplicated",
            "matcher": "standard MPUrge all accepted rounds, original window pattern, no triangle gate"}
        self.key = digest(self.signature)
        self.root = view.run_root / "analysis" / "true_matching"
        self.path = self.root / (self.key + ".json")
        self.status_path = self.root / (self.key + ".status.json")

    def request(self):
        if self.path.exists():
            return add_accuracy_comparisons(read_json(self.path))
        status = read_json(self.status_path) if self.status_path.exists() else {"state": "not_computed"}
        return {"schema": SCHEMA, "status": status, "cache_key": self.key}

    def compute(self):
        if self.path.exists():
            report = add_accuracy_comparisons(read_json(self.path))
            write_json(self.path, report)
            return report
        view, receipt, parameters, room, indices, xyz, pairs, matcher = self.context
        rows, start = [], time.perf_counter()
        levels, repetitions = self.signature["noise_levels"], self.signature["repetitions"]
        target = len(levels) * len(repetitions) * 2
        chunks = self.root / self.key
        clean_counts = {}
        for ap in ("A", "B"):
            clean = self.peaks.profiles(self.context, ap, 0., repetitions[0])["profiles"]
            clean_counts[ap] = [len(p["ranges"]) for p in clean]
        for level in levels:
            for ap in ("A", "B"):
                for repetition in repetitions:
                    chunk = chunks / f"sigma_{level:g}_{ap}_repeat_{repetition}.json"
                    if chunk.exists():
                        row = read_json(chunk)
                    else:
                        data = self.peaks.profiles(self.context, ap, level, repetition)
                        profiles = data["profiles"]
                        row = {"noise": level, "ap": ap, "repetition": repetition,
                            "global": truth_counts(profiles, xyz, self.global_pairs, len(data["sources"])),
                            "neighborhood": truth_counts(profiles, xyz, pairs, len(data["sources"])),
                            "matches": final_match_counts(profiles, xyz, pairs, matcher),
                            "observations": {"fingerprints": len(profiles),
                                "peaks": sum(len(p["ranges"]) for p in profiles),
                                "discarded_negative": sum(clean_counts[ap]) - sum(len(p["ranges"]) for p in profiles),
                                "adjacent_equal_range_pairs": sum(int(np.sum(np.diff(p["ranges"]) == 0)) for p in profiles)}}
                        write_json(chunk, row)
                    rows.append(row)
                    status = {"state": "computing", "completed": len(rows), "target": target,
                              "noise": level, "ap": ap, "repetition": repetition,
                              "elapsed_seconds": time.perf_counter() - start}
                    write_json(self.status_path, status)
                    print(f"{len(rows)}/{target}: sigma={level:g}, AP {ap}, repetition {repetition}", flush=True)
        source_root = Path(__file__).parents[1]
        provenance = {name: file_hash(source_root / name) for name in (
            "execution/true_matching.py", "execution/peak_explorer.py", "execution/presets.py",
            "science/matching.py", "science/grouping.py", "science/reconstruction.py",
            "acquisition/observations.py")}
        report = {"schema": SCHEMA, "status": {"state": "complete"}, "cache_key": self.key,
            "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": self.signature,
            "selection": receipt, "implementation_provenance": provenance,
            "coverage": {"survey_positions": len(xyz), "global_pairs_per_ap_realization": len(self.global_pairs),
                "neighborhood_pairs_per_ap_realization": len(pairs),
                "noise_realizations_per_ap": len(repetitions), "native_source_counts": {
                    ap: len(view.truth_catalog[room["id"]][ap]) for ap in ("A", "B")},
                "room_width_m": room["width_m"], "room_depth_m": room["depth_m"],
                "ceiling_m": room["ceiling_m"], "zero_noise_realizations_identical": True},
            "rows": rows, "pooled": {ap: pool_rows(rows, ap) for ap in ("A", "B", "A+B")},
            "elapsed_seconds": time.perf_counter() - start}
        add_accuracy_comparisons(report)
        write_json(self.path, report)
        write_json(self.root / "latest.json", {"cache_key": self.key, "path": str(self.path.resolve())})
        write_json(self.status_path, {"state": "complete", "completed": target, "target": target})
        return report
