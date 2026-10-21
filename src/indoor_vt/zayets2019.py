"""Explicit delay-only interpretation of the underspecified ICC 2019 VT stage.

This is not author code. Scalar positive delays have identical cosine direction.
The disclosed embedding is unit([range / embedding_scale_m, 1]); it uses only
delay and a globally calibrated length scale. Spherical splits use a Gaussian
BIC on the embedded coordinates. Ward matches [first range, count, span] with
one cluster per reference. See docs/methods.md and the source audit.
"""

from dataclasses import dataclass, asdict
import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .contracts import SourceMap
from .fitting import fit_source
from .reconstruction import merge_candidates


@dataclass(frozen=True)
class Zayets2019Config:
    embedding_scale_m: float = 1.0
    neighborhood_radius_m: float = 1.0
    ward_threshold: float = 1.0
    range_tolerance_m: float = 0.5
    gamma: float = 1.5
    merge_radius_m: float = 0.5


def bic(points, labels):
    n, dimension = points.shape
    groups = [points[labels == k] for k in np.unique(labels)]
    k = len(groups)
    variance = max(
        sum(np.square(g - g.mean(0)).sum() for g in groups) / (max(n - k, 1) * dimension), 1e-12
    )
    log_likelihood = sum(len(g) * np.log(len(g) / n) for g in groups)
    log_likelihood -= 0.5 * n * dimension * (np.log(2 * np.pi * variance) + 1)
    parameters = k * dimension + (k - 1) + 1
    return float(log_likelihood - 0.5 * parameters * np.log(max(n, 2)))


def dynamic_cosine_clusters(row, scale):
    if not len(row):
        return []
    points = np.column_stack((np.asarray(row) / scale, np.ones(len(row))))
    points /= np.linalg.norm(points, axis=1)[:, None]
    pending, result = [np.arange(len(row))], []
    while pending:
        ids = pending.pop()
        local = points[ids]
        if len(ids) < 4:
            result.append(ids)
            continue
        centers = local[[0, -1]].copy()
        labels = np.zeros(len(ids), int)
        for _ in range(50):
            updated = np.argmax(local @ centers.T, axis=1)
            if not (updated == 0).any() or not (updated == 1).any():
                break
            new_centers = np.array([local[updated == k].mean(0) for k in (0, 1)])
            new_centers /= np.linalg.norm(new_centers, axis=1)[:, None]
            converged = np.array_equal(updated, labels)
            centers, labels = new_centers, updated
            if converged:
                break
        if min(np.sum(labels == k) for k in (0, 1)) >= 2 and bic(local, labels) > bic(
            local, np.zeros(len(ids), int)
        ):
            pending.extend([ids[labels == 1], ids[labels == 0]])
        else:
            result.append(ids)
    return sorted(result, key=lambda ids: int(ids[0]))


def ward_groups(features, owners, threshold):
    active = {
        i: {"nodes": [i], "owners": {owner}, "mean": features[i], "n": 1}
        for i, owner in enumerate(owners)
    }

    def cost(a, b):
        return a["n"] * b["n"] / (a["n"] + b["n"]) * np.square(a["mean"] - b["mean"]).sum()

    heap = [
        (float(cost(active[i], active[j])), i, j)
        for i in active
        for j in active
        if i < j and owners[i] != owners[j]
    ]
    heapq.heapify(heap)
    next_id = len(active)
    while heap:
        value, left, right = heapq.heappop(heap)
        if value > threshold:
            break
        if left not in active or right not in active:
            continue
        a, b = active[left], active[right]
        if not a["owners"].isdisjoint(b["owners"]):
            continue
        del active[left], active[right]
        n = a["n"] + b["n"]
        merged = {
            "nodes": a["nodes"] + b["nodes"],
            "owners": a["owners"] | b["owners"],
            "n": n,
            "mean": (a["n"] * a["mean"] + b["n"] * b["mean"]) / n,
        }
        for i, other in active.items():
            if merged["owners"].isdisjoint(other["owners"]):
                heapq.heappush(heap, (float(cost(other, merged)), i, next_id))
        active[next_id] = merged
        next_id += 1
    return [sorted(group["nodes"]) for _, group in sorted(active.items())]


class Zayets2019:
    version = "zayets2019.disclosed_cosine_embedding.v1"

    def __init__(self, config=Zayets2019Config(), fitter=fit_source):
        self.config, self.fitter = config, fitter

    def fit(self, survey):
        c = self.config
        scopes = sorted(
            {
                tuple(sorted(ids))
                for ids in cKDTree(survey.xyz).query_ball_point(survey.xyz, c.neighborhood_radius_m)
            }
        )
        output, diagnostics = {}, {}
        for ap in sorted({ap for o in survey.observations for ap in o.channels}):
            rows = [o.channels.get(ap, np.empty(0)) for o in survey.observations]
            clusters = [dynamic_cosine_clusters(row, c.embedding_scale_m) for row in rows]
            tracks = set()
            for scope in scopes:
                nodes = [(i, ids) for i in scope for ids in clusters[i]]
                if not nodes:
                    continue
                features = np.array(
                    [
                        [
                            rows[i][ids[0]] / c.embedding_scale_m,
                            len(ids),
                            (rows[i][ids[-1]] - rows[i][ids[0]]) / c.embedding_scale_m,
                        ]
                        for i, ids in nodes
                    ]
                )
                for group in ward_groups(features, [i for i, ids in nodes], c.ward_threshold):
                    maximum = max(len(nodes[j][1]) for j in group)
                    target = np.mean(
                        [
                            rows[nodes[j][0]][nodes[j][1]]
                            for j in group
                            if len(nodes[j][1]) == maximum
                        ],
                        axis=0,
                    )
                    slots = [[] for _ in range(maximum)]
                    for j in group:
                        i, ids = nodes[j]
                        cols = (
                            np.arange(maximum)
                            if len(ids) == maximum
                            else linear_sum_assignment(abs(rows[i][ids, None] - target[None]))[1]
                        )
                        for index, slot in zip(ids, cols):
                            slots[slot].append((i, int(index)))
                    tracks.update(tuple(sorted(slot)) for slot in slots if len(slot) >= 4)
            records, rejected = [], []
            for members in sorted(tracks):
                xyz, audit = self.fitter(
                    survey.xyz[[i for i, j in members]], [rows[i][j] for i, j in members]
                )
                if xyz is None:
                    rejected.append({"members": members, **audit})
                    continue
                evidence = []
                predicted = np.linalg.norm(survey.xyz - xyz, axis=1)
                for i, (row, distance) in enumerate(zip(rows, predicted)):
                    if len(row):
                        j = int(abs(row - distance).argmin())
                        if abs(row[j] - distance) <= c.range_tolerance_m:
                            evidence.append((i, j))
                records.append(
                    {
                        "xyz": xyz.tolist(),
                        "members": evidence,
                        "support": len(evidence),
                        "fit": audit,
                        "track": members,
                    }
                )
            threshold = c.gamma * np.mean([r["support"] for r in records]) if records else 0.0
            retained = [r for r in records if r["support"] >= threshold and r["support"] > 0]
            merged = merge_candidates(retained, c.merge_radius_m)
            points = [
                np.mean([retained[j]["xyz"] for j in group["candidate_ids"]], axis=0)
                for group in merged
            ]
            output[ap] = np.asarray(points).reshape(-1, 3)
            diagnostics[ap] = {
                "raw_candidates": records,
                "rejected": rejected,
                "threshold": float(threshold),
                "clusters": [[ids.tolist() for ids in cs] for cs in clusters],
                "merged": merged,
            }
        return SourceMap(
            output, {"version": self.version, "config": asdict(c), "channels": diagnostics}
        )
