"""Exact VT candidate clustering, separate from representative and support rules."""

from collections import deque
from itertools import product
import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import cKDTree

from .provenance import digest
from .stage_cache import cached
from .complete_link import complete_linkage


def greedy_complete(points, radius):
    groups, cells = [], {}
    for index in sorted(range(len(points)), key=lambda i: (*points[i], i)):
        point = points[index]
        cell = tuple(np.floor(point / radius).astype(int))
        nearby = sorted(
            gid
            for offset in product((-1, 0, 1), repeat=3)
            for gid in cells.get(tuple(a + b for a, b in zip(cell, offset)), ())
        )
        for gid in nearby:
            group = groups[gid]
            if np.all(np.linalg.norm(points[group] - point, axis=1) <= radius):
                group.append(index)
                break
        else:
            cells.setdefault(cell, []).append(len(groups))
            groups.append([index])
    return groups


def density_clusters(points, radius, min_samples):
    """DBSCAN with on-demand neighborhoods; no dense all-pairs adjacency matrix."""
    tree = cKDTree(points)
    core = tree.query_ball_point(points, radius, return_length=True) >= min_samples
    labels = np.full(len(points), -1, dtype=int)
    visited = np.zeros(len(points), dtype=bool)
    groups = []
    for start in range(len(points)):
        if visited[start]:
            continue
        visited[start] = True
        if not core[start]:
            continue
        label = len(groups)
        labels[start] = label
        queue = deque([start])
        while queue:
            point = queue.popleft()
            for neighbor in sorted(tree.query_ball_point(points[point], radius)):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if core[neighbor]:
                        queue.append(neighbor)
                if labels[neighbor] < 0:
                    labels[neighbor] = label
        groups.append(np.flatnonzero(labels == label).tolist())
    return groups


def memberships(points, radius, method="greedy_median", min_samples=5, cache=None):
    if radius <= 0 or min_samples < 1:
        raise ValueError("Positive clustering radius and minimum sample count required")
    points = np.asarray(points, float).reshape(-1, 3)
    if not len(points):
        return []
    bank = digest(points)
    algorithm = "greedy" if method in ("greedy_median", "greedy_mean") else method
    identity = {"bank": bank, "method": algorithm, "radius": radius}
    if algorithm == "dbscan":
        identity["min_samples"] = min_samples

    def compute():
        if algorithm == "greedy":
            return greedy_complete(points, radius)
        if algorithm == "agglomerative":
            if len(points) == 1:
                return [[0]]
            hierarchy = cached(
                cache,
                "hierarchies",
                {"bank": bank, "linkage": "complete"},
                lambda: complete_linkage(
                    points, scratch_dir=cache.directory / "scratch" if cache else None
                ).tolist(),
            )
            labels = fcluster(np.asarray(hierarchy), t=radius, criterion="distance")
            return [np.flatnonzero(labels == label).tolist() for label in sorted(set(labels))]
        if algorithm == "dbscan":
            return density_clusters(points, radius, min_samples)
        raise ValueError(f"Unknown VT merger: {method}")

    return cached(cache, "memberships", identity, compute)


def merge(records, radius, method="greedy_median", min_samples=5, cache=None):
    points = np.asarray([record["xyz"] for record in records]).reshape(-1, 3)
    groups = memberships(points, radius, method, min_samples, cache)
    result = []
    for ids in groups:
        members = sorted({tuple(n) for i in ids for n in records[i]["members"]})
        representative = np.mean if method == "greedy_mean" else np.median
        result.append(
            {
                "xyz": representative(points[ids], axis=0).tolist(),
                "members": members,
                "support": len({i for i, _ in members}),
                "candidate_ids": ids,
            }
        )
    return result
