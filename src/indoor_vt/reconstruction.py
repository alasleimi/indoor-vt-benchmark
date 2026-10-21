"""Replaceable scope, graph, grouping, 3D fitting, merging and support filtering."""

from dataclasses import dataclass, asdict
from itertools import combinations
from itertools import product
import numpy as np
from scipy.spatial.distance import cdist

from .contracts import SourceMap
from .fitting import fit_source
from .matching import MatcherConfig, PatternMatcher


@dataclass(frozen=True)
class ReconstructionConfig:
    scope: str = "local"
    neighbors: int = 10
    grouping: str = "components"
    half_window: int = 2
    alpha: float = 0.5
    triangle: bool = False
    triangle_tolerance_m: float = 0.5
    merge_radius_m: float = 0.5
    beta: float = 1.0
    size_norm: str = "l1"


def neighborhoods(positions, scope, count):
    if scope == "full":
        return [tuple(range(len(positions)))]
    if scope != "local" or count < 4:
        raise ValueError("Scope must be full or local with at least four positions")
    order = np.argsort(cdist(positions, positions), axis=1, kind="stable")
    return sorted({tuple(sorted(row[:count])) for row in order})


def build_graph(positions, rows, scopes, matcher, triangle=False, tolerance=0.0):
    pairs = sorted({pair for ids in scopes for pair in combinations(ids, 2)})
    edges = []
    for left, right in pairs:
        separation = float(np.linalg.norm(positions[left] - positions[right]))
        matches = matcher(
            rows[left],
            rows[right],
            separation=separation if triangle else None,
            tolerance=tolerance,
        )
        edges.extend(((left, m.left), (right, m.right), m.cost, separation) for m in matches)
    return edges, len(pairs)


def graph_groups(edges, mode):
    adjacency = {}
    for left, right, cost, _ in edges:
        adjacency.setdefault(left, {})[right] = cost
        adjacency.setdefault(right, {})[left] = cost
    if mode == "star":
        groups = {tuple(sorted([center, *adjacency[center]])) for center in adjacency}
        # Pairwise bijection ensures one neighbor per reference for each center.
        return sorted(g for g in groups if len(g) >= 4)
    if mode != "components":
        raise ValueError(mode)
    unseen, groups = set(adjacency), []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, members = [start], []
        while stack:
            node = stack.pop()
            members.append(node)
            for other in sorted(adjacency[node]):
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        owners = {}
        for node in members:
            owners.setdefault(node[0], []).append(node)

        def priority(node):
            return (-len(adjacency[node]), float(np.mean(list(adjacency[node].values()))), node)

        chosen = tuple(sorted(min(nodes, key=priority) for nodes in owners.values()))
        if len(chosen) >= 4:
            groups.append(chosen)
    return sorted(groups)


def merge_candidates(records, radius):
    """Deterministic complete-link insertion, median position, union of evidence."""
    if radius <= 0:
        raise ValueError("Positive merging radius required")
    points = np.asarray([r["xyz"] for r in records]).reshape(-1, 3)
    groups, cells = [], {}
    for index in sorted(range(len(records)), key=lambda i: (*records[i]["xyz"], i)):
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
    result = []
    for ids in groups:
        members = sorted({tuple(n) for i in ids for n in records[i]["members"]})
        result.append(
            {
                "xyz": np.median([records[i]["xyz"] for i in ids], axis=0).tolist(),
                "members": members,
                "support": len({i for i, _ in members}),
                "candidate_ids": ids,
            }
        )
    return result


def support_filter(candidates, beta):
    if not candidates:
        return [], 0.0
    threshold = float(beta * np.mean([c["support"] for c in candidates]))
    return [c for c in candidates if c["support"] >= threshold], threshold


class MPUrgeReconstructor:
    version = "mpurge.3d.v1"

    def __init__(
        self,
        config=ReconstructionConfig(),
        *,
        matcher=None,
        scope_policy=neighborhoods,
        graph_builder=build_graph,
        group_builder=graph_groups,
        fitter=fit_source,
        merger=merge_candidates,
        candidate_filter=support_filter,
        refinement=None,
        group_fitter=None,
    ):
        self.config = config
        self.matcher = matcher or PatternMatcher(
            MatcherConfig(config.half_window, config.alpha, size_norm=config.size_norm)
        )
        self.scope_policy, self.graph_builder = scope_policy, graph_builder
        self.group_builder, self.fitter = group_builder, fitter
        self.merger, self.candidate_filter, self.refinement = merger, candidate_filter, refinement
        self.group_fitter = group_fitter

    def fit(self, survey):
        c = self.config
        scopes = self.scope_policy(survey.xyz, c.scope, c.neighbors)
        aps = sorted({ap for observation in survey.observations for ap in observation.channels})
        estimates, audit = (
            {},
            {
                "version": self.version,
                "config": asdict(c),
                "matcher": asdict(self.matcher.config),
                "channels": {},
            },
        )
        for ap in aps:
            rows = [o.channels.get(ap, np.empty(0)) for o in survey.observations]
            edges, pair_count = self.graph_builder(
                survey.xyz, rows, scopes, self.matcher, c.triangle, c.triangle_tolerance_m
            )
            # Index once; local scopes reuse the identical pair matches.
            pair_edges = {}
            for edge in edges:
                pair_edges.setdefault((edge[0][0], edge[1][0]), []).append(edge)
            groups = set()
            for ids in scopes:
                local = [edge for pair in combinations(ids, 2) for edge in pair_edges.get(pair, ())]
                groups.update(self.group_builder(local, c.grouping))
            records, rejected = [], []
            for members in sorted(groups):
                ids = [i for i, j in members]
                ranges = [rows[i][j] for i, j in members]
                if self.group_fitter is None:
                    point, fit = self.fitter(survey.xyz[ids], ranges)
                else:
                    point, fit = self.group_fitter(survey.xyz[ids], ranges, members, self.fitter)
                record = {"members": members, "fit": fit}
                if point is None:
                    rejected.append(record)
                else:
                    records.append({**record, "xyz": point.tolist()})
            merged = self.merger(records, c.merge_radius_m)
            selected, threshold = self.candidate_filter(merged, c.beta)
            estimates[ap] = np.asarray([s["xyz"] for s in selected]).reshape(-1, 3)
            audit["channels"][ap] = {
                "pair_count": pair_count,
                "scopes": scopes,
                "edges": edges,
                "groups": sorted(groups),
                "raw_candidates": records,
                "rejected": rejected,
                "merged": merged,
                "selected": selected,
                "support_threshold": threshold,
            }
        sources = SourceMap(estimates, audit)
        return self.refinement.fit(survey, sources) if self.refinement else sources
