"""Geometry selects comparisons; local graphs assemble observation groups."""

from dataclasses import dataclass
from itertools import combinations
import numpy as np
from numba import njit
from scipy.spatial import cKDTree


def nearest_neighborhoods(xyz, size=10, include_ties=True):
    """At least size positions including the center, using 3D distance.

    Include all positions tied at the boundary by default: choosing arbitrary
    lattice neighbors makes scopes asymmetric. Exact-size scopes break ties by
    position index, with the center first. Both policies give nested scopes.
    """
    if size < 1:
        raise ValueError("Neighborhood size must be positive")
    xyz = np.asarray(xyz)
    if not len(xyz):
        return []
    k, scopes = min(size, len(xyz)), set()
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=k)
    boundary = np.asarray(distances).reshape(len(xyz), -1)[:, -1]
    neighborhoods = tree.query_ball_point(xyz, boundary + 1e-12)
    for center, ids in enumerate(neighborhoods):
        ids = np.asarray(ids)
        row = np.round(np.linalg.norm(xyz[ids] - xyz[center], axis=1), 12)
        order = np.lexsort((ids, ids != center, row))
        selected = ids[row <= row[order[k - 1]]] if include_ties else ids[order[:k]]
        scopes.add(tuple(sorted(map(int, selected))))
    return sorted(scopes)


def radius_neighborhoods(xyz, radius_m=1.0):
    return sorted({tuple(sorted(ids)) for ids in cKDTree(xyz).query_ball_point(xyz, radius_m)})


def global_scope(xyz):
    return [tuple(range(len(xyz)))]


def all_pairs_within(scopes):
    return np.asarray(sorted({pair for scope in scopes for pair in combinations(scope, 2)}),
                      dtype=np.int64).reshape(-1, 2)


@dataclass(frozen=True)
class NeighborhoodPairs:
    scope_selector: object

    def __call__(self, xyz):
        return all_pairs_within(self.scope_selector(xyz))


@dataclass(frozen=True)
class PairMatches:
    """One block per fingerprint pair; endpoints index the packed peak array."""
    pairs: np.ndarray
    offsets: np.ndarray
    endpoints: np.ndarray
    costs: np.ndarray
    node_offsets: np.ndarray

    def record(self):
        return vars(self)

    def pair_lookup(self):
        n = len(self.node_offsets) - 1
        lookup = np.full((n, n), -1, dtype=np.int64)
        lookup[self.pairs[:, 0], self.pairs[:, 1]] = np.arange(len(self.pairs))
        return lookup

    def nodes(self, ids):
        owners = np.searchsorted(self.node_offsets, ids, side="right") - 1
        return np.column_stack((owners, ids - self.node_offsets[owners]))


@njit(cache=True)
def _root(parent, node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


@njit(cache=True)
def _components(offsets, endpoints, costs, blocks, count):
    parent = np.arange(count)
    sizes = np.ones(count, np.int64)
    degree = np.zeros(count, np.int64)
    total_cost = np.zeros(count)
    for block in blocks:
        for edge in range(offsets[block], offsets[block + 1]):
            a, b = endpoints[edge]
            degree[a] += 1
            degree[b] += 1
            total_cost[a] += costs[edge]
            total_cost[b] += costs[edge]
            ra, rb = _root(parent, a), _root(parent, b)
            if ra != rb:
                if sizes[ra] < sizes[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                sizes[ra] += sizes[rb]
    for node in range(count):
        if degree[node]:
            parent[node] = _root(parent, node)
    return parent, degree, total_cost


@dataclass(frozen=True)
class ConnectedComponentsGrouper:
    def group(self, matches, blocks=None):
        if blocks is None:
            blocks = np.arange(len(matches.pairs))
        roots, degree, costs = _components(matches.offsets, matches.endpoints, matches.costs,
                                           blocks, int(matches.node_offsets[-1]))
        active = np.flatnonzero(degree)
        nodes = matches.nodes(active)
        # Keep one peak per fingerprint in each component: highest degree,
        # lowest mean edge cost, lowest peak index. Do not regroup after pruning.
        order = np.lexsort((active, costs[active] / degree[active], -degree[active],
                            nodes[:, 0], roots[active]))
        ids, nodes = active[order], nodes[order]
        keep = np.ones(len(ids), bool)
        keep[1:] = (roots[ids[1:]] != roots[ids[:-1]]) | (nodes[1:, 0] != nodes[:-1, 0])
        ids, nodes = ids[keep], nodes[keep]
        boundaries = np.flatnonzero(roots[ids[1:]] != roots[ids[:-1]]) + 1
        return sorted(tuple(map(tuple, group)) for group in np.split(nodes, boundaries) if len(group))


@dataclass(frozen=True)
class StarGrouper:
    def group(self, matches, blocks=None):
        if blocks is None:
            blocks = np.arange(len(matches.pairs))
        graph = {}
        for block in blocks:
            for a, b in matches.endpoints[matches.offsets[block]:matches.offsets[block + 1]]:
                graph.setdefault(int(a), set()).add(int(b))
                graph.setdefault(int(b), set()).add(int(a))
        return sorted({tuple(map(tuple, matches.nodes(np.array(sorted([node, *neighbors])))))
                       for node, neighbors in graph.items()})


def group_within_scopes(matches, scopes, grouper, *, allowed_pairs=None):
    lookup, groups = matches.pair_lookup(), set()
    if allowed_pairs is not None:
        permitted = np.zeros_like(lookup, dtype=bool)
        permitted[allowed_pairs[:, 0], allowed_pairs[:, 1]] = True
        if np.any(lookup[permitted] < 0):
            raise ValueError("Match bank is missing requested fingerprint pairs")
        lookup[~permitted] = -1
    for scope in scopes:
        pairs = np.asarray(list(combinations(scope, 2)), dtype=np.int64).reshape(-1, 2)
        blocks = lookup[pairs[:, 0], pairs[:, 1]]
        groups.update(grouper.group(matches, blocks[blocks >= 0]))
    return sorted(groups)
