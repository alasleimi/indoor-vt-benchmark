"""Local cluster tracks: split ranges, group clusters, align peaks, fit sources."""

from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment

from .costs import absolute_difference, mean
from .models import SourceMap, Ragged


def split_at_gaps(sorted_values, threshold=0.6):
    """Return index runs separated by consecutive gaps at least `threshold`."""
    if threshold <= 0:
        raise ValueError("Gap threshold must be positive")
    if np.any(np.diff(sorted_values) < 0):
        raise ValueError("split_at_gaps expects a sorted sequence")
    if not len(sorted_values):
        return []
    return np.split(np.arange(len(sorted_values)), np.flatnonzero(np.diff(sorted_values) >= threshold) + 1)


def cluster_feature_cost(a, b, *, first_range_weight=.5, peak_count_weight=.5,
                         range_span_weight=.5, difference=absolute_difference):
    def features(row):
        return np.array([row[0], len(row), row[-1] - row[0]])
    return float(np.dot([first_range_weight, peak_count_weight, range_span_weight],
                        difference(features(a), features(b))))


@dataclass(frozen=True)
class ConstrainedSingleLinkageGrouper:
    max_cost: float = .5

    def group(self, owners, edges):
        """Greedy Kruskal: merged components must have disjoint fingerprints.

        Edges are (cost, node index, node index). Intra-component cycle edges do
        nothing. Isolated nodes remain singleton groups.
        """
        parent = list(range(len(owners)))
        fingerprints = [{owner} for owner in owners]
        def root(i):
            while i != parent[i]:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for cost, a, b in sorted(edges):
            if cost > self.max_cost:
                break
            a, b = root(a), root(b)
            if a != b and fingerprints[a].isdisjoint(fingerprints[b]):
                parent[b] = a
                fingerprints[a].update(fingerprints[b])
        groups = {}
        for i in range(len(owners)):
            groups.setdefault(root(i), []).append(i)
        return list(groups.values())


@dataclass(frozen=True)
class LongestClusterTemplateAssigner:
    template_reducer: object = mean
    cost: object = absolute_difference

    def assign(self, clusters):
        """Average longest clusters; assign shorter peaks to distinct slots.

        Return one (cluster index, peak index) list per template slot. Equal
        length clusters align by their sorted peak indices.
        """
        length = max(map(len, clusters), default=0)
        if length == 0:
            return []
        template = self.template_reducer(np.array([c for c in clusters if len(c) == length]))
        slots = [[] for _ in range(length)]
        for i, cluster in enumerate(clusters):
            if len(cluster) == length:
                peak_ids, columns = np.arange(length), np.arange(length)
            else:
                peak_ids, columns = linear_sum_assignment(self.cost(cluster[:, None], template[None, :]))
            for j, column in zip(peak_ids, columns):
                slots[column].append((i, int(j)))
        return slots


@dataclass(frozen=True)
class ZayetsReconstructor:
    scope_selector: object
    range_clusterer: object
    cluster_cost: object
    grouper: object
    peak_assigner: object
    estimator: object
    min_track_positions: int = 4

    def fit(self, survey):
        scopes = self.scope_selector(survey.xyz)
        output, audit = {}, {}
        for ap, rows in survey.channels.items():
            orders = [np.argsort(rows[i], kind="stable") for i in range(len(rows))]
            clusters = [[order[part] for part in self.range_clusterer(rows[i][order])]
                        for i, order in enumerate(orders)]
            tracks, scope_details = set(), []
            for scope in scopes:
                nodes = [(i, ids) for i in scope for ids in clusters[i]]
                edges = [(self.cluster_cost(rows[i][a], rows[j][b]), u, v)
                         for (u, (i, a)), (v, (j, b)) in combinations(enumerate(nodes), 2) if i != j]
                groups = self.grouper.group([i for i, _ in nodes], edges)
                for group in groups:
                    selected = [nodes[i] for i in group]
                    assignments = self.peak_assigner.assign([rows[i][ids] for i, ids in selected])
                    for track in assignments:
                        if len(track) >= self.min_track_positions:
                            members = tuple(sorted((selected[i][0], int(selected[i][1][j])) for i, j in track))
                            tracks.add(members)
                scope_details.append({"positions": np.asarray(scope),
                    "node_owners": np.array([i for i, _ in nodes], dtype=int),
                    "node_peaks": Ragged.pack([ids for _, ids in nodes], dtype=int).record(),
                    "edges": np.array(edges).reshape(-1, 3),
                    "groups": Ragged.pack(groups, dtype=int).record()})
            points, fits = [], []
            for members in sorted(tracks):
                ids, slots = np.array(members).T
                point, detail = self.estimator.fit(survey.xyz[ids], np.array([rows[i][j] for i, j in members]))
                fits.append({"members": np.array(members), "point": point, **detail})
                if point is not None:
                    points.append(point)
            output[ap] = np.array(points).reshape(-1, 3)
            audit[ap] = {"scopes": scope_details, "fits": fits}
        return SourceMap(output, audit)
