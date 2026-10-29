"""Readable reconstruction composition. No persistence or instrumentation."""

from dataclasses import dataclass
import numpy as np

from .grouping import PairMatches, group_within_scopes
from .models import SourceMap


def match_pairs(survey, pairs, matcher):
    graphs = {}
    for ap, rows in survey.channels.items():
        endpoints, costs, offsets = [], [], [0]
        for i, j in pairs:
            separation = np.linalg.norm(survey.xyz[i] - survey.xyz[j])
            matches = matcher.match_array(rows[i], rows[j], separation=separation)
            dtype = np.int32 if rows.offsets[-1] < np.iinfo(np.int32).max else np.int64
            endpoints.append((matches[:, :2] + rows.offsets[[i, j]]).astype(dtype))
            costs.append(matches[:, 2])
            offsets.append(offsets[-1] + len(matches))
        graphs[ap] = PairMatches(pairs, np.asarray(offsets, dtype=np.int64),
            np.concatenate(endpoints) if endpoints else np.empty((0, 2), dtype=np.int64),
            np.concatenate(costs) if costs else np.empty(0), rows.offsets)
    return graphs


def estimate_sources(survey, groups, estimator):
    candidates, diagnostics = {}, {}
    for ap, tracks in groups.items():
        candidates[ap], diagnostics[ap] = [], []
        for members in tracks:
            ids, peaks = np.asarray(members, dtype=int).T
            ranges = np.array([survey.channels[ap][i][j] for i, j in zip(ids, peaks)])
            point, detail = estimator.fit(survey.xyz[ids], ranges)
            diagnostics[ap].append({"members": np.asarray(members), **detail})
            if point is not None:
                candidates[ap].append({"xyz": point, "members": members})
    return candidates, diagnostics


def cluster_with_evidence(candidates, clusterer):
    result = {}
    for ap, records in candidates.items():
        xyz = np.asarray([r["xyz"] for r in records]).reshape(-1, 3)
        points, clusters = clusterer.cluster(xyz)
        members = [sorted({tuple(m) for i in group for m in records[i]["members"]})
                   for group in clusters]
        result[ap] = {"xyz": points, "candidate_indices": clusters, "members": members,
                      "support": np.array([len({i for i, _ in group}) for group in members])}
    return result


@dataclass(frozen=True)
class RelativeSupportFilter:
    beta: float = 1.0

    def select(self, support):
        if self.beta < 0:
            raise ValueError("Support beta must be nonnegative")
        threshold = self.beta * np.mean(support) if len(support) else 0.0
        return np.flatnonzero(support >= threshold), float(threshold)

    def filter(self, clusters):
        output, evidence = {}, {}
        for ap, cluster in clusters.items():
            kept, threshold = self.select(cluster["support"])
            output[ap] = cluster["xyz"][kept]
            evidence[ap] = {"clusters": cluster, "selected": kept, "threshold": threshold}
        return SourceMap(output, evidence)


@dataclass(frozen=True)
class MPUrge:
    pair_selector: object
    group_scope_selector: object
    matcher: object
    grouper: object
    estimator: object
    clusterer: object
    support_filter: object

    def fit(self, survey):
        pairs = self.pair_selector(survey.xyz)
        scopes = self.group_scope_selector(survey.xyz)
        matches = match_pairs(survey, pairs, self.matcher)
        groups = {ap: group_within_scopes(edges, scopes, self.grouper)
                  for ap, edges in matches.items()}
        candidates, fits = estimate_sources(survey, groups, self.estimator)
        clusters = cluster_with_evidence(candidates, self.clusterer)
        sources = self.support_filter.filter(clusters)
        sources.evidence.update({"pairs": pairs, "matches": matches, "groups": groups,
                                 "fits": fits})
        return sources


@dataclass(frozen=True)
class ClusteredReconstructor:
    reconstructor: object
    clusterer: object

    def fit(self, survey):
        sources = self.reconstructor.fit(survey)
        outputs, memberships = {}, {}
        for ap, xyz in sources.by_ap.items():
            outputs[ap], memberships[ap] = self.clusterer.cluster(xyz)
        return SourceMap(outputs, {"memberships": memberships, "input": sources})


@dataclass(frozen=True)
class FilteredReconstructor:
    reconstructor: object
    selector: object

    def fit(self, survey):
        return self.selector.select(survey, self.reconstructor.fit(survey))
