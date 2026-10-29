"""Batch caching beside the readable scientific MPUrge pipeline."""
from dataclasses import replace
import time
import numpy as np

from .artifacts import ArtifactRef
from ..science.models import Ragged, SourceMap, Survey
from ..science.matching import MPUrgeMatcher
from ..science.grouping import PairMatches, group_within_scopes
from ..science.reconstruction import match_pairs


def effective_matcher(matcher, counts):
    """Remove only parameters proven inactive for every comparison in a batch."""
    if not isinstance(matcher, MPUrgeMatcher):
        return matcher
    changes = {}
    if len(np.unique(counts)) <= 1:
        changes["size_unification_norm"] = None
    if matcher.alpha == 1:
        changes.update(half_window=0, normalized_pattern=False)
    return replace(matcher, **changes)


def fit_groups(xyz, rows, groups, estimator):
    points, accepted, statuses, evaluations, residuals = [], [], [], [], []
    for index, track in enumerate(groups):
        ids, slots = np.asarray(track, dtype=int).T
        ranges = np.array([rows[i][j] for i, j in zip(ids, slots)])
        point, detail = estimator.fit(xyz[ids], ranges)
        statuses.append(detail["status"])
        evaluations.append(detail.get("evaluations", 0))
        residuals.append(detail.get("residuals", np.empty(0)))
        if point is not None:
            points.append(point)
            accepted.append(index)
    return {"xyz": np.asarray(points).reshape(-1, 3),
            "accepted_groups": np.asarray(accepted, dtype=np.int64),
            "groups": Ragged.pack(groups, width=2, dtype=np.int64).record(),
            "status": np.asarray(statuses, dtype="U32"),
            "evaluations": np.asarray(evaluations, dtype=np.int32),
            "residuals": Ragged.pack(residuals).record()}


def summarize_clusters(candidates, clusterer):
    xyz = candidates["xyz"]
    representatives, memberships = clusterer.cluster(xyz)
    groups = Ragged(**candidates["groups"])
    accepted = candidates["accepted_groups"]
    support = []
    for indices in memberships:
        owners = {int(i) for candidate in indices for i, _ in groups[accepted[candidate]]}
        support.append(len(owners))
    return {"xyz": representatives,
            "memberships": Ragged.pack(memberships, dtype=np.int64).record(),
            "support": np.asarray(support, dtype=np.int64)}


class MPUrgeExecution:
    def __init__(self, algorithm, store, run="", *, pair_bank=None):
        self.algorithm, self.store, self.run = algorithm, store, run
        self.pair_bank = pair_bank

    def fit(self, survey):
        started = time.perf_counter()
        c = self.algorithm
        geometry, _ = self.store.compute("comparison-geometry.2",
            {"pairs": c.pair_selector, "scopes": c.group_scope_selector}, {"xyz": survey.xyz},
            lambda: {"pairs": c.pair_selector(survey.xyz),
                     "scopes": c.group_scope_selector(survey.xyz)}, persist=False)
        pairs, scopes = geometry["pairs"], geometry["scopes"]
        bank = pairs if self.pair_bank is None else self.pair_bank
        outputs, evidence = {}, {}
        for ap, rows in survey.channels.items():
            matcher = effective_matcher(c.matcher, np.diff(rows.offsets))
            inputs = {"rows": rows.record(), "pairs": bank}
            # Only triangle-constrained matching depends on receiver geometry.
            if not isinstance(matcher, MPUrgeMatcher) or matcher.triangle_tolerance_m is not None:
                inputs["xyz"] = survey.xyz
            one_ap = Survey(survey.xyz, {ap: rows}, survey.position_ids)
            packed, match_key = self.store.compute("pair-matches.2", matcher, inputs,
                lambda: match_pairs(one_ap, bank, matcher)[ap].record())
            matches = PairMatches(**packed)
            def propose():
                groups = group_within_scopes(matches, scopes, c.grouper, allowed_pairs=pairs)
                return fit_groups(survey.xyz, rows, groups, c.estimator)
            candidates, candidate_key = self.store.compute("grouped-candidates.2",
                {"grouper": c.grouper, "estimator": c.estimator},
                {"matches": ArtifactRef(match_key), "pairs": pairs, "scopes": scopes,
                 "xyz": survey.xyz, "ranges": rows.record()}, propose)
            clusters, cluster_key = self.store.compute("source-clusters.2", c.clusterer,
                {"candidates": ArtifactRef(candidate_key)},
                lambda: summarize_clusters(candidates, c.clusterer))
            selected, threshold = c.support_filter.select(clusters["support"])
            outputs[ap] = clusters["xyz"][selected]
            evidence[ap] = {"matches": match_key, "candidates": candidate_key,
                "clusters": cluster_key, "selected": selected, "threshold": threshold,
                "support": clusters["support"], "scope_sizes": np.array(list(map(len, scopes)))}
        evidence["seconds"] = time.perf_counter() - started
        return SourceMap(outputs, evidence)
