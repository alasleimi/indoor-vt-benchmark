"""Cached Zayets/RANSAC proposals and explicit consuming finishers.

Scientific constructors stay separate from execution. The dense nearest-peak
kernel only replaces the reference selector's repeated array allocations.
"""
from functools import partial
from pathlib import Path
import hashlib
import time

import numpy as np
from numba import njit

from .artifacts import ArtifactRef
from .store import digest
from ..science.costs import absolute_difference, mean, gaussian_weights, soft_l1_loss
from ..science.estimation import Multilateration3D
from ..science.grouping import radius_neighborhoods
from ..science.models import SourceMap, Survey, Ragged
from ..science.ransac import (RansacReconstructor, ThreeSphereIntersections,
                              TopPerTrialBatch, uniform_peak_sample)
from ..science.selection import (SupportFirstScorer, SupportAndResidualScorer,
    GreedySourceSelector, NearestPeakRefitter)
from ..science.zayets import (ZayetsReconstructor, split_at_gaps, cluster_feature_cost,
    ConstrainedSingleLinkageGrouper, LongestClusterTemplateAssigner)


ROBUST_FINISHER_PARAMETERS = ("smooth_weight", "median_penalty_weight", "median_clip_factor")


def extension_id():
    return digest({name: Path(__file__).with_name(name).read_text(encoding="utf-8")
                   for name in ("additional.py", "additional_experiment.py")})


def zayets_proposals(p):
    return ZayetsReconstructor(
        partial(radius_neighborhoods, radius_m=p["scope_radius_m"]),
        partial(split_at_gaps, threshold=p["gap_m"]),
        partial(cluster_feature_cost, first_range_weight=.5, peak_count_weight=.5,
                range_span_weight=.5, difference=absolute_difference),
        ConstrainedSingleLinkageGrouper(p["group_cost"]),
        LongestClusterTemplateAssigner(mean, absolute_difference), Multilateration3D())


def ransac_proposals(p, seed):
    return RansacReconstructor(uniform_peak_sample, ThreeSphereIntersections(),
        SupportAndResidualScorer(p["tolerance_m"]), TopPerTrialBatch(400, 12),
        trials=p["trials"], seed=seed)


def finisher(family, p):
    tolerance = p["tolerance_m"]
    scorer = (SupportFirstScorer(tolerance) if family == "zayets"
              else SupportAndResidualScorer(tolerance,
                  **{key: p[key] for key in ROBUST_FINISHER_PARAMETERS if key in p}))
    refitter = None if family == "zayets" else NearestPeakRefitter(
        Multilateration3D(loss=soft_l1_loss, max_evaluations=120,
                         loss_scale_m=tolerance),
        partial(gaussian_weights, scale_m=2 * tolerance), 2 * tolerance)
    return GreedySourceSelector(scorer, refitter=refitter,
        min_support_positions=5, min_support_fraction=p["support_fraction"],
        duplicate_tolerance_m=p["duplicate_m"], consume_support=True, max_sources=None)


@njit(cache=True)
def nearest_many(points, xyz, values, offsets, available):
    errors = np.full((len(points), len(xyz)), np.inf)
    slots = np.full((len(points), len(xyz)), -1, dtype=np.int64)
    for k in range(len(points)):
        for i in range(len(xyz)):
            dx, dy, dz = xyz[i, 0] - points[k, 0], xyz[i, 1] - points[k, 1], xyz[i, 2] - points[k, 2]
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            for j in range(offsets[i], offsets[i + 1]):
                if available[j]:
                    error = abs(values[j] - distance)
                    if error < errors[k, i]:
                        errors[k, i], slots[k, i] = error, j - offsets[i]
    return errors, slots


@njit(cache=True)
def nearest_from_ranges(predicted, values, offsets, available):
    errors = np.full(predicted.shape, np.inf)
    slots = np.full(predicted.shape, -1, dtype=np.int64)
    for k in range(len(predicted)):
        for i in range(predicted.shape[1]):
            for j in range(offsets[i], offsets[i+1]):
                if available[j]:
                    error = abs(values[j] - predicted[k, i])
                    if error < errors[k, i]:
                        errors[k, i], slots[k, i] = error, j-offsets[i]
    return errors, slots


@njit(cache=True)
def repair_nearest(errors, slots, predicted, values, offsets, available, remaining, positions):
    """Only a consumed nearest peak can change a surviving proposal's residual."""
    for k in remaining:
        for i in positions:
            slot = slots[k, i]
            if slot < 0 or available[offsets[i] + slot]:
                continue
            error, best = np.inf, -1
            for j in range(offsets[i], offsets[i+1]):
                if available[j]:
                    distance = abs(values[j] - predicted[k, i])
                    if distance < error:
                        error, best = distance, j-offsets[i]
            errors[k, i], slots[k, i] = error, best


def score_many(errors, scorer):
    tolerance = scorer.inlier_tolerance_m
    support = (errors <= tolerance).sum(axis=1)
    closeness = np.exp(-.5 * (errors / (tolerance * scorer.smooth_scale_factor))**2).sum(axis=1)
    if isinstance(scorer, SupportFirstScorer):
        return support + scorer.tie_break_weight * closeness / (errors.shape[1] + 1)
    penalty = np.median(np.minimum(errors, scorer.median_clip_factor * tolerance), axis=1) / tolerance
    return support + scorer.smooth_weight * closeness - scorer.median_penalty_weight * penalty


def finish_one(survey, candidates, selector):
    """Reference greedy order, scoring, support consumption, refit and suppression."""
    ap = next(iter(survey.channels))
    rows, xyz = survey.channels[ap], survey.xyz
    proposals = candidates.by_ap[ap]
    predicted = np.linalg.norm(proposals[:, None, :] - xyz[None, :, :], axis=2)
    available = np.ones(len(rows.values), dtype=bool)
    remaining = np.arange(len(proposals))
    errors, slots = nearest_from_ranges(predicted, rows.values, rows.offsets, available)
    required = max(selector.min_support_positions,
                   int(np.ceil(selector.min_support_fraction * len(xyz))))
    accepted, accepted_ranges, history = [], [], []
    while len(remaining) and (selector.max_sources is None or len(accepted) < selector.max_sources):
        active = remaining
        if isinstance(selector.scorer, SupportFirstScorer) and 0 <= selector.scorer.tie_break_weight <= 1:
            # Its smooth term is strictly below one: lower support cannot win.
            supports = (errors[remaining] <= selector.scorer.inlier_tolerance_m).sum(axis=1)
            active = remaining[supports == supports.max()]
        scores = score_many(errors[active], selector.scorer)
        winner = int(np.argmax(scores))
        index, score = int(active[winner]), scores[winner]
        best = int(np.searchsorted(remaining, index))
        remaining = np.delete(remaining, best)
        point = proposals[index].copy()
        if np.count_nonzero(errors[index] <= selector.scorer.inlier_tolerance_m) < required:
            history.append({"proposal": index, "score": score, "status": "stop_insufficient_support"})
            break
        refit = None
        if selector.refitter is not None:
            masks = [available[rows.offsets[i]:rows.offsets[i+1]] for i in range(len(rows))]
            updated, refit = selector.refitter.fit(point, xyz, rows, masks)
            if updated is not None:
                point = updated
        error, slot = nearest_many(point.reshape(1, 3), xyz, rows.values, rows.offsets, available)
        error, slot = error[0], slot[0]
        support = np.flatnonzero(error <= selector.scorer.inlier_tolerance_m)
        point_ranges = np.linalg.norm(xyz - point, axis=1)
        duplicate = any(np.sqrt(np.mean((point_ranges - p)**2)) <= selector.duplicate_tolerance_m
                        for p in accepted_ranges)
        status = ("insufficient_support" if len(support) < required else
                  "duplicate" if duplicate else "accepted")
        detail = {"proposal": index, "score": score, "point": point, "errors": error,
                  "support": np.column_stack((support, slot[support])), "refit": refit,
                  "status": status}
        history.append(detail)
        if status != "accepted":
            continue
        accepted.append(point)
        accepted_ranges.append(point_ranges)
        if selector.consume_support:
            available[rows.offsets[support] + slot[support]] = False
        distance = np.sqrt(np.mean((predicted[remaining] - point_ranges)**2, axis=1))
        duplicate_ids = remaining[distance <= selector.duplicate_tolerance_m]
        detail["suppressed_proposals"] = np.asarray(duplicate_ids, dtype=int)
        remaining = remaining[~np.isin(remaining, duplicate_ids)]
        repair_nearest(errors, slots, predicted, rows.values, rows.offsets, available, remaining, support)
    return {"xyz": np.asarray(accepted).reshape(-1, 3),
            "selection": {"required_support": required, "history": history}}


def zayets_one(survey, algorithm):
    """Vectorized feature distances, with the scientific grouping/assignment/fitter."""
    ap, rows = next(iter(survey.channels.items()))
    orders = [np.argsort(rows[i], kind="stable") for i in range(len(rows))]
    clusters = [[order[part] for part in algorithm.range_clusterer(rows[i][order])]
                for i, order in enumerate(orders)]
    scopes = algorithm.scope_selector(survey.xyz)
    tracks = set()
    for scope in scopes:
        nodes = [(i, ids) for i in scope for ids in clusters[i]]
        if not nodes:
            continue
        features = np.array([[rows[i][ids][0], len(ids), np.ptp(rows[i][ids])]
                             for i, ids in nodes])
        owners = np.array([i for i, _ in nodes])
        a, b = np.triu_indices(len(nodes), 1)
        keep = owners[a] != owners[b]
        a, b = a[keep], b[keep]
        costs = .5 * np.abs(features[a] - features[b]).sum(axis=1)
        # Edges above the threshold cannot affect the scientific grouper.
        keep = costs <= algorithm.grouper.max_cost
        edges = [(float(c), int(i), int(j)) for c, i, j in zip(costs[keep], a[keep], b[keep])]
        for group in algorithm.grouper.group(owners.tolist(), edges):
            selected = [nodes[i] for i in group]
            assignments = algorithm.peak_assigner.assign([rows[i][ids] for i, ids in selected])
            for track in assignments:
                if len(track) >= algorithm.min_track_positions:
                    tracks.add(tuple(sorted((selected[i][0], int(selected[i][1][j]))
                                            for i, j in track)))
    points, members, statuses = [], [], []
    for track in sorted(tracks):
        ids, slots = np.array(track).T
        point, detail = algorithm.estimator.fit(survey.xyz[ids],
                                               np.array([rows[i][j] for i, j in track]))
        statuses.append(detail["status"])
        if point is not None:
            points.append(point)
            members.append(np.asarray(track))
    return {"xyz": np.array(points).reshape(-1, 3),
            "tracks": Ragged.pack(members, width=2, dtype=int).record(),
            "fit_status": np.asarray(statuses, dtype="U32"),
            "scopes": Ragged.pack(scopes, dtype=int).record()}


def ransac_pool(survey, algorithm):
    ap, rows = next(iter(survey.channels.items()))
    seed = int.from_bytes(hashlib.sha256(f"{algorithm.seed}:{ap}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    samples, points, counts = [], [], []
    eligible = np.flatnonzero([len(rows[i]) for i in range(len(rows))])
    for _ in range(algorithm.trials):
        if len(eligible) < 3:
            break
        ids = rng.choice(eligible, 3, replace=False)
        slots = [rng.integers(len(rows[i])) for i in ids]
        sample = np.column_stack((ids, slots))
        hypotheses = algorithm.hypothesis_solver.solve(survey.xyz[ids],
            np.array([rows[i][j] for i, j in sample]))
        samples.append(sample)
        counts.append(len(hypotheses))
        points.extend(hypotheses)
    return {"points": np.asarray(points).reshape(-1, 3), "samples": np.asarray(samples),
            "counts": np.asarray(counts, dtype=int)}


def retain_ransac(survey, pool, algorithm):
    rows = next(iter(survey.channels.values()))
    counts = pool["counts"][:algorithm.trials]
    points = pool["points"][:sum(counts)]
    scores = []
    for start in range(0, len(points), 512):
        errors, _ = nearest_many(points[start:start+512], survey.xyz, rows.values, rows.offsets,
                                 np.ones(len(rows.values), dtype=bool))
        scores.extend(score_many(errors, algorithm.scorer))
    def trials():
        order = 0
        for trial, n in enumerate(counts):
            yield [{"point": points[i], "score": scores[i], "order": i, "trial": trial}
                   for i in range(order, order + n)]
            order += n
    retained = algorithm.retainer.retain(trials())
    return {"xyz": np.asarray([r["point"] for r in retained]).reshape(-1, 3),
            "retained": np.asarray([r["order"] for r in retained], dtype=int),
            "scores": np.asarray(scores)}


class AdditionalExecution:
    def __init__(self, family, parameters, store, *, seed=0, max_trials=30000):
        self.family, self.parameters, self.store = family, parameters, store
        self.seed, self.max_trials = seed, max_trials

    def fit(self, survey):
        started = time.perf_counter()
        outputs, evidence = {}, {}
        for ap, rows in survey.channels.items():
            one = Survey(survey.xyz, {ap: rows}, survey.position_ids)
            inputs = {"xyz": survey.xyz, "rows": rows.record(), "ap": ap}
            p = self.parameters
            if self.family == "zayets":
                algorithm = zayets_proposals(p)
                proposals, proposal_key = self.store.compute("zayets-proposals.1",
                    {"algorithm": algorithm}, inputs,
                    lambda: zayets_one(one, algorithm))
                pool_key = None
            else:
                algorithm = ransac_proposals(p, self.seed)
                full = ransac_proposals({**p, "trials": self.max_trials}, self.seed)
                pool, pool_key = self.store.compute("ransac-hypotheses.1",
                    {"solver": full.hypothesis_solver, "trials": full.trials,
                     "seed": full.seed}, inputs,
                    lambda: ransac_pool(one, full))
                proposals, proposal_key = self.store.compute("ransac-retention.1",
                    {"scorer": algorithm.scorer, "retainer": algorithm.retainer,
                     "trials": algorithm.trials},
                    {**inputs, "pool": ArtifactRef(pool_key)},
                    lambda: retain_ransac(one, pool, algorithm))
            selector = finisher(self.family, p)
            selected, selected_key = self.store.compute("consuming-finisher.1",
                {"selector": selector},
                {**inputs, "proposals": ArtifactRef(proposal_key)},
                lambda: finish_one(one, SourceMap({ap: proposals["xyz"]}), selector))
            outputs[ap] = selected["xyz"]
            evidence[ap] = {"proposals": proposal_key, "selection": selected_key,
                            **({"hypotheses": pool_key} if pool_key else {})}
        evidence["seconds"] = time.perf_counter() - started
        return SourceMap(outputs, evidence)
