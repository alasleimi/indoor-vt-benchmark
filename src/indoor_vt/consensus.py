"""SAPP source estimators adapted to signed 3D and a support-based stopping rule.

No physical AP is supplied. The original unsigned-height fitter and arbitrary
20-source output ceiling are removed. See docs/methods.md for differences.
"""

from dataclasses import dataclass, asdict
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .contracts import SourceMap
from .fitting import fit_source


@dataclass(frozen=True)
class ConsensusConfig:
    method: str = "ransac"
    trials: int = 30000
    batch: int = 400
    retained_per_batch: int = 12
    inlier_scale_m: float = 0.18
    support_fraction: float = 0.08
    surface_separation_m: float = 0.12
    neighborhood_radius_m: float = 1.0
    cluster_gap_m: float = 0.6
    group_threshold: float = 0.5


def padded(rows):
    result = np.full((len(rows), max(map(len, rows), default=0)), np.inf)
    for i, row in enumerate(rows):
        result[i, : len(row)] = row
    return result


def score(sources, positions, available, scale, mode):
    curves = np.linalg.norm(sources[:, None, :] - positions[None, :, :], axis=2)
    residual = abs(curves[:, :, None] - available[None]).min(axis=2)
    support = (residual <= scale).sum(axis=1)
    smooth = np.exp(-0.5 * (residual / scale) ** 2).sum(axis=1)
    if mode == "ransac":
        scores = (
            support
            + 0.35 * smooth
            - 0.10 * np.median(np.minimum(residual, 4 * scale), axis=1) / scale
        )
    else:
        scores = support + smooth / (len(positions) + 1)
    return scores, support, residual, curves


def solve_three(positions, ranges):
    """Both intersections of three 3D spheres; full-survey consensus resolves sign.

    This preserves SAPP's three-reference hypothesis size without assuming a
    horizontal receiver plane or discarding the reflected signed solution.
    """
    a, b, c = np.asarray(positions)
    d = np.linalg.norm(b - a)
    if d <= 1e-10:
        return []
    ex = (b - a) / d
    i = np.dot(ex, c - a)
    residual = c - a - i * ex
    j = np.linalg.norm(residual)
    if j <= 1e-10:
        return []
    ey = residual / j
    ez = np.cross(ex, ey)
    x = (ranges[0] ** 2 - ranges[1] ** 2 + d * d) / (2 * d)
    y = (ranges[0] ** 2 - ranges[2] ** 2 + i * i + j * j - 2 * i * x) / (2 * j)
    z2 = ranges[0] ** 2 - x * x - y * y
    if z2 < -1e-10:
        return []
    center, z = a + x * ex + y * ey, np.sqrt(max(z2, 0.0))
    return [center + z * ez, center - z * ez] if z > 1e-10 else [center]


def matched_tracks(positions, rows, gap, threshold):
    nodes = []
    for i, row in enumerate(rows):
        if len(row):
            nodes.extend(
                (i, ids)
                for ids in np.split(np.arange(len(row)), np.flatnonzero(np.diff(row) >= gap) + 1)
            )
    if not nodes:
        return []
    features = np.array(
        [[rows[i][ids[0]], len(ids), rows[i][ids[-1]] - rows[i][ids[0]]] for i, ids in nodes]
    )
    edges = sorted(
        (0.5 * abs(features[a] - features[b]).sum(), a, b)
        for a, b in combinations(range(len(nodes)), 2)
        if nodes[a][0] != nodes[b][0] and 0.5 * abs(features[a] - features[b]).sum() <= threshold
    )
    parent = list(range(len(nodes)))
    owners = [{i} for i, ids in nodes]

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, a, b in edges:
        a, b = root(a), root(b)
        if a != b and owners[a].isdisjoint(owners[b]):
            parent[b] = a
            owners[a].update(owners[b])
    groups = {}
    for j, node in enumerate(nodes):
        groups.setdefault(root(j), []).append(node)
    tracks = []
    for group in groups.values():
        length = max(len(ids) for _, ids in group)
        target = np.mean([rows[i][ids] for i, ids in group if len(ids) == length], axis=0)
        slots = [[] for _ in range(length)]
        for i, ids in group:
            cols = (
                np.arange(length)
                if len(ids) == length
                else linear_sum_assignment(abs(rows[i][ids, None] - target[None]))[1]
            )
            for j, col in zip(ids, cols):
                slots[col].append((i, int(j)))
        tracks.extend(tuple(slot) for slot in slots if len(slot) >= 4)
    return tracks


class ConsensusReconstructor:
    def __init__(self, config=ConsensusConfig(), seed=0, fitter=fit_source):
        if config.method not in ("ransac", "zayets2018"):
            raise ValueError(config.method)
        self.config, self.seed, self.fitter = config, int(seed), fitter

    def fit(self, survey):
        from .provenance import SeedBook

        c = self.config
        outputs, audit = {}, {}
        for ap in sorted({ap for o in survey.observations for ap in o.channels}):
            rows = [o.channels.get(ap, np.empty(0)) for o in survey.observations]
            packed = padded(rows)
            if packed.shape[1] == 0:
                outputs[ap], audit[ap] = np.empty((0, 3)), {"status": "empty_observations"}
                continue
            eligible = np.flatnonzero([len(r) > 0 for r in rows])
            if len(eligible) < 4:
                outputs[ap], audit[ap] = np.empty((0, 3)), {"status": "insufficient_positions"}
                continue
            candidates, memberships, fit_diagnostics = [], [], []
            rng = SeedBook(self.seed, "consensus").rng(ap)
            if c.method == "ransac":
                for start in range(0, c.trials, c.batch):
                    batch, sampled = [], []
                    for _ in range(min(c.batch, c.trials - start)):
                        ids = rng.choice(eligible, 3, replace=False)
                        slots = [int(rng.integers(len(rows[i]))) for i in ids]
                        hypotheses = solve_three(
                            survey.xyz[ids], np.array([rows[i][j] for i, j in zip(ids, slots)])
                        )
                        for point in hypotheses:
                            batch.append(point)
                            sampled.append([(int(i), j) for i, j in zip(ids, slots)])
                    if batch:
                        scores = score(
                            np.array(batch), survey.xyz, packed, c.inlier_scale_m, c.method
                        )[0]
                        keep = np.argsort(-scores, kind="stable")[: c.retained_per_batch]
                        candidates.extend(batch[i] for i in keep)
                        memberships.extend(sampled[i] for i in keep)
            else:
                tree = cKDTree(survey.xyz)
                scopes = sorted(
                    {
                        tuple(sorted(ids))
                        for ids in tree.query_ball_point(survey.xyz, c.neighborhood_radius_m)
                    }
                )
                all_tracks = set()
                for ids in scopes:
                    tracks = matched_tracks(
                        survey.xyz[list(ids)],
                        [rows[i] for i in ids],
                        c.cluster_gap_m,
                        c.group_threshold,
                    )
                    all_tracks.update(tuple((ids[i], j) for i, j in track) for track in tracks)
                for members in sorted(all_tracks):
                    point, detail = self.fitter(
                        survey.xyz[[i for i, j in members]], [rows[i][j] for i, j in members]
                    )
                    fit_diagnostics.append({"members": members, **detail})
                    if point is not None:
                        candidates.append(point)
                        memberships.append(members)
            candidates = np.asarray(candidates).reshape(-1, 3)
            available = packed.copy()
            active = np.ones(len(candidates), bool)
            required = max(5, int(np.ceil(c.support_fraction * len(rows))))
            chosen, support, associations = [], [], []
            # Every successful iteration consumes evidence; every failed one disables a
            # proposal. This is finite without assuming the true number of sources.
            while active.any():
                indices = np.flatnonzero(active)
                best, best_key = None, (-np.inf, -1)
                for start in range(0, len(indices), c.batch):
                    ids = indices[start : start + c.batch]
                    scores, counts, _, _ = score(
                        candidates[ids], survey.xyz, available, c.inlier_scale_m, c.method
                    )
                    local = int(np.argmax(scores))
                    key = (float(scores[local]), int(counts[local]))
                    if key > best_key:
                        best, best_key = int(ids[local]), key
                if best is None or best_key[1] < required:
                    break
                point = candidates[best].copy()
                active[best] = False
                if c.method == "ransac":
                    prediction = np.linalg.norm(survey.xyz - point, axis=1)
                    differences = abs(available - prediction[:, None])
                    nearest = differences.argmin(axis=1)
                    residual = differences[np.arange(len(rows)), nearest]
                    hit = residual <= 2 * c.inlier_scale_m
                    weights = np.exp(-0.5 * (residual[hit] / (2 * c.inlier_scale_m)) ** 2)
                    fitted, _ = self.fitter(
                        survey.xyz[hit],
                        available[np.flatnonzero(hit), nearest[hit]],
                        initial=point,
                        weights=weights,
                        loss="soft_l1",
                        f_scale=max(c.inlier_scale_m, 0.03),
                        max_nfev=120,
                    )
                    if fitted is not None:
                        point = fitted
                _, counts, residuals, curves = score(
                    point[None], survey.xyz, available, c.inlier_scale_m, c.method
                )
                curve = curves[0]
                if counts[0] < required or any(
                    np.sqrt(np.mean((np.linalg.norm(survey.xyz - old, axis=1) - curve) ** 2))
                    < c.surface_separation_m
                    for old in chosen
                ):
                    continue
                chosen.append(point)
                support.append(int(counts[0]))
                hit = np.flatnonzero(residuals[0] <= c.inlier_scale_m)
                slots = abs(available[hit] - curve[hit, None]).argmin(axis=1)
                associations.append(list(zip(hit.tolist(), slots.tolist())))
                available[hit, slots] = np.inf
                for start in range(0, len(candidates), c.batch):
                    ids = np.arange(start, min(start + c.batch, len(candidates)))
                    other_curves = np.linalg.norm(candidates[ids, None] - survey.xyz[None], axis=2)
                    active[
                        ids[
                            np.sqrt(np.mean((other_curves - curve) ** 2, axis=1))
                            < c.surface_separation_m
                        ]
                    ] = False
            outputs[ap] = np.asarray(chosen).reshape(-1, 3)
            audit[ap] = {
                "candidates": candidates.tolist(),
                "candidate_members": memberships,
                "fit_diagnostics": fit_diagnostics,
                "support": support,
                "associations": associations,
                "minimum_support": required,
            }
        return SourceMap(
            outputs,
            {
                "version": "sapp.source_adaptation.3d.v1",
                "config": asdict(c),
                "seed": self.seed,
                "channels": audit,
            },
        )
