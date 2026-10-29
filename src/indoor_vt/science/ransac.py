"""Random source proposals and retention; final selection is an external stage."""

from dataclasses import dataclass
import hashlib
import heapq
import numpy as np

from .models import SourceMap
from .selection import nearest_peaks


def uniform_peak_sample(rows, size, rng):
    eligible = np.flatnonzero([len(rows[i]) for i in range(len(rows))])
    if len(eligible) < size:
        return None
    ids = rng.choice(eligible, size, replace=False)
    return np.column_stack((ids, [rng.integers(len(rows[i])) for i in ids]))


@dataclass(frozen=True)
class ThreeSphereIntersections:
    degeneracy_tolerance: float = 1e-10

    @property
    def sample_size(self):
        return 3

    def solve(self, positions, ranges):
        a, b, c = np.asarray(positions)
        d = np.linalg.norm(b - a)
        if d <= self.degeneracy_tolerance:
            return []
        ex = (b - a) / d
        i = np.dot(ex, c - a)
        normal = c - a - i * ex
        j = np.linalg.norm(normal)
        if j <= self.degeneracy_tolerance:
            return []
        ey = normal / j
        ez = np.cross(ex, ey)
        x = (ranges[0]**2 - ranges[1]**2 + d*d) / (2*d)
        y = (ranges[0]**2 - ranges[2]**2 + i*i + j*j - 2*i*x) / (2*j)
        z2 = ranges[0]**2 - x*x - y*y
        if z2 < -self.degeneracy_tolerance:
            return []
        center, z = a + x*ex + y*ey, np.sqrt(max(z2, 0.))
        return [center + z*ez, center - z*ez] if z > self.degeneracy_tolerance else [center]


@dataclass(frozen=True)
class FittedSourceHypotheses:
    estimator: object
    sample_size: int = 4

    def solve(self, positions, ranges):
        point, _ = self.estimator.fit(positions, ranges)
        return [] if point is None else [point]


def retain_top(records, keep):
    return heapq.nlargest(keep, records, key=lambda r: (r["score"], -r["order"]))


@dataclass(frozen=True)
class TopPerTrialBatch:
    trials_per_batch: int = 400
    keep: int = 12

    def retain(self, trials):
        if self.trials_per_batch < 1 or self.keep < 1:
            raise ValueError("Retention counts must be positive")
        selected, batch = [], []
        for i, proposals in enumerate(trials):
            batch.extend(proposals)
            if (i + 1) % self.trials_per_batch == 0:
                selected.extend(retain_top(batch, self.keep))
                batch = []
        return selected + retain_top(batch, self.keep)


@dataclass(frozen=True)
class GlobalTopK:
    keep: int = 900

    def retain(self, trials):
        if self.keep < 1:
            raise ValueError("Retention count must be positive")
        heap = []
        for proposals in trials:
            for proposal in proposals:
                item = (proposal["score"], -proposal["order"], proposal)
                if len(heap) < self.keep:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
        return [record for _, _, record in sorted(heap, reverse=True)]


@dataclass(frozen=True)
class RansacReconstructor:
    sampler: object
    hypothesis_solver: object
    scorer: object
    retainer: object
    trials: int = 30000
    seed: int = 0

    def fit(self, survey):
        output, audit = {}, {}
        for ap, rows in survey.channels.items():
            ap_seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{ap}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(ap_seed)
            samples, hypotheses, scores, counts = [], [], [], []
            def trials():
                order = 0
                for trial in range(self.trials):
                    members = self.sampler(rows, self.hypothesis_solver.sample_size, rng)
                    if members is None:
                        break
                    ids, slots = members.T
                    points = self.hypothesis_solver.solve(survey.xyz[ids], np.array([rows[i][j] for i, j in members]))
                    samples.append(members)
                    counts.append(len(points))
                    records = []
                    for point in points:
                        score = self.scorer(nearest_peaks(point, survey.xyz, rows)[0])
                        records.append({"point": point, "score": score, "order": order, "trial": trial})
                        hypotheses.append(point)
                        scores.append(score)
                        order += 1
                    yield records
            retained = self.retainer.retain(trials())
            output[ap] = np.asarray([r["point"] for r in retained]).reshape(-1, 3)
            audit[ap] = {"sample_members": np.array(samples), "hypothesis_counts": np.array(counts),
                         "hypotheses": np.array(hypotheses).reshape(-1, 3), "scores": np.array(scores),
                         "retained": np.array([r["order"] for r in retained], dtype=int)}
        return SourceMap(output, audit)
