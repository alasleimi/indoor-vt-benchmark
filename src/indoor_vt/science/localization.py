"""Fingerprint scoring, neighbor estimation, and source-assisted localization."""

from dataclasses import dataclass
import numpy as np

from .models import Prediction


@dataclass(frozen=True)
class MatchBasedScorer:
    matcher: object
    coverage_penalty: bool = True
    higher_is_better: bool = False

    def __call__(self, query, reference):
        if not len(query) or not len(reference):
            return np.inf
        matches = self.matcher.match_array(query, reference)
        if not len(matches):
            return np.inf
        score = float(matches[:, 2].mean())
        if self.coverage_penalty:
            score /= len(matches) / max(len(query), len(reference))
        return score


@dataclass(frozen=True)
class MCAScorer:
    epsilon_m: float = 1.0
    higher_is_better: bool = True

    def __call__(self, query, reference):
        if not len(query) or not len(reference):
            return 0.0
        residual = np.abs(query[:, None] - reference[None, :]).min(axis=1)
        return float(np.square(np.maximum(self.epsilon_m - residual, 0)).sum())


@dataclass(frozen=True)
class ChamferScorer:
    higher_is_better: bool = False

    def __call__(self, query, reference):
        if not len(query) or not len(reference):
            return np.inf
        difference = np.abs(query[:, None] - reference[None, :])
        return float((difference.min(0).mean() + difference.min(1).mean()) / 2)


@dataclass(frozen=True)
class DistanceWeightedKNN:
    neighbors: int = 3
    power: float = 1.0
    higher_is_better: bool = False

    def estimate(self, xyz, scores):
        if self.neighbors < 1 or self.power <= 0:
            raise ValueError("Neighbor count and weighting power must be positive")
        valid = np.flatnonzero(np.isfinite(scores))
        if not len(valid):
            return Prediction(xyz.mean(0), "no_match_centroid")
        ids = valid[np.argsort(scores[valid], kind="stable")[:self.neighbors]]
        values = scores[ids]
        exact = values <= np.finfo(float).eps
        weights = exact.astype(float) if exact.any() else (values.min() / values)**self.power
        weights /= weights.sum()
        return Prediction(weights @ xyz[ids], evidence={"neighbors": ids, "weights": weights})


@dataclass(frozen=True)
class SimilarityWeightedKNN:
    neighbors: int = 3
    higher_is_better: bool = True

    def estimate(self, xyz, scores):
        ids = np.argsort(-scores, kind="stable")[:self.neighbors]
        weights = scores[ids]
        if not np.any(weights > 0):
            return Prediction(xyz[ids[0]], "zero_similarity", {"neighbors": ids})
        return Prediction((weights / weights.sum()) @ xyz[ids], evidence={"neighbors": ids})


@dataclass
class FingerprintLocalizer:
    scorer: object
    estimator: object

    def fit(self, survey):
        if self.scorer.higher_is_better != self.estimator.higher_is_better:
            raise ValueError("Scorer and estimator directions differ")
        self.survey = survey
        return self

    def predict(self, query):
        scores = np.array([np.mean([self.scorer(values, self.survey.channels[ap][i])
                                    for ap, values in query.items()])
                           for i in range(len(self.survey.xyz))])
        return self.estimator.estimate(self.survey.xyz, scores)


@dataclass(frozen=True)
class SmartLEA:
    matcher: object
    estimator: object
    iterations: int = 1

    def step(self, query, point, sources):
        anchors, measured, associations = [], [], []
        for ap, xyz in sources.by_ap.items():
            ranges = query.get(ap, np.empty(0))
            predicted = np.linalg.norm(xyz - point, axis=1)
            for i, j, cost in self.matcher.match_array(ranges, predicted):
                anchors.append(xyz[int(j)])
                measured.append(ranges[int(i)])
                associations.append((ap, int(i), int(j), float(cost)))
        if not anchors:
            return Prediction(point, "fallback_no_matches", {"associations": []})
        updated, detail = self.estimator.fit(np.asarray(anchors), np.asarray(measured), initial=point)
        evidence = {"associations": associations, "fit": detail}
        if updated is None:
            return Prediction(point, "fallback_fit_failed", evidence)
        return Prediction(updated, evidence=evidence)

    def predict(self, query, initial, sources):
        point, trace = initial.xyz.copy(), []
        for _ in range(self.iterations):
            result = self.step(query, point, sources)
            trace.append(result.evidence)
            if result.status != "ok":
                return Prediction(initial.xyz, result.status, {"iterations": trace})
            point = result.xyz
        return Prediction(point, evidence={"iterations": trace})
