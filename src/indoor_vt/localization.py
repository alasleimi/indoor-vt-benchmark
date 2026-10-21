"""Initial localizers and SMART-LEA accept only observable survey/query contracts."""

from dataclasses import dataclass
import numpy as np
from .contracts import DelayObservation, Prediction, SourceMap, Survey
from .fitting import fit_source
from .matching import MatcherConfig, PatternMatcher


@dataclass(frozen=True)
class InitialConfig:
    method: str = "mpurge_map"
    neighbors: int = 3
    half_window: int = 2
    alpha: float = 0.7
    epsilon_m: float = 1.0
    distance_power: float = 1.0


class FingerprintLocalizer:
    def __init__(self, config=InitialConfig(), matcher=None):
        if config.method not in ("mpurge_map", "mpurge", "mca", "chamfer"):
            raise ValueError(config.method)
        self.config = config
        self.matcher = matcher or PatternMatcher(
            MatcherConfig(
                config.half_window, config.alpha, normalized_pattern=config.method == "mpurge_map"
            )
        )

    def fit(self, survey: Survey):
        self.survey = survey
        return self

    def score_pair(self, query, reference):
        c = self.config
        if not len(query) or not len(reference):
            return 0.0 if c.method == "mca" else np.inf
        if c.method == "mca":
            nearest = np.abs(query[:, None] - reference[None]).min(axis=1)
            return float(np.square(np.maximum(c.epsilon_m - nearest, 0)).sum())
        if c.method == "chamfer":
            delta = np.abs(query[:, None] - reference[None])
            return float((delta.min(0).mean() + delta.min(1).mean()) / 2)
        matches = self.matcher.match_array(query, reference)
        if not len(matches):
            return np.inf
        score = float(matches[:, 2].mean())
        if c.method == "mpurge_map":
            score /= len(matches) / max(len(query), len(reference))
        return score

    def predict(self, query: DelayObservation):
        scores = []
        aps = sorted(query.channels)
        for reference in self.survey.observations:
            parts = [
                self.score_pair(query.channels[ap], reference.channels.get(ap, np.empty(0)))
                for ap in aps
            ]
            scores.append(
                float(np.mean(parts)) if parts else (0 if self.config.method == "mca" else np.inf)
            )
        scores = np.asarray(scores)
        c = self.config
        if c.method == "mca":
            selected = np.argsort(-scores, kind="stable")[: c.neighbors]
            weights = scores[selected]
            if c.neighbors == 1 or not np.any(weights > 0):
                return Prediction(
                    self.survey.xyz[selected[0]].copy(),
                    diagnostics={"neighbors": selected.tolist()},
                )
        else:
            valid = np.flatnonzero(np.isfinite(scores))
            if not len(valid):
                return Prediction(self.survey.xyz.mean(0), "no_match_centroid")
            selected = valid[np.argsort(scores[valid], kind="stable")[: c.neighbors]]
            values = scores[selected]
            exact = values <= np.finfo(float).eps
            weights = exact.astype(float) if exact.any() else 1 / values**c.distance_power
        weights = weights / weights.sum()
        return Prediction(
            weights @ self.survey.xyz[selected], diagnostics={"neighbors": selected.tolist()}
        )


@dataclass(frozen=True)
class LEAConfig:
    iterations: int = 1
    half_window: int = 2
    alpha: float = 0.5
    max_nfev: int = 100


class SmartLEA:
    def __init__(self, config=LEAConfig(), matcher=None, fitter=fit_source):
        self.config, self.fitter = config, fitter
        self.matcher = matcher or PatternMatcher(MatcherConfig(config.half_window, config.alpha))

    def predict(self, query, initial, sources: SourceMap):
        position = np.asarray(initial.xyz).copy()
        trace = []
        for _ in range(self.config.iterations):
            selected_sources, selected_ranges = [], []
            for ap, anchors in sources.by_ap.items():
                measured = query.channels.get(ap, np.empty(0))
                predicted = np.linalg.norm(anchors - position, axis=1)
                for match in self.matcher(measured, predicted):
                    selected_sources.append(anchors[match.right])
                    selected_ranges.append(measured[match.left])
            if not selected_sources:
                return Prediction(
                    initial.xyz.copy(), "fallback_no_sources_or_matches", {"trace": trace}
                )
            updated, audit = self.fitter(
                np.array(selected_sources),
                np.array(selected_ranges),
                initial=position,
                max_nfev=self.config.max_nfev,
            )
            trace.append(audit)
            if updated is None:
                return Prediction(initial.xyz.copy(), "fallback_lea_failure", {"trace": trace})
            position = updated
        return Prediction(position, diagnostics={"trace": trace})
