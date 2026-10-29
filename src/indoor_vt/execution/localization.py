"""Reuse match statistics across KNN settings and SMART-LEA iteration prefixes."""
import numpy as np

from .artifacts import ArtifactRef
from .reconstruction import effective_matcher
from ..science.models import Ragged
from ..science.localization import MatchBasedScorer


class LocalizerExecution:
    def __init__(self, localizer, store, run=""):
        self.localizer, self.store, self.run = localizer, store, run

    def predict(self, survey, queries):
        scorer, estimator = self.localizer.scorer, self.localizer.estimator
        if scorer.higher_is_better != estimator.higher_is_better:
            raise ValueError("Scorer and estimator directions differ")
        tables, references = [], {}
        for ap, rows in queries.channels.items():
            reference = survey.channels[ap]
            inputs = {"queries": rows.record(), "references": reference.record()}
            if isinstance(scorer, MatchBasedScorer):
                matcher = effective_matcher(scorer.matcher,
                    np.r_[np.diff(rows.offsets), np.diff(reference.offsets)])
                def match_statistics():
                    values = np.empty((len(rows), len(reference), 2))
                    for q in range(len(rows)):
                        for r in range(len(reference)):
                            matches = matcher.match_array(rows[q], reference[r])
                            values[q, r] = (matches[:, 2].mean() if len(matches) else np.inf, len(matches))
                    return values
                statistics, key = self.store.compute("fingerprint-match-statistics.2", matcher,
                    inputs, match_statistics)
                scores = statistics[:, :, 0].copy()
                if scorer.coverage_penalty:
                    counts = np.maximum(np.diff(rows.offsets)[:, None], np.diff(reference.offsets)[None, :])
                    coverage = np.divide(statistics[:, :, 1], counts,
                        out=np.zeros_like(scores), where=counts > 0)
                    scores = np.divide(scores, coverage, out=np.full_like(scores, np.inf), where=coverage > 0)
            else:
                scores, key = self.store.compute("fingerprint-scores.2", scorer, inputs,
                    lambda: np.array([[scorer(rows[q], reference[r]) for r in range(len(reference))]
                                      for q in range(len(rows))]))
            tables.append(scores)
            references[ap] = key
        scores = np.mean(tables, axis=0)
        def predict():
            predictions = [estimator.estimate(survey.xyz, row) for row in scores]
            return {"xyz": np.array([p.xyz for p in predictions]),
                    "status": np.array([p.status for p in predictions], dtype="U32"),
                    "neighbors": Ragged.pack([p.evidence.get("neighbors", []) for p in predictions],
                                              dtype=np.int64).record(),
                    "weights": Ragged.pack([p.evidence.get("weights", []) for p in predictions]).record()}
        result, key = self.store.compute("neighbor-estimation.2", estimator,
            {"xyz": survey.xyz, "scores": scores}, predict, persist=False)
        return result, {"statistics": references, "predictions": key}


def lea_step(lea, queries, original, previous, sources):
    aps = list(sources.by_ap)
    positions, statuses, members, costs, residuals, evaluations = [], [], [], [], [], []
    for i in range(len(queries.xyz)):
        if previous["status"][i] != "ok":
            # Failed refinement remains terminal, and falls back to the original initializer.
            positions.append(original[i])
            statuses.append(previous["status"][i])
            members.append(np.empty((0, 3), dtype=np.int64))
            costs.append([])
            residuals.append([])
            evaluations.append(0)
            continue
        result = lea.step({ap: rows[i] for ap, rows in queries.channels.items()},
                          previous["xyz"][i], sources)
        positions.append(result.xyz if result.status == "ok" else original[i])
        statuses.append(result.status)
        associations = result.evidence.get("associations", [])
        members.append(np.asarray([(aps.index(ap), q, r) for ap, q, r, cost in associations],
                                   dtype=np.int64).reshape(-1, 3))
        costs.append([cost for ap, q, r, cost in associations])
        detail = result.evidence.get("fit", {})
        residuals.append(detail.get("residuals", []))
        evaluations.append(detail.get("evaluations", 0))
    return {"xyz": np.asarray(positions), "status": np.asarray(statuses, dtype="U32"),
            "aps": aps, "associations": Ragged.pack(members, width=3, dtype=np.int64).record(),
            "costs": Ragged.pack(costs).record(), "residuals": Ragged.pack(residuals).record(),
            "evaluations": np.asarray(evaluations, dtype=np.int32)}


class LEAExecution:
    def __init__(self, lea, store, run=""):
        self.lea, self.store, self.run = lea, store, run

    def predict(self, queries, initial, sources):
        if self.lea.iterations < 1:
            raise ValueError("SMART-LEA iterations must be positive")
        original = initial["xyz"]
        # A fallback initializer can still be refined. Only refinement failures are terminal.
        result = {"xyz": original, "status": np.full(len(original), "ok", dtype="U32")}
        common = {"observations": {ap: rows.record() for ap, rows in queries.channels.items()},
                  "original_xyz": original, "sources": sources.by_ap}
        counts = [len(rows[i]) for rows in queries.channels.values() for i in range(len(rows))]
        counts += [len(xyz) for xyz in sources.by_ap.values()]
        parameters = {"matcher": effective_matcher(self.lea.matcher, counts), "estimator": self.lea.estimator}
        references = []
        for _ in range(self.lea.iterations):
            previous = result
            inputs = {**common, "previous": ArtifactRef(references[-1]) if references else previous}
            result, key = self.store.compute("smart-lea-step.2", parameters, inputs,
                lambda: lea_step(self.lea, queries, original, previous, sources))
            references.append(key)
        return result, {"iterations": references}
