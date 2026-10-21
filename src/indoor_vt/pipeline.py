"""Composition receives components; it has no access to evaluator truth."""

from dataclasses import dataclass
import numpy as np
from .contracts import Prediction


@dataclass
class Pipeline:
    initial_localizer: object
    reconstructor: object
    refiner: object

    def fit(self, survey):
        self.initial_localizer.fit(survey)
        self.sources = self.reconstructor.fit(survey)
        return self

    def predict(self, query):
        initial = self.initial_localizer.predict(query)
        if not np.isfinite(initial.xyz).all():
            return initial, Prediction(initial.xyz, "unresolved_initial_failure")
        final = self.refiner.predict(query, initial, self.sources)
        if not np.isfinite(final.xyz).all():
            final = Prediction(initial.xyz.copy(), "fallback_invalid_refinement")
        return initial, final
