"""Named experiments assemble scientific components once."""

from functools import partial

from ..science.clustering import GreedyCompleteLink
from ..science.costs import absolute_difference, median
from ..science.estimation import Multilateration3D
from ..science.grouping import nearest_neighborhoods, NeighborhoodPairs, ConnectedComponentsGrouper
from ..science.matching import MPUrgeMatcher
from ..science.reconstruction import MPUrge, RelativeSupportFilter
from ..science.localization import FingerprintLocalizer, MatchBasedScorer, DistanceWeightedKNN, SmartLEA


def standard_mpurge(parameters, *, unify):
    neighborhoods = partial(nearest_neighborhoods, size=parameters["neighbors"])
    return MPUrge(
        pair_selector=NeighborhoodPairs(neighborhoods), group_scope_selector=neighborhoods,
        matcher=MPUrgeMatcher(half_window=parameters["half_window"], alpha=parameters["alpha"],
                             size_unification_norm=absolute_difference if unify else None),
        grouper=ConnectedComponentsGrouper(), estimator=Multilateration3D(),
        clusterer=GreedyCompleteLink(radius_m=parameters["radius_m"], representative=median),
        support_filter=RelativeSupportFilter(beta=parameters["beta"]))


def mpurge_map(parameters, *, unify=True):
    return FingerprintLocalizer(
        scorer=MatchBasedScorer(MPUrgeMatcher(half_window=parameters["half_window"],
            alpha=parameters["alpha"], normalized_pattern=True,
            size_unification_norm=absolute_difference if unify else None), coverage_penalty=True),
        estimator=DistanceWeightedKNN(neighbors=parameters["neighbors"], power=parameters["power"]))


def smart_lea(parameters, *, unify=True):
    return SmartLEA(matcher=MPUrgeMatcher(half_window=parameters["half_window"], alpha=parameters["alpha"],
                    size_unification_norm=absolute_difference if unify else None),
                    estimator=Multilateration3D(), iterations=parameters["iterations"])


DEFAULTS = {
    "mpurge_unified": {"half_window": 2, "alpha": 0.5, "neighbors": 10, "radius_m": 0.5, "beta": 1.0},
    "mpurge_ununified": {"half_window": 2, "alpha": 0.5, "neighbors": 10, "radius_m": 0.5, "beta": 1.0},
    "mpurge_map": {"half_window": 2, "alpha": 0.7, "neighbors": 3, "power": 1.0},
    "smart_lea_truth": {"half_window": 2, "alpha": 0.5, "iterations": 1},
}
