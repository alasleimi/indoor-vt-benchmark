"""Explicit proposal/finisher compositions, independent of the frozen studies."""
from dataclasses import dataclass, replace
from pathlib import Path
import time

import numpy as np

from .additional import (finisher, finish_one, zayets_proposals,
    ransac_proposals, zayets_one, ransac_pool, retain_ransac)
from .artifacts import ArtifactRef
from .presets import standard_mpurge
from .reconstruction import MPUrgeExecution, effective_matcher, fit_groups
from ..acquisition.records import digest
from ..science.models import Survey, SourceMap
from ..science.grouping import PairMatches, group_within_scopes
from ..science.reconstruction import match_pairs
from ..science.matching import CrossingAllowedMPUrgeMatcher


VARIANTS = {
    "zayets_robust": ("zayets", "robust"),
    "ransac_consumption": ("ransac", "consumption"),
    "mpurge_consumption": ("mpurge", "consumption"),
    "mpurge_robust": ("mpurge", "robust"),
    "mpurge_triangle": ("mpurge", None),
    "mpurge_allow_crossing": ("mpurge", None),
    "mpurge_allow_crossing_consumption": ("mpurge", "consumption"),
    "mpurge_allow_crossing_robust": ("mpurge", "robust"),
    "zayets_naked": ("zayets", None),
    "ransac_naked": ("ransac", None),
}
FINISHER_NAMES = {"consumption": "support-consumption finisher",
                  "robust": "robust support finisher"}


def composition_id():
    return digest({name: Path(__file__).with_name(name).read_text(encoding="utf-8")
                   for name in ("compositions.py", "composition_experiment.py")})


@dataclass(frozen=True)
class FinishedReconstructor:
    """The selector replaces candidate clustering/filtering, rather than following it."""
    proposals: object
    selector: object


def composed_reconstructor(method, parameters, *, unify, seed=9173):
    """Dependency injection for every new comparison: proposer plus named finisher."""
    family, finishing = VARIANTS[method]
    p = parameters
    if family == "mpurge":
        proposal = standard_mpurge({"radius_m": .5, "beta": 1., **p}, unify=unify)
        if method.startswith("mpurge_allow_crossing"):
            proposal = replace(proposal, matcher=CrossingAllowedMPUrgeMatcher(**vars(proposal.matcher)))
        if finishing is None:
            return (replace(proposal, matcher=replace(proposal.matcher,
                triangle_tolerance_m=p["triangle_tolerance_m"])) if method == "mpurge_triangle" else proposal)
    elif family == "zayets":
        proposal = zayets_proposals(p)
    else:
        proposal = ransac_proposals(p, seed)
    if finishing is None:
        return proposal
    selector = finisher("zayets" if finishing == "consumption" else "ransac", p)
    return FinishedReconstructor(proposal, selector)


class CompositionExecution:
    def __init__(self, method, parameters, store, *, unify, seed=9173, max_trials=30000,
                 pair_bank=None):
        self.method, self.parameters, self.store = method, parameters, store
        self.unify, self.seed, self.max_trials, self.pair_bank = unify, seed, max_trials, pair_bank

    def fit(self, survey):
        started = time.perf_counter()
        family, finishing = VARIANTS[self.method]
        algorithm = composed_reconstructor(self.method, self.parameters,
                                          unify=self.unify, seed=self.seed)
        if family == "mpurge" and finishing is None:
            return MPUrgeExecution(algorithm, self.store, pair_bank=self.pair_bank).fit(survey)
        c, selector = ((algorithm, None) if finishing is None else
                       (algorithm.proposals, algorithm.selector))
        outputs, evidence = {}, {}
        if family == "mpurge":
            pairs, scopes = c.pair_selector(survey.xyz), c.group_scope_selector(survey.xyz)
            bank = pairs if self.pair_bank is None else self.pair_bank
        for ap, rows in survey.channels.items():
            one = Survey(survey.xyz, {ap: rows}, survey.position_ids)
            inputs = {"xyz": survey.xyz, "rows": rows.record(), "ap": ap}
            if family == "mpurge":
                matcher = effective_matcher(c.matcher, np.diff(rows.offsets))
                packed, match_key = self.store.compute("pair-matches.2", matcher,
                    {"rows": rows.record(), "pairs": bank},
                    lambda: match_pairs(one, bank, matcher)[ap].record())
                matches = PairMatches(**packed)
                proposals, proposal_key = self.store.compute("grouped-candidates.2",
                    {"grouper": c.grouper, "estimator": c.estimator},
                    {"matches": ArtifactRef(match_key), "pairs": pairs, "scopes": scopes,
                     "xyz": survey.xyz, "ranges": rows.record()},
                    lambda: fit_groups(survey.xyz, rows, group_within_scopes(
                        matches, scopes, c.grouper, allowed_pairs=pairs), c.estimator))
                evidence[ap] = {"matches": match_key, "candidates": proposal_key}
            elif family == "zayets":
                proposals, proposal_key = self.store.compute("zayets-proposals.1",
                    {"algorithm": c}, inputs, lambda: zayets_one(one, c))
                evidence[ap] = {"proposals": proposal_key}
            else:
                full = ransac_proposals({**self.parameters, "trials": self.max_trials}, self.seed)
                pool, pool_key = self.store.compute("ransac-hypotheses.1",
                    {"solver": full.hypothesis_solver, "trials": full.trials,
                     "seed": full.seed}, inputs, lambda: ransac_pool(one, full))
                proposals, proposal_key = self.store.compute("ransac-retention.1",
                    {"scorer": c.scorer, "retainer": c.retainer, "trials": c.trials},
                    {**inputs, "pool": ArtifactRef(pool_key)},
                    lambda: retain_ransac(one, pool, c))
                evidence[ap] = {"proposals": proposal_key, "hypotheses": pool_key}
            if selector is None:
                outputs[ap] = proposals["xyz"]
                continue
            selected, selected_key = self.store.compute("consuming-finisher.1",
                {"selector": selector},
                {**inputs, "proposals": ArtifactRef(proposal_key)},
                lambda: finish_one(one, SourceMap({ap: proposals["xyz"]}), selector))
            outputs[ap] = selected["xyz"]
            evidence[ap]["selection"] = selected_key
        evidence["seconds"] = time.perf_counter() - started
        return SourceMap(outputs, evidence)
