"""Frozen star proposals with the existing finishers; no parameter optimization."""
from dataclasses import replace

from .additional import finisher, finish_one
from .compositions import FinishedReconstructor
from .presets import standard_mpurge
from .reconstruction import fit_groups
from ..science.grouping import StarGrouper, group_within_scopes
from ..science.models import SourceMap, Survey
from ..science.reconstruction import match_pairs


METHODS = {'mpurge_stars_consumption': 'consumption', 'mpurge_stars_robust': 'robust'}


def star_reconstructor(parameters, finishing):
    proposer = replace(standard_mpurge(parameters, unify=False), grouper=StarGrouper())
    return FinishedReconstructor(proposer, finisher('zayets' if finishing == 'consumption' else 'ransac', parameters))


class StarExecution:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def fit(self, survey):
        c = self.algorithm.proposals
        pairs, scopes = c.pair_selector(survey.xyz), c.group_scope_selector(survey.xyz)
        graphs = match_pairs(survey, pairs, c.matcher)
        output = {}
        for ap, rows in survey.channels.items():
            groups = group_within_scopes(graphs[ap], scopes, c.grouper)
            candidates = fit_groups(survey.xyz, rows, groups, c.estimator)
            one = Survey(survey.xyz, {ap: rows}, survey.position_ids)
            output[ap] = finish_one(one, SourceMap({ap: candidates['xyz']}), self.algorithm.selector)['xyz']
        return SourceMap(output)
