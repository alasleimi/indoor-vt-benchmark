"""Cached PointNet graphs, local proposals, and optional existing finishers."""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time
import numpy as np

from .additional import finisher, finish_one
from .artifacts import ArtifactRef, identity
from .reconstruction import fit_groups
from .store import digest
from ..science.estimation import Multilateration3D
from ..science.grouping import (ConnectedComponentsGrouper, StarGrouper, NeighborhoodPairs,
                               PairMatches, all_pairs_within, nearest_neighborhoods, group_within_scopes)
from ..science.models import Survey, SourceMap
from ..science.pointnet import PointNetLogOddsMatcher

CHECKPOINT_SHA256 = 'acf6d4d7a50219e69b3e9cf5f609dcb7387f685c0e9b0bc27f78e7656bab07a5'
CHECKPOINT = ('runs/pointnet-vt-matching/noise_obstacles_R1_R2_L_from1m_v1/'
              'continuation_4m_batch816/best_validation_log_loss.pt')
GRAPH_ROOT = Path('runs/redesign/pointnet4m-graphs-v1')
METHODS = {f'pointnet4m_{group}_{ending}': (group, ending)
           for group in ('components', 'stars') for ending in ('raw', 'consumption', 'robust')}


@dataclass(frozen=True)
class PointNetReconstructor:
    matcher: PointNetLogOddsMatcher
    pair_selector: object
    group_scope_selector: object
    grouper: object
    estimator: object
    selector: object = None


def pointnet_reconstructor(parameters, grouping, finishing):
    scopes = partial(nearest_neighborhoods, size=parameters['neighbors'])
    return PointNetReconstructor(
        matcher=PointNetLogOddsMatcher(checkpoint_sha256=CHECKPOINT_SHA256),
        pair_selector=NeighborhoodPairs(scopes), group_scope_selector=scopes,
        grouper=ConnectedComponentsGrouper() if grouping == 'components' else StarGrouper(),
        estimator=Multilateration3D(),
        selector=None if finishing == 'raw' else finisher(
            'zayets' if finishing == 'consumption' else 'ransac', parameters))


def graph_identity(xyz, rows, pairs):
    return digest(identity({'checkpoint': CHECKPOINT_SHA256, 'assignment': 'hungarian_log_odds_archived_v1',
                            'xyz': xyz, 'rows': rows.record(), 'pairs': pairs}))


class PointNetExecution:
    def __init__(self, algorithm, store, graph_root=GRAPH_ROOT):
        self.algorithm, self.store, self.graph_root = algorithm, store, Path(graph_root)

    def fit(self, survey):
        started = time.perf_counter()
        c = self.algorithm
        pairs, scopes = c.pair_selector(survey.xyz), c.group_scope_selector(survey.xyz)
        bank = all_pairs_within(nearest_neighborhoods(survey.xyz, size=40))
        output, evidence = {}, {}
        for ap, rows in survey.channels.items():
            graph_key = graph_identity(survey.xyz, rows, bank)
            path = self.graph_root / (graph_key+'.npz')
            if not path.exists():
                raise FileNotFoundError(f'Prepare frozen PointNet inference with scripts/run_pointnet4m_viewer.py: {path}')
            with np.load(path) as saved:
                matches = PairMatches(**dict(saved))
            proposals, proposal_key = self.store.compute('pointnet-local-proposals.1',
                {'grouper': c.grouper, 'estimator': c.estimator},
                {'graph': graph_key, 'pairs': pairs, 'scopes': scopes,
                 'xyz': survey.xyz, 'rows': rows.record()},
                lambda: fit_groups(survey.xyz, rows,
                    group_within_scopes(matches, scopes, c.grouper, allowed_pairs=pairs), c.estimator))
            evidence[ap] = {'graph': graph_key, 'candidates': proposal_key,
                            'candidate_count': len(proposals['xyz']), 'checkpoint': CHECKPOINT_SHA256}
            if c.selector is None:
                output[ap] = proposals['xyz']
            else:
                one = Survey(survey.xyz, {ap: rows}, survey.position_ids)
                finished, selected_key = self.store.compute('consuming-finisher.1',
                    {'selector': c.selector},
                    {'xyz': survey.xyz, 'rows': rows.record(), 'ap': ap,
                     'proposals': ArtifactRef(proposal_key)},
                    lambda: finish_one(one, SourceMap({ap: proposals['xyz']}), c.selector))
                output[ap] = finished['xyz']
                evidence[ap]['selection'] = selected_key
        evidence['seconds'] = time.perf_counter()-started
        return SourceMap(output, evidence)
