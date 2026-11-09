from itertools import permutations

import numpy as np
import pytest

from indoor_vt.science.pointnet import PointNetLogOddsMatcher, graph_from_logits
from indoor_vt.science.models import Ragged
from indoor_vt.execution.pointnet import METHODS, CHECKPOINT_SHA256, pointnet_reconstructor
from indoor_vt.execution.pointnet_experiment import pointnet_config
from indoor_vt.execution.explorer import method_entries
from indoor_vt.execution.presentation import construction, search_spec
from indoor_vt.execution.obstacle_reconstruction import frozen_source_spec
from indoor_vt.execution.experiment import configurations


def test_log_odds_maximum_cardinality_is_not_sum_probability_matching():
    matcher = PointNetLogOddsMatcher(CHECKPOINT_SHA256)
    probabilities = np.array([[.99, .8], [.79, .2]])
    logits = np.log(probabilities/(1-probabilities))
    left, right, costs = matcher.assign(logits)
    # Sum probability would choose the off-diagonal; odds selects the diagonal.
    assert list(zip(left, right)) == [(0, 0), (1, 1)]
    assert logits[left, right].sum() == max(sum(logits[i,j] for i,j in enumerate(p)) for p in permutations(range(2)))
    np.testing.assert_allclose(costs, [.01, .8])
    # Negative evidence does not silently introduce an unmatched option.
    assert len(matcher.assign(np.full((2, 3), -100.))[0]) == 2
    assert len(matcher.assign(np.empty((0, 3)))[0]) == 0


def test_graph_uses_peak_offsets_and_preserves_crossings():
    rows = Ragged.pack([[1., 2.], [], [3., 4., 5.]])
    graph = graph_from_logits(rows, np.array([[0, 1], [0, 2]]),
        [np.empty((2, 0)), np.array([[0., 5., -10.], [6., 0., -10.]])],
        PointNetLogOddsMatcher(CHECKPOINT_SHA256))
    np.testing.assert_array_equal(graph.offsets, [0, 0, 2])
    np.testing.assert_array_equal(graph.endpoints, [[0, 3], [1, 2]])


@pytest.mark.parametrize('variant', METHODS)
@pytest.mark.parametrize('stage', ['', '_lea', '_joint'])
def test_method_details_search_and_constructors_are_consistent(variant, stage):
    base = {'optimization': {'seed': 8181, 'unification': [True, False],
        'mpurge_map': {'neighbors': [1, 3], 'power': [1.], 'alpha': [.7], 'half_window': [2]},
        'smart_lea': {'iterations': [1, 2], 'alpha': [.5], 'half_window': [2]}}}
    config = pointnet_config(base, variant)
    source = configurations(config['pointnet4m']['spaces'][variant])[0]
    initializer = configurations(base['optimization']['mpurge_map'])[0]
    lea = configurations(base['optimization']['smart_lea'])[0]
    parameters = source if not stage else {'source': source, 'map': initializer, 'lea': lea} if stage == '_joint' else lea
    frozen = {'source': source, 'map': initializer} if stage == '_lea' else {}
    receipt = {'id': 'a'*64, 'method': variant+stage, 'label': variant+stage+'_test',
               'parameters': parameters, 'frozen': frozen, 'unify': None if not stage else True}
    entries = method_entries(receipt)
    assert len(entries) == (3 if stage == '_joint' else 1)
    for entry in entries:
        assert entry['family'] == 'pointnet4m'
        assert 'Zayets' not in entry['title']
        assert 'PointNet 4M' in entry['search_text']
        spec = search_spec(entry, config)
        assert 'half_window' not in spec['grids'].get('Reconstruction', {})
        code = construction(receipt, config, entry['component'])['code']
        if entry['component'] != 'initial':
            assert 'PointNetLogOddsMatcher(' in code
            assert ('selector=None' in code) == variant.endswith('_raw')
        if entry['component'] == 'localization':
            assert 'refiner = SmartLEA(' in code
    if stage != '_lea':
        assert frozen_source_spec(receipt, config)['method'] == variant


def test_finishers_receive_the_same_unclustered_proposal_model():
    p = {'neighbors': 5, 'tolerance_m': .1, 'support_fraction': .04, 'duplicate_m': .06}
    algorithms = [pointnet_reconstructor(p, 'stars', e) for e in ('raw', 'consumption', 'robust')]
    for name in ('matcher', 'grouper', 'estimator'):
        assert getattr(algorithms[0], name) == getattr(algorithms[1], name) == getattr(algorithms[2], name)
    assert algorithms[0].selector is None
    assert algorithms[1].selector.refitter is None
    assert algorithms[2].selector.refitter is not None
