import numpy as np
import pytest
from types import SimpleNamespace

from indoor_vt.execution.chamfer_experiment import chamfer_knn, chamfer_config, supplier_constructor
from indoor_vt.execution.explorer import method_entries
from indoor_vt.execution.presentation import construction, search_spec
from indoor_vt.execution.experiment import configurations
from indoor_vt.science.models import Ragged, Survey
from indoor_vt.execution.obstacle_reconstruction import frozen_source_spec
from indoor_vt.execution.peak_explorer import PeakExplorer


def base_config():
    return {'optimization': {'seed': 8181, 'mpurge': {'half_window': [1], 'alpha': [.7],
        'neighbors': [5], 'radius_m': [.1], 'beta': [1.]},
        'smart_lea': {'half_window': [1], 'alpha': [.5, 1.], 'iterations': [1, 4]}}}


def test_chamfer_symmetric_distance_and_knn_exact_ties():
    method = chamfer_knn({'neighbors': 3, 'power': 2.})
    a, b = np.array([1., 4.]), np.array([2., 6., 7.])
    assert method.scorer(a, b) == pytest.approx((1.5 + 2.) / 2)
    assert method.scorer(a, b) == method.scorer(b, a)
    assert np.isinf(method.scorer(a, np.empty(0)))
    xyz = np.array([[0., 0., 0.], [2., 0., 0.], [9., 0., 0.]])
    np.testing.assert_array_equal(method.estimator.estimate(xyz, np.array([0., 0., 5.])).xyz, [1., 0., 0.])


def test_ap_distances_are_averaged_before_nearest_neighbor_selection():
    method = chamfer_knn({'neighbors': 1, 'power': 1.})
    survey = Survey(np.array([[0., 0., 0.], [1., 0., 0.]]),
        {'A': Ragged.pack([[0.], [2.]]), 'B': Ragged.pack([[10.], [3.]])}, np.arange(2))
    result = method.fit(survey).predict({'A': np.array([0.]), 'B': np.array([0.])})
    np.testing.assert_array_equal(result.xyz, [1., 0., 0.])


@pytest.mark.parametrize('family', ['mpurge', 'zayets_robust', 'ransac'])
def test_joint_presentations_expose_three_stages_and_chamfer_only_initializer(family):
    config = chamfer_config(base_config(), family)
    source = configurations(config['chamfer']['suppliers'][family])[0]
    p = {'source': source, 'map': {'neighbors': 3, 'power': 2.},
         'lea': {'half_window': 1, 'alpha': 1., 'iterations': 4}}
    method = 'chamfer_' + family + '_joint'
    receipt = {'id': 'a'*64, 'method': method, 'label': method+'_unified', 'unify': True,
               'parameters': p, 'frozen': {}}
    entries = method_entries(receipt)
    assert [e['component'] for e in entries] == ['sources', 'initial', 'localization']
    assert all(e['initializer'] == 'chamfer_knn' for e in entries)
    assert all('MPUrge-MAP' not in e['title'] for e in entries)
    initial = construction(receipt, config, 'initial')['code']
    assert 'ChamferScorer(' in initial and 'MPUrgeMatcher(' not in initial
    assert 'reconstructor = ' in construction(receipt, config, 'localization')['code']
    assert 'refiner = SmartLEA(' in construction(receipt, config, 'localization')['code']
    spec = search_spec(entries[-1], config)
    assert spec['budget'] == 512
    assert set(spec['grids']['Chamfer-kNN']) == {'neighbors', 'power'}
    assert len(configurations(spec['grids']['Chamfer-kNN'])) == 12
    assert supplier_constructor(family, source, unify=True) is not None
    spec = frozen_source_spec(receipt, config)
    assert spec['method'] == family and spec['parameters'] == source


def test_standalone_has_no_fake_unification_or_reconstruction_search():
    config = chamfer_config(base_config(), 'standalone')
    receipt = {'id': 'b'*64, 'method': 'chamfer_knn', 'label': 'chamfer_knn', 'unify': None,
               'parameters': {'neighbors': 3, 'power': 1.}, 'frozen': {}}
    entry, = method_entries(receipt)
    assert entry['group'] == 'raw' and entry['family'] == 'chamfer'
    assert entry['unify'] is None and not entry['derived']
    spec = search_spec(entry, config)
    assert spec['budget'] == 12 and list(spec['grids']) == ['Chamfer-kNN']
    assert 'ChamferScorer(' in construction(receipt, config, 'localization')['code']
    again, = method_entries(receipt)
    assert again['code'] == entry['code']
    ordinary, = method_entries({**receipt, 'method': 'mpurge_map', 'label': 'mpurge_map_unified', 'unify': True})
    assert 'execution/chamfer_experiment.py' not in ordinary['code']['Execution and experiment configuration']
    assert any(link.get('label') == 'mpurge_map()' for link in ordinary['code']['Composition constructors'])


def test_peak_explorer_uses_chamfer_pipeline_source_parameters(monkeypatch):
    monkeypatch.setattr('indoor_vt.execution.peak_explorer.check_matching_data', lambda view: None)
    p = {'alpha': .7, 'half_window': 2, 'neighbors': 2, 'radius_m': .1, 'beta': 1.}
    receipt = {'method': 'chamfer_mpurge_joint', 'unify': False, 'parameters': {
        'source': p, 'map': {'neighbors': 10, 'power': .5}}}
    room = {'id': 'room', 'survey_spacing_m': .5,
        'survey_lattice': np.array([[0, 0], [1, 0]]),
        'survey_xyz': np.array([[0., 0., 1.], [.5, 0., 1.]])}
    view = SimpleNamespace(config={'observations': {'track': 'direct'},
        'survey': {'spacing_m': .5, 'phase_xy': [0, 0]}},
        observations=SimpleNamespace(rooms=[room]), receipt=lambda key: receipt)
    context = PeakExplorer(SimpleNamespace(owner=lambda key: view)).context('saved')
    assert context[2] == p and context[-1].half_window == 2
    assert context[-2].tolist() == [[0, 1]]
