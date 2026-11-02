"""Chamfer-kNN and jointly selected source-assisted localization.

The query split and observations are shared with the current rectangle study.
Chamfer has no peak matcher or unification parameter. Joint unification applies
only to SMART-LEA and, for the standard MPUrge supplier, reconstruction.
"""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from .additional import AdditionalExecution, ROBUST_FINISHER_PARAMETERS
from .artifacts import file_lock, implementation_id, write_npz
from .composition_experiment import composition_config
from .compositions import CompositionExecution, FinishedReconstructor
from .experiment import Evaluator, configurations, joint_configurations
from .localization import LocalizerExecution, LEAExecution
from .presets import standard_mpurge, smart_lea
from .reconstruction import MPUrgeExecution
from .trials import TrialLog
from ..acquisition.records import read_json, write_json, digest, Seeds
from ..science.grouping import all_pairs_within, nearest_neighborhoods
from ..science.localization import ChamferScorer, DistanceWeightedKNN, FingerprintLocalizer


SUPPLIERS = ('mpurge', 'zayets_robust', 'ransac')
BASELINE_STUDIES = {'mpurge': 'rectangle-clean-selection-v4',
                    'zayets_robust': 'rectangle-compositions-v1',
                    'ransac': 'rectangle-additional-selection-v1'}


def chamfer_knn(parameters):
    return FingerprintLocalizer(
        scorer=ChamferScorer(),
        estimator=DistanceWeightedKNN(neighbors=parameters['neighbors'], power=parameters['power']))


def supplier_constructor(family, parameters, *, unify, seed=9173):
    from .additional import zayets_proposals, ransac_proposals, finisher
    if family == 'mpurge':
        return standard_mpurge(parameters, unify=unify)
    proposer = (zayets_proposals(parameters) if family == 'zayets_robust'
                else ransac_proposals(parameters, seed))
    return FinishedReconstructor(proposer, finisher('ransac', parameters))


def chamfer_config(base, job):
    config = composition_config(deepcopy(base))
    config.update(name=f'rectangle-chamfer-{job}-v1', engine='chamfer')
    config['chamfer'] = {
        'initializer': {'neighbors': [1, 3, 5, 10], 'power': [.5, 1., 2.]},
        'joint_trials': 512,
        'suppliers': {'mpurge': config['optimization']['mpurge'],
                      'zayets_robust': config['compositions']['spaces']['zayets_robust'],
                      'ransac': config['additional']['ransac']},
        'selection_protocol': 'Empty noise-free reference/VT survey; noise-free calibration queries '
            'across all four obstacle layouts (60 per layout, 240 total). Held-out query positions, '
            'noise levels and repetitions never select parameters.',
        'baseline_protocol': 'The independently selected Chamfer initializer replaces MPUrge-MAP '
            'in each supplier\'s previously calibration-selected joint pipeline. Its recorded source '
            'and SMART-LEA settings seed the new joint search; remaining trials sample the full grid.',
        'supplier_selection': 'Standard MPUrge plus the two best distinct supplier families in '
            'existing MPUrge-MAP + SMART-LEA joint calibration: Zayets + robust finisher '
            '(0.0065902887 m) and RANSAC + robust finisher (0.0067743844 m). No held-out metric was used.'}
    return config


class ChamferEvaluator(Evaluator):
    def evaluate(self, method, parameters, *, unify, level, split, frozen=None, retain=False):
        started = time.perf_counter()
        joint = method != 'chamfer_knn'
        family = method.removeprefix('chamfer_').removesuffix('_joint') if joint else None
        if joint and family not in SUPPLIERS:
            raise ValueError('Unknown Chamfer source supplier')
        settings = self.config['observations']
        repetitions = settings[split + '_repetitions']
        if level == 0 and settings['track'] == 'direct':
            repetitions = repetitions[:1]
        initializer = LocalizerExecution(chamfer_knn(parameters['map'] if joint else parameters),
                                         self.store, self.run)
        refiner = (LEAExecution(smart_lea(parameters['lea'], unify=unify), self.store, self.run)
                   if joint else None)
        sources_saved, queries_saved, errors, statuses, timings = [], [], [], [], []
        for room in self.observations.rooms:
            for repetition in repetitions:
                survey = self.observations.survey(room, level, repetition)
                sources = None
                if joint:
                    p = parameters['source']
                    if family == 'mpurge':
                        if room['id'] not in self.banks:
                            self.banks[room['id']] = all_pairs_within(nearest_neighborhoods(survey.xyz,
                                size=max(self.config['optimization']['mpurge']['neighbors'])))
                        source_execution = MPUrgeExecution(standard_mpurge(p, unify=unify), self.store,
                            self.run, pair_bank=self.banks[room['id']])
                    elif family == 'zayets_robust':
                        source_execution = CompositionExecution(family, p, self.store, unify=unify,
                            seed=self.config['additional']['proposal_seed'])
                    else:
                        source_execution = AdditionalExecution(family, p, self.store,
                            seed=self.config['additional']['proposal_seed'],
                            max_trials=max(self.config['additional']['ransac']['trials']))
                    sources = source_execution.fit(survey)
                    timings.append(sources.evidence['seconds'])
                    if retain:
                        sources_saved.append({'room': room['id'], 'repetition': repetition,
                                              'by_ap': sources.by_ap, 'audit': sources.evidence})
                for layout in room['layouts']:
                    queries = self.observations.queries(room, layout, level, repetition, split)
                    initial, initial_audit = initializer.predict(survey, queries)
                    audit = {'initial': initial_audit}
                    result = initial
                    if joint:
                        result, audit['refinement'] = refiner.predict(queries, initial, sources)
                    e = np.linalg.norm(result['xyz'] - queries.xyz, axis=1)
                    errors.extend(e)
                    statuses.extend(result['status'])
                    if retain:
                        queries_saved.append({'room': room['id'], 'layout': layout['id'],
                            'repetition': repetition, 'query_ids': queries.position_ids,
                            'xyz': result['xyz'], 'initial_xyz': initial['xyz'], 'errors_m': e,
                            'status': result['status'], 'audit': audit})
        errors = np.asarray(errors)
        metrics = {'mean_m': float(errors.mean()), 'rmse_m': float(np.sqrt(np.mean(errors**2))),
                   'median_m': float(np.median(errors)), 'p90_m': float(np.quantile(errors, .9)),
                   'fallback_fraction': float(np.mean(np.asarray(statuses) != 'ok'))}
        result = {'loss': metrics['mean_m'], 'metrics': metrics, 'seconds': time.perf_counter()-started,
                  'max_reconstruction_seconds': max(timings, default=0.)}
        if retain:
            result['detail'] = {'sources': sources_saved, 'queries': queries_saved, 'source_metrics': []}
        return result


def baseline_parameters(study_root, family, unify, initializer):
    mode = 'unified' if unify else 'ununified'
    label = ('joint' if family == 'mpurge' else family + '_joint') + '_' + mode
    path = Path(study_root) / BASELINE_STUDIES[family] / 'selected' / (label + '.json')
    receipt = read_json(path)
    p = receipt['parameters']
    source = dict(p.get('source', p.get('mpurge')))
    if family != 'mpurge':
        # Old pre-grid receipts implicitly used these real finisher defaults.
        source = {**dict(zip(ROBUST_FINISHER_PARAMETERS, (.35, .1, 4.))), **source}
    return {'source': source, 'map': initializer, 'lea': p['lea']}, {
        'path': str(path), 'receipt_id': receipt['id'], 'loss': receipt['loss']}


def run_chamfer(data_root, run_root, base_root, job, *, smoke=False):
    data_root, run_root, base_root = map(Path, (data_root, run_root, base_root))
    config = chamfer_config(read_json(base_root / 'config.json'), job)
    if smoke:
        config['chamfer']['joint_trials'] = 2
        config['observations']['noise_levels'] = [0., .1]
        config['observations']['evaluation_repetitions'] = [2]
        config['optimization']['unification'] = [True]
    version = {'id': implementation_id(), 'experiment': digest(Path(__file__).with_name('experiment.py').read_text()),
               'dataset': digest(read_json(data_root / 'dataset.json')),
               'chamfer': digest(Path(__file__).read_text(encoding='utf-8'))}
    run_root.mkdir(parents=True, exist_ok=True)
    with file_lock(run_root / 'run.lock'):
        for name, value in (('config.json', config), ('implementation.json', version)):
            path = run_root / name
            if path.exists() and read_json(path) != value:
                raise ValueError(f'Changed {name}; use a new run directory')
            write_json(path, value)
        evaluator = ChamferEvaluator(data_root, config, config['name'])
        if not evaluator.observations.available():
            raise ValueError('Required native observations are not ready')
        modes = config['optimization']['unification'] if job != 'standalone' else [None]
        budget = config['chamfer']['joint_trials'] if job != 'standalone' else len(configurations(config['chamfer']['initializer']))
        status = {'state': 'selecting', 'completed_trials': 0, 'target_trials': budget*len(modes),
                  'selection_noise_m': 0., 'current': None}
        def update(**values):
            status.update(**values, updated=time.time(), cache=evaluator.store.usage())
            write_json(run_root / 'status.json', status)
        choices, origins = [], {}
        try:
            for unify in modes:
                method = 'chamfer_knn' if job == 'standalone' else 'chamfer_' + job + '_joint'
                label = method if unify is None else method + ('_unified' if unify else '_ununified')
                if job == 'standalone':
                    schedule = configurations(config['chamfer']['initializer'])
                else:
                    initial_path = run_root.parent / ('chamfer-smoke-standalone' if smoke else 'rectangle-chamfer-standalone-v1') / 'selected/chamfer_knn.json'
                    initial = read_json(initial_path)['parameters']
                    baseline, origins[label] = baseline_parameters(base_root.parent, job, unify, initial)
                    spaces = {'source': config['chamfer']['suppliers'][job],
                              'map': config['chamfer']['initializer'], 'lea': config['optimization']['smart_lea']}
                    schedule = joint_configurations(spaces, budget,
                        Seeds(config['optimization']['seed']).seed('chamfer-joint', job, unify), baseline)
                    write_json(run_root / 'baseline_origins.json', origins)
                context = {'method': method, 'unify': unify, 'level': 0., 'split': 'calibration',
                           'frozen': {}, 'label': label}
                log = TrialLog(run_root / 'trials', label, context)
                best = None
                for number, parameters in enumerate(schedule):
                    identity = {k: context[k] for k in ('method', 'unify', 'level', 'split', 'frozen')}
                    identity['parameters'] = parameters
                    key = digest(identity)
                    trial = log.get(key)
                    if trial is None:
                        evaluated = evaluator.evaluate(method, parameters, unify=unify, level=0., split='calibration')
                        trial = {'id': key, 'number': number, 'label': label, **identity, **evaluated}
                        log.append(trial)
                    if best is None or trial['loss'] < best['loss']:
                        best = trial
                    status['completed_trials'] += 1
                    update(current={'method': label, 'trial': number+1, 'total': len(schedule), 'best_loss': best['loss']})
                    if (number+1) % 16 == 0 or number == 0:
                        print(f'{label}: {number+1}/{len(schedule)} best={best["loss"]:.8f}', flush=True)
                choices.append(best)
                write_json(run_root / 'selected' / (label + '.json'), best)
            update(state='evaluating')
            results = []
            for choice in choices:
                for split, levels in (('calibration', [0.]), ('evaluation', config['observations']['noise_levels'])):
                    for level in levels:
                        identity = {k: choice[k] for k in ('method', 'parameters', 'unify', 'frozen')}
                        identity.update(level=level, split=split)
                        key = digest(identity)
                        receipt, outcome = run_root / 'evaluations' / (key+'.json'), run_root / 'outcomes' / (key+'.npz')
                        if receipt.exists() and outcome.exists():
                            row = read_json(receipt)
                        else:
                            value = evaluator.evaluate(choice['method'], choice['parameters'], unify=choice['unify'],
                                                       level=level, split=split, retain=True)
                            write_npz(outcome, value.pop('detail'), metadata=identity)
                            row = {'id': key, 'label': choice['label'], **identity, **value,
                                   'outcome': str(outcome.relative_to(run_root))}
                            write_json(receipt, row)
                        results.append(row)
                        write_json(run_root / 'results.json', results)
                        update(current={'method': choice['label'], 'split': split, 'noise': level})
                        print(f'{choice["label"]} {split} sigma={level:g}: {row["loss"]:.8f}', flush=True)
            update(state='complete', current=None)
            return results
        except Exception as error:
            update(state='failed', error=str(error))
            raise
