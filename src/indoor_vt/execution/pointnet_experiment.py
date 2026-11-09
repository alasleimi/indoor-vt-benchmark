"""Calibrate six fixed-network reconstructors and their stagewise/joint pipelines."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np

from .localization import LocalizerExecution, LEAExecution
from .presets import mpurge_map, smart_lea
from ..metrics import ap_source_metrics

from .additional_experiment import robust_finisher_grid
from .artifacts import file_lock, write_npz
from .composition_experiment import source_schedule
from .experiment import Evaluator, configurations, joint_configurations
from .pointnet import (METHODS, CHECKPOINT, CHECKPOINT_SHA256, GRAPH_ROOT,
                      PointNetExecution, pointnet_reconstructor)
from .trials import TrialLog
from ..acquisition.records import read_json, write_json, digest, Seeds


def pointnet_config(base, method):
    config = deepcopy(base)
    config.update(name=f'rectangle-{method}-v1', engine='pointnet4m')
    grids = {}
    for name, (_, finishing) in METHODS.items():
        space = {'neighbors': [5, 10, 20, 40]}
        if finishing != 'raw':
            space.update(tolerance_m=[.1, .18, .3], support_fraction=[.04, .08, .16],
                         duplicate_m=[.06, .12, .24])
        if finishing == 'robust':
            space.update(robust_finisher_grid())
        grids[name] = space
    config['pointnet4m'] = {'method': method, 'spaces': grids, 'source_trials': 512,
        'joint_trials': 512, 'checkpoint': CHECKPOINT, 'checkpoint_sha256': CHECKPOINT_SHA256,
        'training_updates': 4_000_000, 'graph_root': str(GRAPH_ROOT),
        'prediction_policy': 'Reuse recorded 4M logits for covered pairs; infer only additional pairs with the same checkpoint',
        'protocol': 'Fixed exact 4M weights, trained on archived R1/R2/L with obstacle/noise augmentation. '
            'No benchmark-room network training. Source settings minimize AP-separated A+B OSPA '
            'on the empty noise-free survey. Stagewise SMART-LEA exhausts its grid with frozen source '
            'and independently selected MPUrge-MAP. Joint search includes that baseline and samples '
            '512 distinct source/MAP/LEA configurations. Localization selection uses only noise-free '
            'calibration queries (60 per saved layout); evaluation uses held-out queries and frozen settings. '
            'PointNet matching has no peak unification; pipeline unification applies to MAP/LEA only. '
            'Raw variants return all valid group estimates without candidate clustering or support filtering.'}
    return config


class PointNetEvaluator(Evaluator):
    families = METHODS
    source_metric_aps = (('A', 'B'),)

    def reconstruct(self, family, parameters, unify, survey, room):
        return PointNetExecution(pointnet_reconstructor(parameters, *METHODS[family]),
            self.store, self.config['pointnet4m']['graph_root']).fit(survey)


    def evaluate(self, method, parameters, *, unify, level, split, frozen=None, retain=False):
        started = time.perf_counter()
        family = method.removesuffix("_joint").removesuffix("_lea")
        if family not in self.families:
            raise ValueError("Unknown additional reconstructor")
        frozen = frozen or {}
        source_only, joint = method == family, method.endswith("_joint")
        source_p = parameters if source_only else parameters["source"] if joint else frozen["source"]
        settings = self.config["observations"]
        repetitions = settings[split + "_repetitions"]
        if level == 0 and settings["track"] == "direct":
            repetitions = repetitions[:1]
        source_records, query_records, cases, timings = [], [], [], []
        for room in self.observations.rooms:
            truth = self.observations.truth(room)
            for repetition in repetitions:
                survey = self.observations.survey(room, level, repetition)
                sources = self.reconstruct(family, source_p, unify, survey, room)
                timings.append(sources.evidence["seconds"])
                if retain:
                    source_records.append({"room": room["id"], "repetition": repetition,
                                           "by_ap": sources.by_ap, "audit": sources.evidence})
                if source_only:
                    for aps in self.source_metric_aps:
                        metrics = ap_source_metrics(sources.by_ap, truth.by_ap, aps)
                        cases.append({"ospa": metrics["ospa"], "f1": metrics["f1"]})
                    continue
                map_p = parameters["map"] if joint else frozen["map"]
                lea_p = parameters["lea"] if joint else parameters
                localizer = LocalizerExecution(mpurge_map(map_p, unify=unify), self.store, self.run)
                refiner = LEAExecution(smart_lea(lea_p, unify=unify), self.store, self.run)
                for layout in room["layouts"]:
                    queries = self.observations.queries(room, layout, level, repetition, split)
                    initial, initial_audit = localizer.predict(survey, queries)
                    result, lea_audit = refiner.predict(queries, initial, sources)
                    errors = np.linalg.norm(result["xyz"] - queries.xyz, axis=1)
                    cases.extend({"error_m": float(e), "fallback": status != "ok"}
                                 for e, status in zip(errors, result["status"]))
                    if retain:
                        query_records.append({"room": room["id"], "layout": layout["id"],
                            "repetition": repetition, "query_ids": queries.position_ids,
                            "xyz": result["xyz"], "initial_xyz": initial["xyz"], "errors_m": errors,
                            "status": result["status"],
                            "audit": {"initial": initial_audit, "refinement": lea_audit}})
        if source_only:
            metrics = {key: float(np.mean([c[key] for c in cases])) for key in ("ospa", "f1")}
            loss = metrics["ospa"]
        else:
            errors = np.array([c["error_m"] for c in cases])
            metrics = {"mean_m": float(errors.mean()), "rmse_m": float(np.sqrt(np.mean(errors**2))),
                       "median_m": float(np.median(errors)), "p90_m": float(np.quantile(errors, .9)),
                       "fallback_fraction": float(np.mean([c["fallback"] for c in cases]))}
            loss = metrics["mean_m"]
        result = {"loss": loss, "metrics": metrics, "seconds": time.perf_counter() - started,
                  "max_reconstruction_seconds": max(timings, default=0)}
        if retain:
            result["detail"] = {"sources": source_records, "queries": query_records,
                                "source_metrics": cases if source_only else []}
        return result


def run_pointnet(data_root, root, base_root, method, *, smoke=False):
    root, base_root = Path(root), Path(base_root)
    config = pointnet_config(read_json(base_root/'config.json'), method)
    if smoke:
        config['name'] = root.name
        config['observations']['noise_levels'] = [0.]
        config['optimization']['unification'] = [False]
        config['optimization']['smart_lea'] = {'alpha': [.5], 'half_window': [2], 'iterations': [1]}
        config['pointnet4m'].update(source_trials=1, joint_trials=2)
    settings = config['pointnet4m']
    evaluator = PointNetEvaluator(data_root, config, config['name'])
    seeds = Seeds(config['optimization']['seed'])
    source_grid = source_schedule(settings['spaces'][method], settings['source_trials'],
                                  seeds.seed('pointnet-source', method))
    lea_grid = configurations(config['optimization']['smart_lea'])
    modes = config['optimization']['unification']
    status = {'state': 'selecting', 'completed_trials': 0,
              'target_trials': len(source_grid)+len(modes)*(len(lea_grid)+settings['joint_trials'])}
    results, choices = [], {}

    def update(**values):
        status.update(**values, updated=time.time())
        write_json(root/'status.json', status)

    def select(name, unify, schedule, frozen):
        label = name + ('_fixed_network' if unify is None else '_unified' if unify else '_ununified')
        context = dict(method=name, unify=unify, level=0., split='calibration', frozen=frozen, label=label)
        log, best = TrialLog(root/'trials', label, context), None
        update(state='selecting')
        for number, parameters in enumerate(schedule):
            identity = {k: context[k] for k in ('method', 'unify', 'level', 'split', 'frozen')}
            identity['parameters'] = parameters
            key = digest(identity)
            trial = log.get(key)
            if trial is None:
                trial = {'id': key, 'number': number, 'label': label, **identity,
                         **evaluator.evaluate(**identity)}
                log.append(trial)
            if best is None or trial['loss'] < best['loss']:
                best = trial
            status['completed_trials'] += 1
            update(current={'method': name, 'unify': unify, 'trial': number+1,
                            'total': len(schedule), 'best_loss': best['loss']})
            if number % 16 == 0 or number+1 == len(schedule):
                print(f'{label}: {number+1}/{len(schedule)} best={best["loss"]:.8f}', flush=True)
        choices[name, unify] = best
        write_json(root/'selected'/(label+'.json'), best)
        return best

    def evaluate_choice(choice):
        update(state='evaluating')
        for split, levels in (('calibration', [0.]), ('evaluation', config['observations']['noise_levels'])):
            for level in levels:
                identity = {k: choice[k] for k in ('method', 'parameters', 'unify', 'frozen')}
                identity.update(level=level, split=split)
                key = digest(identity)
                receipt, outcome = root/'evaluations'/(key+'.json'), root/'outcomes'/(key+'.npz')
                if receipt.exists() and outcome.exists():
                    row = read_json(receipt)
                else:
                    value = evaluator.evaluate(**identity, retain=True)
                    write_npz(outcome, value.pop('detail'), metadata=identity)
                    row = {'id': key, 'label': choice['label'], **identity, **value,
                           'outcome': str(outcome.relative_to(root))}
                    write_json(receipt, row)
                results.append(row)
                write_json(root/'results.json', results)
                update(current={'method': choice['label'], 'split': split, 'noise': level})
                print(f'{choice["label"]}: {split} sigma={level:g}, loss={row["loss"]:.8f}', flush=True)

    with file_lock(root/'run.lock'):
        path = root/'config.json'
        if path.exists() and read_json(path) != config:
            raise ValueError('Scientific configuration changed; use a distinct run directory')
        write_json(path, config)
        # Dataset and numerical settings identify the experiment. Unrelated source edits do not invalidate it.
        version = {**read_json(base_root/'implementation.json'), 'pointnet_checkpoint': CHECKPOINT_SHA256}
        write_json(root/'implementation.json', version)
        try:
            independent = select(method, None, source_grid, {})
            evaluate_choice(independent)
            for unify in modes:
                mode = 'unified' if unify else 'ununified'
                map_choice = read_json(base_root/'selected'/('mpurge_map_'+mode+'.json'))
                frozen = {'source': independent['parameters'], 'map': map_choice['parameters']}
                stagewise = select(method+'_lea', unify, lea_grid, frozen)
                evaluate_choice(stagewise)
                baseline = {**frozen, 'lea': stagewise['parameters']}
                schedule = joint_configurations({'source': settings['spaces'][method],
                    'map': config['optimization']['mpurge_map'], 'lea': config['optimization']['smart_lea']},
                    settings['joint_trials'], seeds.seed('pointnet-joint', method, unify), baseline)
                joint = select(method+'_joint', unify, schedule, {})
                evaluate_choice(joint)
            update(state='complete', current=None)
        except Exception as error:
            update(state='failed', error=str(error))
            raise
    return results
