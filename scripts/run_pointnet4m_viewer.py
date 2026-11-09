"""Prepare immutable graphs, run/resume calibrated methods, then evaluate obstacles."""
# ruff: noqa: E402 -- numerical thread limits must precede NumPy/SciPy imports.
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import time

import numpy as np
from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.execution.experiment import Evaluator
from indoor_vt.execution.pointnet import METHODS, CHECKPOINT, CHECKPOINT_SHA256, GRAPH_ROOT, graph_identity
from indoor_vt.science.grouping import all_pairs_within, nearest_neighborhoods

BASE = Path('runs/redesign/rectangle-clean-selection-v4')
DATA = Path('data/redesign/rectangle-20261025')


def prepare():
    from indoor_vt.execution.obstacle_reconstruction import layout_inputs
    config = read_json(BASE/'config.json')
    evaluator = Evaluator(DATA, config, 'pointnet4m-prepare')
    surveys = []
    for room in evaluator.observations.rooms:
        for level in config['observations']['noise_levels']:
            repeats = config['observations']['evaluation_repetitions'] if level else [2]
            for repeat in repeats:
                surveys.append((f'empty_sigma{level}_repeat{repeat}',
                                evaluator.observations.survey(room, level, repeat)))
        calibration = evaluator.observations.survey(room, 0., 0)
        # Identical clean observations are shared by calibration and evaluation.
        for ap in calibration.channels:
            np.testing.assert_array_equal(calibration.channels[ap].values, surveys[0][1].channels[ap].values)
    obstacle = Evaluator('data/redesign/rectangle-matched-obstacles-20261027', config, 'pointnet4m-obstacles')
    for room in obstacle.observations.rooms:
        for layout in room['layouts']:
            surveys.append(('obstacles_'+layout['id'], layout_inputs(obstacle.observations, room, layout)))
    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    archive_root = Path(CHECKPOINT).parent.parent/'reconstruction_4m_vs_3m'
    archive = read_json(archive_root/'prediction_manifest.json')
    assert archive['model']['sha256'] == CHECKPOINT_SHA256
    old_conditions = read_json(archive_root/'conditions.json')
    inputs = []
    for name, survey in surveys:
        bank = all_pairs_within(nearest_neighborhoods(survey.xyz, size=40))
        for ap, rows in survey.channels.items():
            key = graph_identity(survey.xyz, rows, bank)
            path = GRAPH_ROOT/(key+'.input.npz')
            if not path.exists():
                np.savez_compressed(path, xyz=survey.xyz, values=rows.values, offsets=rows.offsets, pairs=bank)
            old = next(c for c in old_conditions if (
                ('empty_sigma'+str(float(c['sigma']))+'_repeat'+str(c['repetition'])
                 if c['population'] == 'empty' else 'obstacles_'+c['name']) == name))
            inputs.append({'case': name+'_'+ap, 'path': str(path), 'output': str(GRAPH_ROOT/(key+'.npz')),
                'archived_input': old['input_file'], 'archived_logits': archive['predictions'][old['case']]['path'],
                'archived_logits_sha256': archive['predictions'][old['case']]['sha256'], 'ap_code': ['A','B'].index(ap)})
    manifest = {'checkpoint': CHECKPOINT, 'checkpoint_sha256': CHECKPOINT_SHA256, 'inputs': inputs}
    write_json(GRAPH_ROOT/'manifest.json', manifest)
    return manifest


def worker(method, smoke):
    from indoor_vt.worker_memory import install_limits, DEFAULT_LIMITS
    from indoor_vt.execution.pointnet_experiment import run_pointnet
    install_limits(**DEFAULT_LIMITS)
    root = BASE.parent/(f'pointnet-viewer-smoke-{method}' if smoke else f'rectangle-{method}-v1')
    return len(run_pointnet(DATA, root, BASE, method, smoke=smoke))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-python', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--methods', nargs='+', choices=list(METHODS), default=list(METHODS))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    manifest = prepare()
    missing = [i for i in manifest['inputs'] if not Path(i['output']).exists()]
    if missing:
        if args.torch_python is None:
            raise ValueError('Pass --torch-python with the existing PyTorch interpreter to compute missing graphs')
        subprocess.run([str(args.torch_python), '-u', 'scripts/prepare_pointnet4m_viewer_graphs.py',
                        '--manifest', str(GRAPH_ROOT/'manifest.json')], check=True)
    if args.prepare_only:
        return
    status_path = BASE.parent/'pointnet4m-viewer-status.json'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {m: pool.submit(worker, m, args.smoke) for m in args.methods}
        while True:
            states = {}
            for m in jobs:
                root = BASE.parent/(f'pointnet-viewer-smoke-{m}' if args.smoke else f'rectangle-{m}-v1')
                states[m] = read_json(root/'status.json') if (root/'status.json').exists() else {'state': 'queued'}
            errors = {m: str(f.exception()) for m, f in jobs.items() if f.done() and f.exception()}
            done = all(f.done() for f in jobs.values())
            write_json(status_path, {'state': 'failed' if errors else 'complete' if done else 'running',
                                    'updated': time.time(), 'studies': states, 'errors': errors})
            if done:
                if errors:
                    raise RuntimeError(str(errors))
                break
            time.sleep(5)
    print('All PointNet reconstruction and localization searches/evaluations complete.', flush=True)


if __name__ == '__main__':
    main()
