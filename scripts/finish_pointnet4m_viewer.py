"""Finish the requested campaign, verify saved results, then publish its snapshot.

Publication is explicit (--publish) and requires the reviewed source commit to
still be HEAD. This never stages unrelated research scripts or training files.
"""
# ruff: noqa: E402 -- keep numerical work single-threaded.
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np
from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.execution.artifacts import read_npz
from indoor_vt.execution.experiment import configurations
from indoor_vt.execution.explorer import query_truth
from indoor_vt.execution.pointnet import METHODS, CHECKPOINT_SHA256
from indoor_vt.execution.pointnet_experiment import PointNetEvaluator
from indoor_vt.execution.trials import read_trials
from indoor_vt.metrics import ap_source_metrics

PARENT = Path('runs/redesign')
DATA = Path('data/redesign/rectangle-20261025')
BASE = PARENT/'rectangle-clean-selection-v4'
STATUS = PARENT/'pointnet4m-publication-status.json'
SOURCES = [
    'src/indoor_vt/science/pointnet.py', 'src/indoor_vt/execution/pointnet.py',
    'src/indoor_vt/execution/pointnet_experiment.py', 'src/indoor_vt/execution/artifacts.py',
    'src/indoor_vt/execution/explorer.py', 'src/indoor_vt/execution/presentation.py',
    'src/indoor_vt/execution/viewer.py', 'src/indoor_vt/execution/obstacle_reconstruction.py',
    'src/indoor_vt/execution/web/app.js', 'src/indoor_vt/execution/web/finisher-explanations.js',
    'scripts/export_viewer_snapshot.py', 'scripts/build_pages.py',
    'scripts/run_pointnet4m_viewer.py', 'scripts/prepare_pointnet4m_viewer_graphs.py',
    'scripts/audit_pointnet4m_viewer.py', 'scripts/finish_pointnet4m_viewer.py',
]


def command(*args, capture=False):
    result = subprocess.run(list(args), check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else None


def audit_study(root):
    config = read_json(root/'config.json')
    method = config['pointnet4m']['method']
    setting = config['pointnet4m']
    evaluator = PointNetEvaluator(DATA, config, 'pointnet4m-publication-audit')
    rooms = {r['id']: r for r in evaluator.observations.rooms}
    truths = {key:evaluator.observations.truth(room).by_ap for key,room in rooms.items()}
    trials = read_trials(root/'trials')
    expected = min(setting['source_trials'],len(configurations(setting['spaces'][method]))) + len(
        config['optimization']['unification'])*(len(configurations(config['optimization']['smart_lea']))+setting['joint_trials'])
    assert len(trials) == expected, (root, len(trials), expected)
    assert all(t['split']=='calibration' and t['level']==0 for t in trials)
    labels = sorted({t['label'] for t in trials})
    selected = {}
    for label in labels:
        chosen = read_json(root/'selected'/(label+'.json'))
        best = min((t for t in trials if t['label']==label), key=lambda t:t['loss'])
        assert chosen['id'] == best['id']
        selected[label] = chosen
    independent = selected[method+'_fixed_network']
    for unify in config['optimization']['unification']:
        mode = 'unified' if unify else 'ununified'
        stage = selected[method+'_lea_'+mode]
        joint = selected[method+'_joint_'+mode]
        original_map = read_json(BASE/'selected'/('mpurge_map_'+mode+'.json'))['parameters']
        assert stage['frozen'] == {'source': independent['parameters'], 'map': original_map}
        first = min((t for t in trials if t['label']==joint['label']), key=lambda t:t['number'])
        assert first['parameters'] == {**stage['frozen'], 'lea': stage['parameters']}
        assert joint['loss'] <= stage['loss'] + 1e-10
    results = read_json(root/'results.json')
    wanted = {(label,split,level) for label in labels for split,levels in (
        ('calibration',[0.]),('evaluation',config['observations']['noise_levels'])) for level in levels}
    assert {(r['label'],r['split'],r['level']) for r in results} == wanted
    query_sets, measured = {}, []
    for row in results:
        chosen = selected[row['label']]
        assert all(row[k]==chosen[k] for k in ('method','parameters','unify','frozen'))
        detail = read_npz(root/row['outcome'])[0]
        expected_repeats = config['observations'][row['split']+'_repetitions']
        if row['level']==0:
            expected_repeats = expected_repeats[:1]
        assert sorted(r['repetition'] for r in detail['sources']) == sorted(expected_repeats)
        by_ap = {ap:[] for ap in ('A','B','A+B')}
        for record in detail['sources']:
            assert all(np.isfinite(v).all() for v in record['by_ap'].values())
            for ap in by_ap:
                values = ap_source_metrics(record['by_ap'],truths[record['room']],ap.split('+'))
                by_ap[ap].append(values)
        if row['method']==method:
            np.testing.assert_allclose(row['loss'],np.mean([v['ospa'] for v in by_ap['A+B']]),rtol=1e-10,atol=1e-12)
        else:
            errors=[]
            assert len(detail['queries']) == len(expected_repeats)*4
            for record in detail['queries']:
                truth = query_truth(record,rooms)
                e = np.linalg.norm(record['xyz']-truth,axis=1)
                np.testing.assert_allclose(record['errors_m'],e,rtol=1e-12,atol=1e-12)
                expected_count=60 if row['split']=='calibration' else 40
                assert len(e)==expected_count
                errors.extend(e)
                key=(record['room'],record['layout'],row['split'])
                ids=set(map(int,record['query_ids']))
                assert key not in query_sets or query_sets[key]==ids
                query_sets[key]=ids
            np.testing.assert_allclose(row['loss'],np.mean(errors),rtol=1e-12,atol=1e-12)
        measured.append({'label':row['label'],'split':row['split'],'sigma':row['level'],
                         'loss':row['loss'],'source_metrics':by_ap})
    for (room,layout,split),ids in query_sets.items():
        if split=='calibration':
            assert ids.isdisjoint(query_sets[room,layout,'evaluation'])
    report={'state':'passed','study':root.name,'trials':len(trials),'selected':len(selected),
            'outcomes':len(results),'checkpoint_sha256':CHECKPOINT_SHA256,'measurements':measured}
    write_json(root/'verification.json',report)
    return {k:v for k,v in report.items() if k!='measurements'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--expected-head')
    parser.add_argument('--audit-root',type=Path)
    args=parser.parse_args()
    if args.audit_root:
        print(audit_study(args.audit_root),flush=True)
        return
    roots=[PARENT/f'rectangle-{method}-v1' for method in METHODS]
    try:
        while True:
            states=[read_json(root/'status.json') if (root/'status.json').exists() else {} for root in roots]
            errors=[s['error'] for s in states if s.get('state')=='failed']
            if errors:
                raise RuntimeError(str(errors))
            if all(s.get('state')=='complete' for s in states):
                break
            if not args.wait:
                raise RuntimeError('Campaign is still running; use --wait for background completion')
            write_json(STATUS,{'state':'waiting_for_searches','completed_trials':sum(s.get('completed_trials',0) for s in states),
                               'target_trials':8436,'updated':time.time()})
            time.sleep(30)
        write_json(STATUS,{'state':'auditing'})
        audits=[audit_study(root) for root in roots]
        assert sum(a['trials'] for a in audits)==8436
        from indoor_vt.execution.obstacle_reconstruction import run_obstacle_reconstruction
        obstacle_root=PARENT/'rectangle-pointnet4m-obstacles-v1'
        protocol=read_json(PARENT/'rectangle-matched-obstacle-reconstruction-v1/protocol.json')
        protocol.pop('methods')
        write_json(STATUS,{'state':'evaluating_obstacles'})
        rows=run_obstacle_reconstruction('data/redesign/rectangle-matched-obstacles-20261027',BASE,
            obstacle_root,selection_data_root=DATA,study=obstacle_root.name,
            protocol_overrides=protocol,method_prefix='pointnet4m_')
        assert len(rows)==72 and all(r['input_count']==270 for r in rows)
        write_json(PARENT/'pointnet4m-viewer-verification.json',{'state':'passed','studies':audits,'obstacle_conditions':len(rows)})
        if args.publish:
            if not args.expected_head or command('git','rev-parse','HEAD',capture=True)!=args.expected_head:
                raise RuntimeError('Reviewed source commit is no longer HEAD; results are retained, publication awaits integration')
            command('git','diff','--quiet','--',*SOURCES)
            command('git','diff','--quiet','--','published/redesign')
            if command('git','ls-files','--others','--exclude-standard','--','published/redesign',capture=True):
                raise RuntimeError('Untracked publication files exist; refusing to include unrelated artifacts')
            command('git','diff','--cached','--quiet')
        write_json(STATUS,{'state':'exporting'})
        command(sys.executable,'scripts/export_viewer_snapshot.py','--incremental')
        command(sys.executable,'scripts/build_pages.py')
        if args.publish:
            assert command('gh','api','user','--jq','.login',capture=True)=='alasleimi'
            assert command('git','remote','get-url','origin',capture=True)=='https://github.com/alasleimi/indoor-vt-benchmark.git'
            command('git','add','--','published/redesign')
            command('git','diff','--cached','--check')
            command('git','commit','-m','Publish calibrated PointNet 4M reconstruction and SMART-LEA results')
            head=command('git','rev-parse','HEAD',capture=True)
            command('git','push','origin','HEAD:main')
            import json
            run=None
            for _ in range(24):
                runs=json.loads(command('gh','run','list','--repo','alasleimi/indoor-vt-benchmark',
                    '--workflow','pages.yml','--limit','5','--json','databaseId,headSha,status,conclusion',capture=True))
                run=next((r for r in runs if r['headSha']==head),None)
                if run:
                    break
                time.sleep(5)
            if run is None:
                raise RuntimeError('Push succeeded but Pages workflow was not discovered')
            command('gh','run','watch',str(run['databaseId']),'--repo','alasleimi/indoor-vt-benchmark','--exit-status','--interval','15')
            import gzip
            url='https://alasleimi.github.io/indoor-vt-benchmark/snapshot/overview.json.gz?release='+head[:12]
            with urlopen(url,timeout=60) as response:
                overview=json.loads(gzip.decompress(response.read()))
            entries=[m for m in overview['methods'] if m['family']=='pointnet4m']
            assert len(entries)==54, len(entries)
            write_json(STATUS,{'state':'published','commit':head,'methods':len(entries),'trials':8436,
                               'url':'https://alasleimi.github.io/indoor-vt-benchmark/','updated':time.time()})
        else:
            write_json(STATUS,{'state':'ready','methods':54,'trials':8436,'updated':time.time()})
    except Exception as error:
        write_json(STATUS,{'state':'failed','error':str(error),'updated':time.time()})
        raise


if __name__=='__main__':
    main()
