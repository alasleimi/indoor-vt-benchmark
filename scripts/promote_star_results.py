"""Package fully measured frozen star variants as a separate exploratory study."""
from pathlib import Path
import numpy as np

from indoor_vt.acquisition.records import read_json, write_json, digest
from indoor_vt.execution.artifacts import read_npz, write_npz
from indoor_vt.execution.star_experiment import METHODS


def main():
    base = Path('runs/redesign/rectangle-clean-selection-v4')
    screen, validation = Path('runs/star-weight-ablation-v1'), Path('runs/star-weight-validation-v1')
    config = read_json(base/'config.json')
    protocol = read_json(screen/'protocol.json')
    rows = [(root, row) for root in (screen, validation) for row in read_json(root/'results.json')]
    rows = [(root, row) for root, row in rows if row['grouping'] == 'stars' and row['weight'] == 'uniform'
            and row['finisher'] in METHODS.values()]
    out = Path('runs/redesign/rectangle-stars-v1')
    packaged = []
    for method, ending in METHODS.items():
        parameters = {**protocol['finisher_parameters'][ending], **protocol['source_parameters']}
        for level in config['observations']['noise_levels']:
            wanted = config['observations']['evaluation_repetitions'][:1] if level == 0 else config['observations']['evaluation_repetitions']
            cases = [(root, r) for root, r in rows if r['finisher'] == ending and r['level'] == level]
            assert sorted(r['repetition'] for _, r in cases) == sorted(wanted), (ending, level, 'incomplete')
            identity = {'method': method, 'parameters': parameters, 'unify': False, 'level': level,
                        'split': 'evaluation', 'frozen': {}, 'grouping': 'local_stars_v1'}
            key = digest(identity)
            records = []
            for root, row in sorted(cases, key=lambda x: x[1]['repetition']):
                predictions = read_npz(root/row['outcome'])[0]['by_ap']
                records.append({'room': 'rectangle_scene_00', 'repetition': row['repetition'], 'by_ap': predictions,
                                'audit': {'grouping': 'Local stars; no edge weighting',
                                          'candidate_counts': row['candidate_counts'],
                                          'source_experiment': root.name, 'source_outcome': row['outcome']}})
            metrics = {name: float(np.mean([r['metrics_by_ap']['A+B'][name] for _, r in cases])) for name in ('ospa', 'f1')}
            receipt = {**identity, 'id': key, 'label': method + '_ununified', 'loss': metrics['ospa'],
                       'metrics': metrics, 'outcome': f'outcomes/{key}.npz',
                       'seconds': sum(r['seconds'] for _, r in cases)}
            packaged.append((receipt, {'sources': records, 'queries': []}))
    # Do not expose a partial study: validate every condition before writing config/implementation.
    out.mkdir(parents=True, exist_ok=True)
    for receipt, detail in packaged:
        write_npz(out/receipt['outcome'], detail)
        write_json(out/'evaluations'/(receipt['id']+'.json'), receipt)
        if receipt['level'] == 0:
            write_json(out/'selected'/(receipt['label']+'.json'), receipt)
    write_json(out/'results.json', [r for r, _ in packaged])
    write_json(out/'config.json', {**config, 'name': out.name, 'engine': 'star_exploration'})
    write_json(out/'protocol.json', {**protocol, 'promotion': 'Exploratory, based on measured same-room comparisons; no recalibration'})
    write_json(out/'status.json', {'state': 'complete', 'completed_trials': 0, 'target_trials': 0})
    write_json(out/'implementation.json', read_json(base/'implementation.json'))
    print(f'Packaged {len(packaged)} saved conditions for {len(METHODS)} star variants')


if __name__ == '__main__':
    main()
