"""Correct AP aggregation from retained coordinates; never rerun scientific stages.

Historical trial logs and frozen parameter choices are left intact. Original
receipt scalar values are retained alongside corrected presentation metrics.
"""
from pathlib import Path
import argparse

from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.acquisition.observations import Observations
from indoor_vt.execution.artifacts import read_npz
from indoor_vt.execution.explorer import saved_stage_metrics
from indoor_vt.execution.obstacle_reconstruction import metrics_for_record

PROTOCOL = 'source-metrics-within-ap-v1'


def refresh(root, default_data):
    config = read_json(root/'config.json')
    rows = read_json(root/'results.json')
    observation = Observations(config.get('data_root',default_data),config)
    rooms = {r['id']:r for r in observation.rooms}
    truth = {key:observation.truth(room) for key,room in rooms.items()}
    corrected, changed = [], 0
    for row in rows:
        if row.get('metric_protocol') == PROTOCOL:
            corrected.append(row)
            continue
        detail = read_npz(root/row['outcome'])[0]
        if not detail.get('sources'):
            corrected.append(row)
            continue
        updated = dict(row)
        if config.get('engine') == 'obstacle_reconstruction':
            metrics, counts = metrics_for_record(detail['sources'][0])
            updated.update(source_metrics_by_ap=metrics,source_counts=[counts],metrics=metrics['A+B'])
        else:
            stages = saved_stage_metrics(detail,rooms,truth)
            updated.update(source_metrics_by_ap=stages['source_metrics_by_ap'],source_counts=stages['source_counts'])
            if 'ospa' in row['metrics']:
                updated['metrics'] = {**row['metrics'],**stages['sources'],
                    'matched_rmse_m':stages['source_metrics_by_ap']['A+B']['matched_rmse_m']}
        updated['metric_protocol'] = PROTOCOL
        updated['recorded_metrics'] = row.get('recorded_metrics',row['metrics'])
        # loss is the historical experiment/selection objective, not a display
        # metric; retaining it does not change the selected hyperparameters.
        updated['metric_note'] = 'Display metrics recomputed from saved coordinates with separate AP assignments. Historical loss and trial logs retain their recorded selection objective.'
        receipt = root/'evaluations'/f"{row['id']}.json"
        if receipt.exists():
            write_json(receipt,updated)
        corrected.append(updated)
        changed += 1
    if changed:
        write_json(root/'results.json',corrected)
    write_json(root/'ap_metric_refresh.json',dict(protocol=PROTOCOL,corrected_receipts=changed,
        saved_source_coordinates_unchanged=True,reconstruction_runs=0,trial_logs_unchanged=True))
    print(f'{root.name}: corrected {changed} saved receipts; no reconstruction',flush=True)
    return changed


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data/redesign/rectangle-20261025'))
    parser.add_argument('--run',type=Path,action='append')
    args=parser.parse_args()
    names=('rectangle-clean-selection-v4','rectangle-additional-selection-v1','rectangle-compositions-v1',
           'rectangle-naked-selection-v1','rectangle-range-rmse-true-count-v1',
           'rectangle-obstacle-reconstruction-v1','rectangle-matched-obstacle-reconstruction-v1')
    roots=args.run or [Path('runs/redesign')/n for n in names]
    for root in roots:
        if (root/'results.json').exists():
            refresh(root,args.data)
