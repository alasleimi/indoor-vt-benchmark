"""After tuning completes, append the new frozen methods on existing obstacle data."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path  # noqa: E402 - numerical libraries must see thread limits before import
import time  # noqa: E402 - numerical libraries must see thread limits before import
from indoor_vt.acquisition.records import read_json,write_json  # noqa: E402 - numerical libraries must see thread limits before import
from indoor_vt.execution.obstacle_reconstruction import run_obstacle_reconstruction  # noqa: E402 - numerical libraries must see thread limits before import
from indoor_vt.worker_memory import install_limits,DEFAULT_LIMITS  # noqa: E402 - numerical libraries must see thread limits before import


if __name__=='__main__':
    limits=install_limits(**DEFAULT_LIMITS)
    parent=Path('runs/redesign')
    status_path=parent/'allow-crossing-obstacles-status.json'
    write_json(status_path,dict(state='waiting_for_frozen_choices'))
    try:
        while True:
            suite=read_json(parent/'allow-crossing-suite-status.json')
            if suite['state']=='failed':
                raise RuntimeError(str(suite['errors']))
            if suite['state']=='complete':
                break
            time.sleep(15)
        root=parent/'rectangle-matched-obstacle-reconstruction-v1'
        protocol=read_json(root/'protocol.json')
        protocol.pop('methods')
        write_json(status_path,dict(state='evaluating',recalibration=False))
        result=run_obstacle_reconstruction('data/redesign/rectangle-matched-obstacles-20261027',
            parent/'rectangle-clean-selection-v4',root,selection_data_root='data/redesign/rectangle-20261025',
            study=root.name,protocol_overrides=protocol)
        added=[r for r in result if r['view_method_id'].startswith('mpurge_allow_crossing')]
        assert len(added)==48, len(added)
        write_json(status_path,dict(state='complete',new_conditions=len(added),recalibration=False))
    except Exception as error:
        write_json(status_path,dict(state='failed',error=str(error)))
        raise
