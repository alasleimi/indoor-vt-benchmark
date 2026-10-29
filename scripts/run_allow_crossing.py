"""Run the three crossing-allowed variants through the established full protocol."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
from concurrent.futures import ProcessPoolExecutor  # noqa: E402 - numerical libraries must see thread limits before import
from pathlib import Path  # noqa: E402 - numerical libraries must see thread limits before import
import argparse  # noqa: E402 - numerical libraries must see thread limits before import
import time  # noqa: E402 - numerical libraries must see thread limits before import

METHODS = {'solo': 'mpurge_allow_crossing', 'consumption': 'mpurge_allow_crossing_consumption',
           'robust': 'mpurge_allow_crossing_robust'}


def worker(name, data, root, base):
    from indoor_vt.worker_memory import install_limits, DEFAULT_LIMITS
    from indoor_vt.execution.composition_experiment import run_compositions
    install_limits(**DEFAULT_LIMITS)
    return len(run_compositions(data, root, base, methods=(METHODS[name],)))


def main(args):
    from indoor_vt.acquisition.records import write_json, read_json
    roots = {name: args.parent / f'rectangle-allow-crossing-{name}-v1' for name in METHODS}
    status_path = args.parent / 'allow-crossing-suite-status.json'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {name: pool.submit(worker, name, args.data, root, args.base) for name,root in roots.items()}
        while True:
            states = {name: read_json(root/'status.json') if (root/'status.json').exists() else {'state':'starting'}
                      for name,root in roots.items()}
            errors = {name: str(f.exception()) for name,f in jobs.items() if f.done() and f.exception()}
            done = all(f.done() for f in jobs.values())
            write_json(status_path, dict(state='failed' if errors else 'complete' if done else 'running',
                updated=time.time(),studies=states,errors=errors,workers=args.workers))
            if done:
                if errors:
                    raise RuntimeError(str(errors))
                print('All three variants completed standalone, stagewise and joint evaluations.',flush=True)
                return
            time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data/redesign/rectangle-20261025'))
    parser.add_argument('--base',type=Path,default=Path('runs/redesign/rectangle-clean-selection-v4'))
    parser.add_argument('--parent',type=Path,default=Path('runs/redesign'))
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    main(parser.parse_args())
