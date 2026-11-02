"""Launch independent Chamfer supplier searches with durable per-study logs."""
import argparse
from pathlib import Path
from indoor_vt.execution.chamfer_experiment import SUPPLIERS, run_chamfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/redesign/rectangle-20261025'))
    parser.add_argument('--base', type=Path, default=Path('runs/redesign/rectangle-clean-selection-v4'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--job', choices=['standalone', *SUPPLIERS], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.run or Path('runs/redesign') / (
        f'chamfer-smoke-{args.job}' if args.smoke else f'rectangle-chamfer-{args.job}-v1')
    run_chamfer(args.data, root, args.base, args.job, smoke=args.smoke)


if __name__ == '__main__':
    main()
