"""Export saved redesign outcomes and inputs; never replay or recalibrate a run."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re

from indoor_vt.acquisition.records import read_json
from indoor_vt.execution.explorer import method_entries
from indoor_vt.execution.peak_explorer import PeakExplorer
from indoor_vt.execution.presentation import construction
from indoor_vt.execution.true_matching import TrueMatchingAnalysis
from indoor_vt.execution.viewer import MultiStudyView, json_value, preview


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(json_value(value), separators=(',', ':'), allow_nan=False).encode()
    path.write_bytes(gzip.compress(encoded, compresslevel=6, mtime=0))


def artifact_keys(value):
    if isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from artifact_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from artifact_keys(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='published/redesign')
    args = parser.parse_args()
    root = Path(args.output)
    view = MultiStudyView('data/redesign/rectangle-20261025', 'runs/redesign/rectangle-clean-selection-v4')
    overview = view.overview()
    lookup, owners, constructors = {}, {}, {}
    for owner in view.views:
        snapshot = owner.overview()
        rows = {r['id']: r for r in snapshot['trials']}
        rows.update({r['id']: r for r in read_json(owner.run_root/'results.json')})
        for key, row in rows.items():
            owners[key] = owner
            lookup[key] = row
        # Read exactly the saved receipt; avoid repeatedly scanning 28k trial files.
        original = owner.receipt
        owner.receipt = lambda key, rows=rows, original=original: rows[key] if key in rows else original(key)
    view.owner = lambda key: owners[key]
    print(f"Exporting {len(lookup)} receipts; {len(overview['methods'])} method rows", flush=True)
    for receipt in lookup.values():
        label = receipt['label']
        codes = {}
        for method in method_entries(receipt):
            component = method['component']
            if component not in codes:
                codes[component] = construction(receipt, owners[receipt['id']].config, component)
        constructors.setdefault(label, {})[receipt['id']] = codes
    for label, records in constructors.items():
        write(root/'constructors'/f'{label}.json.gz', records)
    print('Concrete constructions exported', flush=True)
    exported, artifacts, absent = [], set(), []
    for key, receipt in lookup.items():
        if not receipt.get('outcome'):
            continue
        owner = owners[key]
        if not (owner.run_root / receipt['outcome']).exists():
            absent.append(key)
            continue
        result = owner.result(key)
        write(root/'results'/f'{key}.json.gz', result)
        exported.append(key)
        artifacts.update(artifact_keys([c.get('audit') for c in result['cases']]))
        if len(exported) % 100 == 0:
            print(f'{len(exported)} saved outcomes exported', flush=True)
    retained = []
    for key in sorted(artifacts):
        for owner in view.views:
            try:
                value = owner.store.get(key)
            except FileNotFoundError:
                continue
            write(root/'artifacts'/f'{key}.json.gz', {'id': key, 'value': preview(value)})
            retained.append(key)
            break
    peaks = PeakExplorer(view)
    context = peaks.context('')
    default = peaks.request('init')
    write(root/'matching-init.json.gz', default)
    for repetition in (0, 2, 3):
        for ap in ('A', 'B'):
            for level in default['meta']['noise_levels']:
                data = peaks.profiles(context, ap, level, repetition=repetition)
                write(root/'profiles'/f'{repetition}-{ap}-{level:g}.json.gz',
                      {'profiles': data['profiles'], 'sources': data['sources']})
        print(f'Matching observations exported: realization ID {repetition}', flush=True)
    write(root/'true-matching.json.gz', TrueMatchingAnalysis(view).request())
    overview['publication'] = {'static': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'note': 'Saved outcomes and parameter searches. Matching is computed in your browser from '
                'the saved observations. Replaying evicted numerical stages requires the local Python viewer.'}
    write(root/'overview.json.gz', overview)
    sources = {str(p.relative_to(Path('src/indoor_vt'))).replace('\\', '/'): p.read_text(encoding='utf-8')
               for p in Path('src/indoor_vt').rglob('*.py')}
    write(root/'sources.json.gz', sources)
    manifest = {'created_utc': overview['publication']['created_utc'], 'results': exported,
                'retained_artifact_previews': retained, 'missing_outcomes': absent,
                'receipt_labels': {k: r['label'] for k, r in lookup.items()},
                'default_trial': default['meta']['trial']}
    write(root/'manifest.json.gz', manifest)
    total = sum(p.stat().st_size for p in root.rglob('*.gz'))
    print(f'Complete: {len(exported)} outcomes, {len(retained)} cached previews, {total/1024**2:.1f} MiB', flush=True)


if __name__ == '__main__':
    main()
