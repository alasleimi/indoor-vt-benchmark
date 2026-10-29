"""Save the default matching charts so the public page opens without a room-wide calculation."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import time
import numpy as np

from indoor_vt.execution.peak_explorer import PeakExplorer
from indoor_vt.execution.peak_distributions import SCHEMA, dissimilarity_distributions
from indoor_vt.science.matching import MPUrgeMatcher
from scripts.export_viewer_snapshot import write


def read(name):
    return json.loads(gzip.decompress((Path('published/redesign')/(name+'.json.gz')).read_bytes()))


def compute(condition):
    ap, noise = condition
    out = Path('published/redesign/matching-precomputed')/f'{ap}-{noise:g}.json.gz'
    saved = read(f'matching-precomputed/{ap}-{noise:g}') if out.exists() else {}
    if saved.get('distributions', {}).get('schema') == SCHEMA:
        return str(out)
    start = time.perf_counter()
    init = read('matching-init')
    p = init['meta']
    data = read(f'profiles/0-{ap}-{noise:g}')
    for profile in data['profiles']:
        for key in ('xyz', 'ranges', 'source_ids'):
            profile[key] = np.array(profile[key])
    data['summaries'] = {}
    context = (np.array(init['scene']['survey_xyz']), np.array(init['scene']['pairs']),
               MPUrgeMatcher(half_window=p['half_window'], alpha=p['alpha'], size_unification_norm=None))
    explorer = PeakExplorer(None)
    if not saved:
        saved = {'first': explorer.summary(context, data, 'first'),
                 'final': explorer.summary(context, data, 'final'), 'spatial': explorer.spatial(context, data)}
    saved['distributions'] = dissimilarity_distributions(data['profiles'], context[-2], context[-1])
    write(out, saved)
    return f'{ap}, sigma={noise}: {time.perf_counter()-start:.1f}s'


def main():
    init = read('matching-init')
    conditions = [(ap, noise) for noise in init['meta']['noise_levels'] for ap in ('A', 'B')]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for line in pool.map(compute, conditions):
            print(line, flush=True)
    manifest = read('manifest')
    manifest['matching_precomputed'] = True
    write(Path('published/redesign/manifest.json.gz'), manifest)


if __name__ == '__main__':
    main()
