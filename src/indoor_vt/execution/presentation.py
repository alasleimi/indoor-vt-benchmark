"""Concrete constructions and recorded search spaces, without numerical execution."""
from dataclasses import fields, is_dataclass
from functools import partial
from math import prod
import html

from .presets import standard_mpurge, mpurge_map, smart_lea
from .compositions import VARIANTS, composed_reconstructor, FinishedReconstructor
from .additional import zayets_proposals, ransac_proposals, finisher
from ..acquisition.records import Seeds
from ..science.range_rmse import range_rmse


def expression(value, indent=0):
    """Readable Python calls with named dependencies, including their real defaults."""
    pad = ' ' * indent
    if isinstance(value, partial):
        items = [expression(value.func)] + [expression(v, indent+4) for v in value.args]
        items += [f'{k}={expression(v, indent+4)}' for k, v in (value.keywords or {}).items()]
        name = 'partial'
    elif is_dataclass(value):
        name = type(value).__name__
        items = [f'{f.name}={expression(getattr(value, f.name), indent+4)}'
                 for f in fields(value) if f.init]
    elif callable(value):
        return value.__name__
    else:
        return repr(value)
    compact = f'{name}(' + ', '.join(items) + ')'
    if '\n' not in compact and len(compact) + indent <= 94:
        return compact
    return name + '(\n' + ',\n'.join(pad + '    ' + item for item in items) + ',\n' + pad + ')'


def construction(receipt, config, component='sources'):
    method, p, frozen = receipt['method'], receipt['parameters'], receipt.get('frozen') or {}
    unify = receipt.get('unify')
    variant = method.removesuffix('_joint').removesuffix('_lea')
    joint = method == 'joint' or method.endswith('_joint')
    source_only = method == 'mpurge' or method in VARIANTS or method in ('zayets', 'ransac')
    initial_only = component == 'initial' or method == 'mpurge_map'
    objects, notes = [], []
    if method.startswith('mpurge_stars_'):
        from .star_experiment import star_reconstructor
        objects = [('method', star_reconstructor(p, method.removeprefix('mpurge_stars_')))]
        notes.append('Frozen exploratory variant: same local pair graph; one star per peak instead of '
                     'connected components. The selector replaces the proposer’s clusterer and support_filter.')
    elif method == 'range_rmse_true_count':
        objects = [('method', range_rmse(p))]
        notes.append('The runner derives a deterministic seed per room and AP from this base seed. '
                     'The true count is a separate fit argument; no true coordinates enter this constructor.')
    elif not initial_only and method != 'smart_lea_truth':
        source_p = p if source_only else p.get('source', p.get('mpurge')) if joint else frozen.get('source', frozen.get('mpurge'))
        seed = config.get('additional', {}).get('proposal_seed', 9173)
        if variant in VARIANTS:
            source = composed_reconstructor(variant, source_p, unify=unify, seed=seed)
        elif variant in ('zayets', 'ransac'):
            proposer = zayets_proposals(source_p) if variant == 'zayets' else ransac_proposals(source_p, seed)
            source = FinishedReconstructor(proposer, finisher(variant, source_p))
        else:
            source = standard_mpurge(source_p, unify=unify)
        objects.append(('method' if component == 'sources' else 'reconstructor', source))
        if isinstance(source, FinishedReconstructor) and hasattr(source.proposals, 'clusterer'):
            notes.append('FinishedReconstructor uses the proposer through candidate estimation. '
                         'Its selector replaces the proposer’s clusterer and support_filter; '
                         'those two nested fields are inactive in this composition.')
    if component != 'sources' and method != 'range_rmse_true_count':
        map_p = p if method == 'mpurge_map' else p['map'] if joint else frozen['map']
        objects.append(('method' if initial_only else 'initializer', mpurge_map(map_p, unify=unify)))
        if not initial_only:
            objects.append(('refiner', smart_lea(p['lea'] if joint else p, unify=unify)))
        if method == 'smart_lea_truth':
            notes.append('The refiner receives the ground-truth source map; there is no reconstructor.')
    return {'code': '\n\n'.join(name + ' = ' + expression(value) for name, value in objects),
            'notes': notes, 'receipt_id': receipt['id'], 'component': component,
            'parameters': p, 'frozen': frozen}


def construction_page(value):
    notes = ''.join('<p>' + html.escape(n) + '</p>' for n in value['notes'])
    return ('<!doctype html><html><meta charset="utf-8"><title>Concrete method construction</title>'
            '<style>body{max-width:1100px;margin:32px auto;padding:0 24px;font:15px system-ui;'
            'color:#213540;background:#f8fafb}pre{padding:24px;background:white;border:1px solid #dce4e8;'
            'border-radius:8px;overflow:auto;line-height:1.6;font-size:13px}small{overflow-wrap:anywhere}</style>'
            '<a href="/">Experiment viewer</a><h1>Concrete method construction</h1>'
            '<p>Selected parameter values and injected dependencies. No reconstruction is executed.</p>'
            + notes + '<pre>' + html.escape(value['code']) + '</pre><small>Receipt: '
            + html.escape(value['receipt_id']) + '</small></html>').encode()


def effective_count(space):
    count = prod(len(v) for v in space.values())
    if 'half_window' in space and 1 in space.get('alpha', []):
        count = count // (len(space['half_window'])*len(space['alpha'])) * (
            (len(space['alpha'])-1)*len(space['half_window'])+1)
    if 'median_clip_factor' in space and 0 in space.get('median_penalty_weight', []):
        count = count // (len(space['median_clip_factor'])*len(space['median_penalty_weight'])) * (
            (len(space['median_penalty_weight'])-1)*len(space['median_clip_factor'])+1)
    return count


def search_spec(method, config):
    selection, family = method['selection'], method['family']
    opt = config['optimization']
    variant = method['label'].removesuffix('_ununified').removesuffix('_unified').removesuffix('_joint').removesuffix('_lea')
    if selection == 'fixed':
        return {'strategy': 'Fixed settings; no calibration search', 'grids': {}}
    composed = variant in config.get('compositions', {}).get('spaces', {})
    source = (config['compositions']['spaces'][variant] if composed else
              config['additional'][family] if family in ('zayets', 'ransac') else opt['mpurge'])
    seeds = Seeds(opt['seed'])
    if selection == 'joint':
        grids = {'Reconstruction': source, 'MPUrge-MAP': opt['mpurge_map'], 'SMART-LEA': opt['smart_lea']}
        settings = config['compositions'] if composed else config['additional'] if family != 'mpurge' else opt
        budget = settings['joint_trials']
        seed_args = ('composition-joint', variant, method['unify']) if composed else (
            ('joint', family, method['unify']) if family != 'mpurge' else ('joint', method['unify']))
        strategy = 'Seeded random configurations without replacement, including the independently selected baseline'
        seed = seeds.seed(*seed_args)
    else:
        grids = ({'SMART-LEA': opt['smart_lea']} if selection in ('stagewise', 'truth') else
                 {'MPUrge-MAP': opt['mpurge_map']} if selection == 'localization' else {'Reconstruction': source})
        total = prod(effective_count(s) for s in grids.values())
        budget = min(total, config['compositions']['source_trials']) if composed and selection == 'ospa' else total
        seed = seeds.seed('composition-source', variant) if budget < total else None
        strategy = 'Seeded random sample without replacement from the Cartesian grid' if seed is not None else 'Exhaustive Cartesian grid'
    return {'strategy': strategy, 'grids': grids, 'budget': budget, 'seed': seed,
            'base_seed': opt['seed'], 'effective_configurations': prod(effective_count(s) for s in grids.values()),
            'raw_configurations': prod(prod(len(v) for v in s.values()) for s in grids.values()),
            'notes': 'Duplicate inactive settings are collapsed: half_window when alpha = 1, '
                     'median_clip_factor when median_penalty_weight = 0. Unification is fixed for this method tab. '
                     + ('MAP parameters are frozen and true VT coordinates are supplied; only SMART-LEA is searched.'
                        if selection == 'truth' else
                        'Upstream reconstruction and MAP parameters are frozen; only SMART-LEA is searched.'
                        if selection == 'stagewise' else '')}
