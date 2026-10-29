"""Frozen-parameter, paired MPUrge grouping/estimation ablation; no calibration."""
import argparse
from pathlib import Path
import time

import numpy as np

from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.acquisition.observations import Observations
from indoor_vt.execution.additional import finish_one, finisher
from indoor_vt.execution.artifacts import read_npz, write_npz
from indoor_vt.execution.presets import standard_mpurge
from indoor_vt.execution.reconstruction import fit_groups, summarize_clusters
from indoor_vt.metrics import ap_source_metrics
from indoor_vt.science.grouping import StarGrouper, group_within_scopes
from indoor_vt.science.models import Ragged, Survey, SourceMap
from indoor_vt.science.reconstruction import match_pairs


def edge_weights(graph, xyz, mode):
    """Dimensionless edge quality; scales depend only on observed edges."""
    costs = graph.costs
    positive = costs[costs > 0]
    cost_scale = float(np.median(positive)) if len(positive) else 1.
    owners = graph.nodes(graph.endpoints.ravel())[:, 0].reshape(-1, 2)
    distances = np.linalg.norm(xyz[owners[:, 0]] - xyz[owners[:, 1]], axis=1)
    distance_scale = float(np.median(distances[distances > 0]))
    quality = np.ones(len(costs))
    if mode in ('similarity', 'both'):
        quality /= 1 + costs / cost_scale
    if mode in ('distance', 'both'):
        quality /= 1 + (distances / distance_scale)**2
    adjacency = {}
    for (a, b), q in zip(graph.endpoints, quality):
        adjacency.setdefault(int(a), {})[int(b)] = float(q)
        adjacency.setdefault(int(b), {})[int(a)] = float(q)
    return adjacency, {'cost_scale': cost_scale, 'distance_scale_m': distance_scale}


def weighted_candidates(xyz, rows, groups, estimator, graph, mode):
    if mode == 'uniform':
        return fit_groups(xyz, rows, groups, estimator)
    adjacency, scales = edge_weights(graph, xyz, mode)
    points, accepted = [], []
    for index, track in enumerate(groups):
        ids, slots = np.asarray(track, dtype=int).T
        nodes = graph.node_offsets[ids] + slots
        members = set(nodes)
        weights = np.array([np.mean(q) if q else 0. for node in nodes
                            for q in [[v for neighbor, v in adjacency.get(int(node), {}).items()
                                       if neighbor in members]]])
        keep = weights > 0
        if keep.sum() < 4:
            continue
        weights = weights[keep] / weights[keep].mean()
        ranges = np.array([rows[i][j] for i, j in zip(ids, slots)])
        point, _ = estimator.fit(xyz[ids[keep]], ranges[keep], weights=weights)
        if point is not None:
            points.append(point)
            accepted.append(index)
    return {'xyz': np.asarray(points).reshape(-1, 3),
            'accepted_groups': np.array(accepted, dtype=np.int64),
            'groups': Ragged.pack(groups, width=2, dtype=np.int64).record(), 'weight_scales': scales}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/star-weight-ablation-v1')
    parser.add_argument('--levels', type=float, nargs='+', default=[0., .01, .02, .05, .1, .2, .5, 1., 2., 5.])
    parser.add_argument('--variants', nargs='+', default=[])
    parser.add_argument('--finishers', nargs='+', default=['cluster', 'consumption', 'robust'])
    args = parser.parse_args()
    output = Path(args.output)
    base = Path('runs/redesign/rectangle-clean-selection-v4')
    config = read_json(base / 'config.json')
    selected = read_json(base / 'selected/mpurge_ununified.json')
    observations = Observations(Path('data/redesign/rectangle-20261025'), config)
    room = observations.rooms[0]
    truth = observations.truth(room).by_ap
    algorithm = standard_mpurge(selected['parameters'], unify=False)
    finishers = {name: read_json(Path('runs/redesign/rectangle-compositions-v1/selected') /
                 ('mpurge_' + name + '_ununified.json'))['parameters']
                 for name in ('consumption', 'robust')}
    protocol = {'source_parameters': selected['parameters'], 'unify': False,
                'selection_id': selected['id'], 'finisher_parameters': finishers,
                'note': 'One common frozen OSPA-selected proposer for every finisher. No recalibration. '
                        'The support-first arm uses this common proposer (half_window=2), not its separately '
                        'selected half_window=6. Compare each change with its same-finisher component/uniform arm.',
                'levels': args.levels, 'repetitions': config['observations']['evaluation_repetitions'],
                'weight_formula': 'q_similarity=1/(1+cost/median_positive_cost); '
                                  'q_distance=1/(1+(distance/median_positive_distance)^2); '
                                  'q_both=q_similarity*q_distance; vertex weights=mean incident q '
                                  'among retained group nodes, normalized to mean 1. Zero-degree nodes omitted. '
                                  'Minimize sum_i w_i*(norm(v-x_i)-range_i)^2. '
                                  'Star means center plus its immediate neighbors within each original local scope.'}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', protocol)
    results = read_json(output / 'results.json') if (output / 'results.json').exists() else []
    done = {(r['level'], r['repetition'], r['grouping'], r['weight'], r['finisher']) for r in results}
    # Start with noise levels most useful for detecting a material difference.
    levels = sorted(args.levels, key=lambda x: ([0., .1, .5].index(x) if x in [0., .1, .5] else 3, x))
    for level in levels:
        repetitions = protocol['repetitions'][:1] if level == 0 else protocol['repetitions']
        for rep in repetitions:
            survey = observations.survey(room, level, rep)
            pairs, scopes = algorithm.pair_selector(survey.xyz), algorithm.group_scope_selector(survey.xyz)
            graphs = None
            for grouping in ('components', 'stars'):
                for weight in ('uniform', 'similarity', 'distance', 'both'):
                    if args.variants and grouping + ':' + weight not in args.variants:
                        continue
                    wanted = [f for f in args.finishers
                              if (level, rep, grouping, weight, f) not in done]
                    if not wanted:
                        continue
                    started = time.perf_counter()
                    cache = output / f'candidates-{level}-{rep}-{grouping}-{weight}.npz'
                    if cache.exists():
                        candidates = read_npz(cache)[0]
                    else:
                        if graphs is None:
                            graphs = match_pairs(survey, pairs, algorithm.matcher)
                        grouper = algorithm.grouper if grouping == 'components' else StarGrouper()
                        candidates = {}
                        for ap, rows in survey.channels.items():
                            groups = group_within_scopes(graphs[ap], scopes, grouper)
                            candidates[ap] = weighted_candidates(survey.xyz, rows, groups,
                                algorithm.estimator, graphs[ap], weight)
                        write_npz(cache, candidates)
                    for ending in wanted:
                        predictions = {}
                        for ap, rows in survey.channels.items():
                            if ending == 'cluster':
                                clusters = summarize_clusters(candidates[ap], algorithm.clusterer)
                                keep, _ = algorithm.support_filter.select(clusters['support'])
                                predictions[ap] = clusters['xyz'][keep]
                            else:
                                one = Survey(survey.xyz, {ap: rows}, survey.position_ids)
                                selector = finisher('zayets' if ending == 'consumption' else 'ransac', finishers[ending])
                                predictions[ap] = finish_one(one, SourceMap({ap: candidates[ap]['xyz']}), selector)['xyz']
                        outcome = f'outcomes/{level}-{rep}-{grouping}-{weight}-{ending}.npz'
                        write_npz(output / outcome, {'by_ap': predictions})
                        metrics = {ap: ap_source_metrics(predictions, truth, ap.split('+')) for ap in ('A', 'B', 'A+B')}
                        row = {'level': level, 'repetition': rep, 'grouping': grouping, 'weight': weight,
                               'finisher': ending, 'metrics_by_ap': metrics, 'outcome': outcome,
                               'candidate_counts': {ap: len(c['xyz']) for ap, c in candidates.items()},
                               'seconds': time.perf_counter()-started}
                        results.append(row)
                        write_json(output / 'results.json', results)
                        m = metrics['A+B']
                        print(f"sigma={level} repeat={rep} {grouping}/{weight}/{ending}: "
                              f"OSPA={m['ospa']:.5f}, F1={m['f1']:.5f}", flush=True)
    write_json(output / 'status.json', {'state': 'complete', 'cases': len(results)})


if __name__ == '__main__':
    main()
