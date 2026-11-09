"""Verify archived graph parity, source parity, selected trials and saved metrics."""
from pathlib import Path
import argparse
import numpy as np

from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.execution.artifacts import read_npz, ArtifactCache
from indoor_vt.execution.pointnet import GRAPH_ROOT, CHECKPOINT_SHA256, PointNetExecution, pointnet_reconstructor
from indoor_vt.science.pointnet import PointNetLogOddsMatcher
from indoor_vt.science.grouping import PairMatches
from indoor_vt.science.models import Ragged, Survey


def graphs_and_sources():
    prepared = read_json(GRAPH_ROOT/'manifest.json')['inputs']
    matcher = PointNetLogOddsMatcher(CHECKPOINT_SHA256)
    rows = []
    for item in prepared:
        data = dict(np.load(item['archived_input']))
        logits = np.load(item['archived_logits'], mmap_mode='r')
        graph = PairMatches(**dict(np.load(item['output'])))
        code, n, count = item['ap_code'], len(data['position_ids']), len(data['pairs'])//2
        lookup = graph.pair_lookup()
        edge_count = 0
        for pos, (i,j) in enumerate(data['pairs'][code*count:(code+1)*count]-code*n):
            left, right, costs = matcher.assign(logits[code*count+pos,
                :int(data['mask'][i+code*n].sum()), :int(data['mask'][j+code*n].sum())])
            block = lookup[i,j]
            selection = slice(graph.offsets[block], graph.offsets[block+1])
            np.testing.assert_array_equal(graph.endpoints[selection]-graph.node_offsets[[i,j]],
                                          np.column_stack((left,right)))
            np.testing.assert_array_equal(graph.costs[selection], costs)
            edge_count += len(left)
        rows.append({'case': item['case'], 'exact_archived_edges': edge_count})
    old_root = Path('runs/pointnet-vt-matching/noise_obstacles_R1_R2_L_from1m_v1/reconstruction_4m_vs_3m')
    protocol = read_json(old_root.parent/'reconstruction_3m_vs_2m/protocol.json')
    cases = read_json(old_root/'conditions.json')
    store = ArtifactCache('data/redesign/rectangle-20261025/numerical-cache-v2',
                          disk_bytes=2048*1024**2, ram_bytes=256*1024**2, free_bytes=5120*1024**2)
    parity = []
    for name in ['empty_sigma_0.1_repeat_2', 'obstacles_layout_02']:
        case = next(c for c in cases if c['case'] == name)
        old = read_npz(old_root/'outcomes'/(name+'.npz'))[0]
        data = dict(np.load(case['input_file']))
        n = len(data['position_ids'])
        survey = Survey(data['xyz'][:n], {ap:Ragged.pack([
            data['ranges'][code*n+i,data['mask'][code*n+i]] for i in range(n)])
            for code,ap in enumerate(['A','B'])}, data['position_ids'])
        parameters = {**protocol['proposal_parameters'], **protocol['robust_finisher']}
        for grouping in ['components','stars']:
            actual = PointNetExecution(pointnet_reconstructor(parameters,grouping,'robust'),store).fit(survey)
            for ap in ['A','B']:
                expected = old['outputs']['pointnet_4m_'+grouping][ap]
                np.testing.assert_allclose(actual.by_ap[ap], expected, atol=1e-8, rtol=1e-9)
                parity.append({'case':name,'grouping':grouping,'ap':ap,'source_count':len(expected)})
    report = {'state':'passed','graph_cases':rows,'source_parity':parity}
    write_json(GRAPH_ROOT/'verification.json',report)
    print(f'Exact archived matches verified for {len(rows)} AP/cases; {len(parity)} source-map parity checks passed.',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.parse_args()
    graphs_and_sources()
