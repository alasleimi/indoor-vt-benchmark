"""GPU-only inference worker; compatible with the existing PyTorch environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from indoor_vt.science.models import Ragged
from indoor_vt.science.pointnet import load_network, predict_graph, PointNetLogOddsMatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    checkpoint = Path(manifest['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest['checkpoint_sha256']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.2)
    model = load_network(checkpoint, device)
    matcher = PointNetLogOddsMatcher(manifest['checkpoint_sha256'])
    for index, item in enumerate(manifest['inputs']):
        output = Path(item['output'])
        if output.exists():
            continue
        started = time.perf_counter()
        with np.load(item['path']) as saved:
            rows = Ragged(saved['values'], saved['offsets'])
            pairs = saved['pairs']
            old = dict(np.load(item['archived_input']))
            n, code = len(old['position_ids']), item['ap_code']
            np.testing.assert_array_equal(saved['xyz'], old['xyz'][code*n:(code+1)*n])
            for i in range(n):
                np.testing.assert_array_equal(rows[i], old['ranges'][code*n+i, old['mask'][code*n+i]])
            archive_path = Path(item['archived_logits'])
            assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == item['archived_logits_sha256']
            logits = np.load(archive_path, mmap_mode='r')
            count = len(old['pairs'])//2
            existing = {tuple(pair-code*n): index+code*count for index, pair in
                        enumerate(old['pairs'][code*count:(code+1)*count])}
            missing = np.array([pair for pair in pairs if tuple(pair) not in existing], dtype=np.int64).reshape(-1, 2)
            fresh = predict_graph(model, saved['xyz'], rows, missing, matcher, device) if len(missing) else None
            fresh_lookup = {tuple(pair): index for index, pair in enumerate(missing)}
            offsets, endpoints, costs = [0], [], []
            for a,b in pairs:
                if (a,b) in existing:
                    matrix = logits[existing[a,b], :len(rows[a]), :len(rows[b])]
                    left, right, cost = matcher.assign(matrix)
                    edges = np.column_stack((left,right))+rows.offsets[[a,b]]
                else:
                    fresh_index = fresh_lookup[a,b]
                    block = slice(fresh['offsets'][fresh_index], fresh['offsets'][fresh_index+1])
                    edges, cost = fresh['endpoints'][block], fresh['costs'][block]
                endpoints.append(edges)
                costs.append(cost)
                offsets.append(offsets[-1]+len(cost))
            graph = dict(pairs=pairs, offsets=np.array(offsets,np.int64),
                         endpoints=np.concatenate(endpoints).astype(np.int32), costs=np.concatenate(costs),
                         node_offsets=rows.offsets)
        temporary = output.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **graph)
        os.replace(temporary, output)
        print(json.dumps({'completed': index+1, 'total': len(manifest['inputs']),
                          'case': item['case'], 'seconds': time.perf_counter()-started}), flush=True)


if __name__ == '__main__':
    main()
