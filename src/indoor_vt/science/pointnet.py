"""Frozen PointNet probabilities and maximum-cardinality log-odds assignment.

PyTorch is needed only to prepare predictions, not to read or replay saved graphs.
There is no source identity, AP coordinate, or true source count in these inputs.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

@dataclass(frozen=True)
class PointNetLogOddsMatcher:
    checkpoint_sha256: str
    training_updates: int = 4_000_000

    def assign(self, logits):
        """Maximize sum log(p/(1-p)), matching min(n_left,n_right) pairs."""
        logits = np.asarray(logits, dtype=float)
        if logits.ndim != 2 or not np.isfinite(logits).all():
            raise ValueError("Expected a finite rectangular logit matrix")
        left, right = linear_sum_assignment(-logits)
        return left, right, 1. - expit(logits[left, right])


def graph_from_logits(rows, pairs, logits, matcher):
    from .grouping import PairMatches
    offsets, endpoints, costs = [0], [], []
    for (a, b), matrix in zip(pairs, logits, strict=True):
        left, right, dissimilarity = matcher.assign(matrix[:len(rows[a]), :len(rows[b])])
        endpoints.append(np.column_stack((left, right)) + rows.offsets[[a, b]])
        costs.append(dissimilarity)
        offsets.append(offsets[-1] + len(left))
    return PairMatches(np.asarray(pairs, dtype=np.int64), np.asarray(offsets, dtype=np.int64),
        np.concatenate(endpoints).astype(np.int32) if endpoints else np.empty((0, 2), np.int32),
        np.concatenate(costs) if costs else np.empty(0), rows.offsets)


def load_network(checkpoint, device='cpu'):
    """Architecture identical to the trained 66,433-parameter symmetric network."""
    import torch
    from torch import nn

    class PeakMatcher(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('mean', torch.tensor(0.))
            self.register_buffer('scale', torch.tensor(1.))
            self.pointnet = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64),
                                         nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
            self.head = nn.Sequential(nn.Linear(387, 128), nn.SiLU(), nn.Linear(128, 64),
                                     nn.SiLU(), nn.Linear(64, 1))

        def encode(self, ranges, mask):
            normalized = (ranges-self.mean)/self.scale
            features = self.pointnet(normalized.unsqueeze(-1))
            count = mask.sum(1, keepdim=True)
            if not bool((count > 0).all()):
                raise ValueError('Empty fingerprints must be removed before pooling')
            mean = features.mean(1) if bool(mask.all()) else (features*mask.unsqueeze(-1)).sum(1)/count
            maximum = (features.amax(1) if bool(mask.all()) else
                       features.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1))
            return normalized, features, torch.cat((mean, maximum), dim=-1)

        def score(self, encoded, left, right, ia, ib, separation):
            ranges, features, embedding = encoded
            a = torch.cat((ranges[left, ia, None], features[left, ia], embedding[left]), -1)
            b = torch.cat((ranges[right, ib, None], features[right, ib], embedding[right]), -1)
            distance = separation.unsqueeze(-1)/self.scale
            return .5*(self.head(torch.cat((a, b, distance), -1)) +
                       self.head(torch.cat((b, a, distance), -1))).squeeze(-1)

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved['step'] != 4_000_000 or 'rectangle_scene_00' in saved['config']['protocol']['rooms']:
        raise ValueError('Expected exact 4M checkpoint trained outside the benchmark room')
    model = PeakMatcher().to(device).eval()
    model.load_state_dict(saved['state_dict'])
    assert sum(p.numel() for p in model.parameters()) == 66433
    return model


def predict_graph(model, xyz, rows, pairs, matcher, device='cpu'):
    """Stream matrices into assignments; never retain the full probability cube."""
    import torch
    ids = np.flatnonzero(np.diff(rows.offsets) > 0)
    remap = np.full(len(rows), -1, np.int64)
    remap[ids] = np.arange(len(ids))
    width = max((len(rows[i]) for i in ids), default=0)
    ranges = np.zeros((len(ids), width), np.float32)
    mask = np.zeros_like(ranges, dtype=bool)
    for j, i in enumerate(ids):
        ranges[j, :len(rows[i])] = rows[i]
        mask[j, :len(rows[i])] = True
    valid = np.flatnonzero((remap[pairs] >= 0).all(1))
    blocks = {}
    with torch.inference_mode():
        if len(valid):
            encoded = model.encode(torch.as_tensor(ranges, device=device),
                                   torch.as_tensor(mask, device=device))
            coordinates = torch.as_tensor(xyz[ids], dtype=torch.float32, device=device)
            ia = torch.arange(width, device=device).repeat_interleave(width)
            ib = torch.arange(width, device=device).repeat(width)
            chunk = max(1, min(16, int(16*(25/width)**2)))
            for start in range(0, len(valid), chunk):
                indices = valid[start:start+chunk]
                p = torch.as_tensor(remap[pairs[indices]], device=device)
                left, right = p[:, 0].repeat_interleave(width**2), p[:, 1].repeat_interleave(width**2)
                separation = (coordinates[p[:, 0]]-coordinates[p[:, 1]]).norm(dim=-1).repeat_interleave(width**2)
                scores = model.score(encoded, left, right, ia.repeat(len(p)), ib.repeat(len(p)), separation)
                matrices = scores.reshape(len(p), width, width).cpu().numpy()
                for index, matrix in zip(indices, matrices):
                    a, b = pairs[index]
                    blocks[int(index)] = matcher.assign(matrix[:len(rows[a]), :len(rows[b])])
    offsets, endpoints, costs = [0], [], []
    for index, (a, b) in enumerate(pairs):
        left, right, cost = blocks.get(index, (np.empty(0, int), np.empty(0, int), np.empty(0)))
        endpoints.append(np.column_stack((left, right)) + rows.offsets[[a, b]])
        costs.append(cost)
        offsets.append(offsets[-1]+len(left))
    return dict(pairs=pairs, offsets=np.asarray(offsets, np.int64),
        endpoints=np.concatenate(endpoints).astype(np.int32), costs=np.concatenate(costs),
        node_offsets=rows.offsets)
