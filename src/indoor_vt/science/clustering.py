"""Candidate clustering without dense all-pairs distance matrices."""

from dataclasses import dataclass
from itertools import product
from collections import deque
import numpy as np
from scipy.spatial import cKDTree

from .costs import median


@dataclass(frozen=True)
class GreedyCompleteLink:
    radius_m: float = 0.5
    representative: object = median

    def memberships(self, xyz):
        if self.radius_m <= 0:
            raise ValueError("Clustering radius must be positive")
        groups, cells = [], {}
        for index in sorted(range(len(xyz)), key=lambda i: (*xyz[i], i)):
            point = xyz[index]
            cell = tuple(np.floor(point / self.radius_m).astype(int))
            nearby = sorted(gid for offset in product((-1, 0, 1), repeat=3)
                            for gid in cells.get(tuple(a+b for a,b in zip(cell, offset)), ()))
            for gid in nearby:
                if np.all(np.linalg.norm(xyz[groups[gid]] - point, axis=1) <= self.radius_m):
                    groups[gid].append(index)
                    break
            else:
                cells.setdefault(cell, []).append(len(groups))
                groups.append([index])
        return groups

    def cluster(self, xyz):
        groups = self.memberships(xyz)
        representatives = np.asarray([self.representative(xyz[g]) for g in groups]).reshape(-1, 3)
        return representatives, groups


@dataclass(frozen=True)
class DBSCAN:
    radius_m: float = 0.5
    min_samples: int = 5
    representative: object = median

    def memberships(self, xyz):
        if not len(xyz):
            return []
        if self.radius_m <= 0 or self.min_samples < 1:
            raise ValueError("Invalid DBSCAN radius or minimum count")
        tree = cKDTree(xyz)
        core = tree.query_ball_point(xyz, self.radius_m, return_length=True) >= self.min_samples
        labels = np.full(len(xyz), -1, dtype=int)
        visited = np.zeros(len(xyz), bool)
        groups = []
        for start in range(len(xyz)):
            if visited[start]:
                continue
            visited[start] = True
            if not core[start]:
                continue
            label = len(groups)
            labels[start] = label
            queue = deque([start])
            while queue:
                point = queue.popleft()
                for neighbor in tree.query_ball_point(xyz[point], self.radius_m):
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        if core[neighbor]:
                            queue.append(neighbor)
                    if labels[neighbor] < 0:
                        labels[neighbor] = label
            groups.append(np.flatnonzero(labels == label).tolist())
        return groups

    def cluster(self, xyz):
        groups = self.memberships(xyz)
        return np.asarray([self.representative(xyz[g]) for g in groups]).reshape(-1, 3), groups
