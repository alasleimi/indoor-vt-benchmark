"""Complete linkage with SciPy 1.15.3 ordering and one writable distance array.

The nearest-neighbor chain and label correction are adapted from SciPy's
cluster/_hierarchy.pyx (BSD-3-Clause; see third_party/scipy/LICENSE.txt).
An ascending linked list skips removed slots without changing comparisons.
Large arrays use a temporary file mapping, rather than private committed RAM.
"""

from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
from numba import njit
from scipy.spatial.distance import pdist


@njit(cache=True, inline="always")
def _index(offsets, left, right):
    if left > right:
        left, right = right, left
    return offsets[left] + right - left - 1


@njit(cache=True)
def _chain(distances, n):
    result = np.empty((n - 1, 4))
    sizes = np.ones(n, dtype=np.int64)
    offsets = np.empty(n, dtype=np.int64)
    following = np.arange(1, n + 1, dtype=np.int64)
    preceding = np.arange(-1, n - 1, dtype=np.int64)
    following[-1] = -1
    for i in range(n):
        offsets[i] = n * i - i * (i + 1) // 2
    chain = np.empty(n, dtype=np.int64)
    length, head, y = 0, 0, 0
    for step in range(n - 1):
        if length == 0:
            chain[0] = head
            length = 1
        while True:
            x = chain[length - 1]
            # As in SciPy, prefer the predecessor on an exact distance tie.
            if length > 1:
                y = chain[length - 2]
                minimum = distances[_index(offsets, x, y)]
            else:
                minimum = np.inf
            i = head
            while i != -1:
                if i != x:
                    distance = distances[_index(offsets, x, i)]
                    if distance < minimum:
                        minimum, y = distance, i
                i = following[i]
            if length > 1 and y == chain[length - 2]:
                break
            chain[length] = y
            length += 1
        length -= 2
        if x > y:
            x, y = y, x
        sizes[y] += sizes[x]
        result[step, 0] = x
        result[step, 1] = y
        result[step, 2] = minimum
        result[step, 3] = sizes[y]
        # Remove x while retaining the ascending scan order of active slots.
        before, after = preceding[x], following[x]
        if before == -1:
            head = after
        else:
            following[before] = after
        if after != -1:
            preceding[after] = before
        i = head
        while i != -1:
            if i != y:
                ix, iy = _index(offsets, i, x), _index(offsets, i, y)
                distances[iy] = max(distances[ix], distances[iy])
            i = following[i]
    return result


@njit(cache=True, inline="always")
def _find(parent, node):
    previous = node
    while parent[node] != node:
        node = parent[node]
    while parent[previous] != node:
        after = parent[previous]
        parent[previous] = node
        previous = after
    return node


@njit(cache=True)
def _label(hierarchy, n):
    parent = np.arange(2 * n - 1, dtype=np.int64)
    sizes = np.ones(2 * n - 1, dtype=np.int64)
    for i in range(n - 1):
        left = _find(parent, int(hierarchy[i, 0]))
        right = _find(parent, int(hierarchy[i, 1]))
        hierarchy[i, 0], hierarchy[i, 1] = min(left, right), max(left, right)
        parent[left] = parent[right] = n + i
        sizes[n + i] = sizes[left] + sizes[right]
        hierarchy[i, 3] = sizes[n + i]


def _calculate(points, distances):
    pdist(points, metric="euclidean", out=distances)
    # Avoid the full-size boolean temporary used by a single isfinite call.
    for start in range(0, len(distances), 1024 * 1024):
        if not np.isfinite(distances[start : start + 1024 * 1024]).all():
            raise ValueError("The condensed distance matrix must contain only finite values")
    hierarchy = _chain(distances, len(points))
    hierarchy = hierarchy[np.argsort(hierarchy[:, 2], kind="mergesort")]
    _label(hierarchy, len(points))
    return hierarchy


def complete_linkage(points, *, memory_limit_bytes=256 * 1024 * 1024, scratch_dir=None):
    """Return the full float64 hierarchy, retaining SciPy's merge and tie order.

    The memory limit selects storage only; it cannot change the hierarchy.
    Quadratic distance *storage* remains, but large workspaces are file-backed.
    TemporaryFile also requests deletion on process exit on Windows.
    """
    points = np.asarray(points, dtype=np.float64, order="C")
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Finite two-dimensional observation array required")
    n = len(points)
    if n < 2:
        return np.empty((0, 4))
    count = n * (n - 1) // 2
    if count * 8 <= memory_limit_bytes:
        return _calculate(points, np.empty(count, dtype=np.float64))
    if scratch_dir is not None:
        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryFile(prefix="vt-complete-", suffix=".bin", dir=scratch_dir) as handle:
        handle.truncate(count * 8)
        distances = np.memmap(handle, mode="r+", dtype=np.float64, shape=(count,))
        try:
            return _calculate(points, distances)
        finally:
            distances._mmap.close()
