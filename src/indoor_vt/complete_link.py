"""Complete linkage with SciPy 1.15.3 ordering and bounded distance storage.

The nearest-neighbor chain and label correction are adapted from SciPy's
cluster/_hierarchy.pyx (BSD-3-Clause; see third_party/scipy/LICENSE.txt).
An ascending linked list skips removed slots without changing comparisons.
Large banks cache distance rows and recompute evicted rows from cluster leaves.
"""

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


@njit(cache=True)
def _row(points, x, protected, rows, owners, slots, ages, tick, first, leaves, following, head):
    slot = slots[x]
    if slot != -1:
        ages[slot] = tick
        return slot
    slot = -1
    oldest = np.iinfo(np.int64).max
    for candidate in range(len(rows)):
        if candidate == protected:
            continue
        if owners[candidate] == -1:
            slot = candidate
            break
        if ages[candidate] < oldest:
            slot, oldest = candidate, ages[candidate]
    if owners[slot] != -1:
        slots[owners[slot]] = -1
    # Publish the new owner only after the row is complete. Other cached rows
    # already contain exactly the required symmetric distance to x.
    i = head
    while i != -1:
        if i == x:
            rows[slot, i] = 0.0
        elif slots[i] != -1:
            rows[slot, i] = rows[slots[i], x]
        else:
            maximum = 0.0
            a = first[x]
            while a != -1:
                b = first[i]
                while b != -1:
                    squared = 0.0
                    for dimension in range(points.shape[1]):
                        delta = points[a, dimension] - points[b, dimension]
                        squared += delta * delta
                    maximum = max(maximum, squared)
                    b = leaves[b]
                a = leaves[a]
            distance = np.sqrt(maximum)
            if not np.isfinite(distance):
                raise ValueError("The condensed distance matrix must contain only finite values")
            rows[slot, i] = distance
        i = following[i]
    owners[slot], slots[x], ages[slot] = x, slot, tick
    return slot


@njit(cache=True)
def _bounded_chain(points, capacity):
    n = len(points)
    rows = np.empty((capacity, n), dtype=np.float64)
    owners = np.full(capacity, -1, dtype=np.int64)
    slots = np.full(n, -1, dtype=np.int64)
    ages = np.zeros(capacity, dtype=np.int64)
    first = np.arange(n, dtype=np.int64)
    last = first.copy()
    leaves = np.full(n, -1, dtype=np.int64)
    following = np.arange(1, n + 1, dtype=np.int64)
    preceding = np.arange(-1, n - 1, dtype=np.int64)
    following[-1] = -1
    sizes = np.ones(n, dtype=np.int64)
    chain = np.empty(n, dtype=np.int64)
    result = np.empty((n - 1, 4))
    length, head, y, tick = 0, 0, 0, 0
    for step in range(n - 1):
        if length == 0:
            chain[0], length = head, 1
        while True:
            x = chain[length - 1]
            tick += 1
            sx = _row(points, x, -1, rows, owners, slots, ages, tick, first, leaves, following, head)
            if length > 1:
                y = chain[length - 2]
                minimum = rows[sx, y]
            else:
                minimum = np.inf
            i = head
            while i != -1:
                if i != x and rows[sx, i] < minimum:
                    minimum, y = rows[sx, i], i
                i = following[i]
            if length > 1 and y == chain[length - 2]:
                break
            chain[length] = y
            length += 1
        length -= 2
        if x > y:
            x, y = y, x
        tick += 1
        sx = _row(points, x, -1, rows, owners, slots, ages, tick, first, leaves, following, head)
        tick += 1
        sy = _row(points, y, sx, rows, owners, slots, ages, tick, first, leaves, following, head)
        sizes[y] += sizes[x]
        result[step, 0], result[step, 1] = x, y
        result[step, 2], result[step, 3] = minimum, sizes[y]
        # Complete linkage is the maximum original leaf-pair distance. Updating
        # resident rows and recomputing missing ones therefore agree exactly.
        for slot in range(capacity):
            if owners[slot] != -1 and slot != sx and slot != sy:
                rows[slot, y] = max(rows[slot, x], rows[slot, y])
        i = head
        while i != -1:
            if i != x and i != y:
                rows[sy, i] = max(rows[sx, i], rows[sy, i])
            i = following[i]
        rows[sy, y] = 0.0
        owners[sx], slots[x] = -1, -1
        leaves[last[y]], last[y] = first[x], last[x]
        before, after = preceding[x], following[x]
        if before == -1:
            head = after
        else:
            following[before] = after
        if after != -1:
            preceding[after] = before
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

    The budget selects storage only; it cannot change the hierarchy. Large
    banks use at most the budget (or two rows, whichever is larger) for cached
    distances, plus linear bookkeeping. No distance files are created.
    scratch_dir remains accepted for compatibility with existing callers.
    """
    points = np.asarray(points, dtype=np.float64, order="C")
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Finite two-dimensional observation array required")
    if memory_limit_bytes < 0:
        raise ValueError("Distance cache budget must be nonnegative")
    n = len(points)
    if n < 2:
        return np.empty((0, 4))
    count = n * (n - 1) // 2
    if count * 8 <= memory_limit_bytes:
        return _calculate(points, np.empty(count, dtype=np.float64))
    capacity = max(2, min(n, memory_limit_bytes // (8 * n)))
    hierarchy = _bounded_chain(points, capacity)
    hierarchy = hierarchy[np.argsort(hierarchy[:, 2], kind="mergesort")]
    _label(hierarchy, n)
    return hierarchy
