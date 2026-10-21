"""Exact batched MCA/Chamfer geometry, reusable across MCA tolerance values."""

import numpy as np


def nearest_geometry(queries, references, maximum_work_bytes=16 * 1024 * 1024):
    """Pack query-to-reference minima and the opposite-direction mean distance.

    Grouping references by length avoids padding, so each NumPy reduction has
    the same contiguous row and original length as the scalar implementation.
    The work limit changes chunk size only, never the observations evaluated.
    """
    lengths = np.array([len(row) for row in queries], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    reverse_mean = np.full((len(queries), len(references)), np.inf)
    reference_lengths = np.array([len(row) for row in references], dtype=np.int64)
    maximum_length = int(reference_lengths.max(initial=0))
    # Store exact argmin indices, not millions of high-entropy float distances.
    # Original float64 profiles reconstruct every distance with the same subtraction.
    nearest_indices = np.zeros(
        (len(references), int(offsets[-1])), dtype=np.min_scalar_type(max(0, maximum_length - 1))
    )
    reference_values = np.full((len(references), max(1, maximum_length)), np.inf)
    for index, row in enumerate(references):
        reference_values[index, : len(row)] = row
    for width in sorted(set(reference_lengths) - {0}):
        ids = np.flatnonzero(reference_lengths == width)
        rows = np.stack([references[i] for i in ids])
        for qi, query in enumerate(queries):
            if not len(query):
                continue
            batch = max(1, maximum_work_bytes // (8 * len(query) * int(width)))
            for start in range(0, len(ids), batch):
                selected = ids[start : start + batch]
                delta = np.abs(query[None, :, None] - rows[start : start + batch, None, :])
                nearest_indices[selected, offsets[qi] : offsets[qi + 1]] = delta.argmin(axis=2)
                reverse_mean[qi, selected] = delta.min(axis=1).mean(axis=1)
    return {
        "nearest_indices": nearest_indices,
        "reference_values": reference_values,
        "query_values": np.concatenate(queries) if len(queries) else np.empty(0),
        "reverse_mean": reverse_mean,
        "offsets": offsets,
        "reference_lengths": reference_lengths,
    }


def geometry_scores(geometry, method, epsilon=1.0):
    indices, offsets = geometry["nearest_indices"], geometry["offsets"]
    scores = np.zeros((len(offsets) - 1, len(indices)))
    if method == "chamfer":
        scores[:] = np.inf
    elif method != "mca":
        raise ValueError(method)
    for qi in range(len(offsets) - 1):
        start, stop = offsets[qi : qi + 2]
        if start == stop:
            continue
        reference = np.take_along_axis(geometry["reference_values"], indices[:, start:stop], axis=1)
        values = np.abs(geometry["query_values"][None, start:stop] - reference)
        if method == "mca":
            scores[qi] = np.square(np.maximum(epsilon - values, 0)).sum(axis=1)
        else:
            scores[qi] = (geometry["reverse_mean"][qi] + values.mean(axis=1)) / 2
    return scores
