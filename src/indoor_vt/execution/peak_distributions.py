"""Truth-labelled first-round dissimilarity histograms; no reconstruction changes."""
import numpy as np

from ..science.matching import _delta


SCHEMA = "mpurge.dissimilarity-distributions.1"


def first_dissimilarities(a, b, matcher):
    """Exactly the finite cells shown by the pair inspector, before peak removal."""
    separation = (float(np.linalg.norm(np.asarray(a["xyz"]) - b["xyz"]))
                  if matcher.triangle_tolerance_m is not None else -1.)
    if matcher.size_unification_norm is None:
        ia = np.argsort(a["ranges"], kind="stable")
        ib = np.argsort(b["ranges"], kind="stable")
    else:
        result = matcher.match(a["ranges"], b["ranges"], separation=separation)
        ia, ib = result["left_indices"], result["right_indices"]
    if not len(ia) or not len(ib):
        return np.array([], float), np.array([], bool)
    values = _delta(a["ranges"][ia], b["ranges"][ib], matcher.half_window,
                    matcher.alpha, matcher.normalized_pattern, separation,
                    matcher.triangle_tolerance_m or 0.).ravel()
    truth = (a["source_ids"][ia, None] == b["source_ids"][None, ib]).ravel()
    finite = np.isfinite(values)
    return values[finite], truth[finite]


class Histogram:
    def __init__(self, width):
        self.width = width
        self.counts = np.zeros((2, 256), dtype=np.int64)
        self.pairs = self.candidate_cells = 0

    def add(self, values, truth, candidate_cells):
        self.pairs += 1
        self.candidate_cells += candidate_cells
        ids = np.floor(values / self.width).astype(np.int64)
        length = int(ids.max()) + 1 if len(ids) else 0
        if length > self.counts.shape[1]:
            self.counts = np.pad(self.counts, ((0, 0), (0, length - self.counts.shape[1])))
        for row, mask in enumerate((truth, ~truth)):
            counts = np.bincount(ids[mask])
            self.counts[row, :len(counts)] += counts

    def result(self):
        totals = self.counts.sum(axis=1)
        return {"bin_width": self.width, "pairs": self.pairs,
                "candidate_cells": self.candidate_cells,
                "totals": totals.tolist(),
                "excluded_cells": self.candidate_cells - int(totals.sum()),
                "bins": [[int(i) * self.width, int(self.counts[0, i]), int(self.counts[1, i])]
                         for i in np.flatnonzero(self.counts.sum(axis=0))]}


def dissimilarity_distributions(profiles, local_pairs, matcher):
    """Count all unordered receiver pairs once; normalized local pairs twice.

    For each anchor fingerprint i, m_i is the median of ALL finite candidate
    dissimilarities to its eligible neighbors. Each local cell contributes
    delta/m_i and delta/m_j, one for each anchor. No labels enter m_i.
    """
    local = {tuple(sorted(map(int, pair))) for pair in local_pairs}
    hist_all, hist_local, hist_normalized = Histogram(.25), Histogram(.25), Histogram(.05)
    anchor_values = [[] for _ in profiles]
    stored = []
    for i, a in enumerate(profiles):
        for j in range(i + 1, len(profiles)):
            b = profiles[j]
            x, y = first_dissimilarities(a, b, matcher)
            cells = len(a["ranges"]) * len(b["ranges"])
            hist_all.add(x, y, cells)
            if (i, j) in local:
                hist_local.add(x, y, cells)
                anchor_values[i].append(x)
                anchor_values[j].append(x)
                stored.append((i, j, x, y))
    medians = [float(np.median(np.concatenate(values)))
               if any(len(x) for x in values) else None for values in anchor_values]
    skipped = 0
    for i, j, x, y in stored:
        for anchor in (i, j):
            scale = medians[anchor]
            if scale is not None and scale > 0:
                hist_normalized.add(x / scale, y, len(x))
            else:
                skipped += len(x)
    return {"schema": SCHEMA, "columns": ["low", "correct", "incorrect"],
            "all": hist_all.result(), "neighborhood": hist_local.result(),
            "normalized": hist_normalized.result(),
            "normalization": {"anchor_medians": medians,
                              "skipped_zero_median_cells": skipped,
                              "directions_per_pair": 2}}
