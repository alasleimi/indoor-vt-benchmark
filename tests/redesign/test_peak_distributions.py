"""Check population, truth labels, normalization and published/live parity."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from indoor_vt.execution.peak_distributions import dissimilarity_distributions, first_dissimilarities
from indoor_vt.execution.peak_explorer import pair_data
from indoor_vt.science.matching import MPUrgeMatcher


def profile(ranges, labels, xyz):
    return {"ranges": np.asarray(ranges, float), "source_ids": np.asarray(labels),
            "xyz": np.asarray(xyz, float)}


@pytest.mark.parametrize("unify", [False, True])
@pytest.mark.parametrize("triangle", [None, .03])
def test_distribution_cells_equal_pair_inspector(unify, triangle):
    matcher = MPUrgeMatcher(alpha=.7, triangle_tolerance_m=triangle,
                            **({} if unify else {"size_unification_norm": None}))
    a = profile([4.1, .4, 1.3, 2.8], [3, 0, 1, 2], [0, 0, 0])
    b = profile([.6, 2.2, 3.3], [0, 1, 4], [0, 0, 1])
    x, truth = first_dissimilarities(a, b, matcher)
    values = pair_data(a, b, matcher, [])["matrix"]["values"]
    labels = (a["source_ids"][:, None] == b["source_ids"][None, :]).ravel()
    for correct in (False, True):
        np.testing.assert_allclose(np.sort(x[truth == correct]),
                                   np.sort(values[np.isfinite(values) & (labels == correct)]))


def test_populations_and_per_anchor_medians_without_truth_leakage():
    profiles = [profile([1.11, 2.47], [0, 1], [0, 0, 0]),
                profile([1.38, 3.23], [0, 2], [0, 0, 1]),
                profile([2.71], [1], [0, 0, 2])]
    matcher = MPUrgeMatcher(alpha=1., size_unification_norm=None)
    result = dissimilarity_distributions(profiles, [[0, 1], [1, 2]], matcher)
    assert result["all"]["pairs"] == 3
    assert result["all"]["totals"] == [2, 6]
    assert result["neighborhood"]["totals"] == [1, 5]
    assert result["normalized"]["totals"] == [2, 10]
    assert result["normalized"]["pairs"] == 4
    x01, _ = first_dissimilarities(profiles[0], profiles[1], matcher)
    x12, _ = first_dissimilarities(profiles[1], profiles[2], matcher)
    medians = [np.median(x01), np.median(np.r_[x01, x12]), np.median(x12)]
    np.testing.assert_allclose(result["normalization"]["anchor_medians"], medians)
    expected = np.r_[x01 / medians[0], x01 / medians[1], x12 / medians[1], x12 / medians[2]]
    histogram = np.bincount(np.floor(expected / .05).astype(int))
    for low, correct, incorrect in result["normalized"]["bins"]:
        assert correct + incorrect == histogram[round(low / .05)]
    relabelled = [{**p, "source_ids": p["source_ids"] + i * 10} for i, p in enumerate(profiles)]
    changed = dissimilarity_distributions(relabelled, [[0, 1], [1, 2]], matcher)
    assert changed["normalization"] == result["normalization"]
    assert changed["all"]["totals"][0] == 0


def test_zero_medians_and_empty_fingerprints_are_explicit():
    profiles = [profile([1, 1], [0, 1], [0, 0, 0]),
                profile([1, 1], [0, 1], [0, 0, 1]), profile([], [], [0, 0, 2])]
    matcher = MPUrgeMatcher(alpha=1., size_unification_norm=None)
    result = dissimilarity_distributions(profiles, [[0, 1], [0, 2]], matcher)
    assert result["all"]["totals"] == [2, 2]
    assert result["normalized"]["totals"] == [0, 0]
    assert result["normalization"] == {"anchor_medians": [0., 0., None],
                                       "skipped_zero_median_cells": 8, "directions_per_pair": 2}


def test_browser_and_python_histograms_agree():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for published/live parity")
    rng = np.random.default_rng(12345)
    profiles = [profile(rng.uniform(.2, 9, n), rng.permutation(n), rng.normal(size=3))
                for n in [0, 4, 5, 7, 3]]
    pairs = [[0, 1], [1, 3], [2, 3], [3, 4]]
    cases, expected = [], []
    for unify in (False, True):
        for triangle in (None, .1):
            for alpha in (.7, 1.):
                meta = {"alpha": alpha, "half_window": 2, "unify": unify,
                        "triangle_tolerance_m": triangle, "matcher": "MPUrgeMatcher"}
                cases.append({"data": {"profiles": profiles}, "init": {"meta": meta, "scene": {"pairs": pairs}}})
                matcher = MPUrgeMatcher(alpha=alpha, triangle_tolerance_m=triangle,
                                       **({} if unify else {"size_unification_norm": None}))
                expected.append(dissimilarity_distributions(profiles, pairs, matcher))
    module = (Path(__file__).parents[2] / "src/indoor_vt/execution/web/static-distributions.mjs").as_uri()
    script = (f"import {{ distributions }} from {json.dumps(module)};"
              "let input='';for await(const chunk of process.stdin)input+=chunk;"
              "console.log(JSON.stringify(JSON.parse(input).map(c=>distributions(c.data,c.init))));")
    process = subprocess.run([node, "--input-type=module", "-e", script],
                             input=json.dumps(cases, default=lambda x: x.tolist()),
                             capture_output=True, text=True, check=True)
    actual = json.loads(process.stdout)
    for wanted, got in zip(expected, actual):
        for population in ("all", "neighborhood", "normalized"):
            assert got[population] == wanted[population]
        np.testing.assert_allclose([v or 0 for v in got["normalization"]["anchor_medians"]],
                                   [v or 0 for v in wanted["normalization"]["anchor_medians"]])
        assert got["normalization"]["skipped_zero_median_cells"] == wanted["normalization"]["skipped_zero_median_cells"]
