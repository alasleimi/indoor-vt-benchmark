"""Audit the saved exhaustive analysis without recalibration or reconstruction."""
import json
from pathlib import Path

import numpy as np

from indoor_vt.acquisition.records import write_json
from indoor_vt.execution.true_matching import (TrueMatchingAnalysis, TRUTH_FIELDS, pool_rows,
                                              truth_counts, TOLERANCE_M, violation_classes)
from indoor_vt.execution.viewer import MultiStudyView


def brute_truth(profiles, xyz, pairs):
    result = dict.fromkeys(TRUTH_FIELDS, 0)
    for a, b in pairs:
        pa, pb = profiles[a], profiles[b]
        true_edges = [(i, j) for i, label in enumerate(pa["source_ids"])
                      for j, other in enumerate(pb["source_ids"]) if label == other]
        distance = np.linalg.norm(xyz[a] - xyz[b])
        for i, j in true_edges:
            result["available"] += 1
            result["same_rank"] += i == j
            result["cross_any"] += any((i-k)*(j-other_rank) < 0 for k, other_rank in true_edges)
            x, y = pa["ranges"][i], pb["ranges"][j]
            result["difference_satisfies"] += abs(x-y) <= distance + TOLERANCE_M
            result["sum_satisfies"] += x+y+TOLERANCE_M >= distance
    return {**{k: int(v) for k, v in result.items()}, "pairs": len(pairs)}


analysis = TrueMatchingAnalysis(MultiStudyView(Path("data/redesign/rectangle-20261025"),
                                               Path("runs/redesign/rectangle-clean-selection-v4")))
report = analysis.request()
assert len(report["rows"]) == 40
assert report["coverage"]["global_pairs_per_ap_realization"] == 39060
assert report["coverage"]["neighborhood_pairs_per_ap_realization"] == 2112
for row in report["rows"]:
    for scope in ("global", "neighborhood"):
        c = row[scope]
        assert all(0 <= c[name] <= c["available"] for name in TRUTH_FIELDS)
    c = row["matches"]
    assert c["correct"] <= c["total"]
    assert c["correct"] <= row["neighborhood"]["available"]
    assert c["same_rank_correct"] <= c["correct"]
    assert c["same_rank_correct"] <= row["neighborhood"]["same_rank"]
    assert c["same_rank_correct"] + c["same_rank_incorrect"] <= c["total"]
    for bound in ("difference", "sum"):
        v = c[bound + "_violation_edges"]
        assert v == c[bound + "_violation_correct"] + c[bound + "_violation_incorrect"]
        assert 2*v == sum(c[bound + "_counterpart_" + end] for end in ("violates", "satisfies", "unavailable"))
        assert c[bound + "_counterpart_violates"] >= 2*c[bound + "_violation_correct"]
        assert c[bound + "_violation_correct"] <= row["neighborhood"]["available"] - row["neighborhood"][bound + "_satisfies"]
        if row["noise"] == 0:
            assert c[bound + "_violation_correct"] == c[bound + "_counterpart_violates"] == 0
    assert row["observations"]["adjacent_equal_range_pairs"] == 0
for ap in ("A", "B", "A+B"):
    assert [{k:v for k,v in r.items() if k != "identity_accuracy"} for r in report["pooled"][ap]] == pool_rows(report["rows"], ap)
    for row in report["pooled"][ap]:
        for bound in ("difference", "sum"):
            c = row["identity_accuracy"][bound]
            v = row["matches"][bound + "_violation_edges"]
            assert c["violating"]["total"] == v
            if not v:
                assert c["violating"]["fraction_correct"] is None
    if ap != "A+B":
        clean = [r for r in report["rows"] if r["ap"] == ap and r["noise"] == 0]
        assert {k:v for k,v in clean[0].items() if k != "repetition"} == {k:v for k,v in clean[1].items() if k != "repetition"}

# Independent nested-loop reference on deterministic global-pair samples.
rng = np.random.default_rng(20261027)
cases = []
for ap, noise, repetition in (("A",0.,2),("B",0.,3),("A",.1,2),("B",1.,3),("A",5.,2),("B",5.,3)):
    data = analysis.peaks.profiles(analysis.context, ap, noise, repetition)
    profiles, xyz = data["profiles"], analysis.context[5]
    pairs = analysis.global_pairs[rng.choice(len(analysis.global_pairs), 160, replace=False)]
    expected = brute_truth(profiles, xyz, pairs)
    assert truth_counts(profiles, xyz, pairs, len(data["sources"])) == expected
    matcher = analysis.context[-1]
    # Reverse the orientation: endpoint classifications must be identical.
    for a, b in pairs[:20]:
        result = matcher.match(profiles[a]["ranges"], profiles[b]["ranges"])
        edges = result["matches"][:, :2].astype(int)
        assert np.array_equal(edges, result["trace"][result["trace"][:,4] == 1, :2].astype(int))
        ia, ib = edges.T
        for bound in ("difference", "sum"):
            d = np.linalg.norm(xyz[a]-xyz[b])
            assert violation_classes(profiles[a],profiles[b],ia,ib,d,bound) == violation_classes(profiles[b],profiles[a],ib,ia,d,bound)
    cases.append({"ap":ap,"noise":noise,"repetition":repetition,"pairs":len(pairs)})

zero = report["pooled"]["A+B"][0]["neighborhood"]
assert abs(100*zero["same_rank"]/zero["available"] - 49.19412878787879) < 1e-10
assert 100*zero["cross_any"]/zero["available"] == 53.125
verification = {"state":"passed", "cache_key":report["cache_key"], "conditions":40,
    "count_invariants":"all conditions and all pooled AP selections",
    "reference_truth_cases":cases, "independent_truth_pairs":960,
    "production_trace_checks":120, "symmetric_endpoint_checks":240,
    "baseline_neighborhood_same_rank_percent":100*zero["same_rank"]/zero["available"],
    "baseline_neighborhood_cross_any_percent":53.125,
    "noiseless_duplicate_realizations":"identical", "exact_range_ties":0,
    "conditional_accuracy_zero_denominator":"null, never 0 or 100"}
write_json(analysis.root / "verification.json", verification)
print(json.dumps(verification, indent=2))
