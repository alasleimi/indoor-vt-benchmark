import test from "node:test";
import assert from "node:assert/strict";
import {strictMetricWinners} from "../../src/indoor_vt/execution/web/pareto.mjs";

function condition(method_id, metrics, extra = {}) {
  return {method_id, split: "evaluation", level: 0,
    source_counts: [{room: "rectangle", repetition: 2}],
    source_metrics_by_ap: {"A+B": metrics}, ...extra};
}
function fixture(conditions, extra = {}) {
  return {methods: [...new Set(conditions.map((c) => c.method_id))].map((id) => ({id, group: "vt"})),
    conditions, ...extra};
}
const winners = (data, options) => strictMetricWinners(data, options).wins;

test("strict minimum uses full precision, not table rounding", () => {
  const result = winners(fixture([condition("a", {ospa: .1234000000001}), condition("b", {ospa: .1234})]));
  assert.deepEqual([...result.keys()], ["b"]);
  assert.equal(result.get("b")[0].runnerUpValue, .1234000000001);
  assert.equal(result.get("b")[0].comparableMethods, 2);
});

test("tied best values do not qualify; being nondominated alone is insufficient", () => {
  const result = winners(fixture([
    condition("a", {ospa: .1, f1: .5}), condition("b", {ospa: .1, f1: .5}),
    condition("c", {ospa: .2, f1: .5}),
  ]));
  assert.equal(result.size, 0);
});

test("all three F1 thresholds maximize independently", () => {
  const result = winners(fixture([
    condition("a", {f1_0_1m: .6, f1_0_3m: .6, f1: .6}),
    condition("b", {f1_0_1m: .5, f1_0_3m: .7, f1: .7}),
    condition("c", {f1_0_1m: .4, f1_0_3m: .5, f1: .8}),
  ]));
  assert.equal(result.get("a")[0].metric, "f1_0_1m");
  assert.equal(result.get("b")[0].metric, "f1_0_3m");
  assert.equal(result.get("c")[0].metric, "f1");
});

test("missing, nonfinite, and singleton metric values cannot create a winner", () => {
  const result = winners(fixture([
    condition("a", {ospa: .1, f1: null}),
    condition("b", {ospa: NaN, f1: Infinity}),
    condition("c", {ospa: null}),
  ]));
  assert.equal(result.size, 0);
});

test("RMSE without matches is excluded even if a stale value is finite", () => {
  for (const missingMatches of [{tp: 0}, {f1: 0}, {matches: []}]) {
    const result = winners(fixture([
      condition("bad", {matched_rmse_m: 0, ...missingMatches}),
      condition("a", {matched_rmse_m: .2, f1: .5}),
      condition("b", {matched_rmse_m: .3, f1: .5}),
    ]));
    assert.deepEqual([...result.keys()], ["a"]);
    assert.equal(result.get("a")[0].comparableMethods, 2);
  }
});

test("eligibility is the union across noise and obstacle conditions", () => {
  const obstacle = {level: 0, obstacle_count: 3, input_protocol: "matched_survey", input_count: 270, layout: "layout_03"};
  const result = winners(fixture([
    condition("a", {ospa: .1}), condition("b", {ospa: .2}),
    condition("a", {ospa: .8}, {level: .5}), condition("b", {ospa: .6}, {level: .5}),
    condition("c", {ospa: .05}, obstacle), condition("b", {ospa: .2}, obstacle),
    condition("unmeasured-elsewhere", {ospa: 0}, {level: 99}),
  ]));
  assert.deepEqual([...result.keys()].sort(), ["a", "b", "c"]);
  assert.equal(result.get("b")[0].condition.level, .5);
  assert.equal(result.get("c")[0].condition.obstacle_count, 3);
});

test("AP changes select already AP-separated metrics, with no pooling in the filter", () => {
  const data = fixture([
    condition("a", {}, {source_metrics_by_ap: {A: {ospa: .1}, B: {ospa: .9}, "A+B": {ospa: .5}}}),
    condition("b", {}, {source_metrics_by_ap: {A: {ospa: .2}, B: {ospa: .3}, "A+B": {ospa: .25}}}),
  ]);
  assert.deepEqual([...winners(data, {ap: "A"}).keys()], ["a"]);
  assert.deepEqual([...winners(data, {ap: "B"}).keys()], ["b"]);
  assert.deepEqual([...winners(data, {ap: "A+B"}).keys()], ["b"]);
});

test("split, room, dataset, layout, receiver set, and repeat mismatches stay separate", () => {
  const changes = [
    {split: "calibration"}, {dataset_id: "other"},
    {source_counts: [{room: "other", repetition: 2}]},
    {source_counts: [{room: "rectangle", repetition: 3}]},
    {obstacle_count: 0, input_protocol: "matched_survey", input_count: 270},
    {layout: "layout_01"}, {input_count: 100},
  ];
  for (const change of changes) {
    assert.equal(winners(fixture([condition("a", {ospa: .1}), condition("b", {ospa: .2}, change)])).size, 0);
  }
  const data = fixture([
    condition("a", {ospa: .1}), condition("b", {ospa: .2}),
    condition("a", {ospa: .9}, {split: "calibration"}), condition("b", {ospa: .2}, {split: "calibration"}),
  ]);
  assert.deepEqual([...winners(data).keys()], ["a"]);
  assert.deepEqual([...winners(data, {split: "calibration"}).keys()], ["b"]);
});

test("run names do not partition methods measured on identical inputs", () => {
  const data = fixture([condition("a", {ospa: .1}, {run: "run-a"}), condition("b", {ospa: .2}, {run: "run-b"})]);
  assert.deepEqual([...winners(data).keys()], ["a"]);
});

test("duplicate rows are not extra competitors; conflicting duplicates are not cherry-picked", () => {
  const a = condition("a", {ospa: .1});
  assert.equal(winners(fixture([a, a])).size, 0);
  const result = winners(fixture([a, a, condition("b", {ospa: .2})]));
  assert.equal(result.get("a")[0].comparableMethods, 2);
  assert.equal(winners(fixture([a, condition("a", {ospa: .3}), condition("b", {ospa: .2})])).size, 0);
});

test("raw and pipeline localization rows do not enter VT reconstruction comparisons", () => {
  const data = fixture([condition("a", {ospa: .1}), condition("b", {ospa: .2}), condition("raw", {ospa: 0})]);
  data.methods.find((m) => m.id === "raw").group = "raw";
  assert.deepEqual([...winners(data).keys()], ["a"]);
});

test("oracle baselines neither win nor suppress ordinary methods, even in old snapshots", () => {
  for (const id of ["range_rmse_true_count", "another_oracle"]) {
    for (const oracleMetrics of [{ospa: 0, f1: 1}, {ospa: .1, f1: .7}]) {
      const data = fixture([
        condition("a", {ospa: .1, f1: .7}), condition("b", {ospa: .2, f1: .6}),
        condition(id, oracleMetrics),
      ]);
      if (id === "another_oracle") data.methods.find(m => m.id === id).oracle = true;
      const result = winners(data);
      assert.deepEqual([...result.keys()], ["a"]);
      assert.equal(result.get("a").length, 2);
      assert.ok(result.get("a").every(win => win.comparableMethods === 2 && win.runnerUpMethod === "b"));
      assert.equal(data.methods.length, 3); // All methods keeps the diagnostic baseline.
    }
  }
});

test("one ordinary method plus an oracle is not a comparable competition", () => {
  const data = fixture([condition("a", {ospa: .1}), condition("range_rmse_true_count", {ospa: .2})]);
  const result = strictMetricWinners(data);
  assert.equal(result.wins.size, 0);
  assert.equal(result.comparableCells, 0);
});
