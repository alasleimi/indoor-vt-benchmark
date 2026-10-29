// "Pareto" here means a unique best value in any tested metric/condition,
// as requested by the experiment owner, not the nondominated frontier.
export const paretoMetrics = [
  { key: "ospa", label: "OSPA (c = 1 m)", direction: "min" },
  { key: "f1_0_1m", label: "F1 @ 0.1 m", direction: "max" },
  { key: "f1_0_3m", label: "F1 @ 0.3 m", direction: "max" },
  { key: "f1", label: "F1 @ 0.5 m", direction: "max" },
  { key: "matched_rmse_m", label: "Matched RMSE @ 0.5 m", direction: "min" },
];

function comparisonKey(condition) {
  // Run names identify different methods' studies, not different inputs. The
  // overview has already validated the common base dataset. Preserve explicitly
  // supplied dataset IDs, receiver protocols, layouts and repetition sets.
  const cases = (condition.source_counts || []).map((record) =>
    JSON.stringify([record.room, record.layout ?? null, record.repetition]),
  ).sort();
  return JSON.stringify([
    condition.dataset_id ?? condition.dataset ?? null,
    condition.split,
    condition.level,
    condition.obstacle_count ?? null,
    condition.input_protocol ?? "survey",
    condition.input_count ?? null,
    condition.layout ?? null,
    cases,
  ]);
}

function metricValue(condition, ap, key) {
  const metrics = condition.source_metrics_by_ap?.[ap];
  if (!metrics || !Number.isFinite(metrics[key])) return null;
  if (key === "matched_rmse_m" &&
      (metrics.tp === 0 || metrics.f1 === 0 || metrics.matches?.length === 0))
    return null;
  return metrics[key];
}

export function strictMetricWinners(data, { ap = "A+B", split = "evaluation" } = {}) {
  // Oracle baselines remain in All methods, but must neither qualify nor
  // suppress ordinary methods. The ID fallback also covers older snapshots.
  const methods = new Set(data.methods.filter((m) => m.group === "vt" &&
    m.oracle !== true && m.id !== "range_rmse_true_count").map((m) => m.id));
  const groups = new Map();
  for (const condition of data.conditions) {
    if (!methods.has(condition.method_id) || condition.split !== split ||
        !Number.isFinite(condition.level)) continue;
    const key = comparisonKey(condition);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(condition);
  }
  const wins = new Map();
  let comparableCells = 0;
  for (const conditions of groups.values()) {
    for (const metric of paretoMetrics) {
      // Repeated identical records cannot count as additional competitors.
      // Ambiguous duplicate values are excluded rather than cherry-picked.
      const byMethod = new Map();
      for (const condition of conditions) {
        const value = metricValue(condition, ap, metric.key);
        if (value === null) continue;
        if (!byMethod.has(condition.method_id))
          byMethod.set(condition.method_id, { condition, values: new Set() });
        byMethod.get(condition.method_id).values.add(value);
      }
      const candidates = [...byMethod.values()]
        .filter(({ values }) => values.size === 1)
        .map(({ condition, values }) => ({ condition, value: [...values][0] }));
      if (candidates.length < 2) continue;
      comparableCells++;
      candidates.sort((a, b) => (a.value - b.value) * (metric.direction === "min" ? 1 : -1));
      // Full stored precision, with no display rounding or arbitrary epsilon.
      if (candidates[0].value === candidates[1].value) continue;
      const first = candidates[0], runnerUp = candidates[1];
      const id = first.condition.method_id;
      if (!wins.has(id)) wins.set(id, []);
      wins.get(id).push({
        metric: metric.key,
        label: metric.label,
        direction: metric.direction,
        value: first.value,
        runnerUpValue: runnerUp.value,
        runnerUpMethod: runnerUp.condition.method_id,
        comparableMethods: candidates.length,
        condition: first.condition,
      });
    }
  }
  return { wins, conditionCount: groups.size, comparableCells };
}
