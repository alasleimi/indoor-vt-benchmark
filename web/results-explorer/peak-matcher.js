// First round of the standard preset only. Kept separate for Python parity checks.
function orderedSubset(longer, target) {
  const n = longer.length,
    m = target.length;
  const cost = Array.from({ length: m + 1 }, () =>
    new Float64Array(n + 1).fill(Infinity),
  );
  const take = Array.from({ length: m + 1 }, () => new Uint8Array(n + 1));
  cost[0].fill(0);
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++) {
      const choose =
        cost[i - 1][j - 1] + Math.abs(longer[j - 1] - target[i - 1]);
      if (choose <= cost[i][j - 1]) {
        cost[i][j] = choose;
        take[i][j] = 1;
      } else cost[i][j] = cost[i][j - 1];
    }
  const result = new Array(m);
  let i = m,
    j = n;
  while (i) {
    if (take[i][j]) {
      result[i - 1] = j - 1;
      i--;
    }
    j--;
  }
  return result;
}
function windows(a) {
  const tiny =
    Number.EPSILON * Math.max(a.at(-1) - a[0], ...a.map(Math.abs), 1) * 16;
  return a.map((center, i) =>
    Array.from({ length: 5 }, (_, k) => {
      const j = i + k - 2;
      const value =
        j < 0
          ? a[0] + j * tiny
          : j >= a.length
            ? a.at(-1) + (j - a.length + 1) * tiny
            : a[j];
      return value - center;
    }),
  );
}
export function firstRound(a0, b0) {
  let ia = a0.map((_, i) => i).sort((i, j) => a0[i] - a0[j] || i - j);
  let ib = b0.map((_, i) => i).sort((i, j) => b0[i] - b0[j] || i - j);
  let a = ia.map((i) => a0[i]),
    b = ib.map((i) => b0[i]);
  if (a.length > b.length) {
    const keep = orderedSubset(a, b);
    a = keep.map((i) => a[i]);
    ia = keep.map((i) => ia[i]);
  } else if (b.length > a.length) {
    const keep = orderedSubset(b, a);
    b = keep.map((i) => b[i]);
    ib = keep.map((i) => ib[i]);
  }
  const values = new Array(a0.length * b0.length).fill(null);
  const pattern = new Array(values.length).fill(null);
  const delta = [],
    matches = [],
    rejected = [],
    candidates = [];
  if (!a.length || !b.length)
    return { ia, ib, delta, values, pattern, matches, rejected };
  const wa = windows(a),
    wb = windows(b);
  for (let i = 0; i < a.length; i++) {
    const row = [];
    for (let j = 0; j < b.length; j++) {
      let p = 0;
      for (let l = 0; l < 5; l++)
        for (let r = l + 1; r < 5; r++)
          p += Math.abs(
            Math.abs(wa[i][l] - wa[i][r]) - Math.abs(wb[j][l] - wb[j][r]),
          );
      const d = 0.5 * Math.abs(a[i] - b[j]) + 0.5 * p;
      row.push(d);
      values[ia[i] * b0.length + ib[j]] = d;
      pattern[ia[i] * b0.length + ib[j]] = p;
    }
    delta.push(row);
  }
  for (let i = 0; i < a.length; i++) {
    let j = 0;
    for (let k = 1; k < b.length; k++) if (delta[i][k] < delta[i][j]) j = k;
    let best = 0;
    for (let k = 1; k < a.length; k++)
      if (delta[k][j] < delta[best][j]) best = k;
    if (best === i) candidates.push([ia[i], ib[j], delta[i][j]]);
  }
  candidates.sort((x, y) => x[2] - y[2] || x[0] - y[0] || x[1] - y[1]);
  for (const pair of candidates) {
    const [i, j] = pair;
    const crosses = matches.some(
      ([u, v]) => (a0[i] - a0[u]) * (b0[j] - b0[v]) < 0,
    );
    (crosses ? rejected : matches).push(pair);
  }
  return { ia, ib, delta, values, pattern, matches, rejected };
}
export function quality(a, b, matches) {
  const available = new Set(
    a.source_ids.filter((id) => b.source_ids.includes(id)),
  ).size;
  const tp = matches.filter(
    ([i, j]) => a.source_ids[i] === b.source_ids[j],
  ).length;
  const predicted = matches.length;
  return {
    tp,
    fp: predicted - tp,
    fn: available - tp,
    available,
    precision: predicted ? tp / predicted : 0,
    recall: available ? tp / available : 0,
    f1: predicted + available ? (2 * tp) / (predicted + available) : 0,
  };
}
