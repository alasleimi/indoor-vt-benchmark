import { delta, match } from './static-matcher.mjs';

class Histogram {
  constructor(width) { this.width = width; this.bins = new Map(); this.pairs = 0; this.cells = 0; }
  add(values, truth, cells, scale = 1) {
    this.pairs++; this.cells += cells;
    values.forEach((v, k) => {
      const id = Math.floor((v / scale) / this.width), row = this.bins.get(id) || [0, 0];
      row[truth[k] ? 0 : 1]++; this.bins.set(id, row);
    });
  }
  result() {
    const bins = [...this.bins].sort((a, b) => a[0] - b[0]).map(([i, row]) => [i * this.width, ...row]);
    const totals = bins.reduce((out, row) => [out[0] + row[1], out[1] + row[2]], [0, 0]);
    return { bin_width: this.width, pairs: this.pairs, candidate_cells: this.cells,
      totals, excluded_cells: this.cells - totals[0] - totals[1], bins };
  }
}

export function distributions(data, init) {
  const p = init.meta, profiles = data.profiles;
  const local = new Set(init.scene.pairs.map(([i, j]) => [Math.min(i, j), Math.max(i, j)].join(',')));
  const all = new Histogram(.25), neighborhood = new Histogram(.25), normalized = new Histogram(.05);
  const anchorValues = profiles.map(() => []), stored = [];
  for (let i = 0; i < profiles.length; i++) for (let j = i + 1; j < profiles.length; j++) {
    const a = profiles[i], b = profiles[j], separation = p.triangle_tolerance_m == null ? -1 : Math.hypot(...a.xyz.map((v, k) => v - b.xyz[k]));
    let ia, ib;
    if (p.unify) {
      const result = match(a.ranges, b.ranges, p, separation);
      ia = result.left_indices; ib = result.right_indices;
    } else {
      ia = a.ranges.map((_, k) => k).sort((x, y) => a.ranges[x] - a.ranges[y]);
      ib = b.ranges.map((_, k) => k).sort((x, y) => b.ranges[x] - b.ranges[y]);
    }
    const x = [], y = [];
    if (ia.length && ib.length) {
      const matrix = delta(ia.map(k => a.ranges[k]), ib.map(k => b.ranges[k]), p.half_window, p.alpha, false, separation, p.triangle_tolerance_m || 0);
      ia.forEach((ai, u) => ib.forEach((bi, v) => {
        if (Number.isFinite(matrix[u][v])) { x.push(matrix[u][v]); y.push(a.source_ids[ai] === b.source_ids[bi]); }
      }));
    }
    const cells = a.ranges.length * b.ranges.length;
    all.add(x, y, cells);
    if (local.has(i + ',' + j)) {
      neighborhood.add(x, y, cells); anchorValues[i].push(x); anchorValues[j].push(x); stored.push([i, j, x, y]);
    }
  }
  const medians = anchorValues.map(chunks => {
    const values = Float64Array.from(chunks.flat()).sort(), n = values.length;
    return n ? (values[Math.floor((n - 1) / 2)] + values[Math.floor(n / 2)]) / 2 : null;
  });
  let skipped = 0;
  for (const [i, j, x, y] of stored) for (const anchor of [i, j]) {
    const scale = medians[anchor];
    if (scale != null && scale > 0) normalized.add(x, y, x.length, scale);
    else skipped += x.length;
  }
  return { schema: 'mpurge.dissimilarity-distributions.1', columns: ['low', 'correct', 'incorrect'],
    all: all.result(), neighborhood: neighborhood.result(), normalized: normalized.result(),
    normalization: { anchor_medians: medians, skipped_zero_median_cells: skipped, directions_per_pair: 2 } };
}
