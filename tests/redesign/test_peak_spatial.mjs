import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
const source = await readFile(new URL("../../src/indoor_vt/execution/web/peak-spatial.js", import.meta.url), "utf8");
const { matchingMetrics, distanceBins, anchorRows, neighborhoodBoundary } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

const m = matchingMetrics([6, 8, 10, 100]);
assert.equal(m.precision, .75);
assert.equal(m.recall, .6);
assert.equal(m.f1, 2 / 3);
assert.equal(m.accuracy, .94);
assert.deepEqual([m.fp, m.fn, m.tn], [2, 4, 88]);
assert.equal(matchingMetrics([0, 0, 4, 20]).precision, null);
assert.equal(matchingMetrics([0, 0, 4, 20]).recall, 0);
assert.equal(matchingMetrics([0, 0, 0, 0]).f1, null);
assert.equal(matchingMetrics([0, 0, 0, 0]).accuracy, null);
assert.equal(matchingMetrics([0, 2, 0, 20]).recall, null);

const data = { pairs: [[0, 1], [0, 2], [1, 2]], distance_m: [.1, .4, .5],
  neighborhood: [true, false, true], first_counts: [[1, 1, 2, 10], [0, 9, 10, 100], [0, 0, 0, 0]],
  final_counts: [[2, 2, 2, 10], [5, 10, 10, 100], [0, 0, 0, 0]] };
const bins = distanceBins(data, "first", "all", .5);
assert.equal(bins.length, 2);
assert.equal(bins[0].precision, .1); // Pool counts, not the mean of 100% and 0%.
assert.equal(bins[0].recall, 1 / 12);
assert.equal(bins[0].pairs, 2);
assert.equal(bins[1].low, .5); // Boundary belongs to the next half-open bin.
assert.equal(bins[1].f1, null);
assert.equal(distanceBins(data, "final", "all", .5)[0].f1, 14 / 24);
assert.equal(distanceBins(data, "first", "neighborhood", .5)[0].precision, 1);
assert.deepEqual([...anchorRows(data, 2)], [[0, 1], [1, 2]]); // Anchor may be either endpoint.
assert.deepEqual(neighborhoodBoundary(data), { max: .5, pairs: 2, excludedWithin: 1 });
assert.equal(neighborhoodBoundary({ distance_m: [1], neighborhood: [false] }), null);
const distributionSource = await readFile(new URL("../../src/indoor_vt/execution/web/peak-distributions.js", import.meta.url), "utf8");
const { histogramRows } = await import(`data:text/javascript;base64,${Buffer.from(distributionSource).toString("base64")}`);
const rows = histogramRows({ bin_width: .25, totals: [3, 9], bins: [[0, 1, 4], [.25, 2, 5]] });
assert.equal(rows.reduce((sum, r) => sum + r.density[0] * .25, 0), 1);
assert.equal(rows.reduce((sum, r) => sum + r.density[1] * .25, 0), 1);
assert.equal(histogramRows({ bin_width: .25, totals: [0, 1], bins: [[0, 0, 1]] })[0].density[0], null);
console.log("Spatial matching metric, binning and anchor checks passed.");
