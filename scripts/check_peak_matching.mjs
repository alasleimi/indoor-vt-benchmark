// Verify the browser first-round matcher against the published Python counts/bins.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import assert from "node:assert/strict";
import crypto from "node:crypto";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const base = path.join(root, "published/v6");
const source = fs.readFileSync(
  path.join(root, "web/results-explorer/peak-matcher.js"),
  "utf8",
);
const { firstRound, quality } = await import(
  "data:text/javascript;base64," + Buffer.from(source).toString("base64")
);
const index = JSON.parse(
  fs.readFileSync(path.join(base, "matching/index.json")),
);
const sha = (buffer) =>
  crypto.createHash("sha256").update(buffer).digest("hex");
function unpack(d) {
  const packed = fs.readFileSync(path.join(base, d.file));
  assert.equal(sha(packed), d.sha256);
  const raw = gunzipSync(packed);
  assert.equal(sha(raw), d.raw_sha256);
  assert.equal(raw.length, d.raw_bytes);
  return JSON.parse(raw);
}
function near(actual, expected, tolerance = 1e-10) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} != ${expected}`,
  );
}
const scene = unpack(index.scene),
  natives = Object.fromEntries(
    Object.entries(index.native).map(([ap, d]) => [ap, unpack(d)]),
  );
const fixtureFile = path.join(
  root,
  "runs/development/publication-checkpoint/peak-matching/validation.json",
);
const fixtures = fs.existsSync(fixtureFile)
  ? JSON.parse(fs.readFileSync(fixtureFile))
  : [];
let pairsChecked = 0,
  cellsChecked = 0,
  fixturesChecked = 0;
for (const condition of index.conditions) {
  const data = unpack(condition.data),
    native = natives[condition.ap];
  const profiles = data.profiles.map((p, i) => ({
    ranges: p.ranges,
    source_ids: p.native_indices.map((k) => native.profiles[i].source_ids[k]),
  }));
  let tp = 0,
    fp = 0,
    fn = 0;
  const bins = new Map();
  for (let id = 0; id < scene.pairs.length; id++) {
    const [left, right] = scene.pairs[id],
      a = profiles[left],
      b = profiles[right];
    const m = firstRound(a.ranges, b.ranges),
      q = quality(a, b, m.matches);
    assert.deepEqual(
      [q.tp, m.matches.length, q.available],
      data.pair_counts[id],
    );
    tp += q.tp;
    fp += q.fp;
    fn += q.fn;
    pairsChecked++;
    for (let i = 0; i < m.ia.length; i++)
      for (let j = 0; j < m.ib.length; j++) {
        const value = m.delta[i][j],
          bin = Math.floor(value / 0.25);
        const counts = bins.get(bin) ?? [0, 0];
        counts[0]++;
        counts[1] += a.source_ids[m.ia[i]] === b.source_ids[m.ib[j]] ? 1 : 0;
        bins.set(bin, counts);
        cellsChecked++;
      }
  }
  assert.deepEqual(
    [tp, fp, fn],
    [condition.summary.tp, condition.summary.fp, condition.summary.fn],
  );
  near(tp / (tp + fp), condition.summary.precision);
  near(tp / (tp + fn), condition.summary.recall);
  near((2 * tp) / (2 * tp + fp + fn), condition.summary.f1);
  assert.equal(bins.size, data.bins.length);
  for (const [lower, , count, correct] of data.bins)
    assert.deepEqual(bins.get(Math.round(lower / 0.25)), [count, correct]);
  for (const fixture of fixtures.filter(
    (f) => f.ap === condition.ap && f.noise === condition.noise,
  )) {
    const m = firstRound(
      profiles[fixture.left].ranges,
      profiles[fixture.right].ranges,
    );
    assert.deepEqual(m.ia, fixture.ia);
    assert.deepEqual(m.ib, fixture.ib);
    assert.deepEqual(m.matches, fixture.matches);
    assert.deepEqual(m.delta, fixture.delta);
    fixturesChecked++;
  }
  console.log(
    `${condition.ap} sigma=${condition.noise}: every pair, cell bin and summary agrees`,
  );
}
assert.deepEqual(firstRound([], []).matches, []);
console.log(
  `PASS: ${pairsChecked} pairs; ${cellsChecked} cell bins; ${fixturesChecked} bit-for-bit Python matrix fixtures.`,
);
