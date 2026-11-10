import test from 'node:test';
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { decodeSnapshot, restoreSnapshot } from '../../src/indoor_vt/execution/web/static-snapshot.mjs';

test('compact source maps preserve exact AP coordinates, metrics, and localization cases', () => {
  const original = { cases: [
    { kind: 'sources', predicted: { A: [[1.000000000000003, -2, 3]], B: [[4, 5, 6], [7, 8, 9]] },
      maps: { A: {predicted: [1.000000000000003, -2, 3], metrics: {f1: 0.7}},
        B: {predicted: [4, 5, 6, 7, 8, 9]},
        'A+B': {predicted: [1.000000000000003, -2, 3, 4, 5, 6, 7, 8, 9]} } },
    { kind: 'localization', predicted: [[2, 3, 4]], errors: [0.12] },
    { kind: 'sources', predicted: { A: [], B: [] }, maps: {A: {predicted: []}, 'A+B': {predicted: []}} },
  ]};
  const compact = structuredClone(original);
  for (const record of compact.cases.filter(r => r.kind === 'sources')) {
    for (const payload of Object.values(record.maps)) delete payload.predicted;
    record.map_predictions_from_case = true;
  }
  assert.deepEqual(restoreSnapshot(compact), original);
  assert.deepEqual(restoreSnapshot(structuredClone(original)), original);
});

test('overview and legacy result payloads remain compatible', () => {
  const overview = {methods: [{id: 'method'}]};
  assert.equal(restoreSnapshot(overview), overview);
  assert.deepEqual(restoreSnapshot({cases: []}), {cases: []});
});

test('transport pieces reconstruct the exact gzip stream before restoring AP maps', async () => {
  const saved = {cases: [{kind: 'sources', predicted: {A: [[1.123456789012345, 2, 3]], B: []},
    maps: {A: {}, 'A+B': {}}, map_predictions_from_case: true}]};
  const compressed = gzipSync(JSON.stringify(saved));
  const pieces = [compressed.subarray(0, 7), compressed.subarray(7, 19), compressed.subarray(19)];
  const names = pieces.map((_, i) => `results/result.part-${i}`);
  const manifest = new Blob([gzipSync(JSON.stringify({gzip_parts: names}))]);
  const requested = [];
  const result = await decodeSnapshot(manifest, async name => {
    requested.push(name);
    return new Blob([pieces[names.indexOf(name)]]);
  });
  assert.deepEqual(requested, names);
  assert.deepEqual(result, restoreSnapshot(structuredClone(saved)));
  assert.deepEqual(await decodeSnapshot(new Blob([compressed]), () => assert.fail('Unexpected part request')),
    restoreSnapshot(structuredClone(saved)));
});
