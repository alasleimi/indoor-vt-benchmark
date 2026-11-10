// Restore redundant display arrays omitted by the static exporter.
// Saved predictions and metrics retain their original numeric precision.
async function readGzipJSON(blob) {
  return JSON.parse(await new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).text());
}

export async function decodeSnapshot(blob, loadPart) {
  let value = await readGzipJSON(blob);
  if (Array.isArray(value.gzip_parts)) {
    const parts = await Promise.all(value.gzip_parts.map(loadPart));
    value = await readGzipJSON(new Blob(parts));
  }
  return restoreSnapshot(value);
}

export function restoreSnapshot(value) {
  for (const record of value.cases || []) {
    if (!record.map_predictions_from_case) continue;
    for (const [ap, payload] of Object.entries(record.maps)) {
      payload.predicted = ap.split('+').flatMap(channel => record.predicted[channel].flat());
    }
    delete record.map_predictions_from_case;
  }
  return value;
}
