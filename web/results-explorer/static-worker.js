"use strict";

// Lossless exported columns; all filtering/statistics run off the UI thread.
const metadata = fetch("meta.json").then(async (response) => {
  if (!response.ok) throw Error("Could not load the published result index.");
  const data = await response.json();
  if (data.export_schema !== "indoor-vt.pages.v1")
    throw Error("Unsupported result snapshot.");
  return data;
});
const arrays = new Map(),
  loading = new Map();
let cachedBytes = 0,
  downloads = 0;
const waiters = [],
  CACHE_BYTES = 96 * 1024 * 1024;

async function slot() {
  if (downloads >= 4) await new Promise((resolve) => waiters.push(resolve));
  else downloads++;
}
function release() {
  const next = waiters.shift();
  if (next) next();
  else downloads--;
}
async function digest(buffer) {
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", buffer))]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
async function column(descriptor) {
  const key = `${descriptor.file}:${descriptor.dtype}:${descriptor.length}`;
  if (arrays.has(key)) {
    const value = arrays.get(key);
    arrays.delete(key);
    arrays.set(key, value);
    return value;
  }
  if (loading.has(key)) return loading.get(key);
  const promise = (async () => {
    await slot();
    try {
      const response = await fetch(descriptor.file);
      if (!response.ok)
        throw Error(
          `Could not load a result column (HTTP ${response.status}).`,
        );
      const compressed = await response.arrayBuffer();
      if ((await digest(compressed)) !== descriptor.sha256)
        throw Error("Result column checksum mismatch.");
      const raw = await new Response(
        new Blob([compressed])
          .stream()
          .pipeThrough(new DecompressionStream("gzip")),
      ).arrayBuffer();
      if (
        raw.byteLength !== descriptor.raw_bytes ||
        (await digest(raw)) !== descriptor.raw_sha256
      )
        throw Error("Invalid decompressed result column.");
      const value = descriptor.dtype.endsWith("f8")
        ? new Float64Array(raw)
        : new Uint8Array(raw);
      if (value.length !== descriptor.length)
        throw Error("Invalid result column length.");
      while (cachedBytes + raw.byteLength > CACHE_BYTES && arrays.size) {
        const oldest = arrays.keys().next().value;
        cachedBytes -= arrays.get(oldest).byteLength;
        arrays.delete(oldest);
      }
      arrays.set(key, value);
      cachedBytes += raw.byteLength;
      return value;
    } finally {
      release();
      loading.delete(key);
    }
  })();
  loading.set(key, promise);
  return promise;
}
async function selection(job, filters) {
  const meta = await metadata;
  const conditions = await column(job.columns.conditions);
  const fields =
    job.stage === "vt"
      ? ["ospa", "f1", "predicted", "truth"]
      : ["final", "initial", "fallback"];
  const values = Object.fromEntries(
    await Promise.all(
      fields.map(async (name) => [name, await column(job.columns[name])]),
    ),
  );
  const selected = [];
  for (let i = 0; i < conditions.length / 4; i++) {
    const offset = 4 * i,
      room = meta.rooms[conditions[offset]];
    if (filters.family && room.family !== filters.family) continue;
    if (filters.room && room.id !== filters.room) continue;
    if (filters.ap && ["A", "B", "A+B"][conditions[offset + 1]] !== filters.ap)
      continue;
    if (
      filters.noise &&
      meta.levels[job.track][conditions[offset + 2]] !== Number(filters.noise)
    )
      continue;
    if (job.stage !== "vt") {
      const layout = conditions[offset + 3] - 1;
      if (filters.layout && layout !== Number(filters.layout)) continue;
      if (
        filters.obstacles &&
        room.layouts[layout] !== Number(filters.obstacles)
      )
        continue;
    }
    selected.push(i);
  }
  return { conditions, values, selected };
}
function average(values, indices, squared = false) {
  // Compensated accumulation avoids drift across hundreds of thousands of cases.
  let sum = 0,
    correction = 0;
  for (const i of indices) {
    const term = (squared ? values[i] * values[i] : values[i]) - correction;
    const next = sum + term;
    correction = next - sum - term;
    sum = next;
  }
  return sum / indices.length;
}
function quantile(ordered, q) {
  const position = (ordered.length - 1) * q,
    lower = Math.floor(position);
  const fraction = position - lower;
  return (
    ordered[lower] +
    fraction *
      (ordered[Math.min(lower + 1, ordered.length - 1)] - ordered[lower])
  );
}
function summarize(data, stage) {
  const { values, selected } = data;
  if (!selected.length) return null;
  if (stage === "vt")
    return {
      count: selected.length,
      ...Object.fromEntries(
        ["ospa", "f1", "predicted", "truth"].map((k) => [
          k,
          average(values[k], selected),
        ]),
      ),
    };
  const ordered = Float64Array.from(selected, (i) => values.final[i]).sort();
  const mean = average(values.final, selected),
    initial = average(values.initial, selected);
  return {
    count: selected.length,
    mean,
    initial,
    rmse: Math.sqrt(average(values.final, selected, true)),
    median: quantile(ordered, 0.5),
    p90: quantile(ordered, 0.9),
    improvement: initial ? (100 * (initial - mean)) / initial : null,
    fallbacks: selected.reduce((sum, i) => sum + values.fallback[i], 0),
  };
}
async function results(filters) {
  const meta = await metadata;
  const stages =
    filters.view === "vt"
      ? ["vt"]
      : filters.view === "raw"
        ? ["coarse"]
        : ["coarse", "pipeline"];
  const jobs = meta.jobs.filter((job) => {
    if (
      job.track !== (filters.track || "receiver") ||
      !stages.includes(job.stage)
    )
      return false;
    if (filters.view === "all" && filters.stage && job.stage !== filters.stage)
      return false;
    if (
      (filters.scope && job.scope !== filters.scope) ||
      (filters.merger && job.merger !== filters.merger)
    )
      return false;
    return [job.id, job.name, job.reconstruction, job.merger]
      .join(" ")
      .toLowerCase()
      .includes((filters.q || "").toLowerCase());
  });
  const rows = await Promise.all(
    jobs.map(async (job) => {
      const metrics = summarize(await selection(job, filters), job.stage);
      return metrics
        ? {
            ...Object.fromEntries(
              [
                "id",
                "name",
                "stage",
                "method",
                "scope",
                "merger",
                "reconstruction",
              ].map((k) => [k, job[k]]),
            ),
            ...metrics,
          }
        : null;
    }),
  );
  return { rows: rows.filter(Boolean), filters };
}
async function detail(filters) {
  const meta = await metadata,
    job = meta.jobs.find((j) => j.id === filters.id);
  if (!job) throw Error("Unknown result.");
  const data = await selection(job, filters),
    rooms = [];
  meta.rooms.forEach((room, index) => {
    const selected = data.selected.filter(
      (i) => data.conditions[i * 4] === index,
    );
    const metrics = summarize({ ...data, selected }, job.stage);
    if (metrics) rooms.push({ room: room.id, family: room.family, ...metrics });
  });
  const ordered = Float64Array.from(
    data.selected,
    (i) => data.values[job.stage === "vt" ? "ospa" : "final"][i],
  ).sort();
  const ecdf = [],
    count = Math.min(201, ordered.length);
  for (let j = 0; j < count; j++) {
    const i =
      count === 1 ? 0 : Math.floor((j * (ordered.length - 1)) / (count - 1));
    const value = ordered[i];
    let lower = i,
      upper = ordered.length;
    while (lower < upper) {
      const middle = (lower + upper) >>> 1;
      if (ordered[middle] <= value) lower = middle + 1;
      else upper = middle;
    }
    ecdf.push([value, lower / ordered.length]);
  }
  const { columns, ...publicJob } = job;
  return { job: publicJob, metrics: summarize(data, job.stage), rooms, ecdf };
}
self.onmessage = async ({ data }) => {
  try {
    let result;
    if (data.endpoint === "meta") {
      const { jobs, ...meta } = await metadata;
      result = meta;
    } else if (data.endpoint === "results")
      result = await results(data.filters);
    else if (data.endpoint === "detail") result = await detail(data.filters);
    else throw Error("Unknown request.");
    self.postMessage({ id: data.id, result });
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message });
  }
};
