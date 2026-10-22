"use strict";

// Lossless exported columns; all filtering/statistics run off the UI thread.
const metadata = fetch("meta.json").then(async (response) => {
  if (!response.ok) throw Error("Could not load the published result index.");
  const data = await response.json();
  if (
    !["indoor-vt.pages.v1", "indoor-vt.pages.v2"].includes(data.export_schema)
  )
    throw Error("Unsupported result snapshot.");
  return data;
});
const arrays = new Map(),
  loading = new Map();
let cachedBytes = 0,
  downloads = 0;
const waiters = [],
  CACHE_BYTES = 16 * 1024 * 1024;
const selections = new Map(),
  jsonCache = new Map();
let selectionBytes = 0;
const pause = () => new Promise((resolve) => setTimeout(resolve, 0));

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
async function buffer(descriptor, signal) {
  signal.throwIfAborted();
  const key = descriptor.file;
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
      signal.throwIfAborted();
      const response = await fetch(descriptor.file, { signal });
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
      signal.throwIfAborted();
      while (cachedBytes + raw.byteLength > CACHE_BYTES && arrays.size) {
        const oldest = arrays.keys().next().value;
        cachedBytes -= arrays.get(oldest).byteLength;
        arrays.delete(oldest);
      }
      arrays.set(key, raw);
      cachedBytes += raw.byteLength;
      return raw;
    } finally {
      release();
      loading.delete(key);
    }
  })();
  loading.set(key, promise);
  return promise;
}
async function column(descriptor, signal) {
  const raw = await buffer(descriptor, signal);
  const Type = descriptor.dtype.endsWith("f8") ? Float64Array : Uint8Array;
  const offset = descriptor.byte_offset || 0;
  if (offset + descriptor.length * Type.BYTES_PER_ELEMENT > raw.byteLength)
    throw Error("Invalid result column length.");
  return new Type(raw, offset, descriptor.length);
}
async function jsonData(descriptor, signal) {
  signal.throwIfAborted();
  if (jsonCache.has(descriptor.file)) {
    const value = jsonCache.get(descriptor.file);
    jsonCache.delete(descriptor.file);
    jsonCache.set(descriptor.file, value);
    return value;
  }
  const value = JSON.parse(
    new TextDecoder().decode(await buffer(descriptor, signal)),
  );
  signal.throwIfAborted();
  while (jsonCache.size >= 2) jsonCache.delete(jsonCache.keys().next().value);
  jsonCache.set(descriptor.file, value);
  return value;
}
function hasCaseFilters(job, filters) {
  return [
    "family",
    "room",
    "noise",
    "ap",
    ...(job.stage === "vt" ? [] : ["layout", "obstacles"]),
  ].some((key) => filters[key] !== undefined && filters[key] !== "");
}
async function selection(job, filters, signal) {
  const meta = await metadata;
  if (
    job.stage !== "vt" &&
    meta.partitions &&
    (filters.room || filters.family)
  ) {
    const rooms = meta.rooms.filter(
      (r) =>
        (!filters.room || r.id === filters.room) &&
        (!filters.family || r.family === filters.family),
    );
    const index = await jsonData(meta.partitions, signal);
    const pieces = [];
    for (const room of rooms) {
      pieces.push(
        await selectColumns(
          { ...job, columns: index[job.track][room.id][job.id] },
          filters,
          signal,
        ),
      );
    }
    if (pieces.length === 1) return pieces[0];
    const count = pieces.reduce((n, piece) => n + piece.selected.length, 0);
    const conditions = new Uint8Array(count * 4),
      selected = new Uint32Array(count);
    const values = {
      final: new Float64Array(count),
      initial: new Float64Array(count),
      fallback: new Uint8Array(count),
    };
    let offset = 0;
    for (const piece of pieces)
      for (const i of piece.selected) {
        conditions.set(piece.conditions.subarray(i * 4, i * 4 + 4), offset * 4);
        for (const name of Object.keys(values))
          values[name][offset] = piece.values[name][i];
        selected[offset] = offset++;
      }
    return { conditions, values, selected };
  }
  return selectColumns(job, filters, signal);
}
async function selectColumns(job, filters, signal) {
  const meta = await metadata;
  const conditions = await column(job.columns.conditions, signal);
  const keys = [
    "family",
    "room",
    "noise",
    "ap",
    ...(job.stage === "vt" ? [] : ["layout", "obstacles"]),
  ];
  const selectionKey = JSON.stringify([
    job.columns.conditions,
    keys.map((k) => filters[k] || ""),
  ]);
  let selected = selections.get(selectionKey);
  if (!selected) {
    const indices = new Uint32Array(conditions.length / 4);
    let count = 0;
    for (let i = 0; i < conditions.length / 4; i++) {
      const offset = 4 * i,
        room = meta.rooms[conditions[offset]];
      if (filters.family && room.family !== filters.family) continue;
      if (filters.room && room.id !== filters.room) continue;
      if (
        filters.ap &&
        ["A", "B", "A+B"][conditions[offset + 1]] !== filters.ap
      )
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
      indices[count++] = i;
    }
    selected = indices.slice(0, count);
    while (
      selectionBytes + selected.byteLength > 4 * 1024 * 1024 &&
      selections.size
    ) {
      const oldest = selections.keys().next().value;
      selectionBytes -= selections.get(oldest).byteLength;
      selections.delete(oldest);
    }
    if (selections.size >= 16) {
      const oldest = selections.keys().next().value;
      selectionBytes -= selections.get(oldest).byteLength;
      selections.delete(oldest);
    }
    selections.set(selectionKey, selected);
    selectionBytes += selected.byteLength;
  }
  if (!selected.length) return { conditions, values: {}, selected };
  const fields =
    job.stage === "vt"
      ? ["ospa", "f1", "predicted", "truth"]
      : ["final", "initial", "fallback"];
  const values = Object.fromEntries(
    await Promise.all(
      fields.map(async (name) => [
        name,
        await column(job.columns[name], signal),
      ]),
    ),
  );
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
async function results(filters, signal) {
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
  const rows = [];
  for (const job of jobs) {
    signal.throwIfAborted();
    const metrics =
      !hasCaseFilters(job, filters) && job.metrics
        ? job.metrics
        : summarize(await selection(job, filters, signal), job.stage);
    if (metrics)
      rows.push({
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
      });
    if (hasCaseFilters(job, filters)) await pause();
  }
  return { rows, filters };
}
async function detail(filters, signal) {
  const meta = await metadata,
    job = meta.jobs.find((j) => j.id === filters.id);
  if (!job) throw Error("Unknown result.");
  const { columns, ...publicJob } = job;
  if (!hasCaseFilters(job, filters) && meta.details) {
    const saved = await jsonData(meta.details, signal);
    return { job: publicJob, ...saved[job.id] };
  }
  const data = await selection(job, filters, signal),
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
  return { job: publicJob, metrics: summarize(data, job.stage), rooms, ecdf };
}
async function sourceMap(filters, signal) {
  const meta = await metadata,
    job = meta.jobs.find((j) => j.id === filters.id);
  if (!job || job.stage !== "vt")
    throw Error("Select a VT reconstruction result.");
  const descriptor = meta.visualizer?.maps[job.track]?.[filters.room];
  if (!descriptor) throw Error("No saved source map for this room.");
  const scenes = await jsonData(meta.visualizer.scenes, signal);
  const pack = await jsonData(descriptor, signal);
  const records = pack.methods[job.id]?.[filters.ap];
  const level = Object.keys(records || {}).find(
    (k) => Number(k) === Number(filters.noise),
  );
  if (level === undefined)
    throw Error("No saved source map for this selection.");
  const { points, matches, ...metrics } = records[level];
  const channels = [],
    truth = [];
  let count = 0;
  for (const ap of filters.ap.split("+")) {
    const descriptor = pack.points[points[ap]];
    channels.push({ ap, start: count, count: descriptor.count, descriptor });
    count += descriptor.count;
  }
  const predicted = new Float64Array(count * 3);
  for (const channel of channels) {
    const raw = await buffer(channel.descriptor, signal);
    if (raw.byteLength !== channel.count * 3 * 8)
      throw Error("Invalid source coordinate data.");
    predicted.set(new Float64Array(raw), channel.start * 3);
    delete channel.descriptor;
  }
  for (const ap of filters.ap.split("+")) {
    truth.push(
      ...scenes[filters.room].truth[ap].map((source) => ({ ...source, ap })),
    );
  }
  return {
    scene: scenes[filters.room],
    predicted,
    channels,
    truth,
    matches,
    metrics,
    track: job.track,
    room: filters.room,
    ap: filters.ap,
    level: Number(level),
  };
}
const tasks = new Map(),
  queue = [];
let running = false;
async function pump() {
  if (running) return;
  running = true;
  while (queue.length) {
    const data = queue.shift(),
      controller = tasks.get(data.id);
    if (!controller) continue;
    const signal = controller.signal;
    try {
      signal.throwIfAborted();
      let result;
      if (data.endpoint === "meta") {
        const { jobs, ...meta } = await metadata;
        result = meta;
      } else if (data.endpoint === "results")
        result = await results(data.filters, signal);
      else if (data.endpoint === "detail")
        result = await detail(data.filters, signal);
      else if (data.endpoint === "map")
        result = await sourceMap(data.filters, signal);
      else if (data.endpoint === "diagnostics")
        result = {
          cachedBytes,
          selectionBytes,
          downloads,
          queued: queue.length,
          jsonEntries: jsonCache.size,
        };
      else throw Error("Unknown request.");
      signal.throwIfAborted();
      self.postMessage(
        { id: data.id, result },
        data.endpoint === "map" ? [result.predicted.buffer] : [],
      );
    } catch (error) {
      if (!signal.aborted)
        self.postMessage({ id: data.id, error: error.message });
    } finally {
      controller.abort();
      await Promise.allSettled([...loading.values()]);
      tasks.delete(data.id);
    }
  }
  running = false;
}
self.onmessage = ({ data }) => {
  if (data.cancel) {
    tasks.get(data.id)?.abort();
    tasks.delete(data.id);
    return;
  }
  tasks.set(data.id, new AbortController());
  queue.push(data);
  pump();
};
