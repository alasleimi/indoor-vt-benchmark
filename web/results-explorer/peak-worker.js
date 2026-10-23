import {
  firstRound,
  finalMatching,
  quality,
  summarizeFinalRoom,
} from "./peak-matcher.js";
const cache = new Map();
const finalCache = new Map();
const finalSummaries = new Map();
async function sha(raw) {
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", raw))]
    .map((x) => x.toString(16).padStart(2, "0"))
    .join("");
}
async function data(d) {
  if (cache.has(d.file)) {
    const value = cache.get(d.file);
    cache.delete(d.file);
    cache.set(d.file, value);
    return value;
  }
  const response = await fetch(d.file);
  if (!response.ok)
    throw Error(`Could not load matching data (${response.status}).`);
  const compressed = await response.arrayBuffer();
  if ((await sha(compressed)) !== d.sha256)
    throw Error("Matching data checksum mismatch.");
  const raw = await new Response(
    new Blob([compressed])
      .stream()
      .pipeThrough(new DecompressionStream("gzip")),
  ).arrayBuffer();
  if (raw.byteLength !== d.raw_bytes || (await sha(raw)) !== d.raw_sha256)
    throw Error("Invalid matching data.");
  const result = JSON.parse(new TextDecoder().decode(raw));
  while (cache.size >= 3) cache.delete(cache.keys().next().value);
  cache.set(d.file, result);
  return result;
}
const index = fetch("matching/index.json").then(async (r) => {
  if (!r.ok) throw Error("The matching snapshot is unavailable.");
  const value = await r.json();
  if (value.schema !== "indoor-vt.peak-matching.v1")
    throw Error("Unsupported matching snapshot.");
  return value;
});
let scenePromise;
async function condition(ap, noise) {
  const meta = await index;
  const item = meta.conditions.find(
    (x) => x.ap === ap && x.noise === Number(noise),
  );
  if (!item) throw Error("Unsupported AP or noise condition.");
  const [native, observed] = await Promise.all([
    data(meta.native[ap]),
    data(item.data),
  ]);
  return { item, native, observed };
}
async function roomResult(item, native, observed, scene, scope) {
  const first = {
    summary: item.summary,
    pair_counts: observed.pair_counts,
    bins: observed.bins,
  };
  if (scope !== "final") return first;
  const key = item.data.file;
  if (!finalCache.has(key)) {
    const pending = (async () => {
      const profiles = observed.profiles.map((p, i) => ({
        ranges: p.ranges,
        source_ids: p.native_indices.map(
          (k) => native.profiles[i].source_ids[k],
        ),
      }));
      const result = await summarizeFinalRoom(
        profiles,
        scene.pairs,
        () => new Promise((resolve) => setTimeout(resolve, 0)),
      );
      const summary = { ...item.summary, ...result.summary };
      finalSummaries.set(key, summary);
      return { ...result, summary };
    })();
    while (finalCache.size >= 3)
      finalCache.delete(finalCache.keys().next().value);
    finalCache.set(key, pending);
    pending.catch(() => {
      if (finalCache.get(key) === pending) finalCache.delete(key);
    });
  }
  const pending = finalCache.get(key);
  finalCache.delete(key);
  finalCache.set(key, pending);
  return { ...(await pending), bins: observed.bins };
}
self.onmessage = async ({ data: message }) => {
  const { id, action, args = {} } = message;
  try {
    const meta = await index;
    scenePromise ??= data(meta.scene);
    const scene = await scenePromise;
    let result;
    if (action === "init") result = { meta, scene };
    else if (action === "summaries") {
      result = [];
      for (const item of meta.conditions) {
        let summary = item.summary;
        if (args.matches === "final") {
          if (!finalSummaries.has(item.data.file)) {
            const { native, observed } = await condition(item.ap, item.noise);
            await roomResult(item, native, observed, scene, "final");
          }
          summary = finalSummaries.get(item.data.file);
        }
        result.push({ ap: item.ap, noise: item.noise, summary });
      }
    } else {
      const { item, native, observed } = await condition(args.ap, args.noise);
      if (action === "condition")
        result = await roomResult(item, native, observed, scene, args.matches);
      else if (action === "pair") {
        const [left, right] = [Number(args.left), Number(args.right)].sort(
          (a, b) => a - b,
        );
        if (
          !Number.isInteger(left) ||
          !Number.isInteger(right) ||
          left < 0 ||
          right >= observed.profiles.length ||
          left === right
        )
          throw Error("Choose two different survey positions.");
        const profile = (i) => {
          const p = observed.profiles[i],
            n = native.profiles[i];
          return {
            reference: i,
            ranges: p.ranges,
            source_ids: p.native_indices.map((k) => n.source_ids[k]),
            gains: p.native_indices.map((k) => n.gain_db[k]),
            xyz: scene.survey_xyz[i],
          };
        };
        const a = profile(left),
          b = profile(right),
          matrix = firstRound(a.ranges, b.ranges),
          final = finalMatching(a.ranges, b.ranges, matrix);
        result = {
          a,
          b,
          matrix,
          final,
          sources: native.sources,
          quality: quality(a, b, matrix.matches),
          finalQuality: quality(a, b, final.matches),
        };
      } else throw Error("Unknown matching request.");
    }
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
