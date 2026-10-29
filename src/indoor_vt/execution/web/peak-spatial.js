// Pair identity metrics: shared by the distance plot and the survey map.
export function matchingMetrics([tp, selected, truth, cells]) {
  const fp = selected - tp, fn = truth - tp, tn = cells - tp - fp - fn;
  return {
    precision: selected ? tp / selected : null,
    recall: truth ? tp / truth : null,
    f1: selected + truth ? 2 * tp / (selected + truth) : null,
    accuracy: cells ? (tp + tn) / cells : null,
    tp, fp, fn, tn, selected, truth, cells,
  };
}

export function distanceBins(data, stage, population, width) {
  const bins = new Map(), counts = data[`${stage}_counts`];
  data.pairs.forEach((_, i) => {
    if (population === "neighborhood" && !data.neighborhood[i]) return;
    const id = Math.floor(data.distance_m[i] / width);
    if (!bins.has(id)) bins.set(id, { id, pairs: 0, counts: [0, 0, 0, 0] });
    const bin = bins.get(id);
    bin.pairs++;
    counts[i].forEach((v, k) => bin.counts[k] += v);
  });
  return [...bins.values()].sort((a, b) => a.id - b.id).map(b => ({
    ...b, low: b.id * width, high: (b.id + 1) * width, ...matchingMetrics(b.counts),
  }));
}

export function anchorRows(data, anchor) {
  const rows = new Map();
  data.pairs.forEach(([a, b], i) => {
    if (a === anchor) rows.set(b, i);
    else if (b === anchor) rows.set(a, i);
  });
  return rows;
}

export function neighborhoodBoundary(data) {
  const included = data.distance_m.filter((_, i) => data.neighborhood[i]);
  if (!included.length) return null;
  const max = Math.max(...included);
  return { max, pairs: included.length,
    excludedWithin: data.distance_m.filter((d, i) => d <= max + 1e-12 && !data.neighborhood[i]).length };
}

const labels = { precision: "Precision", recall: "Recall", f1: "F1", accuracy: "Pairwise accuracy" };
const colors = { precision: "#2071aa", recall: "#b66b18", f1: "#087e70" };
const fmt = (v, digits = 2) => Number(v).toLocaleString("en-US", { maximumFractionDigits: digits });
const pct = v => v == null ? "Undefined" : `${fmt(v * 100)}%`;
const ref = i => `S${String(i + 1).padStart(3, "0")}`;
const escape = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
// ColorBrewer Reds, with the same piecewise-linear scale for points and legend.
const reds = ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"];
const redStops = reds.map(hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)));
const ramp = v => {
  if (v == null) return "#b8c2ca";
  const t = Math.max(0, Math.min(1, v)) * (redStops.length - 1);
  const i = Math.min(Math.floor(t), redStops.length - 2), f = t - i;
  return `rgb(${redStops[i].map((x, k) => Math.round(x + f * (redStops[i + 1][k] - x))).join(",")})`;
};

export function mountMatchingSpatial(container, { scene, request, saveLocation, inspect }) {
  const $ = s => container.querySelector(s);
  container.style.setProperty("--matching-ramp", `linear-gradient(90deg, ${reds.join(",")})`);
  container.innerHTML = `
    <div class="match-section-heading"><div><h2>Matching quality versus distance</h2>
      <p data-spatial="condition" aria-live="polite">Loading pair identities…</p></div></div>
    <div class="spatial-controls">
      <label>Pairs to compare<select data-spatial="population"><option value="all">All survey pairs</option><option value="neighborhood">MPUrge neighborhood pairs</option></select></label>
      <label>Distance bin width<select data-spatial="bin"><option>0.25</option><option selected>0.5</option><option>1</option><option>2</option></select></label>
      <fieldset class="spatial-series"><legend>Show metrics</legend>${["precision", "recall", "f1"].map(k => `<label><input type="checkbox" data-series="${k}" checked><i style="background:${colors[k]}"></i>${labels[k]}</label>`).join("")}</fieldset>
      <button class="button subtle" data-spatial="export" disabled>Export distance bins</button>
    </div>
    <p class="spatial-caption">Distance is the 3D separation between survey positions, in meters. Within each bin, correct, selected and true matches are pooled before calculating the three metrics. All survey pairs includes diagnostic pairs outside the neighborhoods used by reconstruction.</p>
    <div class="spatial-chart" data-spatial="chart" aria-live="polite"></div>
    <p class="spatial-caption" data-spatial="boundary"></p>
    <p class="spatial-caption" data-spatial="bin-detail">Hover a point for its counts and distance interval.</p>
    <details class="matching-explanation"><summary>What counts as correct?</summary>
      <p>A selected pair is correct when both peaks have the same true source identity. Let C be the number of correct selected pairs, M the number of selected pairs, and T the number of true pairs present in both observed fingerprints.</p>
      <p><strong>Precision = C / M · Recall = C / T · F1 = 2C / (M + T).</strong> A source removed by measurement filtering is unavailable; a true pair excluded by size unification remains a missed match. A zero denominator is shown as undefined, never as 0%.</p>
      <p>The optional map accuracy counts every possible peak pair, including nonmatches: <strong>accuracy = (TP + TN) / (TP + FP + FN + TN)</strong>. TP = C, FP = M − C, FN = T − C, and TN = number of candidate peak pairs − TP − FP − FN. It can be high even when few true matches are recovered.</p>
    </details>
    <div class="spatial-map-heading"><h2>Matching quality from one survey point</h2><p>Click a point to use it as the reference. Hover another point to inspect the matching between them.</p></div>
    <div class="spatial-controls">
      <label>Reference survey point<select data-spatial="anchor" aria-label="Reference survey point"></select></label>
      <label>Color by<select data-spatial="metric"><option value="f1">F1</option><option value="precision">Precision · correct among selected matches</option><option value="recall">Recall · recovered true matches</option><option value="accuracy">Pairwise accuracy · includes nonmatches</option></select></label>
    </div>
    <div class="spatial-camera" role="group" aria-label="Map camera"><button data-camera="3d">3D</button><button data-camera="top">Top</button><button data-camera="side">Side</button><button data-camera="reset">Reset view</button><button data-camera="in" aria-label="Zoom in">+</button><button data-camera="out" aria-label="Zoom out">−</button></div>
    <div class="spatial-canvas-wrap"><canvas tabindex="0" aria-label="3D survey matching map. Click a point to select a reference. Drag or use arrow keys to rotate; scroll or use plus and minus to zoom."></canvas><div class="spatial-tooltip" hidden></div></div>
    <div class="spatial-legend"><strong data-spatial="legend">F1</strong><span>0%</span><i class="spatial-ramp"></i><span>100%</span><span><i class="spatial-grey"></i>Undefined / outside selected pairs</span><span><i class="spatial-anchor-key"></i>Reference</span><span>◆ AP</span></div>
    <p class="spatial-caption" data-spatial="map-note"></p>
    <div class="spatial-point-detail" data-spatial="point-detail" aria-live="polite">Choose a reference and hover another survey point.</div>
    <button class="button subtle" data-spatial="inspect" disabled>Inspect hovered pair below</button>
    <p class="spatial-caption">Drag to rotate · Shift-drag to pan · Scroll to zoom · Double-click a point to open its peak matrix against the reference. The population selector also controls which points receive colors.</p>`;

  const xyz = scene.survey_xyz, canvas = $("canvas"), ctx = canvas.getContext("2d");
  const opts = xyz.map((p, i) => `<option value="${i}">${ref(i)} · (${p.map(v => fmt(v)).join(", ")}) m</option>`).join("");
  $('[data-spatial="anchor"]').innerHTML = opts;
  let selected = { anchor: 0, metric: "f1", population: "all", bin: 0.5 };
  let data = null, settings = null, revision = 0, rows = new Map(), hovered = null, bins = [];
  const cache = new Map();
  let width = 800, height = 460, yaw = -0.65, pitch = 0.65, zoom = 1, pan = [0, 0], drag = null, screen = [];
  const ceiling = scene.ceiling_m ?? Math.max(...xyz.map(p => p[2]));
  const corners = scene.polygon_xy.flatMap(([x, y]) => [[x, y, 0], [x, y, ceiling]]);
  const bounds = [...corners, ...xyz, ...Object.values(scene.aps)];
  const center = [0, 1, 2].map(k => (Math.min(...bounds.map(p => p[k])) + Math.max(...bounds.map(p => p[k]))) / 2);
  const extent = Math.max(1, ...bounds.map(p => Math.hypot(...p.map((v, k) => v - center[k]))));

  function restore() {
    const q = new URLSearchParams(location.hash.slice(1)), a = Number(q.get("anchor"));
    selected.anchor = q.has("anchor") && Number.isInteger(a) && a >= 0 && a < xyz.length ? a : 0;
    selected.metric = Object.hasOwn(labels, q.get("color")) ? q.get("color") : "f1";
    selected.population = q.get("population") === "neighborhood" ? "neighborhood" : "all";
    selected.bin = [0.25, 0.5, 1, 2].includes(Number(q.get("bin"))) ? Number(q.get("bin")) : 0.5;
    for (const [id, key] of [["anchor", "anchor"], ["metric", "metric"], ["population", "population"], ["bin", "bin"]])
      $(`[data-spatial="${id}"]`).value = selected[key];
  }
  function eligible(i) { return selected.population === "all" || data.neighborhood[i]; }
  function counts(i) { return data[`${settings.matches}_counts`][i]; }
  function pointValue(id) {
    const i = rows.get(id);
    return data && i !== undefined && eligible(i) ? matchingMetrics(counts(i))[selected.metric] : null;
  }
  function binText(b, key) {
    return `${fmt(b.low)} ≤ distance < ${fmt(b.high)} m · ${labels[key]} ${pct(b[key])} · ${fmt(b.pairs, 0)} fingerprint pairs · ${fmt(b.tp, 0)} correct / ${fmt(b.selected, 0)} selected · ${fmt(b.truth, 0)} available true matches · ${fmt(b.fp, 0)} incorrect · ${fmt(b.fn, 0)} missed`;
  }
  function renderChart() {
    if (!data) return;
    bins = distanceBins(data, settings.matches, selected.population, selected.bin);
    const boundary = neighborhoodBoundary(data);
    const max = Math.max(selected.bin, ...bins.map(b => b.high), (boundary?.max || 0) + selected.bin * .1);
    const x = v => 58 + v / max * 780, y = v => 292 - v * 250;
    const marker = boundary ? `<g data-neighborhood-boundary="${boundary.max}"><path d="M${x(boundary.max)} 42V292" stroke="#79539a" stroke-width="2" stroke-dasharray="2 5"/><text x="${Math.min(x(boundary.max) + 6, 832)}" y="36" text-anchor="${x(boundary.max) > 590 ? 'end' : 'start'}" fill="#79539a">Farthest neighborhood pair: ${fmt(boundary.max, 3)} m</text><title>No receiver pair beyond ${fmt(boundary.max, 6)} m is considered by this neighborhood rule. Membership is defined by shared local groups, not by distance alone.</title></g>` : '';
    let paths = "", dots = "";
    for (const key of ["precision", "recall", "f1"]) {
      if (!$(`[data-series="${key}"]`).checked) continue;
      let d = "", last = -2;
      for (const b of bins) {
        if (b[key] == null) { last = -2; continue; }
        const cx = x((b.low + b.high) / 2), cy = y(b[key]);
        d += `${b.id === last + 1 ? "L" : "M"}${cx},${cy} `;
        last = b.id;
        dots += `<circle cx="${cx}" cy="${cy}" r="4" fill="${colors[key]}" stroke="white" stroke-width="1" tabindex="0" data-bin="${b.id}" data-key="${key}" aria-label="${escape(binText(b, key))}"><title>${escape(binText(b, key))}</title></circle>`;
      }
      paths += `<path d="${d}" fill="none" stroke="${colors[key]}" stroke-width="2.4"/>`;
    }
    $('[data-spatial="chart"]').innerHTML = `<svg viewBox="0 0 880 355" role="img" aria-label="Precision, recall and F1 versus 3D survey-point distance">
      ${[0, .25, .5, .75, 1].map(v => `<path d="M58 ${y(v)}H838" stroke="#e2e8eb"/><text x="48" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join("")}
      ${[0, 1, 2, 3, 4, 5, 6].map(i => `<text x="${x(i * max / 6)}" y="315" text-anchor="middle">${fmt(i * max / 6)}</text>`).join("")}
      <text x="58" y="22">Correct peak identities · ${selected.population === "all" ? "all survey pairs" : "MPUrge neighborhood pairs"}</text>
      ${paths}${marker}${dots}<text x="448" y="343" text-anchor="middle">Distance between survey points (m)</text></svg>`;
    $('[data-spatial="boundary"]').textContent = boundary ? `Dotted line: the farthest receiver pair actually considered (${fmt(boundary.max, 3)} m). Pairs are considered when both receivers belong to at least one common nearest-neighbor group, including boundary ties. All pairs beyond this line are excluded; ${fmt(boundary.excludedWithin, 0)} pairs at or below it are also excluded by group membership. The line uses exact pair distances, not distance-bin edges.` : 'There are no eligible neighborhood pairs for this configuration.';
    $('[data-spatial="bin-detail"]').textContent = `${fmt(bins.reduce((n, b) => n + b.pairs, 0), 0)} fingerprint pairs · ${selected.bin} m bins · Hover or focus a plotted point for counts. Empty bins and undefined metrics leave gaps.`;
  }
  function showPoint(id) {
    hovered = id;
    const i = rows.get(id), button = $('[data-spatial="inspect"]');
    button.disabled = !data || id == null || id === selected.anchor;
    if (!data || id == null) return "Select a reference and hover another survey point.";
    if (id === selected.anchor) {
      const text = `${ref(id)} · Reference · (${xyz[id].map(v => fmt(v, 3)).join(", ")}) m. Self-pairs are excluded.`;
      $('[data-spatial="point-detail"]').textContent = text;
      return text;
    }
    const m = matchingMetrics(counts(i));
    const text = `${ref(selected.anchor)} ↔ ${ref(id)} · ${fmt(data.distance_m[i], 3)} m apart · (${xyz[id].map(v => fmt(v, 3)).join(", ")}) m\nPrecision ${pct(m.precision)} · Recall ${pct(m.recall)} · F1 ${pct(m.f1)} · Accuracy ${pct(m.accuracy)}\n${m.tp} correct / ${m.selected} selected · ${m.truth} true matches available · ${m.fp} incorrect · ${m.fn} missed${eligible(i) ? "" : "\nOutside the selected neighborhood pairs; shown in grey."}`;
    $('[data-spatial="point-detail"]').textContent = text;
    return text;
  }
  function selectAnchor(id) {
    selected.anchor = id;
    $('[data-spatial="anchor"]').value = id;
    rows = data ? anchorRows(data, id) : new Map();
    hovered = null;
    $('[data-spatial="inspect"]').disabled = true;
    $('[data-spatial="point-detail"]').textContent = `${ref(id)} selected as reference. Hover a different point for matching counts.`;
    $('.spatial-tooltip').hidden = true;
    saveLocation(); renderMap();
  }
  function project(p) {
    const [x, y, z] = p.map((v, k) => v - center[k]);
    const horizontal = Math.cos(yaw) * x - Math.sin(yaw) * y;
    const depth = Math.sin(yaw) * x + Math.cos(yaw) * y;
    const scale = Math.min(width - 90, height - 70) / (2 * extent) * zoom;
    return [width / 2 + pan[0] + horizontal * scale,
      height / 2 + pan[1] + (Math.sin(pitch) * depth - Math.cos(pitch) * z) * scale,
      Math.cos(pitch) * depth + Math.sin(pitch) * z];
  }
  function line(a, b, color, lineWidth = 1) {
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.stroke();
  }
  function renderMap() {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f8fbfc"; ctx.fillRect(0, 0, width, height);
    const floor = scene.polygon_xy.map(([x, y]) => project([x, y, 0]));
    const roof = scene.polygon_xy.map(([x, y]) => project([x, y, ceiling]));
    floor.forEach((p, i) => { const j = (i + 1) % floor.length; line(p, floor[j], "#b8ced3"); line(roof[i], roof[j], "#d4e1e5"); line(p, roof[i], "#d4e1e5"); });
    screen = xyz.map((p, id) => ({ id, p: project(p) })).sort((a, b) => a.p[2] - b.p[2]);
    if (hovered != null && hovered !== selected.anchor) line(project(xyz[selected.anchor]), project(xyz[hovered]), "#768797", 1.5);
    for (const { id, p } of screen) {
      const anchor = id === selected.anchor;
      ctx.beginPath(); ctx.arc(p[0], p[1], anchor ? 7.5 : 5, 0, Math.PI * 2);
      ctx.fillStyle = anchor ? "#8e469c" : ramp(pointValue(id)); ctx.fill();
      ctx.strokeStyle = anchor ? "#56275f" : "#997d7d"; ctx.lineWidth = anchor ? 2 : .75; ctx.stroke();
      if (anchor || id === hovered) {
        ctx.beginPath(); ctx.arc(p[0], p[1], anchor ? 11 : 8, 0, Math.PI * 2);
        ctx.strokeStyle = anchor ? "#8e469c" : "#374b57"; ctx.stroke();
        ctx.fillStyle = "#243e4b"; ctx.font = "12px Segoe UI, sans-serif"; ctx.fillText(ref(id), p[0] + 13, p[1] - 9);
      }
    }
    const ap = scene.aps[settings?.ap];
    if (ap) {
      const p = project(ap); ctx.beginPath(); ctx.moveTo(p[0], p[1] - 7); ctx.lineTo(p[0] + 7, p[1]); ctx.lineTo(p[0], p[1] + 7); ctx.lineTo(p[0] - 7, p[1]); ctx.closePath(); ctx.fillStyle = "#243e4b"; ctx.fill(); ctx.fillText(`AP ${settings.ap}`, p[0] + 10, p[1]);
    }
    const origin = [34, height - 38];
    ["x", "y", "z"].forEach((name, k) => {
      const unit = [...center]; unit[k] += 1;
      const a = project(center), b = project(unit), dx = b[0] - a[0], dy = b[1] - a[1];
      const length = Math.max(1, Math.hypot(dx, dy));
      const end = [origin[0] + 25 * dx / length, origin[1] + 25 * dy / length];
      line(origin, end, ["#aa544d", "#087e70", "#2071aa"][k], 2);
      ctx.fillStyle = "#52666e"; ctx.fillText(name, end[0] + 3, end[1] - 3);
    });
    $('[data-spatial="legend"]').textContent = labels[selected.metric];
    const n = data ? [...rows.values()].filter(eligible).length : 0;
    $('[data-spatial="map-note"]').textContent = `${ref(selected.anchor)} reference · ${n} comparison points · ${labels[selected.metric]} on a fixed 0–100% scale.${selected.metric === "accuracy" ? " Accuracy includes correctly rejected nonmatches and can be high despite poor recall." : ""}`;
  }
  function refresh() { rows = data ? anchorRows(data, selected.anchor) : new Map(); renderChart(); renderMap(); }
  const observer = new ResizeObserver(() => {
    const rect = canvas.getBoundingClientRect(); width = rect.width; height = rect.height;
    const ratio = devicePixelRatio || 1; canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); renderMap();
  });
  observer.observe(canvas);
  const hit = e => {
    const box = canvas.getBoundingClientRect(), x = e.clientX - box.left, y = e.clientY - box.top;
    let best = null, distance = 12;
    for (const item of [...screen].reverse()) {
      const d = Math.hypot(x - item.p[0], y - item.p[1]);
      if (d < distance) { best = item.id; distance = d; }
    }
    return { id: best, x, y };
  };
  canvas.onpointerdown = e => { drag = { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, moved: false, anchor: selected.anchor }; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = e => {
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.moved ||= Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) > 4;
      if (drag.moved) {
        if (e.shiftKey) { pan[0] += dx; pan[1] += dy; }
        else { yaw += dx * .009; pitch = Math.max(-1.5, Math.min(Math.PI / 2, pitch + dy * .009)); }
        $('.spatial-tooltip').hidden = true;
      }
      drag.x = e.clientX; drag.y = e.clientY; renderMap(); return;
    }
    const { id, x, y } = hit(e), tip = $('.spatial-tooltip');
    if (id != null && data) { tip.textContent = showPoint(id); tip.hidden = false; tip.style.left = `${Math.max(8, Math.min(x + 16, width - 330))}px`; tip.style.top = `${Math.max(8, Math.min(y + 14, height - 145))}px`; }
    else tip.hidden = true;
    renderMap();
  };
  let clickTimer = null, clickAnchor = null;
  canvas.onpointerup = e => {
    if (drag && !drag.moved) {
      const { id } = hit(e);
      if (id != null) {
        clickAnchor = drag.anchor;
        clearTimeout(clickTimer);
        clickTimer = setTimeout(() => selectAnchor(id), 240);
      }
    }
    drag = null;
  };
  canvas.onpointercancel = () => { drag = null; };
  canvas.ondblclick = e => { clearTimeout(clickTimer); const { id } = hit(e); if (id != null && id !== clickAnchor) inspect(clickAnchor ?? selected.anchor, id); };
  canvas.onpointerleave = () => { $('.spatial-tooltip').hidden = true; };
  canvas.onwheel = e => { e.preventDefault(); zoom = Math.max(.3, Math.min(8, zoom * Math.exp(-e.deltaY * .001))); renderMap(); };
  canvas.onkeydown = e => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-"].includes(e.key)) return;
    e.preventDefault();
    if (e.key.startsWith("Arrow")) { yaw += (e.key === "ArrowRight" ? .1 : e.key === "ArrowLeft" ? -.1 : 0); pitch = Math.max(-1.5, Math.min(Math.PI / 2, pitch + (e.key === "ArrowUp" ? .1 : e.key === "ArrowDown" ? -.1 : 0))); }
    else zoom = Math.max(.3, Math.min(8, zoom * (e.key === "-" ? .85 : 1.15)));
    renderMap();
  };
  container.querySelectorAll('[data-camera]').forEach(b => b.onclick = () => {
    const mode = b.dataset.camera;
    if (["3d", "reset"].includes(mode)) { yaw = -.65; pitch = .65; }
    if (mode === "top") { yaw = 0; pitch = Math.PI / 2; }
    if (mode === "side") { yaw = 0; pitch = 0; }
    if (mode === "reset") { zoom = 1; pan = [0, 0]; }
    if (["in", "out"].includes(mode)) zoom = Math.max(.3, Math.min(8, zoom * (mode === "in" ? 1.2 : 1 / 1.2)));
    renderMap();
  });
  $('[data-spatial="anchor"]').onchange = e => selectAnchor(Number(e.target.value));
  for (const [id, key] of [["metric", "metric"], ["population", "population"], ["bin", "bin"]]) {
    $(`[data-spatial="${id}"]`).onchange = e => {
      selected[key] = key === "bin" ? Number(e.target.value) : e.target.value;
      $('.spatial-tooltip').hidden = true;
      saveLocation(); refresh();
      if (hovered != null) showPoint(hovered);
    };
  }
  container.querySelectorAll('[data-series]').forEach(input => input.onchange = renderChart);
  for (const event of ["pointerover", "focusin"]) $('[data-spatial="chart"]').addEventListener(event, e => {
    const p = e.target.closest('[data-bin]');
    if (p) $('[data-spatial="bin-detail"]').textContent = binText(bins.find(b => b.id === Number(p.dataset.bin)), p.dataset.key);
  });
  $('[data-spatial="inspect"]').onclick = () => { if (hovered != null && hovered !== selected.anchor) inspect(selected.anchor, hovered); };
  $('[data-spatial="export"]').onclick = () => {
    const records = [["ap", "noise_sigma_m", "stage", "population", "distance_low_m", "distance_high_exclusive_m", "pairs", "correct", "selected", "true", "precision", "recall", "f1"],
      ...bins.map(b => [settings.ap, settings.noise, settings.matches, selected.population, b.low, b.high, b.pairs, b.tp, b.selected, b.truth, b.precision, b.recall, b.f1])];
    const url = URL.createObjectURL(new Blob([records.map(r => r.join(",")).join("\n")], { type: "text/csv" }));
    const a = document.createElement("a"); a.href = url; a.download = `mpurge-distance-${settings.ap}-${settings.noise}-${settings.matches}-${selected.population}.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  restore();
  return {
    state: () => ({ anchor: selected.anchor, color: selected.metric, population: selected.population, bin: selected.bin }),
    async update(next) {
      restore(); settings = { ...next };
      const current = ++revision, key = `${next.ap}:${next.noise}`;
      data = null; rows = new Map(); hovered = null;
      $('.spatial-tooltip').hidden = true;
      $('[data-spatial="inspect"]').disabled = true;
      $('[data-spatial="export"]').disabled = true;
      $('[data-spatial="chart"]').textContent = "Calculating MPUrge matches for every survey pair…";
      $('[data-spatial="boundary"]').textContent = '';
      $('[data-spatial="condition"]').textContent = `AP ${next.ap} · σ ${next.noise} m · ${next.matches === "final" ? "Final matches (all rounds)" : "First-round matches"} · Empty room · Loading…`;
      $('[data-spatial="point-detail"]').textContent = "Loading the selected condition…";
      renderMap();
      try {
        if (!cache.has(key)) cache.set(key, request("spatial", { ap: next.ap, noise: next.noise }).catch(e => { cache.delete(key); throw e; }));
        const result = await cache.get(key);
        if (current !== revision) return;
        data = result;
        while (cache.size > 3) cache.delete(cache.keys().next().value);
        $('[data-spatial="condition"]').textContent = `AP ${next.ap} · σ ${next.noise} m · ${next.matches === "final" ? "Final matches (all rounds)" : "First-round matches"} · Empty room · Saved selected parameters`;
        $('[data-spatial="export"]').disabled = false;
        $('[data-spatial="point-detail"]').textContent = "Choose a reference and hover another survey point.";
        refresh();
      } catch (e) {
        if (current !== revision) return;
        $('[data-spatial="chart"]').textContent = `Spatial diagnostic unavailable: ${e.message}`;
        $('[data-spatial="condition"]').textContent = "Could not load this matching condition.";
      }
    },
  };
}
