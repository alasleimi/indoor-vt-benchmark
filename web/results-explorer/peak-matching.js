const $ = (s) => document.querySelector(s);
const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const num = (x, d = 2) =>
  x == null
    ? "—"
    : Number(x).toLocaleString("en-US", {
        minimumFractionDigits: d,
        maximumFractionDigits: d,
      });
const pct = (x) => num(x * 100, 2) + "%";
const ref = (i) => "S" + String(i + 1).padStart(3, "0");
const worker = new Worker(new URL("./peak-worker.js", import.meta.url), {
  type: "module",
});
let serial = 0,
  version = 0,
  pairVersion = 0,
  meta,
  scene,
  condition,
  pair,
  ordered = [];
const pending = new Map(),
  pairIds = new Map();
const state = { ap: "A", noise: 0, left: 0, right: 1, matches: "first" };
const finalOverlay = () => state.matches === "final";
const selectedMatches = () =>
  finalOverlay() ? pair.final.matches : pair.matrix.matches;
function request(action, args = {}) {
  return new Promise((resolve, reject) => {
    const id = ++serial;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, action, args });
  });
}
worker.onmessage = ({ data }) => {
  const p = pending.get(data.id);
  if (!p) return;
  pending.delete(data.id);
  data.error ? p.reject(Error(data.error)) : p.resolve(data.result);
};
worker.onerror = () => {
  for (const p of pending.values())
    p.reject(Error("The matching worker could not load. Refresh the page."));
  pending.clear();
};
function error(e) {
  $("#matching-error").textContent = e.message;
  $("#matching-error").hidden = false;
  $("#matrix-scroll").setAttribute("aria-busy", "false");
}
function saveLocation() {
  history.replaceState(
    null,
    "",
    location.pathname + "#" + new URLSearchParams(state),
  );
}
function readLocation() {
  const q = new URLSearchParams(location.hash.slice(1));
  state.matches = q.get("matches") === "final" ? "final" : "first";
  state.ap = ["A", "B"].includes(q.get("ap")) ? q.get("ap") : "A";
  state.noise = meta.noise_levels.includes(Number(q.get("noise")))
    ? Number(q.get("noise"))
    : 0;
  for (const name of ["left", "right"]) {
    const n = Number(q.get(name));
    state[name] =
      q.has(name) &&
      Number.isInteger(n) &&
      n >= 0 &&
      n < scene.survey_xyz.length
        ? n
        : name === "left"
          ? 0
          : 1;
  }
  if (state.left === state.right)
    state.right = (state.left + 1) % scene.survey_xyz.length;
  [state.left, state.right] = [state.left, state.right].sort((a, b) => a - b);
}
function enablePairs(value) {
  for (const id of [
    "reference-left",
    "reference-right",
    "pair-order",
    "previous-pair",
    "next-pair",
    "export-pair",
    "match-overlay",
  ])
    $("#" + id).disabled = !value;
}
function metric(label, value) {
  return `<div class="match-metric"><span>${esc(label)}</span><strong>${value}</strong></div>`;
}
function renderRoom() {
  const m = condition.summary;
  $("#room-summary-note").textContent =
    `AP ${state.ap} · σ = ${state.noise} m · All ${num(m.pairs, 0)} unique pairs in the room’s nearest-10 neighborhoods · Round 1 only`;
  $("#room-metrics").innerHTML = [
    ["Precision", pct(m.precision)],
    ["Recall", pct(m.recall)],
    ["F1", pct(m.f1)],
    ["Correct matches", num(m.tp, 0)],
    ["Incorrect matches", num(m.fp, 0)],
    ["Missed correspondences", num(m.fn, 0)],
  ]
    .map(([k, v]) => metric(k, v))
    .join("");
  $("#correlation-note").innerHTML =
    `<strong>Correlation of δ with same-source truth:</strong> ${num(m.candidate_correlation, 3)} across all ${num(m.candidate_cells, 0)} candidate cells; ${num(m.accepted_correlation, 3)} among accepted matches. Negative is the intended direction. These populations differ: an absolute δ is not a calibrated probability of a correct accepted match.`;
  const max = Math.max(0.25, ...condition.bins.map((b) => b[0] + 0.25));
  const x = (v) => 45 + (v / max) * 400,
    y = (v) => 205 - v * 1.7;
  const points = condition.bins.map((b) => [b[1], (100 * b[3]) / b[2], b]);
  const path = points
    .map(([a, b], i) => `${i ? "L" : "M"}${x(a)},${y(b)}`)
    .join(" ");
  const baseline = (100 * m.candidate_same_source) / m.candidate_cells;
  $("#reliability-chart").innerHTML =
    `<svg viewBox="0 0 480 260" role="img" aria-label="First-round dissimilarity versus percentage of candidate peak pairs sharing the same true source"><text x="45" y="18" style="font-size:12px;fill:#203c46">First-round δ → same-source percentage</text>${[0, 25, 50, 75, 100].map((v) => `<path d="M45 ${y(v)}H445" stroke="#e5ebee"/><text x="36" y="${y(v) + 3}" text-anchor="end">${v}%</text>`).join("")}<path d="M45 ${y(baseline)}H445" stroke="#7d8994" stroke-dasharray="4 3"/><path d="${path}" stroke="#087e70" stroke-width="1.5"/>${points.map(([a, b, v]) => `<circle cx="${x(a)}" cy="${y(b)}" r="2" fill="${v[2] < 100 ? "#9ca3af" : "#087e70"}"><title>δ ${num(v[0])}–${num(v[0] + 0.25)}: ${num(b)}% same source; ${num(v[2], 0)} candidate cells</title></circle>`).join("")}${[0, 0.25, 0.5, 0.75, 1].map((f) => `<text x="${x(f * max)}" y="225" text-anchor="middle">${num(f * max, 1)}</text>`).join("")}<text x="245" y="246" text-anchor="middle">δ (m) · 0.25-wide bins · all candidate cells</text></svg>`;
}
function orderPairs() {
  ordered = scene.pairs.map((_, i) => i);
  const f1 = (i) => {
    const [tp, n, t] = condition.pair_counts[i];
    return n + t ? (2 * tp) / (n + t) : 0;
  };
  const mode = $("#pair-order").value;
  if (mode === "errors")
    ordered.sort(
      (a, b) =>
        condition.pair_counts[b][1] -
          condition.pair_counts[b][0] -
          (condition.pair_counts[a][1] - condition.pair_counts[a][0]) || a - b,
    );
  if (mode === "worst" || mode === "best")
    ordered.sort(
      (a, b) => (f1(a) - f1(b)) * (mode === "best" ? -1 : 1) || a - b,
    );
}
async function loadCondition() {
  const current = ++version;
  ++pairVersion;
  $("#matching-error").hidden = true;
  enablePairs(false);
  $("#export-room").disabled = true;
  $("#matrix-scroll").setAttribute("aria-busy", "true");
  $("#matrix-content").replaceChildren();
  $("#room-metrics").replaceChildren();
  $("#reliability-chart").replaceChildren();
  $("#pair-location").replaceChildren();
  $("#room-summary-note").textContent =
    "Loading selected AP and noise condition…";
  $("#pair-status").textContent = "Loading fingerprints…";
  $("#pair-metrics").replaceChildren();
  $("#cell-detail").textContent = "Loading selected condition…";
  $("#match-ap").value = state.ap;
  $("#match-noise").value = String(state.noise);
  try {
    const result = await request("condition", state);
    if (current !== version) return;
    condition = result;
    renderRoom();
    orderPairs();
    $("#export-room").disabled = false;
    await loadPair();
  } catch (e) {
    if (current === version) error(e);
  }
}
async function loadPair() {
  const current = ++pairVersion;
  $("#export-pair").disabled = true;
  $("#match-overlay").disabled = true;
  [state.left, state.right] = [state.left, state.right].sort((a, b) => a - b);
  $("#reference-left").value = String(state.left);
  $("#reference-right").value = String(state.right);
  $("#matrix-scroll").setAttribute("aria-busy", "true");
  saveLocation();
  try {
    const result = await request("pair", state);
    if (current !== pairVersion) return;
    pair = result;
    renderPair();
    enablePairs(true);
    $("#matrix-scroll").setAttribute("aria-busy", "false");
  } catch (e) {
    if (current === pairVersion) error(e);
  }
}
function sceneSVG() {
  const points = scene.polygon_xy,
    xs = points.map((p) => p[0]),
    ys = points.map((p) => p[1]);
  const minX = Math.min(...xs),
    maxX = Math.max(...xs),
    minY = Math.min(...ys),
    maxY = Math.max(...ys);
  const scale = Math.min(190 / (maxX - minX), 110 / (maxY - minY));
  const xy = (p) => [25 + (p[0] - minX) * scale, 125 - (p[1] - minY) * scale];
  const a = xy(pair.a.xyz),
    b = xy(pair.b.xyz),
    ap = xy(scene.aps[state.ap]);
  const surveyXY = [
    ...new Map(
      scene.survey_xyz.map((p) => [p.slice(0, 2).join(","), p]),
    ).values(),
  ];
  return `<svg viewBox="0 0 240 145" role="img" aria-label="Top view showing the two survey positions and selected physical AP"><defs><pattern id="survey-grid" width="${scale}" height="${scale}" patternUnits="userSpaceOnUse"><path d="M${scale} 0H0V${scale}" fill="none" stroke="#e9eef1" stroke-width="1"/></pattern></defs><rect width="240" height="145" fill="url(#survey-grid)"/><polygon points="${points
    .map(xy)
    .map((p) => p.join(","))
    .join(" ")}" fill="#ecf5f3" stroke="#91aaa8"/>${surveyXY
    .map((p) => {
      const [x, y] = xy(p);
      return `<circle cx="${x}" cy="${y}" r="1" fill="#bed0d5"/>`;
    })
    .join(
      "",
    )}<path d="M${a.join(" ")}L${b.join(" ")}" stroke="#516879"/><circle cx="${a[0]}" cy="${a[1]}" r="4" fill="#3b6c8c"/><rect x="${b[0] - 4}" y="${b[1] - 4}" width="8" height="8" fill="#9b5c2c"/><circle cx="${ap[0]}" cy="${ap[1]}" r="3" fill="#087e70"/><text x="${a[0] + 7}" y="${a[1] - 5}" font-size="9">${ref(state.left)}</text><text x="${b[0] + 7}" y="${b[1] + 12}" font-size="9">${ref(state.right)}</text><text x="${ap[0] + 5}" y="${ap[1] - 5}" font-size="9">AP ${state.ap}</text></svg>`;
}
function renderPair() {
  const { a, b, matrix: m } = pair;
  const q = finalOverlay() ? pair.finalQuality : pair.quality;
  const matches = selectedMatches();
  $("#match-overlay").value = state.matches;
  const id = pairIds.get(`${state.left},${state.right}`),
    position = ordered.indexOf(id);
  const separation = Math.hypot(...a.xyz.map((x, k) => x - b.xyz[k]));
  $("#pair-status").textContent =
    `${ref(state.left)} ↔ ${ref(state.right)} · ${num(separation)} m apart · ${id === undefined ? "Custom pair: outside the room-summary neighborhoods" : `Room pair ${position + 1} of ${num(scene.pairs.length, 0)}`}`;
  $("#pair-metrics").innerHTML = [
    ["Pair precision", pct(q.precision)],
    ["Pair recall", pct(q.recall)],
    ["Pair F1", pct(q.f1)],
    ["Correct / incorrect", `${q.tp} / ${q.fp}`],
    [finalOverlay() ? "Missed after all rounds" : "Missed in round 1", q.fn],
  ]
    .map(([k, v]) => `<div><strong>${v}</strong><span>${k}</span></div>`)
    .join("");
  $("#pair-location").innerHTML = sceneSVG();
  const accepted = new Set(matches.map(([i, j]) => `${i},${j}`));
  const max = Math.max(1e-12, ...m.values.filter((x) => x != null));
  const maxGainA = Math.max(...a.gains),
    maxGainB = Math.max(...b.gains);
  const bar = (gain, maxGain) =>
    Math.max(3, 45 * Math.max(0, 1 - (maxGain - gain) / 40));
  let html = `<table class="peak-matrix" aria-label="First-round peak-pair dissimilarities"><thead><tr><th class="corner" scope="col">${ref(state.left)} →<br/>${ref(state.right)} ↓<br/>Range (m)</th>`;
  for (let i = 0; i < a.ranges.length; i++)
    html += `<th scope="col" class="${m.ia.includes(i) ? "" : "excluded-peak"}" title="${ref(state.left)} peak ${i + 1}; native gain ${num(a.gains[i])} dB"><span>P${i + 1}</span><div class="header-stem"><i style="height:${bar(a.gains[i], maxGainA)}px"></i></div><span class="peak-range">${num(a.ranges[i])}</span></th>`;
  html += "</tr></thead><tbody>";
  for (let j = 0; j < b.ranges.length; j++) {
    html += `<tr><th scope="row" class="row-peak ${m.ib.includes(j) ? "" : "excluded-peak"}" title="${ref(state.right)} peak ${j + 1}; native gain ${num(b.gains[j])} dB"><div class="row-peak-inner"><div class="row-stem"><i style="width:${bar(b.gains[j], maxGainB)}px"></i></div><span>P${j + 1} · ${num(b.ranges[j])}</span></div></th>`;
    for (let i = 0; i < a.ranges.length; i++) {
      const value = m.values[i * b.ranges.length + j],
        matched = accepted.has(`${i},${j}`),
        truth = a.source_ids[i] === b.source_ids[j];
      const shade = value == null ? 1 : Math.min(1, value / max),
        r = Math.round(162 + 85 * shade),
        g = Math.round(201 + 50 * shade),
        blue = Math.round(227 + 26 * shade);
      const label = `Horizontal peak ${i + 1}, vertical peak ${j + 1}; ${value == null ? "not scored" : `first-round delta ${num(value, 6)}`}; ${matched ? "accepted match" : "not accepted"} (${finalOverlay() ? "final matching" : "first round"}); ${truth ? "same true source" : "different true sources"}`;
      html += `<td class="${value == null ? "excluded" : ""}"><button type="button" class="${matched ? "matched " : ""}${truth ? "true-pair" : ""}" style="background:rgb(${r},${g},${blue})" data-i="${i}" data-j="${j}" tabindex="${i === 0 && j === 0 ? 0 : -1}" aria-label="${label}" title="${label}"><span class="cell-value">${value == null ? "—" : num(value, 2)}</span></button></td>`;
    }
    html += "</tr>";
  }
  $("#matrix-content").innerHTML = html + "</tbody></table>";
  $("#heatmap-range").textContent =
    `Round-1 δ = 0 → ${num(max)} (lower is better)`;
  const excluded =
    a.ranges.length - m.ia.length + b.ranges.length - m.ib.length;
  $("#matrix-caption").textContent =
    `Circles: ${finalOverlay() ? "final matches (all rounds)" : "first-round matches"}. Values and colors: first-round δ. ${a.ranges.length} horizontal peaks × ${b.ranges.length} vertical peaks · ${excluded} excluded by size unification. Peaks are ordered by range; header spacing is uniform. Bars show native gain on a relative dB scale.`;
  const first = matches[0] ?? [0, 0];
  selectCell(first[0], first[1], false);
}
function sourceLabel(id) {
  const s = pair.sources[id];
  return `${s.id} · ${s.surface_sequences.map((seq) => (seq.length ? seq.join(" → ") : "direct AP")).join(" / ")}`;
}
function selectCell(i, j, focus = false) {
  const target = $(`#matrix-content button[data-i="${i}"][data-j="${j}"]`);
  if (!target) return;
  const old = $("#matrix-content button.selected");
  if (old) {
    old.classList.remove("selected");
    old.tabIndex = -1;
  }
  target.classList.add("selected");
  target.tabIndex = 0;
  if (focus) target.focus();
  const { a, b, matrix: m } = pair,
    value = m.values[i * b.ranges.length + j],
    pattern = m.pattern[i * b.ranges.length + j];
  const matched = selectedMatches().some(([x, y]) => x === i && y === j),
    rejected = m.rejected.some(([x, y]) => x === i && y === j),
    same = a.source_ids[i] === b.source_ids[j];
  const finalIndex = pair.final.matches.findIndex(
    ([x, y]) => x === i && y === j,
  );
  const finalRejected = pair.final.rejected.find(
    ([x, y]) => x === i && y === j,
  );
  let decision = matched
    ? `Accepted in round ${finalOverlay() ? pair.final.rounds[finalIndex] : 1}`
    : finalOverlay()
      ? value == null
        ? "Excluded by size unification"
        : finalRejected
          ? `Reciprocal candidate rejected by the order check in round ${finalRejected[2]}`
          : "Not accepted in the final matching"
      : rejected
        ? "Reciprocal candidate rejected by the order check"
        : value == null
          ? "Excluded by size unification"
          : "Not a reciprocal-best candidate";
  $("#cell-detail").innerHTML =
    `<h3>${ref(state.left)} peak ${i + 1} ↔ ${ref(state.right)} peak ${j + 1} · ${same ? "Same true source" : "Different true sources"}</h3><p>${decision}.</p><div class="cell-facts">${[
      [a, i],
      [b, j],
    ]
      .map(
        ([p, k]) =>
          `<div><strong>${ref(p.reference)}</strong> at (${p.xyz.map((x) => num(x)).join(", ")}) m<br/>Range ${num(p.ranges[k], 6)} m · delay ${num(p.ranges[k] / 0.3, 3)} ns · native gain ${num(p.gains[k], 2)} dB<br/>${esc(sourceLabel(p.source_ids[k]))}</div>`,
      )
      .join(
        "",
      )}</div>${value == null ? "<p>No dissimilarity was calculated for this cell.</p>" : `<p class="score-formula">First-round δ = 0.5 × ${num(Math.abs(a.ranges[i] - b.ranges[j]), 6)} + 0.5 × ${num(pattern, 6)} = <strong>${num(value, 6)}</strong><br/><small>Absolute range difference + unnormalized pattern discrepancy. The pattern uses five-peak windows after size unification. This stays at round 1 even for matches accepted later.</small></p>`}`;
}
function download(name, fields, rows) {
  const quote = (x) => '"' + String(x ?? "").replaceAll('"', '""') + '"';
  const body = [fields, ...rows]
    .map((r) => r.map(quote).join(","))
    .join("\r\n");
  const url = URL.createObjectURL(
    new Blob([body], { type: "text/csv;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("#export-room").onclick = () => {
  const fields = [
    "ap",
    "noise_sigma_m",
    "round",
    "pairs",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "candidate_correlation",
    "accepted_correlation",
  ];
  download(
    "rectangle-01-first-round-matching.csv",
    fields,
    meta.conditions.map((c) =>
      fields.map((k) =>
        k === "ap"
          ? c.ap
          : k === "noise_sigma_m"
            ? c.noise
            : k === "round"
              ? 1
              : c.summary[k],
      ),
    ),
  );
};
$("#export-pair").onclick = () => {
  const { a, b, matrix: m } = pair,
    rows = [];
  for (let i = 0; i < a.ranges.length; i++)
    for (let j = 0; j < b.ranges.length; j++)
      rows.push([
        meta.room,
        state.ap,
        state.noise,
        1,
        state.matches,
        a.reference,
        b.reference,
        i + 1,
        j + 1,
        a.ranges[i],
        b.ranges[j],
        m.values[i * b.ranges.length + j],
        a.source_ids[i] === b.source_ids[j],
        selectedMatches().some(([x, y]) => x === i && y === j),
        m.matches.some(([x, y]) => x === i && y === j),
        pair.final.matches.some(([x, y]) => x === i && y === j),
        pair.final.rounds[
          pair.final.matches.findIndex(([x, y]) => x === i && y === j)
        ] ?? "",
        sourceLabel(a.source_ids[i]),
        sourceLabel(b.source_ids[j]),
      ]);
  download(
    `matching-${state.matches}-${state.ap}-${state.noise}-${ref(a.reference)}-${ref(b.reference)}.csv`,
    [
      "room",
      "ap",
      "noise_sigma_m",
      "score_round",
      "matching_scope",
      "horizontal_survey_index",
      "vertical_survey_index",
      "horizontal_peak",
      "vertical_peak",
      "horizontal_range_m",
      "vertical_range_m",
      "delta",
      "same_true_source",
      "accepted",
      "accepted_first_round",
      "accepted_final",
      "accepted_in_round",
      "horizontal_source",
      "vertical_source",
    ],
    rows,
  );
};
$("#copy-matching").onclick = async () => {
  try {
    await navigator.clipboard.writeText(location.href);
    $("#copy-matching").textContent = "Link copied";
    setTimeout(
      () => ($("#copy-matching").textContent = "Copy view link ↗"),
      1400,
    );
  } catch {
    $("#copy-matching").textContent = "Copy this page’s address";
  }
};
$("#match-overlay").onchange = () => {
  const selected = $("#matrix-content button.selected");
  const cell = selected
    ? [Number(selected.dataset.i), Number(selected.dataset.j)]
    : null;
  state.matches = $("#match-overlay").value;
  saveLocation();
  renderPair();
  if (cell) selectCell(...cell);
};
for (const id of ["match-ap", "match-noise"])
  $("#" + id).onchange = () => {
    state.ap = $("#match-ap").value;
    state.noise = Number($("#match-noise").value);
    loadCondition();
  };
for (const name of ["left", "right"])
  $("#reference-" + name).onchange = () => {
    const value = Number($("#reference-" + name).value),
      other = name === "left" ? "right" : "left";
    if (value === state[other]) {
      $("#reference-" + name).value = String(state[name]);
      return;
    }
    state[name] = value;
    loadPair();
  };
$("#pair-order").onchange = () => {
  orderPairs();
  [state.left, state.right] = scene.pairs[ordered[0]];
  loadPair();
};
for (const [id, step] of [
  ["previous-pair", -1],
  ["next-pair", 1],
])
  $("#" + id).onclick = () => {
    const position = ordered.indexOf(
      pairIds.get(`${state.left},${state.right}`),
    );
    const next = (position + step + ordered.length) % ordered.length;
    [state.left, state.right] = scene.pairs[ordered[next]];
    loadPair();
  };
for (const [id, cls] of [
  ["show-matches", "hide-matches"],
  ["show-truth", "hide-truth"],
  ["show-values", "hide-values"],
])
  $("#" + id).onchange = () =>
    $("#matrix-content").classList.toggle(cls, !$("#" + id).checked);
$("#matrix-content").onclick = (e) => {
  const b = e.target.closest("button[data-i]");
  if (b) selectCell(Number(b.dataset.i), Number(b.dataset.j));
};
$("#matrix-content").onfocusin = (e) => {
  const b = e.target.closest("button[data-i]");
  if (b) selectCell(Number(b.dataset.i), Number(b.dataset.j));
};
$("#matrix-content").onkeydown = (e) => {
  const b = e.target.closest("button[data-i]");
  if (!b) return;
  const moves = {
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0],
    ArrowUp: [0, -1],
    ArrowDown: [0, 1],
  };
  if (!moves[e.key]) return;
  e.preventDefault();
  selectCell(
    Number(b.dataset.i) + moves[e.key][0],
    Number(b.dataset.j) + moves[e.key][1],
    true,
  );
};
window.addEventListener("hashchange", () => {
  if (meta) {
    readLocation();
    loadCondition();
  }
});
try {
  ({ meta, scene } = await request("init"));
  scene.pairs.forEach((p, i) => pairIds.set(p.join(","), i));
  $("#match-noise").innerHTML = meta.noise_levels
    .map((n) => `<option value="${n}">σ = ${n} m</option>`)
    .join("");
  const opts = scene.survey_xyz
    .map(
      (p, i) =>
        `<option value="${i}">${ref(i)} · (${p.map((x) => num(x, 2)).join(", ")}) m</option>`,
    )
    .join("");
  $("#reference-left").innerHTML = opts;
  $("#reference-right").innerHTML = opts;
  $("#match-ap").disabled = false;
  $("#match-noise").disabled = false;
  readLocation();
  await loadCondition();
} catch (e) {
  error(e);
}
