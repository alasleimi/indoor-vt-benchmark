"use strict";
let methodCatalog;
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const fmt = (x, n = 4) =>
  x == null
    ? "—"
    : Number(x).toLocaleString("en-US", {
        minimumFractionDigits: n,
        maximumFractionDigits: n,
      });
const integer = (x) => Number(x).toLocaleString("en-US");
const defaults = {
  view: "vt",
  track: "receiver",
  family: "",
  room: "",
  noise: "",
  ap: "",
  obstacles: "",
  layout: "",
  scope: "",
  merger: "",
  stage: "",
  q: "",
  sort: "",
  direction: "asc",
};
let state = { ...defaults },
  meta,
  rows = [],
  request,
  detailRequest,
  detail,
  detailTab = "overview",
  lastFocus;
const viewText = {
  vt: [
    "VT reconstruction",
    "Compare reconstructed source maps across the same survey conditions.",
  ],
  raw: [
    "Raw localization",
    "Compare initial position estimates before SMART-LEA refinement.",
  ],
  all: [
    "All localization",
    "Compare raw baselines and complete pipelines with SMART-LEA.",
  ],
};
const sortNames = {
  ospa: "OSPA",
  f1: "F1 at 0.5 m",
  mean: "mean 3D error",
  rmse: "root mean squared 3D error",
  median: "median 3D error",
  p90: "90th-percentile error",
  predicted: "predicted source count",
  truth: "true source count",
  count: "case count",
  initial: "initial mean error",
  improvement: "improvement",
  fallbacks: "fallback count",
  name: "method name",
};
function readLocation() {
  const query = new URLSearchParams(location.hash.slice(1));
  state = { ...defaults };
  for (const k of Object.keys(defaults))
    if (query.has(k)) state[k] = query.get(k);
  if (!viewText[state.view]) state.view = "vt";
  if (!["direct", "receiver"].includes(state.track)) state.track = "receiver";
  if (!["asc", "desc"].includes(state.direction)) state.direction = "asc";
}
function saveLocation() {
  const query = new URLSearchParams();
  for (const [k, v] of Object.entries(state))
    if (v !== defaults[k] && v !== "") query.set(k, v);
  history.replaceState(
    null,
    "",
    location.pathname + (query.size ? "#" + query : ""),
  );
}
function options(select, values, label) {
  const first = select.options[0];
  select.replaceChildren(first);
  for (const value of values) {
    const o = document.createElement("option");
    o.value = String(value);
    o.textContent = label(value);
    select.append(o);
  }
}
function roomLabel(id) {
  const room = meta.rooms.find((r) => r.id === id);
  return room
    ? `${room.family === "rectangle" ? "Rectangle" : room.family} ${String(room.index + 1).padStart(2, "0")}`
    : id;
}
function populateFilters() {
  const rooms = meta.rooms.filter(
    (r) => !state.family || r.family === state.family,
  );
  if (state.room && !rooms.some((r) => r.id === state.room)) state.room = "";
  options(
    $("#room"),
    rooms.map((r) => r.id),
    roomLabel,
  );
  $("#room").options[0].textContent = `All ${rooms.length} rooms`;
  const levels = meta.levels[state.track];
  if (state.noise && !levels.some((n) => String(n) === state.noise))
    state.noise = "";
  options($("#noise"), levels, (n) =>
    state.track === "receiver" ? `${n} dB SNR` : `σ = ${n} m`,
  );
  const layoutMap = meta.rooms[0].layouts;
  const layouts = Object.keys(layoutMap).filter(
    (l) => !state.obstacles || String(layoutMap[l]) === state.obstacles,
  );
  if (state.layout && !layouts.includes(state.layout)) state.layout = "";
  options(
    $("#layout"),
    layouts,
    (l) => `${String(l).padStart(2, "0")} · ${layoutMap[l]} obstacles`,
  );
  $("#layout").options[0].textContent = `All ${layouts.length} layouts`;
  for (const k of [
    "family",
    "room",
    "noise",
    "ap",
    "obstacles",
    "layout",
    "scope",
    "merger",
    "stage",
  ])
    $("#" + k).value = state[k];
  $("#search").value = state.q;
}
function syncView() {
  populateFilters();
  const [title, description] = viewText[state.view];
  $("#page-title").textContent = title;
  $("#page-description").textContent = description;
  document.title = `${title} · Indoor VT`;
  $$("[data-view]").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === state.view);
    b.setAttribute(
      "aria-current",
      b.dataset.view === state.view ? "page" : "false",
    );
  });
  $$("[data-track]").forEach((b) => {
    b.classList.toggle("selected", b.dataset.track === state.track);
    b.setAttribute("aria-pressed", b.dataset.track === state.track);
  });
  $$(".local-filter").forEach((e) => (e.hidden = state.view === "vt"));
  $$(".vt-filter").forEach((e) => (e.hidden = state.view !== "vt"));
  $$(".all-filter").forEach((e) => (e.hidden = state.view !== "all"));
  $("#track-note").textContent =
    state.track === "receiver"
      ? "Noise: receiver SNR in dB"
      : "Noise: Gaussian range σ in meters";
  const conditions = [
    state.family,
    state.room && roomLabel(state.room),
    state.noise &&
      (state.track === "receiver"
        ? state.noise + " dB SNR"
        : "σ = " + state.noise + " m"),
    state.ap && "AP " + state.ap,
  ];
  if (state.view !== "vt")
    conditions.push(
      state.obstacles && state.obstacles + " obstacles",
      state.layout && "Layout " + state.layout,
    );
  $("#filter-summary").textContent =
    conditions.filter(Boolean).join(" · ") ||
    "All conditions · equal weight per case";
  $("#metric-note").textContent =
    state.view === "vt"
      ? "OSPA: p = 1, c = 1 m. F1: one-to-one matching at 0.5 m, averaged over cases."
      : "Exact statistics over selected query cases. Fallback estimates are included.";
}
async function loadResults() {
  closeDetail();
  syncView();
  saveLocation();
  request?.abort();
  request = new AbortController();
  $("#results-panel").setAttribute("aria-busy", "true");
  $("#export").disabled = true;
  $("#error").hidden = true;
  try {
    const data = await resultsRequest("results", state, request.signal);
    rows = data.rows;
    render();
    $("#results-panel").setAttribute("aria-busy", "false");
    $("#export").disabled = !rows.length;
  } catch (error) {
    if (error.name !== "AbortError") {
      $("#error").textContent = error.message;
      $("#error").hidden = false;
      $("#results-panel").setAttribute("aria-busy", "false");
    }
  }
}
function activeSort() {
  const allowed =
    state.view === "vt"
      ? ["ospa", "f1", "predicted", "truth", "count", "name"]
      : [
          "mean",
          "rmse",
          "median",
          "p90",
          "initial",
          "improvement",
          "fallbacks",
          "count",
          "name",
        ];
  return allowed.includes(state.sort)
    ? state.sort
    : state.view === "vt"
      ? "ospa"
      : "mean";
}
function sortedRows() {
  const key = activeSort(),
    sign = state.direction === "desc" ? -1 : 1;
  return [...rows].sort((a, b) => {
    if (a[key] == null) return b[key] == null ? a.id.localeCompare(b.id) : 1;
    if (b[key] == null) return -1;
    return (
      sign *
        (typeof a[key] === "number"
          ? a[key] - b[key]
          : String(a[key]).localeCompare(String(b[key]))) ||
      a.id.localeCompare(b.id)
    );
  });
}
function card(label, value, unit, caption, accent = false) {
  return `<div class="stat-card ${accent ? "accent" : ""}"><div class="stat-label">${esc(label)}</div><div class="stat-value">${value}<small>${esc(unit)}</small></div><div class="stat-caption" title="${esc(caption)}">${esc(caption)}</div></div>`;
}
function variantLabel(row) {
  if (!row) return "No matching results";
  return [
    row.name,
    row.stage === "pipeline" ? `+ SMART-LEA · VT: ${row.reconstruction}` : "",
    row.scope === "full" ? "Full room" : row.scope === "local" ? "Local" : "",
    row.merger ? meta.mergers[row.merger] : "",
  ]
    .filter(Boolean)
    .join(" · ");
}
function methodComponents(job) {
  if (job.stage !== "pipeline")
    return [{ key: `${job.stage}/${job.method}`, label: job.name }];
  const [initial, reconstruction] = job.method.split("__");
  return [
    { key: "pipeline", label: "Pipeline composition" },
    { key: `coarse/${initial}`, label: `Initial localization: ${job.name}` },
    {
      key: `vt/${reconstruction}`,
      label: `VT reconstruction: ${job.reconstruction}`,
    },
    { key: "lea", label: "Final refinement: SMART-LEA" },
  ];
}
function primarySource(job) {
  return methodCatalog?.methods[methodComponents(job)[0].key]?.code[0];
}
function methodDescription(job) {
  const components = methodComponents(job);
  let html =
    '<p class="detail-hint">Implementation links point to the exact code revision recorded by this catalog. Paper citations identify the underlying methods; the benchmark includes the adaptations described below.</p>';
  for (const component of components) {
    const entry = methodCatalog.methods[component.key];
    if (!entry) continue;
    html += `<section class="method-section"><h3>${esc(component.label)}</h3><div class="source-links">${entry.code.map((link) => `<a href="${esc(link.url)}" target="_blank" rel="noopener noreferrer">${esc(link.label)} ↗<small>${esc(link.path)}:${link.line}</small></a>`).join("")}</div>`;
    if (entry.papers.length)
      html += `<div class="paper-links">${entry.papers.map((paper) => `<a href="${esc(paper.url)}" target="_blank" rel="noopener noreferrer">Paper: ${esc(paper.title)} ↗</a>`).join("")}</div>`;
    if (entry.steps.length) {
      const parameters =
        job.stage === "pipeline" ? job.parameters.vt : job.parameters;
      const interpolate = (text) =>
        text.replace(/\{\{(\w+)\}\}/g, (_, key) =>
          String(parameters[key] ?? "not recorded"),
        );
      html += `<h3>${esc(entry.title)}</h3><p>${esc(entry.summary)}</p><h3>Inputs and separation of channels</h3><p>${esc(entry.inputs)}</p><h3>Provenance labels</h3><dl class="origin-legend">${Object.entries(
        entry.legend,
      )
        .map(
          ([label, description]) =>
            `<dt><span class="origin-tag">${esc(label)}</span></dt><dd>${esc(description)}</dd>`,
        )
        .join(
          "",
        )}</dl><h3>Parameters for this result</h3><div class="parameter-list">${entry.parameters.map(([key, label, unit, origin]) => `<div><strong>${esc(label)}</strong><span>${esc(parameters[key])} ${esc(unit)}</span><small>${esc(origin)}</small></div>`).join("")}</div><ol class="algorithm-steps">${entry.steps
        .map((step) => {
          const source = entry.code.find(
            (link) => link.symbol === step.code_symbol,
          );
          return `<li><h4>${esc(step.title)}</h4><div class="origin-tags">${step.tags.map((tag) => `<span class="origin-tag">${esc(tag)}</span>`).join("")}</div><p>${esc(interpolate(step.text))}</p>${source ? `<a class="step-source" href="${esc(source.url)}" target="_blank" rel="noopener noreferrer">Implementation: ${esc(source.symbol)} ↗</a>` : ""}</li>`;
        })
        .join(
          "",
        )}</ol><h3>Limitations and interpretation</h3><p>${esc(entry.limitations)}</p>`;
    } else
      html +=
        '<p class="description-pending">A step-by-step walkthrough has not been written for this component yet. Its implementation links and exact parameters are available.</p>';
    html += "</section>";
  }
  return html;
}
function render() {
  const vt = state.view === "vt",
    primary = vt ? "ospa" : "mean",
    best = [...rows].sort(
      (a, b) => a[primary] - b[primary] || a.id.localeCompare(b.id),
    )[0];
  const counts = [...new Set(rows.map((r) => r.count))];
  const countText =
    counts.length === 1
      ? integer(counts[0])
      : counts.length
        ? integer(Math.min(...counts)) + "–" + integer(Math.max(...counts))
        : "—";
  const bestF1 = vt
    ? [...rows].sort((a, b) => b.f1 - a.f1 || a.id.localeCompare(b.id))[0]
    : null;
  $("#cards").innerHTML =
    card(
      vt ? "LOWEST OSPA IN VIEW" : "LOWEST MEAN ERROR IN VIEW",
      fmt(best?.[primary]),
      "m",
      variantLabel(best),
      true,
    ) +
    card(
      "METHODS & VARIANTS",
      String(rows.length),
      "",
      state.track === "receiver" ? "SAPP receiver track" : "Direct NIST track",
    ) +
    (vt
      ? card(
          "HIGHEST F1 IN VIEW",
          bestF1 ? fmt(bestF1.f1 * 100, 2) : "—",
          "%",
          variantLabel(bestF1),
        )
      : card(
          "P90 OF LOWEST-MEAN METHOD",
          fmt(best?.p90),
          "m",
          variantLabel(best),
        )) +
    card(
      vt ? "SOURCE-MAP CASES / METHOD" : "QUERY CASES / METHOD",
      countText,
      "",
      vt ? "Room × noise × AP placement" : "Room × noise × AP × layout × query",
    );
  $("#row-count").textContent = rows.length;
  $("#ranking-note").textContent =
    `Ordered by ${sortNames[activeSort()]} · ${state.direction === "asc" ? "ascending" : "descending"}`;
  const columns = vt
    ? [
        ["name", "Method / scope"],
        [null, "VT merger"],
        ["ospa", "OSPA (m)"],
        ["f1", "F1 @ 0.5 m"],
        ["predicted", "Predicted"],
        ["truth", "Truth"],
        ["count", "Cases"],
      ]
    : [
        ["name", "Localization method"],
        [null, "Stage"],
        ["mean", "Mean (m)"],
        ["rmse", "RMSE (m)"],
        ["median", "Median (m)"],
        ["p90", "P90 (m)"],
        ["improvement", "Gain vs raw"],
        ["count", "Queries"],
      ];
  $("#table-head").innerHTML =
    "<tr><th scope='col'>#</th>" +
    columns
      .map(
        ([key, title], i) =>
          `<th scope="col" class="${i === 0 ? "method-head" : i === 1 ? "merger-head" : ""}" ${key ? `aria-sort="${activeSort() === key ? (state.direction === "asc" ? "ascending" : "descending") : "none"}"` : ""}>${key ? `<button data-sort="${key}" class="${activeSort() === key ? "active" : ""}">${esc(title)} ${activeSort() === key ? (state.direction === "asc" ? "↓" : "↑") : "↕"}</button>` : esc(title)}</th>`,
      )
      .join("") +
    "</tr>";
  const sorted = sortedRows(),
    max = Math.max(1, ...rows.map((r) => r[primary]));
  $("#table-body").innerHTML = sorted.length
    ? sorted
        .map((r, i) => {
          const scope = r.scope
            ? `<span class="scope">${r.scope === "full" ? "Full room" : "Local"}</span>`
            : "";
          const sub =
            r.stage === "pipeline"
              ? `${scope}VT: ${esc(r.reconstruction)}`
              : r.stage === "coarse"
                ? "Initial position estimate"
                : `${scope}${esc(r.merger ? "Survey source map" : "Published-method adapter")}`;
          const source = primarySource(r);
          const method = `<td class="method-cell"><div class="method-name-row"><button class="method-button" data-id="${esc(r.id)}">${esc(r.name)}</button>${source ? `<a class="code-link" href="${esc(source.url)}" target="_blank" rel="noopener noreferrer" title="${esc(source.label)}: ${esc(source.path)}">Code ↗</a>` : ""}</div><div class="method-meta">${sub}</div></td>`;
          const number = vt
            ? `<td class="merger-cell">${esc(meta.mergers[r.merger] || "Method default")}</td><td class="metric-primary">${fmt(r.ospa)}<span class="mini-bar"><span style="width:${Math.min(100, r.ospa * 100)}%"></span></span></td><td>${fmt(r.f1 * 100, 2)}<span class="unit">%</span></td><td title="Mean predicted source count per selected map">${fmt(r.predicted, 1)}</td><td title="Mean true source count per selected map">${fmt(r.truth, 1)}</td>`
            : `<td class="merger-cell"><span class="stage-badge ${r.stage === "coarse" ? "raw" : ""}">${r.stage === "coarse" ? "Raw" : "+ SMART-LEA"}</span></td><td class="metric-primary">${fmt(r.mean)}<span class="mini-bar"><span style="width:${(100 * r.mean) / max}%"></span></span></td><td title="Square root of the mean squared 3D position error">${fmt(r.rmse)}</td><td>${fmt(r.median)}</td><td>${fmt(r.p90)}</td><td title="100 × (initial mean − final mean) / initial mean; positive means improvement" class="${r.improvement >= 0 ? "positive" : "negative"}">${r.stage === "coarse" ? "—" : r.improvement == null ? "—" : `${r.improvement > 0 ? "+" : ""}${fmt(r.improvement, 2)}%`}</td>`;
          return `<tr class="${r.id === best?.id ? "row-best" : ""}"><td class="rank">${i === 0 ? '<span class="top-rank">1</span>' : i + 1}</td>${method}${number}<td>${integer(r.count)}</td></tr>`;
        })
        .join("")
    : `<tr><td colspan="${columns.length + 1}" class="empty">No results match these filters. Try another selection.</td></tr>`;
  $$("[data-sort]").forEach(
    (b) =>
      (b.onclick = () => {
        const key = b.dataset.sort;
        state.direction =
          activeSort() === key && state.direction === "asc" ? "desc" : "asc";
        state.sort = key;
        saveLocation();
        render();
      }),
  );
  $$("[data-id]").forEach((b) => (b.onclick = () => openDetail(b.dataset.id)));
}
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  setTimeout(() => ($("#toast").hidden = true), 2500);
}
function closeDetail() {
  detailRequest?.abort();
  if (!$("#drawer").hidden) {
    $("#drawer").hidden = true;
    $("#drawer-backdrop").hidden = true;
    document.body.style.overflow = "";
    lastFocus?.focus();
  }
}
async function openDetail(id) {
  lastFocus = document.activeElement;
  detail = null;
  detailTab = "overview";
  detailRequest?.abort();
  detailRequest = new AbortController();
  $("#drawer").hidden = false;
  $("#drawer-backdrop").hidden = false;
  document.body.style.overflow = "hidden";
  $("#detail-title").textContent = "Loading result…";
  $("#detail-subtitle").textContent = "";
  $("#detail-content").textContent = "Reading exact case statistics…";
  $("#close-detail").focus();
  try {
    detail = await resultsRequest(
      "detail",
      { ...state, id },
      detailRequest.signal,
    );
    $("#detail-title").textContent = detail.job.name;
    $("#detail-subtitle").textContent = [
      detail.job.track === "receiver" ? "SAPP receiver" : "Direct NIST",
      detail.job.stage === "pipeline"
        ? "VT: " + detail.job.reconstruction + " → SMART-LEA"
        : detail.job.stage === "vt"
          ? "VT reconstruction"
          : "Raw localization",
      detail.job.scope === "full"
        ? "Full room"
        : detail.job.scope === "local"
          ? "Local"
          : "",
    ]
      .filter(Boolean)
      .join(" · ");
    renderDetail();
  } catch (error) {
    if (error.name !== "AbortError")
      $("#detail-content").textContent = error.message;
  }
}
function chart(points, vt) {
  if (!points.length) return "";
  const max = Math.max(vt ? 1 : 0.001, points.at(-1)[0]);
  const x = (v) => 42 + (v / max) * 424,
    y = (v) => 151 - v * 129;
  let path = `M ${x(points[0][0])} 151`;
  for (const [a, b] of points) path += ` H ${x(a)} V ${y(b)}`;
  return `<svg class="ecdf" viewBox="0 0 490 192" role="img" aria-label="Cumulative distribution of ${vt ? "OSPA" : "3D localization error"}; exact ranks sampled for display"><path class="fill" d="${path} L ${x(points.at(-1)[0])} 151 Z"/><path class="axis" d="M42 22V151H466"/>${[0, 0.5, 1].map((n) => `<text x="33" y="${y(n) + 3}" text-anchor="end">${n * 100}%</text>`).join("")}<path class="curve" d="${path}"/>${[0, 0.25, 0.5, 0.75, 1].map((n) => `<text x="${x(n * max)}" y="169" text-anchor="middle">${fmt(n * max, 1)}</text>`).join("")}<text x="254" y="188" text-anchor="middle">${vt ? "OSPA" : "3D error"} (m)</text></svg>`;
}
function detailMetric(label, value) {
  return `<div class="detail-metric">${esc(label)}<strong>${value}</strong></div>`;
}
function renderDetail() {
  if (!detail) return;
  $$("[data-tab]").forEach((b) => {
    b.setAttribute("aria-selected", b.dataset.tab === detailTab);
    b.tabIndex = b.dataset.tab === detailTab ? 0 : -1;
  });
  $("#detail-content").setAttribute("aria-labelledby", "tab-" + detailTab);
  const j = detail.job,
    m = detail.metrics,
    vt = j.stage === "vt";
  if (detailTab === "method") {
    $("#detail-content").innerHTML = methodDescription(j);
    return;
  }
  if (detailTab === "parameters") {
    $("#detail-content").innerHTML =
      `<div class="detail-hint">Exact preset parameters recorded with this result. These settings have not been selected by a completed hyperparameter search.</div><pre>${esc(JSON.stringify(j.parameters, null, 2))}</pre>`;
    return;
  }
  if (detailTab === "provenance") {
    $("#detail-content").innerHTML =
      `<div class="detail-hint">Imported from a completed calibration receipt. The objective file’s SHA-256 and its full-case metrics were verified when building the explorer index. Historical artifacts retain their recorded compatibility evidence.</div><dl class="provenance">${[
        ["Result ID", j.id],
        ["Preset status", j.status],
        ["Receipt", j.receipt],
        ["Objective artifact", j.artifact],
        ["Artifact SHA-256", j.artifact_sha256],
        ["Receipt code hash", j.code_hash],
        ["Objective code hash", j.objective_identity.code_hash],
        ["Data hash", j.objective_identity.data_hash],
        ["Index built", meta.built_utc],
      ]
        .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`)
        .join(
          "",
        )}</dl>${j.compatibility_reuse ? `<h3>Recorded compatibility reuse</h3><pre>${esc(JSON.stringify(j.compatibility_reuse, null, 2))}</pre>` : ""}`;
    return;
  }
  if (!m) {
    $("#detail-content").textContent = "No cases match the current filters.";
    return;
  }
  const measures = vt
    ? [
        ["OSPA (m)", fmt(m.ospa)],
        ["F1 @ 0.5 m", fmt(m.f1 * 100, 2) + "%"],
        ["Source-map cases", integer(m.count)],
        ["Mean predicted", fmt(m.predicted, 1)],
        ["Mean truth", fmt(m.truth, 1)],
        ["Matching gate", "0.5 m"],
      ]
    : [
        ["Mean error (m)", fmt(m.mean)],
        ["RMSE (m)", fmt(m.rmse)],
        ["Median error (m)", fmt(m.median)],
        ["P90 error (m)", fmt(m.p90)],
        ["Initial mean (m)", fmt(m.initial)],
        ["Query cases", integer(m.count)],
        ["Fallbacks", integer(m.fallbacks)],
      ];
  const max = Math.max(
    vt ? 1 : 0.001,
    ...detail.rooms.map((r) => r[vt ? "ospa" : "mean"]),
  );
  $("#detail-content").innerHTML =
    `<div class="detail-hint">${esc($("#filter-summary").textContent)}. All statistics below use this selection.${vt ? " The source bag includes real APs and reflected VTs." : " Query errors use 3D Euclidean distance."}</div><div class="detail-metrics">${measures.map(([k, v]) => detailMetric(k, v)).join("")}</div>${j.stage === "pipeline" ? `<h3>Change after SMART-LEA</h3><p class="${m.improvement >= 0 ? "positive" : "negative"}">${m.improvement == null ? "Undefined for a zero initial mean" : fmt(Math.abs(m.improvement), 2) + "% " + (m.improvement >= 0 ? "lower" : "higher") + " mean error than this pipeline’s initial estimates."}</p>` : ""}<h3>${vt ? "OSPA" : "Mean 3D error"} by room (m)</h3>${detail.rooms.map((r) => `<div class="room-row"><span>${esc(roomLabel(r.room))}</span><span class="room-bar"><span style="width:${(r[vt ? "ospa" : "mean"] / max) * 100}%"></span></span><strong>${fmt(r[vt ? "ospa" : "mean"])}</strong></div>`).join("")}<h3>Cumulative distribution</h3>${chart(detail.ecdf, vt)}<p class="stat-caption">Curve uses up to 201 exact order statistics. Summary metrics use every selected case.</p><h3>How to read these results</h3><p>${vt ? "Lower OSPA is better (p = 1, cut-off = 1 m). F1 uses one-to-one source matching within 0.5 m and is averaged over maps. Predicted and true source counts are also averages over maps. Obstacle layouts do not change the survey-map evaluation." : "Lower error is better. Mean, median and P90 are calculated directly from selected query errors, including fallback outputs. Positive gain means SMART-LEA reduced mean error relative to the same pipeline’s initial estimates."}</p>`;
}
function exportCSV() {
  const fields =
    state.view === "vt"
      ? ["ospa", "f1", "predicted", "truth", "count"]
      : [
          "mean",
          "rmse",
          "median",
          "p90",
          "initial",
          "improvement",
          "fallbacks",
          "count",
        ];
  const names = {
    ospa: "ospa_p1_c1_m",
    f1: "macro_f1_gate_0.5m",
    predicted: "mean_predicted_sources",
    truth: "mean_true_sources",
    count: state.view === "vt" ? "source_map_cases" : "query_cases",
    mean: "mean_error_m",
    rmse: "rmse_error_m",
    median: "median_error_m",
    p90: "p90_error_m",
    initial: "initial_mean_error_m",
    improvement: "improvement_percent",
    fallbacks: "fallback_count",
  };
  const header = [
    "rank",
    "result_id",
    "method",
    "reconstruction",
    "scope",
    "merger",
    "stage",
    "track",
    "split",
    "preset_status",
    "filters_json",
    ...fields.map((k) => names[k]),
  ];
  const data = sortedRows().map((r, i) => [
    i + 1,
    r.id,
    r.name,
    r.reconstruction,
    r.scope,
    r.merger,
    r.stage,
    state.track,
    "calibration",
    "uncalibrated preset",
    JSON.stringify(state),
    ...fields.map((k) => r[k] ?? ""),
  ]);
  const quote = (v) => '"' + String(v).replace(/"/g, '""') + '"';
  const csv =
    "\ufeff" +
    [header, ...data].map((r) => r.map(quote).join(",")).join("\r\n");
  const url = URL.createObjectURL(
    new Blob([csv], { type: "text/csv;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `indoor-vt-${state.view}-${state.track}-calibration.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function init() {
  try {
    [meta, methodCatalog] = await Promise.all([
      resultsRequest("meta"),
      fetch("./method-catalog.json").then((response) => {
        if (!response.ok)
          throw Error("Could not load implementation references.");
        return response.json();
      }),
    ]);
    $("#count-vt").textContent = meta.counts.vt;
    $("#count-raw").textContent = meta.counts.coarse;
    $("#count-all").textContent = meta.counts.coarse + meta.counts.pipeline;
    $("#run-state").textContent =
      `${meta.counts.vt + meta.counts.coarse + meta.counts.pipeline} presets complete · ${meta.mode === "static" ? "published snapshot" : meta.status.running?.length ? "workers active" : "workers stopped"}`;
    $("#source-note").textContent =
      `Source: ${meta.revision} preset receipts · ${meta.counts.vt + meta.counts.coarse + meta.counts.pipeline} verified artifacts`;
    const footer = document.createElement("div");
    footer.className = "publication-links";
    footer.innerHTML = `<a href="${esc(methodCatalog.repository)}" target="_blank" rel="noopener noreferrer">Repository & code ↗</a>${meta.inputs ? `<a href="${esc(methodCatalog.repository)}/blob/main/published/${esc(meta.revision)}/${esc(meta.inputs.file)}" target="_blank" rel="noopener noreferrer">Calibration input data · ${(meta.inputs.bytes / 1024 ** 2).toFixed(1)} MiB (Git LFS) ↗</a>` : ""}`;
    $("main").append(footer);
    options($("#family"), [...new Set(meta.rooms.map((r) => r.family))], (f) =>
      f === "rectangle" ? "Rectangle" : f,
    );
    options($("#merger"), Object.keys(meta.mergers), (m) => meta.mergers[m]);
    readLocation();
    await loadResults();
  } catch (error) {
    $("#error").textContent = error.message;
    $("#error").hidden = false;
    $("#results-panel").setAttribute("aria-busy", "false");
  }
}
$$("[data-view]").forEach(
  (b) =>
    (b.onclick = () => {
      state.view = b.dataset.view;
      state.sort = "";
      state.direction = "asc";
      state.scope = "";
      state.merger = "";
      state.stage = "";
      loadResults();
    }),
);
$$("[data-track]").forEach(
  (b) =>
    (b.onclick = () => {
      state.track = b.dataset.track;
      state.noise = "";
      loadResults();
    }),
);
for (const k of [
  "family",
  "room",
  "noise",
  "ap",
  "obstacles",
  "layout",
  "scope",
  "merger",
  "stage",
])
  $("#" + k).onchange = (e) => {
    state[k] = e.target.value;
    loadResults();
  };
let debounce;
$("#search").oninput = (e) => {
  state.q = e.target.value;
  clearTimeout(debounce);
  debounce = setTimeout(loadResults, 180);
};
$("#reset").onclick = () => {
  state = { ...defaults, view: state.view, track: state.track };
  loadResults();
};
$("#export").onclick = exportCSV;
$("#copy-link").onclick = async () => {
  try {
    await navigator.clipboard.writeText(location.href);
    toast("View link copied");
  } catch {
    toast("Copy the address from your browser to share this view locally.");
  }
};
$("#close-detail").onclick = closeDetail;
$("#drawer-backdrop").onclick = closeDetail;
$$("[data-tab]").forEach(
  (b) =>
    (b.onclick = () => {
      detailTab = b.dataset.tab;
      renderDetail();
    }),
);
$(".detail-tabs").onkeydown = (e) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
  e.preventDefault();
  const tabs = $$("[data-tab]"),
    i = tabs.indexOf(document.activeElement),
    next =
      e.key === "Home"
        ? 0
        : e.key === "End"
          ? tabs.length - 1
          : (i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  tabs[next].click();
  tabs[next].focus();
};
document.addEventListener("keydown", (e) => {
  if ($("#drawer").hidden) return;
  if (e.key === "Escape") closeDetail();
  if (e.key === "Tab") {
    const focusable = [
      ...$("#drawer").querySelectorAll(
        'button:not([tabindex="-1"]),a[href],input,select',
      ),
    ].filter((x) => !x.hidden);
    const first = focusable[0],
      last = focusable.at(-1);
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }
});
window.addEventListener("hashchange", () => {
  if (meta) {
    readLocation();
    loadResults();
  }
});
init();
