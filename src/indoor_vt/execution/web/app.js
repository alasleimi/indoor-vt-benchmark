import "./site-runtime.js";
import {finisherNames, finisherParameterSummary, sourceParameters,
  hasReconstructionExplanation, reconstructionExplanation,
  connectedComponentSourceOverview, sourceMetricExplanation} from "/finisher-explanations.js";

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const fmt = (v, places = 3) =>
  Number.isFinite(v)
    ? v.toLocaleString("en-US", {
        minimumFractionDigits: places,
        maximumFractionDigits: places,
      })
    : "—";
const count = (v) => Number(v || 0).toLocaleString("en-US");
const palette = [
  "#16836e",
  "#b67a36",
  "#527ab5",
  "#966294",
  "#b05965",
  "#6c8450",
  "#4c93a1",
  "#ae754a",
  "#6966a3",
  "#587965",
  "#b15893",
  "#4d7994",
  "#a18b39",
  "#774d65",
];
const selectionNames = {
  fixed: "Fixed settings (no tuning)",
  ospa: "OSPA",
  localization: "Independent",
  stagewise: "Stagewise",
  truth: "LEA only (ground-truth VTs)",
  joint: "Joint pipeline tuning",
};
const viewNames = {
  vt: "VT reconstruction",
  raw: "Raw localization",
  all: "Localization",
};
const descriptions = {
  vt: "Compare reconstructed source maps across noise levels, with each method's selection protocol shown below.",
  raw: "MPUrge-MAP estimates before SMART-LEA refinement. Joint labels identify the pipeline that selected the MAP parameters.",
  all: "MPUrge-MAP initializes each position; SMART-LEA refines it using the listed VT source. Raw rows show MAP only.",
};
let data,
  rows = [],
  currentMethod = null,
  currentRow = null,
  activeReceipt = null;
let detailTab = "overview",
  loaded = null,
  loading = null,
  disposeMap = null,
  tabVersion = 0;
let selectedCase = 0,
  auditHistory = [],
  currentArtifact = null,
  focusReturn = null;
let mapChoice = { room: "", layout: "", repetition: null };
let searchPage = 0,
  trialSearch = "",
  trialSort = "loss",
  refreshing = false;
let state = {
  view: "vt",
  split: "evaluation",
  noise: "0",
  obstacles: "",
  ap: "A+B",
  unification: "",
  tags: "",
  selection: "",
  stage: "",
  search: "",
  sort: "",
  direction: "asc",
  hidden: "",
  mapMode: "2d",
};
const allTabs = [
  "overview",
  "map",
  "method",
  "code",
  "parameters",
  "search",
  "evidence",
];

async function api(path, post = false) {
  const response = await fetch(path, { method: post ? "POST" : "GET" });
  const value = await response.json();
  if (!response.ok)
    throw Error(value.error || `Request failed (${response.status})`);
  return value;
}
function readURL() {
  const query = new URLSearchParams(location.search);
  for (const key of Object.keys(state))
    if (query.has(key)) state[key] = query.get(key);
  if (!(state.view in viewNames)) state.view = "vt";
  if (!["evaluation", "calibration"].includes(state.split))
    state.split = "evaluation";
  if (!["", "on", "off"].includes(state.unification)) state.unification = "";
  if (state.unification && !state.tags)
    state.tags = "unification:" + state.unification;
  state.unification = "";
  if (!["", "pipeline", "raw"].includes(state.stage)) state.stage = "";
  if (!["asc", "desc"].includes(state.direction)) state.direction = "asc";
  if (!["2d", "3d"].includes(state.mapMode)) state.mapMode = "2d";
  if (!["A", "B", "A+B"].includes(state.ap)) state.ap = "A+B";
  if (state.selection && !(state.selection in selectionNames))
    state.selection = "";
  if (state.sort.startsWith("obstacles_") && !query.has("obstacles")) {
    state.obstacles = state.sort.slice(10);
    state.sort = "mean_m";
  }
  return query;
}
function saveURL() {
  const query = new URLSearchParams(state);
  if (currentMethod) {
    query.set("method", currentMethod.id);
    query.set("tab", detailTab);
    if (activeReceipt?.id !== currentRow?.receipt_id)
      query.set("trial", activeReceipt.id);
  }
  history.replaceState(null, "", `${location.pathname}?${query}`);
}
function matchesView(method, view = state.view) {
  return view === "all" ? method.group !== "vt" : method.group === view;
}
function visibleMethods() {
  return data.methods.filter(
    (m) =>
      matchesView(m) &&
      matchesTags(m) &&
      (!state.selection || m.selection === state.selection) &&
      (state.view !== "all" || !state.stage || m.group === state.stage) &&
      (m.search_text || m.title)
        .toLowerCase()
        .includes(state.search.toLowerCase()),
  );
}
function displayName(method) {
  return state.view === "all" && method.group === "raw"
    ? "MPUrge-MAP only"
    : method.name;
}
const tagGroups = {
  "Method family": [
    ["family:range_rmse", "Range RMSE (true count)"],
    ["family:mpurge", "MPUrge"],
    ["family:zayets", "Zayets"],
    ["family:ransac", "RANSAC"],
  ],
  Finishing: [
    ["finisher:consumption", finisherNames.consumption],
    ["finisher:robust", finisherNames.robust],
    ["finisher:none", "No finisher"],
    ["finisher:cluster", "Cluster support / no consuming finisher"],
  ],
  "Peak matching": [
    ["matching:hungarian", "Hungarian range assignment"],
    ["matching:standard", "Standard"],
    ["matching:triangle", "Triangle gate"],
    ["matching:crossing_allowed", "Crossings allowed"],
  ],
  "Peak unification": [
    ["unification:on", "On"],
    ["unification:off", "Off"],
    ["unification:n/a", "Not used"],
  ],
};
function matchesTags(method) {
  const selected = new Set(state.tags.split(",").filter(Boolean));
  return Object.values(tagGroups).every((options) => {
    const group = options
      .map(([tag]) => tag)
      .filter((tag) => selected.has(tag));
    return !group.length || group.some((tag) => method.tags?.includes(tag));
  });
}
function renderTags() {
  const selected = new Set(state.tags.split(",").filter(Boolean));
  $("tag-filter-summary").textContent = selected.size
    ? `Filter tags · ${selected.size} selected`
    : "Filter tags";
  $("tag-filter-menu").innerHTML =
    '<p class="inline-note">Choose any within a group; groups combine.</p>' +
    Object.entries(tagGroups)
      .map(
        ([label, options]) =>
          `<fieldset><legend>${esc(label)}</legend>${options.map(([tag, title]) => `<label><input type="checkbox" value="${tag}" ${selected.has(tag) ? "checked" : ""}><span class="config-badge">${esc(title)}</span></label>`).join("")}</fieldset>`,
      )
      .join("") +
    '<button class="text-button" id="clear-tags">Clear tags</button>';
  $("tag-filter-menu").onchange = () => {
    state.tags = [...$("tag-filter-menu").querySelectorAll("input:checked")]
      .map((input) => input.value)
      .join(",");
    $("tag-filter-summary").textContent = state.tags
      ? `Filter tags · ${state.tags.split(",").length} selected`
      : "Filter tags";
    render();
    saveURL();
  };
  $("clear-tags").onclick = () => {
    state.tags = "";
    renderTags();
    render();
    saveURL();
  };
}
function selectionBadge(method) {
  return `<span class="config-badge" title="${esc(method.selection_description || method.selection_label)}">${esc(method.tuning)}</span>`;
}
function unificationBadge(method) {
  if (method.unify === null) return "";
  return `<span class="config-badge unification-tag" title="${esc(method.unification_scope)}">${esc(method.unification_label)}</span>`;
}
function detailBadges(method) {
  return `<span class="config-badges">${method.vt_source ? `<span class="config-badge source-badge">VTs: ${esc(method.vt_source)}</span>` : ""}<span class="config-badge" title="${esc(method.selection_description || method.selection_label)}">${esc(method.selection_label)}</span>${unificationBadge(method)}</span>`;
}
function renderRoomProtocol(room) {
  const obstacleReconstruction = state.view === "vt" && state.obstacles !== "";
  $("room-label").textContent =
    `${room.id} · ${fmt(room.width_m)} × ${fmt(room.depth_m)} m · ${fmt(room.ceiling_m)} m high · ${obstacleReconstruction ? obstacleInputLabel() : `${count(room.survey_xyz.length)} survey positions`}`;
  if (obstacleReconstruction) {
    $("query-counts").textContent = `${state.obstacles} obstacles · ${obstacleInputCount()} measured fingerprints per AP · σ 0 m · no new calibration`;
    if (matchedObstacleSurvey()) {
      const mask = data.obstacle_reconstruction_protocol.common_survey_mask;
      $("query-origin").innerHTML = `<p>Every layout uses the exact same ${mask.retained_count} positions from the original ${mask.original_count}-position survey, with unchanged coordinates, heights, order and IDs. ${mask.excluded_count} positions are excluded because they are inside at least one box in any layout.</p><p>Occupancy includes a box footprint's boundary and heights from the floor through its top. No extra clearance is added. A point above a box remains eligible unless another layout blocks it.</p><p>Obstacle fingerprints were freshly acquired with the same NIST simulator settings. The zero-box control reuses the original empty-room fingerprints at these same positions. Parameters remain frozen; no calibration or search was repeated.</p><p>Truth is the union of direct, first-order and second-order source images visible at this receiver set in each layout, including obstacle reflections. Only the true-count RMSE solver receives the count; identities and coordinates remain evaluator-only. The earlier 100-query-position experiment is preserved in its original run directory and is not mixed into this comparison.</p>`;
      return;
    }
    $("query-origin").innerHTML = `<p>Reconstruction uses the 100 existing receiver positions for this saved layout, all outside its boxes. Their coordinates are known inputs. The 280 empty-room reference fingerprints are not used.</p><p>The 60 positions previously used for localization calibration and the 40 localization validation positions are combined here as reconstruction inputs. Parameters remain frozen from the original selections. The zero-obstacle layout is a matching 100-position control; changing layouts also changes receiver positions.</p><p>Truth is the union of distinct direct, first-order and second-order image sources visible in the archived simulator paths at these positions. It includes obstacle reflections and does not claim to include sources invisible at every input receiver. Only the true-count RMSE baseline receives the resulting source count; no solver receives identities or true coordinates.</p>`;
    return;
  }
  const layouts = room.query_layouts,
    total = layouts.reduce((sum, layout) => sum + layout.total, 0),
    calibration = layouts.reduce((sum, layout) => sum + layout.calibration, 0),
    evaluation = total - calibration,
    scene = data.config.scene,
    observations = data.config.observations;
  $("query-counts").textContent =
    `${count(calibration)} calibration queries · ${count(evaluation)} validation queries (held-out evaluation) · ${count(layouts.length)} layouts`;
  $("query-origin").innerHTML =
    `<p>These are ${count(total)} distinct query positions, sampled separately for each obstacle layout in the same room, independently of the survey grid. Horizontal positions are drawn uniformly inside the room, rejecting positions within ${fmt(scene.clearance_m.query_box, 2)} m of an obstacle. Heights are sampled uniformly from ${fmt(scene.query_height_m[0], 1)} to ${fmt(scene.query_height_m[1], 1)} m. The saved samples use seed ${data.config.seed}; NIST simulates the measurements at each position.</p>` +
    `<p>The first ${fmt(100 * observations.query_calibration_fraction, 0)}% of each layout's saved query list is used for calibration and hyperparameter selection. The remainder is held out for validation (called evaluation in the results). Both splits include the same obstacle layouts; the query positions are held out, not the layouts. Query IDs do not overlap between splits, and all displayed studies use the same split.</p>` +
    `<div class="table-scroll"><table class="query-splits"><thead><tr><th>Layout</th><th>Obstacles</th><th>Calibration queries</th><th>Validation queries</th></tr></thead><tbody>${layouts.map((layout) => `<tr><td>${esc(layout.id)}</td><td>${count(layout.box_count)}</td><td>${count(layout.calibration)}</td><td>${count(layout.total - layout.calibration)}</td></tr>`).join("")}</tbody></table></div>` +
    `<p>Counts refer to distinct positions, not repeated measurements. Noise evaluation reuses the same ${count(evaluation)} held-out positions with ${count(observations.evaluation_repetitions.length)} noise realizations per level${observations.track === "direct" ? "; at zero noise, identical repeats are evaluated once" : ""}. The ${count(room.survey_xyz.length)} survey positions form a ${fmt(data.config.survey.spacing_m, 1)} m grid at heights ${scene.survey_heights_m.map((height) => fmt(height, 1)).join(" and ")} m.</p>`;
}
function primaryMetric() {
  return state.view === "vt" ? "ospa" : "mean_m";
}
function matchedObstacleSurvey() {
  return data.obstacle_reconstruction_protocol?.input_protocol === "matched_survey";
}
function obstacleInputCount() {
  return data.obstacle_reconstruction_protocol?.common_survey_mask?.retained_count || 100;
}
function obstacleInputLabel() {
  return `${obstacleInputCount()} ${matchedObstacleSurvey() ? "common survey" : "layout"} positions`;
}
function obstacleEvaluationNote() {
  if (activeReceipt?.obstacle_count === undefined) return "";
  const matched = activeReceipt.input_protocol === "matched_survey";
  return `<p class="detail-hint">${activeReceipt.obstacle_count} obstacles · ${activeReceipt.input_count || 100} ${matched ? "common survey" : "saved layout"} positions · σ 0 m. ${matched ? "All layouts use identical receiver coordinates and original position IDs, outside every layout's boxes. Obstacle measurements come from fresh NIST simulation; the empty control reuses original measurements at those same positions." : "This earlier experiment uses different random receiver positions in each layout."} No calibration was repeated. Truth includes all distinct source images visible in these measurements, including room and obstacle reflections. Identities and coordinates are evaluator-only; the RMSE baseline additionally receives the scene/AP source count.</p><div class="source-links">${matched ? '<a href="/source?file=acquisition%2Fmatched_obstacles.py" target="_blank" rel="noopener">Common survey and simulator acquisition ↗</a>' : ""}<a href="/source?file=execution%2Fobstacle_reconstruction.py" target="_blank" rel="noopener">Frozen reconstruction and visible-source truth ↗</a></div>`;
}
function obstacleCounts() {
  return [
    ...new Set(
      data.rooms.flatMap((room) =>
        room.query_layouts.map((layout) => layout.box_count),
      ),
    ),
  ].sort((a, b) => a - b);
}
function conditionMetric(condition, key) {
  if (state.view === "vt")
    return condition.source_metrics_by_ap?.[state.ap]?.[key];
  return state.view !== "vt" && state.obstacles !== ""
    ? condition.obstacle_metrics?.[state.obstacles]?.[key]
    : condition.metrics[key];
}
function conditionContextMatches(condition) {
  return state.view !== "vt" || (state.obstacles === ""
    ? condition.obstacle_count === undefined
    : condition.obstacle_count === Number(state.obstacles));
}
function metricColumns(view = state.view) {
  return view === "vt"
    ? [
        ["ospa", "OSPA (c = 1 m)"],
        ["f1", "F1 @ 0.5 m (%)"],
        ["matched_rmse_m", "Matched RMSE @ 0.5 m (m)"],
      ]
    : [
        ["mean_m", "Mean (m)"],
        ["rmse_m", "RMSE (m)"],
        ["median_m", "Median (m)"],
        ["p90_m", "P90 (m)"],
      ];
}
function sourceCounts(condition, kind) {
  const cases = condition?.source_counts || [];
  if (!cases.length) return "—";
  const values = cases.map((record) => record[kind][state.ap]);
  if (values.every((value) => value === values[0])) return count(values[0]);
  const multipleRooms = new Set(cases.map((record) => record.room)).size > 1;
  return cases
    .map(
      (record) =>
        `<div class="source-count-case"><small title="Stored noise seed ID ${record.repetition}">${multipleRooms ? esc(record.room) + " · " : ""}Realization ${data.config.observations[condition.split + "_repetitions"].indexOf(record.repetition) + 1}</small><strong>${count(record[kind][state.ap])}</strong></div>`,
    )
    .join("");
}
function obstacleLabel() {
  return state.obstacles === ""
    ? "All obstacle layouts"
    : `${state.obstacles} obstacles`;
}
function chartMetricLabel() {
  const label = metricColumns().find(([key]) => key === primaryMetric())[1];
  return state.view === "vt"
    ? `${label} · AP ${state.ap}${state.obstacles !== "" ? ` · ${obstacleLabel()} · ${obstacleInputLabel()}` : ""}`
    : `${label} · ${obstacleLabel()}`;
}
function chartMetricValue(condition) {
  return conditionMetric(condition, primaryMetric());
}
function metricValue(key, value) {
  const places = key === "matched_rmse_m" ? 6 : key.includes("count") ? 1 : 3;
  return fmt(key === "f1" && Number.isFinite(value) ? value * 100 : value, places);
}
function receiptFor(id) {
  return (
    data.evaluation.find((r) => r.id === id) ||
    data.calibration.find((r) => r.id === id) ||
    data.trials.find((r) => r.id === id)
  );
}
function rowFor(method) {
  return data.conditions.find(
    (r) =>
      r.method_id === method.id &&
      conditionContextMatches(r) &&
      r.split === state.split &&
      r.level === Number(state.split === "calibration" ? 0 : state.noise),
  );
}
function fillControls() {
  renderTags();
  const obstacleValues = obstacleCounts().map(String);
  if (!obstacleValues.includes(state.obstacles)) state.obstacles = "";
  if (state.view === "vt" && state.obstacles !== "") {
    state.split = "evaluation";
    state.noise = "0";
  }
  $("obstacles").innerHTML =
    `<option value="">${state.view === "vt" ? "Empty-room survey (280 positions)" : "All layouts"}</option>` +
    obstacleValues
      .map(
        (n) =>
          `<option value="${n}">${n === "0" ? "0 (empty room)" : `${n} obstacles`}${state.view === "vt" ? " · " + obstacleInputLabel() : ""}</option>`,
      )
      .join("");
  $("noise").innerHTML = data.config.observations.noise_levels
    .map((v) => `<option value="${v}">${v}</option>`)
    .join("");
  if (!data.config.observations.noise_levels.includes(Number(state.noise)))
    state.noise = String(data.config.observations.noise_levels[0]);
  $("selection").innerHTML =
    '<option value="">All selection types</option>' +
    [
      ...new Set(
        data.methods.filter((m) => matchesView(m)).map((m) => m.selection),
      ),
    ]
      .map(
        (s) =>
          `<option value="${s}">${s === "joint" && state.view === "vt" ? "Joint with MPUrge-MAP + SMART-LEA" : selectionNames[s]}</option>`,
      )
      .join("");
  if (![...$("selection").options].some((o) => o.value === state.selection))
    state.selection = "";
  for (const id of [
    "split",
    "noise",
    "obstacles",
    "ap",
    "selection",
    "stage",
    "search",
  ])
    $(id).value = state[id];
}
function render() {
  if (!data) return;
  const calibration = state.split === "calibration",
    vt = state.view === "vt";
  $("page-title").textContent = viewNames[state.view];
  $("page-description").textContent = descriptions[state.view];
  document.querySelectorAll("[data-view]").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === state.view);
    b.setAttribute(
      "aria-current",
      b.dataset.view === state.view ? "page" : "false",
    );
  });
  for (const view of ["vt", "raw", "all"])
    $(`count-${view}`).textContent = data.methods.filter((m) =>
      matchesView(m, view),
    ).length;
  const obstacleReconstruction = vt && state.obstacles !== "";
  $("noise").disabled = calibration || obstacleReconstruction;
  $("split").disabled = obstacleReconstruction;
  $("stage-filter").hidden = state.view !== "all";
  $("obstacles-filter").hidden = false;
  $("ap-filter").hidden = !vt;
  $("split-badge").textContent = calibration
    ? "NOISE-FREE SELECTION"
    : "FROZEN PARAMETERS";
  $("split-note").textContent = obstacleReconstruction
    ? (matchedObstacleSurvey() ? `Frozen settings, no new calibration. Every layout uses the same ${obstacleInputCount()} original survey positions outside all obstacles. Actual obstacle measurements were simulated at these positions; the zero-box control reuses their original empty-room fingerprints.` : "Frozen settings, no new calibration. Reconstruction uses all 100 saved receiver fingerprints from the selected layout at zero noise. Truth includes distinct room and obstacle image sources visible in these fingerprints. This is a different receiver set from the 280-point survey.")
    : calibration
    ? "Calibration results select the parameters. These are not held-out performance estimates. Full-pipeline selection minimizes final localization error."
    : vt
      ? "Settings are held fixed across these noise tests. Tuned methods use zero-noise selection; fixed baselines are labeled in the table. Source maps use the same room and survey; this is not an unseen-room evaluation."
      : "Parameters were selected at zero noise and held fixed for these noise tests. Query positions are held out within the same room; this is not an unseen-room evaluation.";
  $("split").options[0].textContent = vt
    ? "Fixed-parameter noise evaluation"
    : "Held-out query evaluation";
  if (vt && state.ap === "A+B")
    $("split-note").textContent += " A+B matches within each AP, then aggregates; cross-AP matches are forbidden.";
  const room = data.rooms[0],
    status = data.status;
  $("run-state").textContent =
    status.state === "complete"
      ? "Experiment complete"
      : status.state.replaceAll("_", " ");
  $("trial-state").textContent =
    `${count(data.trials.length)} / ${count(status.target_trials)} selection trials`;
  $("opt-progress").max = status.target_trials || 1;
  $("opt-progress").value = data.trials.length;
  $("room-description").textContent =
    `${data.rooms.length} ${data.rooms.length === 1 ? "room" : "rooms"} · ${data.config.survey.spacing_m} m survey`;
  const studies = data.studies || [{ name: data.config.name }];
  $("dataset-description").textContent = studies.map((s) => s.name).join(" · ");
  renderRoomProtocol(room);
  $("track-label").textContent =
    data.config.observations.track === "direct"
      ? "Direct NIST"
      : "Receiver observations";
  $("filter-summary").textContent =
    `${calibration ? "Calibration · σ 0 m" : `Evaluation · σ ${state.noise} m`} · ${vt ? `AP ${state.ap}${obstacleReconstruction ? ` · ${obstacleLabel()} · ${obstacleInputLabel()}` : " · 280 survey positions"}` : obstacleLabel()}`;
  $("source-note").textContent =
    `Source: ${studies.map((s) => s.name).join(" + ")} · ${data.methods.length} measured method configurations`;
  $("run-details").innerHTML =
    `<p>Acquisition: ${count(data.acquisition.completed)} / ${count(data.acquisition.total)} chunks. Numerical cache: ${fmt(data.cache.disk_bytes / 1048576, 1)} / ${data.config.storage.cache_mib} MiB.</p><p>${studies.map((s) => `${esc(s.name)}: ${esc(s.state || status.state)}`).join("<br>")}</p><p>${status.current ? esc(JSON.stringify(status.current)) : "No active work."}${status.error ? " " + esc(status.error) : ""}</p><p>Refreshes every 15 seconds. All ${count(data.trials.length)} trial summaries are available in method details.</p>`;
  rows = visibleMethods()
    .map((method) => ({ method, condition: rowFor(method) }))
    .filter((r) => r.condition);
  const columns = metricColumns();
  if (![...columns.map((c) => c[0]), "name"].includes(state.sort))
    state.sort = primaryMetric();
  rows.sort((a, b) => {
    const x = conditionMetric(a.condition, state.sort),
      y = conditionMetric(b.condition, state.sort);
    if (state.sort !== "name") {
      if (!Number.isFinite(x)) return Number.isFinite(y) ? 1 : 0;
      if (!Number.isFinite(y)) return -1;
    }
    const result =
      state.sort === "name"
        ? a.method.title.localeCompare(b.method.title)
        : (x ?? Infinity) - (y ?? Infinity);
    return result * (state.direction === "asc" ? 1 : -1);
  });
  $("row-count").textContent = rows.length;
  $("ranking-note").textContent =
    `Ordered by ${state.sort === "name" ? "method name" : columns.find((c) => c[0] === state.sort)?.[1]} · ${state.direction === "asc" ? "ascending" : "descending"}`;
  const showSource = state.view === "all";
  const sortHeader = (key, label) =>
    `<th class="${key === "name" ? "method-head" : ""}" aria-sort="${state.sort === key ? (state.direction === "asc" ? "ascending" : "descending") : "none"}"><button data-sort="${key}" class="${state.sort === key ? "active" : ""}">${label}${state.sort === key ? (state.direction === "asc" ? " ↑" : " ↓") : ""}</button></th>`;
  $("table-head").innerHTML =
    "<tr><th>#</th>" +
    sortHeader(
      "name",
      vt
        ? "Reconstructor"
        : state.view === "raw"
          ? "Localizer"
          : "Localization pipeline",
    ) +
    (showSource ? '<th class="label-head">VTs supplied by</th>' : "") +
    '<th class="label-head">Parameter selection</th>' +
    columns.map(([key, label]) => sortHeader(key, label)).join("") +
    (vt
      ? '<th class="label-head">Predicted</th><th class="label-head">True</th>'
      : "") +
    "<th>Search trials</th></tr>";
  const maximum = Math.max(
    ...rows.map((r) => conditionMetric(r.condition, primaryMetric()) || 0),
    0.001,
  );
  $("table-body").innerHTML =
    rows
      .map(
        ({ method: m, condition: r }, i) =>
          `<tr><td class="rank">${i + 1}</td><td class="method-cell"><div class="method-identity"><button class="method-button" data-method="${m.id}" aria-label="${esc(m.title)}">${esc(displayName(m))}</button>${unificationBadge(m)}</div></td>${showSource ? `<td class="label-cell" data-column="vt-source">${esc(m.vt_source || "Not used")}</td>` : ""}<td class="label-cell" data-column="selection">${selectionBadge(m)}</td>${columns.map(([key]) => `<td data-metric="${key}" class="${key === primaryMetric() ? "metric-primary" : ""}">${metricValue(key, conditionMetric(r, key))}${key === primaryMetric() ? `<span class="mini-bar"><span style="width:${(100 * conditionMetric(r, key)) / maximum}%"></span></span>` : ""}</td>`).join("")}${vt ? `<td class="source-counts" data-count="predicted">${sourceCounts(r, "predicted")}</td><td class="source-counts" data-count="true">${sourceCounts(r, "true")}</td>` : ""}<td>${count(m.trial_count)}</td></tr>`,
      )
      .join("") ||
    `<tr><td class="empty" colspan="${columns.length + 4 + Number(showSource) + 2 * Number(vt)}">No completed results match these filters.</td></tr>`;
  $("export").disabled = !rows.length;
  $("metric-note").textContent = vt
    ? `AP ${state.ap} · All metrics and counts use only this AP selection. OSPA and F1 average noise repetitions. Matched RMSE pools squared 3D distances over one-to-one matches within 0.5 m; no matches = —. Counts that vary are listed per repetition.`
    : `${obstacleLabel()} · All metrics use saved 3D query errors, including noise repetitions. The obstacle selector also controls the chart and CSV.`;
  $("results-panel").setAttribute("aria-busy", "false");
  const best = rows
    .filter((r) =>
      Number.isFinite(conditionMetric(r.condition, primaryMetric())),
    )
    .sort(
      (a, b) =>
        conditionMetric(a.condition, primaryMetric()) -
        conditionMetric(b.condition, primaryMetric()),
    )[0];
  const cards = [
    [
      "Configurations",
      count(rows.length),
      "Completed results in this selection",
    ],
    [
      `Lowest ${metricColumns().find(([key]) => key === primaryMetric())[1]}`,
      best
        ? metricValue(
            primaryMetric(),
            conditionMetric(best.condition, primaryMetric()),
          ) + " <small>m</small>"
        : "—",
      best ? best.method.title : "No matching results",
    ],
    [
      rows.length && rows.every((r) => r.method.selection === "fixed") ? "Parameter protocol" : "Selection noise",
      rows.length && rows.every((r) => r.method.selection === "fixed") ? "Fixed" : "0 <small>m</small>",
      "Fixed baselines or zero-noise tuning; settings frozen across noise levels",
    ],
    [
      "Search coverage",
      count(data.trials.length),
      "Completed selection trials across the experiment",
    ],
  ];
  $("cards").innerHTML = cards
    .map(
      ([label, value, note], i) =>
        `<article class="stat-card ${i === 1 ? "accent" : ""}"><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-caption">${esc(note)}</div></article>`,
    )
    .join("");
  renderNoiseChart();
}
function lineChart(series, levels, yLabel, unit = "m") {
  series = series
    .map((s) => ({
      ...s,
      values: s.values.filter(([, value]) => Number.isFinite(value)),
    }))
    .filter((s) => s.values.length);
  if (!series.some((s) => s.values.length))
    return '<div class="chart-empty">No completed measurements yet.</div>';
  const w = 760,
    h = 270,
    L = 55,
    R = 20,
    T = 18,
    B = 52;
  const max =
    Math.max(0.01, ...series.flatMap((s) => s.values.map((v) => v[1]))) * 1.08;
  const x = (i) => L + (i * (w - L - R)) / Math.max(1, levels.length - 1),
    y = (v) => h - B - (v * (h - T - B)) / max;
  let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(yLabel)} across noise levels">`;
  for (let i = 0; i < 5; i++) {
    const v = (max * i) / 4;
    svg += `<line x1="${L}" x2="${w - R}" y1="${y(v)}" y2="${y(v)}" stroke="#e3eaed"/><text x="${L - 9}" y="${y(v) + 4}" text-anchor="end" fill="#73848a" font-size="10">${fmt(v, 2)}</text>`;
  }
  levels.forEach(
    (v, i) =>
      (svg += `<text x="${x(i)}" y="${h - 31}" text-anchor="middle" fill="#73848a" font-size="10">${v}</text>`),
  );
  for (const s of series) {
    svg += `<polyline points="${s.values.map((v) => `${x(levels.indexOf(v[0]))},${y(v[1])}`).join(" ")}" stroke="${s.color}" stroke-width="2" fill="none"/>`;
    s.values.forEach(
      (v) =>
        (svg += `<circle cx="${x(levels.indexOf(v[0]))}" cy="${y(v[1])}" r="3" fill="${s.color}"><title>${esc(s.title)} · σ ${v[0]} m · ${fmt(v[1])} ${unit}</title></circle>`),
    );
  }
  return (
    svg +
    `<text x="${w / 2}" y="${h - 7}" text-anchor="middle" font-size="11" fill="#73848a">Range noise σ (m; discrete levels)</text><text x="${L}" y="11" font-size="10" fill="#73848a">${esc(yLabel)}</text></svg>`
  );
}
function renderNoiseChart() {
  const hidden = new Set(state.hidden.split(",").filter(Boolean));
  const methods = visibleMethods();
  const series = methods.map((m) => ({
    id: m.id,
    title: m.title,
    color: palette[data.methods.indexOf(m) % palette.length],
    values: data.conditions
      .filter((r) => r.method_id === m.id && r.split === "evaluation" && conditionContextMatches(r))
      .sort((a, b) => a.level - b.level)
      .map((r) => [r.level, chartMetricValue(r)]),
  }));
  $("noise-chart").innerHTML = lineChart(
    series.filter((s) => !hidden.has(s.id)),
    data.config.observations.noise_levels,
    chartMetricLabel(),
  );
  if (
    series.some((s) => s.values.length) &&
    !series.some((s) => s.values.length && !hidden.has(s.id))
  )
    $("noise-chart").innerHTML =
      '<div class="chart-empty">Select one or more curves below.</div>';
  $("chart-legend").innerHTML =
    '<div class="curve-actions"><strong>Display curves</strong><button class="text-button" data-curves="all">Select all</button><button class="text-button" data-curves="none">Clear all</button></div>' +
    (state.view !== "vt"
      ? '<p class="curve-key">MAP = MPUrge-MAP · LEA = SMART-LEA</p>'
      : "") +
    series
      .filter((s) => s.values.length)
      .map(
        (s) =>
          `<label class="curve-option"><input type="checkbox" data-series="${s.id}" ${hidden.has(s.id) ? "" : "checked"}><i class="color-dot" style="background:${s.color}"></i><span>${esc(s.title)}</span></label>`,
      )
      .join("");
  $("chart-legend").onchange = (e) => {
    const id = e.target.dataset.series;
    if (!id) return;
    e.target.checked ? hidden.delete(id) : hidden.add(id);
    state.hidden = [...hidden].join(",");
    renderNoiseChart();
    saveURL();
  };
  $("chart-legend").onclick = (e) => {
    const action = e.target.dataset.curves;
    if (!action) return;
    series.forEach((s) =>
      action === "all" ? hidden.delete(s.id) : hidden.add(s.id),
    );
    state.hidden = [...hidden].join(",");
    renderNoiseChart();
    saveURL();
  };
}
function resetEvidence() {
  loaded = null;
  loading = null;
  selectedCase = 0;
  auditHistory = [];
  currentArtifact = null;
}
function openMethod(id, tab = "overview", trial = null) {
  const method = data.methods.find((m) => m.id === id),
    row = method && rowFor(method);
  if (!row) return;
  focusReturn = document.activeElement;
  currentMethod = method;
  currentRow = row;
  mapChoice = { room: "", layout: "", repetition: null };
  activeReceipt = receiptFor(row.receipt_id);
  if (trial)
    activeReceipt =
      data.trials.find((t) => t.id === trial && t.label === method.label) ||
      activeReceipt;
  detailTab = allTabs.includes(tab) ? tab : "overview";
  searchPage = 0;
  trialSearch = "";
  trialSort = "loss";
  resetEvidence();
  $("drawer").hidden = false;
  $("drawer-backdrop").hidden = false;
  $("main").inert = true;
  document.querySelector(".sidebar").inert = true;
  document.body.style.overflow = "hidden";
  $("drawer").focus();
  renderDetail();
}
function closeDetail(updateURL = true) {
  tabVersion++;
  disposeMap?.();
  disposeMap = null;
  currentMethod = null;
  activeReceipt = null;
  resetEvidence();
  $("drawer").hidden = true;
  $("drawer-backdrop").hidden = true;
  $("main").inert = false;
  document.querySelector(".sidebar").inert = false;
  document.body.style.overflow = "";
  focusReturn?.focus();
  if (updateURL) saveURL();
}
function activateTrial(id) {
  const trial = data.trials.find(
    (t) => t.id === id && t.label === currentMethod.label,
  );
  if (!trial) return;
  activeReceipt = trial;
  resetEvidence();
  detailTab = "parameters";
  renderDetail();
}
function restoreChoice() {
  activeReceipt = receiptFor(currentRow.receipt_id);
  resetEvidence();
  renderDetail();
}
function selectedMetrics() {
  if (currentMethod.group === "vt")
    return activeReceipt.id === currentRow.receipt_id
      ? currentRow.source_metrics_by_ap?.[state.ap]
      : null;
  if (activeReceipt.id === currentRow.receipt_id)
    return currentMethod.group !== "vt" && state.obstacles !== ""
      ? currentRow.obstacle_metrics?.[state.obstacles]
      : currentRow.metrics;
  if (currentMethod.group !== "vt" && state.obstacles !== "") return null;
  if (!currentMethod.derived) return activeReceipt.metrics;
  // A joint trial's objective is final localization error, never its source-map OSPA.
  return null;
}
function metricCards(metrics) {
  if (!metrics)
    return '<p class="detail-hint">Metrics for this selection are available from the Map tab’s replayed cases. The full-pipeline selection loss is not a component score.</p>';
  const columns = metricColumns(currentMethod.group);
  return `<div class="detail-metrics">${columns
    .filter(([k]) => Number.isFinite(metrics[k]) || k === "matched_rmse_m")
    .map(
      ([k, label]) =>
        `<div class="detail-metric">${label}<strong>${metricValue(k, metrics[k])}</strong></div>`,
    )
    .join("")}</div>`;
}
function showError(error) {
  return `<p class="error-message" role="alert">${esc(error.message)}</p>`;
}
function obstacleBreakdown() {
  const condition = data.conditions.find(
    (row) =>
      row.method_id === currentMethod.id && row.receipt_id === activeReceipt.id,
  );
  const groups = Object.entries(condition?.obstacle_metrics || {});
  if (!groups.length) return "";
  return `<h3>Localization by obstacle count</h3><div class="table-scroll"><table class="obstacle-breakdown"><thead><tr><th>Obstacles</th><th>Query estimates</th><th>Mean (m)</th><th>RMSE (m)</th><th>Median (m)</th><th>P90 (m)</th></tr></thead><tbody>${groups.map(([n, metrics]) => `<tr><td>${n}</td><td>${count(metrics.query_count)}</td>${["mean_m", "rmse_m", "median_m", "p90_m"].map((key) => `<td>${fmt(metrics[key])}</td>`).join("")}</tr>`).join("")}</tbody></table></div><p class="inline-note">Each column uses the saved predictions for that obstacle count. Query estimates include noise repetitions. Calibration used all four layouts at zero noise.</p>`;
}
async function renderDetail() {
  const version = ++tabVersion;
  disposeMap?.();
  disposeMap = null;
  $("detail-title").textContent = displayName(currentMethod);
  $("detail-subtitle").innerHTML = detailBadges(currentMethod);
  $("drawer").classList.toggle("map-open", detailTab === "map");
  document.querySelectorAll("[data-tab]").forEach((b) => {
    b.setAttribute("aria-selected", b.dataset.tab === detailTab);
    b.tabIndex = b.dataset.tab === detailTab ? 0 : -1;
  });
  $("detail-content").setAttribute("aria-labelledby", "tab-" + detailTab);
  const historical = activeReceipt.id !== currentRow.receipt_id;
  $("trial-context").hidden = false;
  $("trial-context").innerHTML =
    `${historical ? `Inspecting calibration trial ${activeReceipt.number + 1}` : `${activeReceipt.split === "calibration" ? "Noise-free calibration" : "Frozen-parameter evaluation"} · σ ${activeReceipt.level} m`}${historical ? ` · ${currentMethod.objective}: ${fmt(activeReceipt.loss)} <button class="text-button" id="restore-choice">Return to selected result</button>` : ""}`;
  $("restore-choice")?.addEventListener("click", restoreChoice);
  saveURL();
  const panel = $("detail-content");
  if (detailTab === "overview") {
    panel.innerHTML =
      metricCards(selectedMetrics()) +
      (currentMethod.group === "vt" &&
      activeReceipt.id === currentRow.receipt_id
        ? `<p class="inline-note">AP ${state.ap}</p><div class="detail-metrics"><div class="detail-metric">Predicted<strong>${sourceCounts(currentRow, "predicted")}</strong></div><div class="detail-metric">True<strong>${sourceCounts(currentRow, "true")}</strong></div></div>`
        : "") +
      (currentMethod.group === "vt"
        ? ""
        : `<p class="inline-note">${obstacleLabel()}</p>`) +
      obstacleBreakdown() +
      `<h3>How this configuration was selected</h3><p>${selectionExplanation()}</p>${activeReceipt.obstacle_count !== undefined ? obstacleEvaluationNote() : currentMethod.derived ? '<p class="detail-hint">This is a saved intermediate stage of the joint experiment. Its parameters were selected by final localization error, not by this component’s own score. No additional experiment was run.</p>' : ""}<h3>Evaluation across noise levels</h3>` +
      lineChart(
        [
          {
            title: currentMethod.title,
            color: "#16836e",
            values: data.conditions
              .filter(
                (r) =>
                  r.method_id === currentMethod.id && r.split === "evaluation" && conditionContextMatches(r),
              )
              .sort((a, b) => a.level - b.level)
              .map((r) => [r.level, chartMetricValue(r)]),
          },
        ],
        data.config.observations.noise_levels,
        chartMetricLabel(),
      ) +
      `<h3>Saved evidence</h3><p>Inspect the map, ${count(currentMethod.trial_count)} search trials, and numerical stages using the tabs above.</p><details class="receipt-details"><summary>Result provenance</summary><pre>${esc(JSON.stringify({ run: activeReceipt.run || currentMethod.run || data.config.name, receipt_id: activeReceipt.id, search: currentMethod.label, split: activeReceipt.split, noise_m: activeReceipt.level, outcome: activeReceipt.outcome || "Recreated on demand" }, null, 2))}</pre></details>`;
  } else if (detailTab === "method") renderMethod(panel);
  else if (detailTab === "code") renderCode(panel);
  else if (detailTab === "parameters") renderParameters(panel);
  else if (detailTab === "search") renderSearch(panel);
  else {
    panel.innerHTML =
      '<div class="chart-empty" role="status">Loading saved evidence or recreating this trial…</div>';
    try {
      await loadEvidence();
      if (version !== tabVersion) return;
      if (detailTab === "map") await renderMap(panel, version);
      else renderEvidence(panel);
    } catch (error) {
      if (version === tabVersion) panel.innerHTML = showError(error);
    }
  }
  if (["method", "code"].includes(detailTab)) {
    const url = `/construction?trial=${activeReceipt.id}&component=${currentMethod.component}`;
    panel.querySelectorAll('a[href*="execution%2Fcompositions.py"], a[href*="execution%2Fpresets.py"], a[href*="execution/compositions.py"], a[href*="execution/presets.py"]').forEach(a => {
      if (/constructor|composed_reconstructor|standard_mpurge|mpurge_map|smart_lea/i.test(a.textContent)) {
        a.href = url;
        a.textContent = "Concrete method construction ↗";
      }
    });
    panel.querySelectorAll('.algorithm-constructor a').forEach(a => {
      a.href = url;
      a.textContent = "Concrete method construction ↗";
    });
    const card = document.createElement("section");
    card.className = "method-section construction-card";
    card.innerHTML = `<h3>Concrete method construction</h3><a href="${url}" target="_blank" rel="noopener">Open constructor with the selected values ↗</a><details><summary>Show injected dependencies</summary><pre>Loading construction…</pre></details>`;
    panel.prepend(card);
    try {
      const value = await api(`/api/construction?trial=${activeReceipt.id}&component=${currentMethod.component}`);
      if (version !== tabVersion) return;
      card.querySelector("pre").textContent = value.code;
      value.notes.forEach(note => {
        const p = document.createElement("p"); p.textContent = note; card.append(p);
      });
    } catch (error) { card.querySelector("pre").textContent = error.message; }
  }
}
function selectionExplanation() {
  if (currentMethod.family === "range_rmse")
    return `Fixed numerical settings, with no calibration search. At each noise realization and for each AP, run 16 starts and keep the one with the lowest measured range residual. OSPA, F1, matched RMSE and true source coordinates do not select the result. Only the true source count for this scene and AP is supplied: A=${currentRow?.source_counts?.[0]?.true.A ?? 25}, B=${currentRow?.source_counts?.[0]?.true.B ?? 25}, including the direct AP when visible. Its position is also estimated.`;
  if (currentMethod.selection === "joint")
    return (
      esc(currentMethod.selection_description) +
      (currentMethod.group === "raw"
        ? " This row measures the MAP position estimate before SMART-LEA refinement. The pipeline name identifies the tuning context; MAP itself matches query and survey fingerprints."
        : currentMethod.group === "vt"
          ? " This row measures the saved reconstruction stage; its OSPA was not the joint selection objective."
          : "")
    );
  if (currentMethod.family && currentMethod.family !== "mpurge") {
    const family = currentMethod.source_full_name;
    if (currentMethod.selection === "ospa")
      return `${family} parameters were selected on noise-free calibration surveys to minimize OSPA. Size unification does not apply to this reconstructor.`;
    if (currentMethod.selection === "stagewise")
      return `The independently OSPA-tuned ${family} and the independently tuned MPUrge-MAP settings are frozen while SMART-LEA is tuned on noise-free calibration queries.`;
    return `${family}, MPUrge-MAP and SMART-LEA parameters were selected together to minimize final localization error on noise-free calibration queries.`;
  }
  return {
    ospa: `${currentMethod.source_full_name} parameters were selected on noise-free calibration data to minimize OSPA. This source map is then frozen into the stage-wise localization experiment.`,
    localization:
      "MPUrge-MAP parameters were selected on noise-free calibration queries to minimize mean 3D localization error.",
    stagewise: `SMART-LEA was tuned on noise-free calibration queries. ${currentMethod.source_full_name} and MPUrge-MAP use their independently selected, frozen settings.`,
    truth:
      "SMART-LEA was tuned using true VT coordinates and the independently tuned, frozen MPUrge-MAP initializer. This is an oracle source-map comparison.",
    joint:
      "MPUrge, MPUrge-MAP and SMART-LEA were selected together on noise-free calibration queries to minimize final mean 3D localization error.",
  }[currentMethod.selection];
}
const parameterLabels = {
  starts: "Independent numerical starts",
  max_iterations: "Maximum assignment / refit iterations",
  max_nfev: "Maximum evaluations per source refit",
  relative_tolerance: "Relative squared-residual stopping tolerance",
  seed: "Initialization seed",
  half_window: "Pattern half-window",
  alpha: "Range weight (α)",
  neighbors: "Nearest neighbors",
  radius_m: "Clustering radius (m)",
  beta: "Cluster support multiplier",
  power: "Distance-weighting power",
  iterations: "Refinement iterations",
  scope_radius_m: "Survey neighborhood radius (m)",
  gap_m: "Range-cluster gap threshold (m)",
  group_cost: "Cluster-grouping cost threshold",
  tolerance_m: "Finisher inlier tolerance (m)",
  support_fraction: "Minimum supporting survey fraction",
  duplicate_m: "Duplicate range-curve RMS threshold (m)",
  smooth_weight: "Smooth range agreement weight",
  median_penalty_weight: "Median normalized range error penalty weight",
  median_clip_factor: "Median error clipping threshold (multiples of tolerance)",
  trials: "RANSAC proposal trials",
  triangle_tolerance_m: "Triangle gate slack (m)",
};
function parameterTable(parameters, title) {
  const nested = Object.entries(parameters).filter(
    ([, value]) => value !== null && typeof value === "object",
  );
  const scalar = Object.entries(parameters).filter(
    ([, value]) => value === null || typeof value !== "object",
  );
  const isMap =
    title.includes("MPUrge-MAP") || activeReceipt.method === "mpurge_map";
  const groups = [
    ["Numerical optimization", ["starts", "max_iterations", "max_nfev", "relative_tolerance", "seed"]],
    [
      "Survey neighborhoods",
      ["scope_radius_m", ...(isMap ? [] : ["neighbors"])],
    ],
    ["Range clusters and grouping", ["gap_m", "group_cost"]],
    ["Peak matching", ["half_window", "alpha", "triangle_tolerance_m"]],
    ["RANSAC proposals", ["trials", ...(currentMethod.naked && currentMethod.family === "ransac"
      ? ["tolerance_m"] : [])]],
    ["Candidate clustering and support", ["radius_m", "beta"]],
    [
      finisherNames[currentMethod.finisher] || "Source selection",
      [...(currentMethod.naked ? [] : ["tolerance_m"]), "smooth_weight", "median_penalty_weight", "median_clip_factor",
       "support_fraction", "duplicate_m"],
    ],
    [
      "Fingerprint position estimate",
      [...(isMap ? ["neighbors"] : []), "power"],
    ],
    ["SMART-LEA refinement", ["iterations"]],
  ];
  const known = new Set(groups.flatMap(([, keys]) => keys));
  groups.push([
    "Other settings",
    scalar.map(([key]) => key).filter((key) => !known.has(key)),
  ]);
  const rows = (keys) =>
    keys
      .filter((key) => scalar.some(([k]) => k === key))
      .map((key) => {
        const value = parameters[key];
        const label = key === "tolerance_m" && currentMethod.naked && currentMethod.family === "ransac"
          ? "RANSAC proposal range tolerance (m)" : parameterLabels[key] || key;
        return `<tr><th>${esc(label)}<br><code>${esc(key)}</code></th><td class="parameter-value">${esc(value)}${key === "alpha" ? `<div class="method-meta">Pattern weight (1 − α): ${esc(Number((1 - value).toPrecision(12)))}</div>` : ""}</td></tr>`;
      })
      .join("");
  return (
    `<section class="parameter-section"><h3>${esc(title)}</h3>` +
    groups
      .map(([label, keys]) => {
        const content = rows(keys);
        return content
          ? `<h4 class="parameter-group">${esc(label)}</h4><table class="parameter-table"><tbody>${content}</tbody></table>`
          : "";
      })
      .join("") +
    nested
      .map(([key, value]) =>
        parameterTable(
          value,
          {
            mpurge: "MPUrge reconstruction",
            source: currentMethod.source_full_name,
            map: "MPUrge-MAP",
            lea: "SMART-LEA",
          }[key] || key,
        ),
      )
      .join("") +
    "</section>"
  );
}
function renderParameters(panel) {
  if (hasReconstructionExplanation(currentMethod)) {
    const other = currentMethod.selection === "ospa" ? {} :
      Object.fromEntries(Object.entries(activeReceipt.parameters).filter(([key]) => key !== "source"));
    const frozen = Object.fromEntries(Object.entries(activeReceipt.frozen || {}).filter(([key]) => key !== "source"));
    panel.innerHTML = finisherParameterSummary(currentMethod, activeReceipt) +
      (Object.keys(other).length ? parameterTable(other, "Localization parameters") : "") +
      (Object.keys(frozen).length ? parameterTable(frozen, "Frozen localization parameters") : "") +
      `<details class="receipt-details"><summary>Parameter selection and exact record</summary><p>${selectionExplanation()}</p><pre>${esc(JSON.stringify({ parameters: activeReceipt.parameters, frozen: activeReceipt.frozen, unify: activeReceipt.unify }, null, 2))}</pre></details>` +
      codeLinks(["Composition constructors"]);
    return;
  }
  const source = sourceParameters(currentMethod, activeReceipt);
  const fixedRobust = currentMethod.finisher === "robust" && source
    ? Object.fromEntries(Object.entries({smooth_weight: 0.35,
        median_penalty_weight: 0.10, median_clip_factor: 4})
        .filter(([key]) => !(key in source)))
    : {};
  panel.innerHTML =
    `<p class="detail-hint">${activeReceipt.id === currentRow.receipt_id ? "Parameters of this selected configuration." : "Parameters of the inspected calibration trial."} Size unification is <strong>${activeReceipt.unify === null ? "not applicable to this reconstructor" : activeReceipt.unify ? "on" : "off"}</strong>.</p>` +
    parameterTable(
      activeReceipt.parameters,
      activeReceipt.method === "joint" ||
        activeReceipt.method.endsWith("_joint")
        ? "Jointly selected pipeline parameters"
        : currentMethod.family === "range_rmse" ? "Fixed numerical settings (not tuned)" : "Tuned parameters",
    ) +
    (Object.keys(activeReceipt.frozen || {}).length
      ? parameterTable(activeReceipt.frozen, "Frozen upstream parameters")
      : "") +
    (Object.keys(fixedRobust).length
      ? parameterTable(fixedRobust, "Fixed robust-finisher values in this saved run (not optimized)")
      : "") +
    finisherParameterSummary(currentMethod, activeReceipt) +
    `<h3>Selection protocol</h3><p>${selectionExplanation()}</p><details class="receipt-details"><summary>Exact parameter record</summary><pre>${esc(JSON.stringify({ parameters: activeReceipt.parameters, frozen: activeReceipt.frozen, unify: activeReceipt.unify }, null, 2))}</pre></details>`;
  panel.innerHTML += codeLinks([
    "Composition constructors",
    "Finisher implementation",
    "Execution and experiment configuration",
  ]);
  if (currentMethod.search_description)
    panel.insertAdjacentHTML(
      "afterbegin",
      `<p class="detail-hint">${esc(currentMethod.search_description)}</p>`,
    );
  if (currentMethod.family !== "mpurge" && activeReceipt.unify !== null)
    panel.insertAdjacentHTML(
      "afterbegin",
      '<p class="detail-hint">Size unification applies to the MPUrge-MAP and SMART-LEA matchers. The source reconstructor and its finisher do not use it.</p>',
    );
}
function codeLinks(groups = Object.keys(currentMethod.code || {})) {
  return groups
    .filter((group) => currentMethod.code?.[group])
    .map(
      (group) =>
        group === "Composition constructors" ? `<section class="method-section"><h3>Concrete construction</h3><a href="/construction?trial=${activeReceipt.id}&component=${currentMethod.component}" target="_blank" rel="noopener">${esc(currentMethod.name)} — injected dependencies ↗</a></section>` :
        `<section class="method-section"><h3>${esc(group)}</h3><div class="source-links">${currentMethod.code[
          group
        ]
          .map((link) => {
            const item =
              typeof link === "string" ? { file: link, label: link } : link;
            return `<a href="/source?file=${encodeURIComponent(item.file)}${item.line ? "#L" + item.line : ""}" target="_blank" rel="noopener">${esc(item.label)} ↗</a>`;
          })
          .join("")}</div></section>`,
    )
    .join("");
}
function renderCode(panel) {
  if (hasReconstructionExplanation(currentMethod)) {
    panel.innerHTML = reconstructionExplanation(currentMethod, activeReceipt) +
      `<section class="algorithm-cache"><h3>Execution &amp; caching</h3><p>Step links above open the numerical functions. The constructor assembles the proposer${currentMethod.finisher ? " and selected finisher" : ""}; the execution layer caches ${currentMethod.family === "mpurge" ? "pair matches, source proposals and final source maps" : "source proposals and final source maps"} separately.</p><p>Saved outcomes are read from this run's receipt. This explanation does not rerun calibration or reconstruction.</p><p class="inline-note">Run: ${esc(activeReceipt.run || currentMethod.run || data.config.name)} · Receipt: <code>${esc(activeReceipt.id)}</code></p>` +
      codeLinks(["Execution and experiment configuration", "Caching and saved evidence"]) + "</section>";
    return;
  }
  if (currentMethod.family === "range_rmse") {
    panel.innerHTML = `<h3>Execution and saved evidence</h3><p>The constructor below builds the range solver. Its fit call receives only receiver coordinates, noisy range arrays and an integer source count. The evaluator reads the count from truth and uses coordinates only after fitting to calculate the displayed metrics.</p><p>This study saves its own source maps, assignments, per-start residual histories and receipts. Completed conditions are reused on restart; existing methods and their cached outcomes are untouched. Saved outputs remain readable after unrelated code changes.</p><p class="inline-note">Run: ${esc(activeReceipt.run || currentMethod.run)} · Receipt: <code>${esc(activeReceipt.id)}</code></p>` + codeLinks();
    return;
  }
  panel.innerHTML = connectedComponentSourceOverview(currentMethod) +
    `<h3>Execution and caching</h3><p>These links open the current local code used to configure and execute this method. Computation identities include the scientific implementation, effective parameters and numerical inputs. Repeated stages reuse cached typed arrays.</p><p>The shared numerical cache is limited to ${data.config.storage.cache_mib} MiB on disk and ${data.config.storage.ram_mib} MiB in memory. Selected outcomes are saved with the run; evicted intermediate evidence can be recreated from its receipt after checking the implementation and dataset versions.</p><p class="inline-note">Run: ${esc(activeReceipt.run || currentMethod.run || data.config.name)} · Receipt: <code>${esc(activeReceipt.id)}</code>. Open Evidence to inspect individual cached stages and array pages.</p>` +
    codeLinks([
      "Composition constructors",
      "Finisher implementation",
      "Execution and experiment configuration",
      "Caching and saved evidence",
    ]);
}
function renderMethod(panel) {
  if (hasReconstructionExplanation(currentMethod)) {
    panel.innerHTML = reconstructionExplanation(currentMethod, activeReceipt) +
      obstacleEvaluationNote() +
      `<details class="algorithm-detail"><summary>Parameter selection and displayed metrics</summary><p>${selectionExplanation()}</p><p>${currentMethod.group === "vt" ? "OSPA: p = 1, cutoff 1 m. F1: one-to-one matches within 0.5 m. Matched RMSE: square root of the mean squared 3D distance over maximum-cardinality matches, using the assignment with the smallest squared distance. Unmatched sources are excluded from matched RMSE. Metrics use the selected APs." : "This reconstruction supplies the source map for the selected localization pipeline. Query errors compare estimated and true 3D receiver coordinates."}</p></details>`;
    if (currentMethod.group === "vt") panel.innerHTML += sourceMetricExplanation();
    return;
  }
  if (currentMethod.family === "range_rmse") {
    panel.innerHTML = `<h3>Range RMSE (cheating true count)</h3>
      <p>Estimate exactly <strong>K source positions per AP</strong>, using the scene/AP count shown in the table, including the direct AP when visible. Neither true source positions nor peak identities are supplied. AP A and B are solved independently. Source coordinates are unrestricted in 3D, so virtual sources may lie outside the room.</p>${obstacleEvaluationNote()}
      <h3>Objective and unknowns</h3>
      <p><code>min over q and π: L = √[(1/N) Σₛ Σᵢ (yₛᵢ − ‖xₛ − q[πₛ(i)]‖₂)²]</code></p>
      <p><strong>xₛ</strong> is the known 3D receiver position; <strong>yₛᵢ</strong> is its observed noisy range for peak i; <strong>qₖ</strong> is unknown source position k; <strong>πₛ(i)</strong> assigns that peak to a source, with no source used twice within one fingerprint; <strong>N</strong> is the number of observed peaks across all survey fingerprints. Minimizing L is equivalent to minimizing the sum of squared residuals inside it.</p>
      <ol class="method-steps"><li><strong>Initialize without source truth.</strong> One start pairs equal ranks across fingerprints and uses algebraic multilateration, when all fingerprints are complete. Other starts sample a receiver, take its measured ranges as sphere radii, and place candidate sources along random directions. Missing radii are sampled from the observed range distribution.</li>
      <li><strong>Assign peaks.</strong> With q fixed, Hungarian matching minimizes <code>Σᵢ (yₛᵢ − ‖xₛ − q[πₛ(i)]‖₂)²</code> independently at each receiver. Assignments may cross in range order across receivers.</li>
      <li><strong>Move sources.</strong> With π fixed, each source minimizes the sum of its squared range residuals using ordinary nonlinear least squares (Levenberg–Marquardt), with at most 60 evaluations per refit.</li>
      <li><strong>Repeat and select.</strong> Stop after 150 assignment/refit iterations or relative squared-residual improvement ≤ 10⁻⁸. Run 16 starts; retain the smallest measured residual. This numerical solution can be a local minimum and is not a certified global optimum.</li></ol>
      <h3>Noise and missing observations</h3><p>The same saved survey positions, Gaussian range-noise seeds and noise levels are used as the other methods. The benchmark removes ranges that become negative after noise. The resulting rectangular assignment matches every remaining observation to a distinct source; unused source slots contribute no residual. No clean measurements or noise-free solution initialize a noisy condition. There are two evaluation realizations per positive noise level and one at zero noise.</p>
      <h3>What the three reconstruction metrics mean</h3><p>These compare reconstructed coordinates with truth after fitting; they are different from the measured-range RMSE minimized above. OSPA uses p = 1 and c = 1 m. F1 uses the maximum number of one-to-one source matches within 0.5 m. Matched RMSE is √(sum of squared 3D distances / matched count), using the least squared distance among assignments with that maximum match count. It excludes unmatched sources. Metrics follow the selected A, B or A+B view; OSPA and F1 average realizations, while matched RMSE pools matched pairs.</p>
      <h3>True-count advantage</h3><p>${selectionExplanation()}</p>` + codeLinks(["Composition constructors", "Scientific algorithms", "Observation conversion and metrics"]);
    if (currentMethod.group === "vt") panel.innerHTML += sourceMetricExplanation();
    return;
  }
  const steps = {
    sources: [
      "Match peaks between neighboring survey fingerprints using the selected pattern window and range weight.",
      "Build connected components within each local receiver scope.",
      "Fit a source to each group, cluster candidates, and retain clusters meeting the selected support threshold.",
    ],
    initial: [
      "Match each query fingerprint to survey fingerprints and compute a coverage-normalized mean match cost.",
      "Average costs across APs and estimate the position with distance-weighted nearest neighbors.",
    ],
    localization:
      activeReceipt.method === "mpurge_map"
        ? [
            "Match query and survey fingerprints using MPUrge.",
            "Average coverage-normalized match costs across APs and estimate position with weighted nearest neighbors.",
          ]
        : [
            "Initialize each query with MPUrge-MAP using the selected fingerprint parameters.",
            activeReceipt.method === "smart_lea_truth"
              ? "Use the true source coordinates for this oracle comparison."
              : "Use the MPUrge reconstructed source map.",
            "Match measured ranges to ranges predicted from the current position, then refit the position for the selected number of iterations.",
          ],
  };
  if (currentMethod.family === "zayets")
    steps.sources = [
      "Split each fingerprint into range clusters and compare cluster features within local survey neighborhoods.",
      "Group clusters using constrained single linkage; align peaks to the longest-cluster template, then estimate source coordinates by the ordinary range least-squares objective defined below.",
    ];
  if (currentMethod.family === "ransac")
    steps.sources = [
      "Sample three survey positions and peaks, solve sphere intersections, and retain proposals in trial batches using the RANSAC retention formula defined below.",
    ];
  if (currentMethod.finisher) {
    steps.sources = steps.sources.filter(
      (s) => !s.includes("finisher") && !s.startsWith("Consume"),
    );
    if (currentMethod.family === "mpurge")
      steps.sources[2] =
        "Estimate a 3D source proposal for each connected peak group by the ordinary range least-squares objective defined below. The selected finisher replaces candidate clustering and the relative-support filter.";
    steps.sources.push(
      `Apply the ${esc(finisherNames[currentMethod.finisher])} described below.`,
    );
  }
  if (currentMethod.naked) {
    steps.sources = steps.sources.filter(
      (s) => !s.includes("finisher") && !s.startsWith("Consume"),
    );
    steps.sources.push("Return every retained candidate directly as a source, with no final support threshold, coordinate refinement, duplicate suppression or peak consumption.");
  }
  if (currentMethod.triangle)
    steps.sources[0] +=
      " Before matching, apply both triangle inequalities: the range difference must not exceed the survey-position separation plus slack, and the range sum plus slack must reach that separation.";
  if (currentMethod.allow_crossing)
    steps.sources[0] += " Accept reciprocal-minimum matches even when they cross in range order; purge both endpoints as usual. This changes reconstruction matching only; MPUrge-MAP and SMART-LEA retain their standard matchers.";
  if (
    activeReceipt.method !== "mpurge_map" &&
    activeReceipt.method !== "smart_lea_truth" &&
    currentMethod.component === "localization"
  )
    steps.localization[1] = `Use the source map from ${currentMethod.source_full_name}.`;
  const sourceOverview = connectedComponentSourceOverview(currentMethod);
  const methodSteps = `<ol class="method-steps">${steps[currentMethod.component].map((s) => `<li>${s}</li>`).join("")}</ol>`;
  const algorithm = sourceOverview
    ? sourceOverview + (currentMethod.component === "localization" ? `<h3>Query localization</h3>${methodSteps}` : "")
    : `<h3>${esc(currentMethod.group === "vt" ? currentMethod.source_full_name : currentMethod.name)}</h3>${methodSteps}`;
  panel.innerHTML = algorithm + `<h3>Parameter selection</h3><p>${selectionExplanation()}</p><h3>What is compared</h3><p>${currentMethod.group === "vt" ? "OSPA uses p = 1 and a 1 m cutoff. F1 uses maximum-cardinality, one-to-one matches within 0.5 m. Matched RMSE is the square root of the mean squared 3D distance over those matches, choosing the smallest squared distance among assignments with the maximum match count. It excludes unmatched sources and is unavailable when there are no matches. All displayed metrics use only the AP selection above. OSPA and F1 average noise repetitions; matched RMSE pools matched pairs across repetitions. Search history retains the recorded selection objective." : "Localization metrics are calculated over individual 3D query errors, including fallback estimates. The evaluation split uses held-out query positions in the same room."}</p><p class="inline-note">${currentMethod.derived ? "This row displays an intermediate output retained by the joint pipeline experiment." : "Only this redesign experiment supplies the displayed results."}</p>`;
  panel.innerHTML += obstacleEvaluationNote() + codeLinks([
    "Composition constructors",
    "Finisher implementation",
    "Scientific algorithms",
    "Observation conversion and metrics",
  ]);
  if (currentMethod.group === "vt") panel.innerHTML += sourceMetricExplanation();
}

function searchPlot(trials) {
  if (!trials.length) return "";
  const ordered = [...trials].sort((a, b) => a.number - b.number),
    w = 640,
    h = 180,
    L = 48,
    B = 30,
    T = 15,
    R = 15;
  const max = Math.max(0.01, ...ordered.map((t) => t.loss)) * 1.05;
  const x = (n) => L + (n * (w - L - R)) / Math.max(1, ordered.at(-1).number),
    y = (v) => h - B - (v * (h - B - T)) / max;
  let best = Infinity,
    points = [];
  const dots = ordered
    .map((t) => {
      best = Math.min(best, t.loss);
      points.push(`${x(t.number)},${y(best)}`);
      return `<circle cx="${x(t.number)}" cy="${y(t.loss)}" r="1.7" fill="#a9bbb5" opacity=".6"><title>Trial ${t.number + 1}: ${fmt(t.loss)}</title></circle>`;
    })
    .join("");
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Calibration trial loss and best-so-far loss"><path d="M${L} ${T}V${h - B}H${w - R}" fill="none" stroke="#d6e0dc"/><text x="${L - 6}" y="${T + 8}" text-anchor="end" font-size="10" fill="#6c7c76">${fmt(max, 2)}</text><text x="${L - 6}" y="${h - B}" text-anchor="end" font-size="10" fill="#6c7c76">0</text>${dots}<polyline points="${points.join(" ")}" fill="none" stroke="#16836e" stroke-width="2"/><text x="${L}" y="${h - 8}" font-size="10" fill="#6c7c76">Trial 1</text><text x="${w - R}" y="${h - 8}" text-anchor="end" font-size="10" fill="#6c7c76">Trial ${ordered.at(-1).number + 1}</text></svg>`;
}
function renderSearch(panel) {
  if (currentMethod.family === "range_rmse") {
    panel.innerHTML = `<h3>No hyperparameter search</h3><p>${selectionExplanation()}</p><p>The 16 starts are attempts to minimize the same measured-range objective, not configurations selected using reconstruction truth. The Evidence tab contains each start's initialization, iterations, stopping reason and squared-residual history for the selected noise condition.</p>`;
    return;
  }
  const trials = data.trials.filter((t) => t.label === currentMethod.label);
  panel.innerHTML = `<p class="detail-hint">${count(trials.length)} completed trials · objective: <strong>${currentMethod.objective}</strong> · noise-free calibration. ${currentMethod.selection === "joint" ? "These are shared full-pipeline trials; the loss measures final localization, including for component rows." : "The lowest calibration loss selects the configuration."}</p><div class="search-plot">${searchPlot(trials)}</div><p class="inline-note">Dots: trial losses. Green line: best loss so far.</p><div class="search-controls"><label>Find parameters <input id="trial-search" type="search" placeholder="e.g. half_window or 0.7" value="${esc(trialSearch)}"></label><label>Order <select id="trial-sort"><option value="loss">Lowest loss</option><option value="number">Trial order</option></select></label><button id="export-trials" class="button">Export all trials ↓</button></div><div class="table-scroll"><table class="search-table"><thead><tr><th>Trial</th><th>${currentMethod.objective}</th><th>Parameters</th><th></th></tr></thead><tbody id="trial-rows"></tbody></table></div><div class="pager"><button id="trials-prev" class="button">← Previous</button><span id="trial-page"></span><button id="trials-next" class="button">Next →</button></div>`;
  $("trial-sort").value = trialSort;
  if (currentMethod.search_description)
    panel.insertAdjacentHTML(
      "afterbegin",
      `<p class="detail-hint">${esc(currentMethod.search_description)}</p>`,
    );
  const spec = currentMethod.search_spec;
  if (spec) panel.insertAdjacentHTML("afterbegin", `<section class="method-section search-space"><h3>${esc(spec.strategy)}</h3><p>${count(spec.budget)} planned configurations from ${count(spec.effective_configurations)} effective combinations (${count(spec.raw_configurations)} before collapsing inactive parameters).${spec.seed == null ? "" : ` RNG seed: <code>${spec.seed}</code> (base seed ${spec.base_seed}).`}</p>${Object.entries(spec.grids).map(([stage, grid]) => `<h4>${esc(stage)}</h4><pre>${Object.entries(grid).map(([key, values]) => `${key}: [${values.join(", ")}]`).map(esc).join("\n× ")}</pre>`).join("")}<p class="inline-note">${esc(spec.notes)}</p>${Object.keys(activeReceipt.frozen || {}).length ? `<details><summary>Frozen upstream settings</summary><pre>${esc(JSON.stringify(activeReceipt.frozen, null, 2))}</pre></details>` : ""}</section>`);
  $("trial-search").oninput = () => {
    trialSearch = $("trial-search").value;
    searchPage = 0;
    renderTrialRows(trials);
  };
  $("trial-sort").onchange = () => {
    trialSort = $("trial-sort").value;
    searchPage = 0;
    renderTrialRows(trials);
  };
  $("trials-prev").onclick = () => {
    searchPage--;
    renderTrialRows(trials);
  };
  $("trials-next").onclick = () => {
    searchPage++;
    renderTrialRows(trials);
  };
  $("trial-rows").onclick = (e) => {
    const b = e.target.closest("[data-trial]");
    if (b) activateTrial(b.dataset.trial);
  };
  $("export-trials").onclick = () =>
    downloadCSV(
      `${currentMethod.label}-trials.csv`,
      [
        "trial",
        "id",
        "search",
        "objective",
        "loss",
        "seconds",
        "unification",
        "parameters",
        "frozen",
      ],
      trials.map((t) => [
        t.number + 1,
        t.id,
        t.label,
        currentMethod.objective,
        t.loss,
        t.seconds,
        t.unify,
        JSON.stringify(t.parameters),
        JSON.stringify(t.frozen),
      ]),
    );
  renderTrialRows(trials);
}
function renderTrialRows(trials) {
  const found = trials
    .filter((t) =>
      `${t.number + 1} ${JSON.stringify(t.parameters)}`
        .toLowerCase()
        .includes(trialSearch.toLowerCase()),
    )
    .sort((a, b) => a[trialSort] - b[trialSort] || a.number - b.number);
  const pages = Math.max(1, Math.ceil(found.length / 50));
  searchPage = Math.min(Math.max(searchPage, 0), pages - 1);
  $("trial-rows").innerHTML =
    found
      .slice(searchPage * 50, (searchPage + 1) * 50)
      .map(
        (t) =>
          `<tr class="${activeReceipt.id === t.id ? "active-trial" : ""}"><td>${t.number + 1}${t.id === currentMethod.selected_id ? ' <span class="stage-badge">Selected</span>' : ""}</td><td>${fmt(t.loss, 6)}</td><td class="trial-parameters">${esc(JSON.stringify(t.parameters))}</td><td><button class="text-button" data-trial="${t.id}" aria-label="Inspect trial ${t.number + 1}">Inspect ↗</button></td></tr>`,
      )
      .join("") ||
    '<tr><td colspan="4" class="empty">No matching trials.</td></tr>';
  $("trial-page").textContent =
    `Page ${searchPage + 1} / ${pages} · ${count(found.length)} trials`;
  $("trials-prev").disabled = searchPage === 0;
  $("trials-next").disabled = searchPage >= pages - 1;
}
async function loadEvidence() {
  const id = activeReceipt.id;
  if (loaded?.receipt.id === id) return loaded;
  if (loading?.id === id) return loading.promise;
  const promise = api("/api/result?id=" + id, true).then((value) => {
    if (activeReceipt?.id === id) loaded = value;
    return value;
  });
  loading = { id, promise };
  try {
    return await promise;
  } finally {
    if (loading?.id === id) loading = null;
  }
}
function casesForMethod() {
  return loaded.cases.filter(
    (c) =>
      c.kind === currentMethod.component &&
      (c.kind === "sources" ||
        state.obstacles === "" ||
        c.obstacles.length === Number(state.obstacles)),
  );
}
function caseSelector(cases) {
  selectedCase = Math.min(selectedCase, Math.max(cases.length - 1, 0));
  return `<div class="case-controls"><label>Saved case <select id="evidence-case" aria-label="Saved case">${cases.map((c, i) => `<option value="${i}">${esc(c.room + " · " + c.title)}</option>`).join("")}</select></label></div>`;
}
async function renderMap(panel, version) {
  const cases = loaded.cases.filter((c) => c.kind === currentMethod.component);
  if (!cases.length) {
    panel.innerHTML =
      '<p class="detail-hint">No map was produced by this trial.</p>';
    return;
  }
  const rooms = [...new Set(cases.map((c) => c.room))];
  if (!rooms.includes(mapChoice.room)) mapChoice.room = rooms[0];
  const roomCases = cases.filter((c) => c.room === mapChoice.room);
  const source = currentMethod.component === "sources";
  const layouts = [
    ...new Map(
      roomCases.filter((c) => c.layout).map((c) => [c.layout, c]),
    ).values(),
  ];
  if (!layouts.some((c) => c.layout === mapChoice.layout))
    mapChoice.layout =
      (
        layouts.find((c) => String(c.obstacles.length) === state.obstacles) ||
        layouts[0]
      )?.layout || "";
  const layoutCases = roomCases.filter(
    (c) => source || c.layout === mapChoice.layout,
  );
  if (!layoutCases.some((c) => c.repetition === mapChoice.repetition))
    mapChoice.repetition = layoutCases[0].repetition;
  const record = layoutCases.find((c) => c.repetition === mapChoice.repetition);
  const historical = activeReceipt.id !== currentRow.receipt_id;
  const levels = historical
    ? [activeReceipt.level]
    : [
        ...new Set(
          data.conditions
            .filter(
              (c) =>
                c.method_id === currentMethod.id &&
                c.split === activeReceipt.split && conditionContextMatches(c),
            )
            .map((c) => c.level),
        ),
      ].sort((a, b) => a - b);
  const selector = `<div class="map-case-controls">
    <label>Room<select id="map-room" ${rooms.length === 1 ? "disabled" : ""}>${rooms.map((room) => `<option value="${esc(room)}">${esc(room)}</option>`).join("")}</select></label>
    <label>Range noise σ (m)<select id="map-noise" ${levels.length === 1 ? "disabled" : ""}>${levels.map((level) => `<option value="${level}">${level}</option>`).join("")}</select></label>
    ${source ? `<label>AP<select id="map-ap"><option>A</option><option>B</option><option>A+B</option></select></label><label>Obstacles<select id="map-source-layout"><option value="">Empty-room survey (280 positions)</option>${data.rooms.find((r) => r.id === record.room).query_layouts.map((l) => `<option value="${l.box_count}">${l.box_count} obstacles · ${obstacleInputLabel()}</option>`).join("")}</select></label>` : `<label>Obstacles<select id="map-layout">${layouts.map((c) => `<option value="${c.layout}">${c.obstacles.length === 0 ? "0 (empty room)" : c.obstacles.length + " obstacles"} · ${c.layout}</option>`).join("")}</select></label>`}
    <label>Noise realization<select id="map-realization" ${layoutCases.length === 1 ? "disabled" : ""}>${layoutCases.map((c) => `<option value="${c.repetition}">${c.realization}</option>`).join("")}</select></label></div>`;
  panel.innerHTML =
    selector +
    (record.kind === "sources"
      ? ""
      : `<div class="map-view-switch" role="group" aria-label="Localization map view"><button class="button ${state.mapMode === "2d" ? "selected" : ""}" data-map-mode="2d" aria-pressed="${state.mapMode === "2d"}">2D top view</button><button class="button ${state.mapMode === "3d" ? "selected" : ""}" data-map-mode="3d" aria-pressed="${state.mapMode === "3d"}">Interactive 3D</button></div>`) +
    '<div id="map-content"></div>' +
    `<details class="map-context"><summary>About this map and noise realization</summary><p>Realization ${record.realization} within the ${activeReceipt.split} split. The stored noise seed ID is ${record.repetition}; calibration and evaluation use separate seed IDs. At zero noise, identical realizations are stored once.</p>${source ? "" : `<p>${record.obstacles.length} obstacles · ${record.covered_survey_indices.length} empty-room reference positions now inside boxes · ${record.inside_obstacle_estimates.length} estimates inside boxes. References remain eligible for matching; the localizers do not enforce obstacle boundaries. The amber boxes show the saved obstacle geometry.</p>`}</details>`;
  panel.querySelectorAll("[data-map-mode]").forEach(
    (button) =>
      (button.onclick = () => {
        state.mapMode = button.dataset.mapMode;
        renderDetail();
      }),
  );
  $("map-room").value = record.room;
  $("map-room").onchange = () => {
    mapChoice.room = $("map-room").value;
    renderDetail();
  };
  $("map-noise").value = String(activeReceipt.level);
  $("map-noise").onchange = () => {
    const level = Number($("map-noise").value);
    const row = data.conditions.find(
      (c) =>
        c.method_id === currentMethod.id &&
        c.split === activeReceipt.split &&
        c.level === level && conditionContextMatches(c),
    );
    if (!row) return;
    state.noise = String(level);
    state.split = row.split;
    currentRow = row;
    activeReceipt = receiptFor(row.receipt_id);
    resetEvidence();
    fillControls();
    render();
    renderDetail();
  };
  $("map-realization").value = String(record.repetition);
  $("map-realization").onchange = () => {
    mapChoice.repetition = Number($("map-realization").value);
    renderDetail();
  };
  if (source) {
    $("map-source-layout").value = activeReceipt.obstacle_count === undefined ? "" : String(activeReceipt.obstacle_count);
    $("map-source-layout").onchange = () => {
      state.obstacles = $("map-source-layout").value;
      fillControls();
      currentRow = rowFor(currentMethod);
      if (!currentRow) return;
      activeReceipt = receiptFor(currentRow.receipt_id);
      resetEvidence();
      render();
      renderDetail();
      saveURL();
    };
    $("map-ap").value = state.ap;
    $("map-ap").onchange = () => {
      state.ap = $("map-ap").value;
      $("ap").value = state.ap;
      render();
      renderDetail();
    };
  } else {
    $("map-layout").value = record.layout;
    $("map-layout").onchange = () => {
      mapChoice.layout = $("map-layout").value;
      state.obstacles = String(
        layouts.find((c) => c.layout === mapChoice.layout).obstacles.length,
      );
      $("obstacles").value = state.obstacles;
      render();
      renderDetail();
    };
  }
  if (record.kind === "sources" || state.mapMode === "3d") {
    const { mountVTMap } = await import("/vt-map.js");
    if (version !== tabVersion) return;
    disposeMap = mountVTMap($("map-content"), {
      job: { id: activeReceipt.id, track: "direct" },
      externalControls: true,
      kind: record.kind === "sources" ? "sources" : "localization",
      meta: {
        rooms: [data.rooms.find((r) => r.id === record.room)],
        levels: { direct: [activeReceipt.level] },
      },
      filters: { noise: String(activeReceipt.level), ap: state.ap },
      roomLabel: (id) => id,
      request: async (_kind, filters) => {
        if (record.kind !== "sources") return localization3D(record);
        if (state.ap !== filters.ap) {
          state.ap = filters.ap;
          $("ap").value = state.ap;
          render();
          saveURL();
        }
        return record.maps[filters.ap];
      },
    });
  } else {
    $("map-content").innerHTML =
      metricCards(record.metrics) +
      `<div class="query-map">${queryMap(record)}</div><p class="inline-note">Top view. Amber: obstacle footprints (labels show height). Green: true query positions. Blue: estimates. Purple: empty-room reference positions now inside boxes. Lines connect each query to its estimate. Distances and scores use all three coordinates.</p><h3>Query error distribution</h3><div class="distribution">${distribution(record.errors)}</div>`;
  }
}
function localization3D(record) {
  const room = data.rooms.find((r) => r.id === record.room);
  return {
    localization: true,
    ap: "A+B",
    predicted: record.predicted.flat(),
    truth: record.truth.map((xyz, i) => ({
      xyz,
      id: "Query ID " + record.query_ids[i],
      surface_sequences: [[]],
    })),
    channels: [{ ap: "query", start: 0, count: record.predicted.length }],
    matches: record.truth.map((_, i) => [i, i]),
    metrics: record.metrics,
    errors: record.errors,
    scene: {
      polygon_xy: room.polygon_xy,
      ceiling_m: room.ceiling_m,
      access_points_xyz: room.aps,
      survey_xyz: room.survey_xyz,
      obstacles: record.obstacles,
      covered_survey_indices: record.covered_survey_indices,
    },
  };
}
function queryMap(record) {
  const room = data.rooms.find((r) => r.id === record.room),
    all = [...room.polygon_xy, ...record.predicted, ...record.truth];
  const xs = all.map((p) => p[0]),
    ys = all.map((p) => p[1]),
    minx = Math.min(...xs),
    miny = Math.min(...ys),
    maxx = Math.max(...xs),
    maxy = Math.max(...ys);
  const w = 640,
    h = 380,
    scale = Math.min(
      (w - 70) / (maxx - minx || 1),
      (h - 60) / (maxy - miny || 1),
    ),
    x = (v) => 35 + (v - minx) * scale,
    y = (v) => h - 30 - (v - miny) * scale;
  let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Query positions and estimated positions"><polygon points="${room.polygon_xy.map((p) => `${x(p[0])},${y(p[1])}`).join(" ")}" fill="#eff5f1" stroke="#94aa9d"/>`;
  record.obstacles.forEach((box, i) => {
    const center = box.footprint_xy.reduce(
      (sum, p) => [
        sum[0] + p[0] / box.footprint_xy.length,
        sum[1] + p[1] / box.footprint_xy.length,
      ],
      [0, 0],
    );
    svg += `<polygon class="map-obstacle" data-obstacle="${i}" points="${box.footprint_xy.map((p) => `${x(p[0])},${y(p[1])}`).join(" ")}" fill="#d5a25b66" stroke="#a6702d"><title>Obstacle ${i + 1} · height ${fmt(box.height_m)} m</title></polygon><text x="${x(center[0])}" y="${y(center[1])}" text-anchor="middle" font-size="9" fill="#754c21">${i + 1}: ${fmt(box.height_m, 2)} m</text>`;
  });
  const covered = new Set(record.covered_survey_indices);
  room.survey_xyz.forEach((p, i) => {
    const blocked = covered.has(i);
    svg += `<circle class="${blocked ? "covered-reference" : "survey-reference"}" cx="${x(p[0])}" cy="${y(p[1])}" r="${blocked ? 2.5 : 1.1}" fill="${blocked ? "#a04b75" : "#c4d2c9"}"><title>Empty-room reference ${i} · height ${fmt(p[2])} m${blocked ? " · now inside an obstacle" : ""}</title></circle>`;
  });
  Object.entries(room.aps).forEach(
    ([ap, p]) =>
      (svg += `<rect x="${x(p[0]) - 4}" y="${y(p[1]) - 4}" width="8" height="8" fill="#243f3a"><title>AP ${ap}</title></rect>`),
  );
  record.truth.forEach((p, i) => {
    const q = record.predicted[i];
    svg += `<line x1="${x(p[0])}" y1="${y(p[1])}" x2="${x(q[0])}" y2="${y(q[1])}" stroke="#bacad9"/><circle cx="${x(p[0])}" cy="${y(p[1])}" r="2.5" fill="#55967a"/><circle cx="${x(q[0])}" cy="${y(q[1])}" r="3" fill="#527ab5"><title>Query ID ${record.query_ids[i]} · 3D error ${fmt(record.errors[i])} m · xyz ${q.map((v) => fmt(v)).join(", ")}</title></circle>`;
  });
  return svg + "</svg>";
}
function distribution(errors) {
  if (!errors.length) return "";
  const sorted = [...errors].sort((a, b) => a - b),
    w = 620,
    h = 210,
    L = 45,
    B = 34,
    T = 18,
    R = 20,
    max = Math.max(sorted.at(-1), 0.001);
  const x = (v) => L + (v / max) * (w - L - R),
    y = (v) => h - B - v * (h - B - T);
  const points = sorted
    .map((e, i) => `${x(e)},${y((i + 1) / sorted.length)}`)
    .join(" ");
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Cumulative distribution of 3D query errors"><path d="M${L} ${T}V${h - B}H${w - R}" fill="none" stroke="#d7e2dc"/>${[0, 0.5, 1].map((v) => `<text x="${L - 8}" y="${y(v) + 4}" text-anchor="end" font-size="10" fill="#78867f">${v * 100}%</text>`).join("")}<polyline points="${L},${h - B} ${points}" stroke="#16836e" stroke-width="2" fill="none"/><text x="${L}" y="${h - 12}" font-size="10">0 m</text><text x="${w - R}" y="${h - 12}" text-anchor="end" font-size="10">${fmt(max)} m</text></svg>`;
}
function tree(value) {
  if (value && typeof value === "object") {
    if (value.shape)
      return `<details open><summary>Array [${value.shape.join(", ")}] · ${value.dtype} · ${count(value.bytes)} bytes · row ${value.start}</summary><pre>${esc(JSON.stringify(value.preview, null, 2))}</pre>${currentArtifact && value.shape[0] > value.start + 30 ? `<button class="button" data-field="${esc(value.field)}" data-start="${value.start + 30}">Next 30 rows</button>` : ""}</details>`;
    return `<div class="tree">${Object.entries(value)
      .map(
        ([k, v]) => `<div><span class="key">${esc(k)}</span>: ${tree(v)}</div>`,
      )
      .join("")}</div>`;
  }
  if (typeof value === "string" && /^[a-f0-9]{64}$/.test(value))
    return `<button class="ref" data-ref="${value}">Open batch ${value.slice(0, 12)} ↗</button>`;
  return esc(value);
}
function showAudit(value) {
  $("audit").innerHTML = tree(value);
  $("audit").onclick = (e) => {
    const ref = e.target.closest("[data-ref]"),
      page = e.target.closest("[data-field]");
    if (ref) openArtifact(ref.dataset.ref);
    if (page)
      openArtifact(
        currentArtifact,
        false,
        page.dataset.field,
        Number(page.dataset.start),
      );
  };
  $("audit-back").disabled = auditHistory.length < 2;
}
async function openArtifact(id, remember = true, field = "", start = 0) {
  const version = tabVersion,
    trial = activeReceipt.id;
  $("audit").textContent = "Loading or recreating numerical stage…";
  try {
    const value = await api(
      "/api/artifact?" + new URLSearchParams({ id, trial, field, start }),
      true,
    );
    if (version !== tabVersion) return;
    currentArtifact = id;
    if (remember) auditHistory.push(id);
    showAudit(value);
  } catch (error) {
    if (version === tabVersion) $("audit").innerHTML = showError(error);
  }
}
function renderEvidence(panel) {
  const cases = casesForMethod();
  if (!cases.length) {
    panel.innerHTML =
      '<p class="detail-hint">No numerical cases were produced by this trial.</p>';
    return;
  }
  if (currentMethod.family === "range_rmse") {
    const record = cases[selectedCase];
    panel.innerHTML = obstacleEvaluationNote() + `<p class="detail-hint">Each AP is optimized independently. The selected start has the lowest measured range RMSE, without using reconstruction truth.</p>${caseSelector(cases)}<label>AP <select id="rmse-evidence-ap"><option>A</option><option>B</option><option>A+B</option></select></label>` +
      state.ap.split("+").map((ap) => {
        const audit = record.audit[ap];
        return `<h3>AP ${ap}</h3><p>${count(audit.known_source_count)} sources · ${count(audit.observation_count)} observed peaks · ${count(audit.missing_observations)} missing observations · selected start ${audit.selected_start + 1} · measured range RMSE ${fmt(audit.range_rmse_m, 6)} m</p><div class="table-scroll"><table><thead><tr><th>Start</th><th>Initialization</th><th>Iterations</th><th>Stopping reason</th><th>Measured range RMSE (m)</th></tr></thead><tbody>${audit.starts.map((s) => `<tr><td>${s.start + 1}${s.start === audit.selected_start ? ' <span class="stage-badge">Selected</span>' : ''}</td><td>${esc(s.initialization)}</td><td>${s.iterations}</td><td>${esc(s.stop.replaceAll('_', ' '))}</td><td>${fmt(s.range_rmse_m, 6)}</td></tr>`).join('')}</tbody></table></div><details><summary>Assignment matrix and residual histories (first 30 rows)</summary>${tree(audit)}</details>`;
      }).join('') + codeLinks(["Execution and experiment configuration", "Caching and saved evidence"]);
    $("evidence-case").value = selectedCase;
    $("evidence-case").onchange = () => { selectedCase = Number($("evidence-case").value); renderDetail(); };
    $("rmse-evidence-ap").value = state.ap;
    $("rmse-evidence-ap").onchange = () => { state.ap = $("rmse-evidence-ap").value; $("ap").value = state.ap; render(); renderDetail(); };
    return;
  }
  panel.innerHTML = `<p class="detail-hint">Evidence for the configuration shown above. Intermediate arrays are recreated if evicted. Each array page contains up to 30 rows.</p>${caseSelector(cases)}<div class="evidence-actions"><button id="audit-back" class="button" disabled>← Back</button><button id="audit-root" class="text-button">Case overview</button></div><div id="audit" class="audit-tree"></div><section class="evidence-section"><h3>Peak matching · merge / purge rounds</h3><p class="inline-note">Replay one fingerprint pair using this trial’s actual parameters. Query matching inspects the MPUrge-MAP initializer.</p><div class="pair-controls"><label>Pair type<select id="pair-kind"></select></label><label>AP<select id="pair-ap"><option>A</option><option>B</option></select></label><label>Left row<input id="pair-left" type="number" min="0" step="1" value="0"></label><label>Survey row<input id="pair-right" type="number" min="0" step="1" value="1"></label><label>Query layout<select id="pair-layout"></select></label></div><button id="pair-run" class="button">Replay pair match</button><div id="pair-result" class="pair-results" aria-live="polite"></div></section>`;
  $("evidence-case").value = selectedCase;
  const record = cases[selectedCase];
  $("evidence-case").onchange = () => {
    selectedCase = Number($("evidence-case").value);
    auditHistory = [];
    renderDetail();
  };
  const root = () => {
    currentArtifact = null;
    auditHistory = [];
    showAudit(record.audit);
  };
  $("audit-root").onclick = root;
  $("audit-back").onclick = () => {
    auditHistory.pop();
    openArtifact(auditHistory.at(-1), false);
  };
  root();
  const hasSurvey =
    activeReceipt.method === "mpurge" ||
    activeReceipt.method.startsWith("mpurge_triangle") ||
    activeReceipt.method.startsWith("mpurge_allow_crossing") ||
    activeReceipt.method.startsWith("mpurge_consumption") ||
    activeReceipt.method.startsWith("mpurge_robust") ||
    activeReceipt.method === "joint" ||
    !!activeReceipt.frozen?.mpurge;
  const hasQuery = activeReceipt.obstacle_count === undefined && (
    activeReceipt.method === "mpurge_map" ||
    activeReceipt.method === "joint" ||
    activeReceipt.method.endsWith("_joint") ||
    !!activeReceipt.frozen?.map);
  $("pair-kind").innerHTML =
    (hasSurvey ? '<option value="survey">Survey / survey</option>' : "") +
    (hasQuery ? '<option value="query">Query / survey</option>' : "");
  if (currentMethod.component !== "sources" && hasQuery)
    $("pair-kind").value = "query";
  const room = data.rooms.find((r) => r.id === record.room);
  $("pair-layout").innerHTML = room.layouts
    .map((id) => `<option value="${esc(id)}">${esc(id)}</option>`)
    .join("");
  if (record.layout !== undefined) $("pair-layout").value = record.layout;
  const changeKind = () => {
    $("pair-layout").disabled = $("pair-kind").value === "survey";
  };
  $("pair-kind").onchange = changeKind;
  changeKind();
  $("pair-run").onclick = () => replayPair(record);
  if (!hasSurvey && !hasQuery)
    panel.querySelector(".evidence-section").hidden = true;
  if (hasSurvey && activeReceipt.obstacle_count === undefined)
    panel.insertAdjacentHTML(
      "beforeend",
      `<p class="inline-note"><a class="button" href="/matching.html?trial=${activeReceipt.id}" target="_blank" rel="noopener">Open full peak-matching explorer ↗</a></p>`,
    );
  panel.insertAdjacentHTML(
    "beforeend",
    obstacleEvaluationNote() + codeLinks([
      "Execution and experiment configuration",
      "Caching and saved evidence",
    ]),
  );
}
async function replayPair(record) {
  const version = tabVersion;
  const left = Number($("pair-left").value),
    right = Number($("pair-right").value);
  if (
    !Number.isInteger(left) ||
    !Number.isInteger(right) ||
    left < 0 ||
    right < 0
  ) {
    $("pair-result").innerHTML = showError(
      Error("Enter nonnegative integer row numbers."),
    );
    return;
  }
  $("pair-run").disabled = true;
  $("pair-result").textContent = "Replaying selected pair…";
  try {
    const q = new URLSearchParams({
      trial: activeReceipt.id,
      ap: $("pair-ap").value,
      left,
      right,
      kind: $("pair-kind").value,
      layout: $("pair-layout").value,
      room: record.room,
      repetition: record.repetition,
    });
    const value = await api("/api/match?" + q, true);
    if (version !== tabVersion) return;
    $("pair-result").innerHTML =
      '<div class="pair-matrix-controls"><label>Match overlay <select id="match-overlay"><option value="final">Final accepted matches</option><option value="first">First-round accepted matches</option></select></label><button id="export-pair" class="button">Export pair CSV ↓</button></div><p class="inline-note">Matrix cells always show first-round dissimilarity; lower is better. Circles mark accepted matches in the selected overlay. Grey cells were excluded by unification or the triangle check. Select a cell for its exact score and acceptance round.</p><div id="pair-matrix" class="pair-matrix"></div><p id="pair-cell" class="inline-note" aria-live="polite">Select a matrix cell to inspect it.</p>' +
      `<p class="inline-note">${value.matches.length} accepted matches · ${value.left_ranges_m.length} left peaks · ${value.right_ranges_m.length} right peaks. Costs below are from each candidate’s decision round.</p><table><thead><tr><th>Round</th><th>Left peak / range</th><th>Right peak / range</th><th>Cost</th><th>Decision</th></tr></thead><tbody>${value.trace.map(([a, b, c, round, accepted]) => `<tr><td>${round}</td><td>${a} / ${fmt(value.left_ranges_m[a])} m</td><td>${b} / ${fmt(value.right_ranges_m[b])} m</td><td>${fmt(c, 6)}</td><td class="${accepted ? "accepted" : "rejected"}">${accepted ? "Accepted" : "Purged"}</td></tr>`).join("") || '<tr><td colspan="5">No match candidates.</td></tr>'}</tbody></table><details class="receipt-details"><summary>Full match record</summary><pre>${esc(JSON.stringify(value, null, 2))}</pre></details>`;
    const matrix = () => renderPairMatrix(value);
    $("match-overlay").onchange = matrix;
    matrix();
    $("export-pair").onclick = () => {
      const trace = new Map(
        value.trace.map((row) => [row[0] + ":" + row[1], row]),
      );
      const cells = value.first_round_costs.flatMap((row, a) =>
        row.map((cost, b) => {
          const decision = trace.get(a + ":" + b);
          return [
            activeReceipt.id,
            q.get("kind"),
            q.get("ap"),
            q.get("left"),
            q.get("right"),
            a,
            b,
            value.left_ranges_m[a],
            value.right_ranges_m[b],
            cost,
            decision?.[3],
            Boolean(decision?.[4]),
            decision?.[2],
            $("match-overlay").value,
            value.unify,
            JSON.stringify(value.parameters),
          ];
        }),
      );
      downloadCSV(
        "peak-pair.csv",
        [
          "receipt_id",
          "pair_kind",
          "ap",
          "left_row",
          "survey_row",
          "left_peak",
          "right_peak",
          "left_range_m",
          "right_range_m",
          "first_round_cost",
          "decision_round",
          "accepted",
          "decision_cost",
          "overlay",
          "unification",
          "parameters",
        ],
        cells,
      );
    };
  } catch (error) {
    if (version === tabVersion) $("pair-result").innerHTML = showError(error);
  } finally {
    if (version === tabVersion) $("pair-run").disabled = false;
  }
}
function renderPairMatrix(value) {
  const final = $("match-overlay").value === "final";
  const accepted = new Set(
    value.trace
      .filter((t) => t[4] && (final || t[3] === 1))
      .map((t) => t[0] + ":" + t[1]),
  );
  const finite = value.first_round_costs.flat().filter(Number.isFinite);
  const max = Math.max(...finite, 0.001);
  $("pair-matrix").innerHTML =
    "<table><thead><tr><th>Left ↓ / right →</th>" +
    value.right_ranges_m
      .map((r, i) => `<th>${i}<small>${fmt(r, 2)} m</small></th>`)
      .join("") +
    "</tr></thead><tbody>" +
    value.first_round_costs
      .map(
        (row, a) =>
          `<tr><th>${a}<small>${fmt(value.left_ranges_m[a], 2)} m</small></th>${row.map((cost, b) => `<td><button data-cell="${a}:${b}" class="matrix-cell ${accepted.has(a + ":" + b) ? "match-cell" : ""}" style="background:${Number.isFinite(cost) ? `hsl(160 30% ${96 - 28 * (1 - Math.min(cost / max, 1))}%)` : "#e8ebed"}" aria-label="Left peak ${a}, right peak ${b}, first-round cost ${cost === null ? "excluded" : cost}${accepted.has(a + ":" + b) ? ", accepted match" : ""}">${Number.isFinite(cost) ? fmt(cost, 2) : "—"}</button></td>`).join("")}</tr>`,
      )
      .join("") +
    "</tbody></table>";
  $("pair-matrix").onclick = (e) => {
    const cell = e.target.closest("[data-cell]");
    if (!cell) return;
    const [a, b] = cell.dataset.cell.split(":").map(Number),
      cost = value.first_round_costs[a][b];
    const decision = value.trace.find((t) => t[0] === a && t[1] === b);
    $("pair-cell").textContent =
      `Left peak ${a} (${fmt(value.left_ranges_m[a], 6)} m) / right peak ${b} (${fmt(value.right_ranges_m[b], 6)} m). First-round cost: ${Number.isFinite(cost) ? fmt(cost, 9) : "excluded"}. ${decision ? `${decision[4] ? "Accepted" : "Purged"} in round ${decision[3]}, decision cost ${fmt(decision[2], 9)}.` : "No reciprocal candidate decision."}`;
  };
}
function downloadCSV(name, headers, values) {
  const cell = (v) => '"' + String(v ?? "").replaceAll('"', '""') + '"';
  const text = [headers, ...values]
    .map((row) => row.map(cell).join(","))
    .join("\r\n");
  const url = URL.createObjectURL(
    new Blob(["\ufeff" + text], { type: "text/csv;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function exportResults() {
  const metrics = metricColumns().map((c) => c[0]);
  downloadCSV(
    `redesign-${state.view}-${state.split}.csv`,
    [
      "run",
      "method",
      "algorithm",
      "vt_source",
      "parameter_selection",
      "tuning_pipeline",
      "unification_scope",
      "selection",
      "unification",
      "split",
      "noise_sigma_m",
      "obstacle_count",
      "ap_scope",
      ...metrics,
      ...(state.view === "vt" ? ["source_counts_by_repetition"] : []),
      "search_label",
      "trial_count",
      "receipt_id",
      "parameters",
      "frozen",
    ],
    rows.map(({ method: m, condition: r }) => {
      const receipt = receiptFor(r.receipt_id);
      return [
        r.run || m.run || data.config.name,
        m.title,
        m.name,
        m.vt_source,
        m.tuning,
        m.tuning_pipeline,
        m.unification_scope,
        m.selection,
        m.unify,
        r.split,
        r.level,
        state.view === "vt"
          ? (r.obstacle_count ?? "empty_room_survey")
          : state.obstacles === ""
            ? "all"
            : state.obstacles,
        state.view === "vt" ? state.ap : "A+B",
        ...metrics.map((k) => conditionMetric(r, k)),
        ...(state.view === "vt"
          ? [
              JSON.stringify(
                r.source_counts.map((record) => ({
                  room: record.room,
                  repetition: record.repetition,
                  predicted: record.predicted[state.ap],
                  true: record.true[state.ap],
                })),
              ),
            ]
          : []),
        m.label,
        m.trial_count,
        r.receipt_id,
        JSON.stringify(receipt.parameters),
        JSON.stringify(receipt.frozen),
      ];
    }),
  );
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  setTimeout(() => ($("toast").hidden = true), 3500);
}
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const first = !data;
    data = await api("/api/overview");
    if (first) {
      const query = readURL();
      fillControls();
      render();
      if (query.has("method"))
        openMethod(query.get("method"), query.get("tab"), query.get("trial"));
    } else {
      render();
    }
    $("error").hidden = true;
  } catch (error) {
    $("error").hidden = false;
    $("error").textContent = error.message;
    $("results-panel").setAttribute("aria-busy", "false");
  } finally {
    refreshing = false;
  }
}
document.querySelectorAll("[data-view]").forEach(
  (b) =>
    (b.onclick = () => {
      closeDetail(false);
      state.view = b.dataset.view;
      state.selection = "";
      state.stage = "";
      state.sort = "";
      state.search = "";
      fillControls();
      state.direction = "asc";
      render();
      saveURL();
    }),
);
for (const id of ["split", "noise", "selection", "stage", "obstacles", "ap"])
  $(id).onchange = () => {
    state[id] = $(id).value;
    if (id === "obstacles") fillControls();
    render();
    saveURL();
  };
$("search").oninput = () => {
  state.search = $("search").value;
  render();
  saveURL();
};
$("reset").onclick = () => {
  state = {
    ...state,
    split: "evaluation",
    noise: "0",
    obstacles: "",
    ap: "A+B",
    unification: "",
    tags: "",
    selection: "",
    stage: "",
    search: "",
    sort: "",
    direction: "asc",
  };
  fillControls();
  render();
  saveURL();
};
$("table-head").onclick = (e) => {
  const b = e.target.closest("[data-sort]");
  if (!b) return;
  state.direction =
    state.sort === b.dataset.sort
      ? state.direction === "asc"
        ? "desc"
        : "asc"
      : b.dataset.sort === "f1"
        ? "desc"
        : "asc";
  state.sort = b.dataset.sort;
  render();
  saveURL();
};
$("table-body").onclick = (e) => {
  const b = e.target.closest("[data-method]");
  if (b) openMethod(b.dataset.method);
};
$("export").onclick = exportResults;
$("copy-link").onclick = async () => {
  saveURL();
  try {
    await navigator.clipboard.writeText(location.href);
    toast("View link copied");
  } catch {
    toast("Copy the current address from your browser to share this view.");
  }
};
$("close-detail").onclick = () => closeDetail();
$("drawer-backdrop").onclick = () => closeDetail();
document.querySelectorAll("[data-tab]").forEach(
  (b) =>
    (b.onclick = () => {
      detailTab = b.dataset.tab;
      renderDetail();
    }),
);
document.querySelector(".detail-tabs").onkeydown = (e) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
  e.preventDefault();
  const buttons = [...document.querySelectorAll("[data-tab]")],
    i = buttons.indexOf(document.activeElement);
  const next =
    e.key === "Home"
      ? 0
      : e.key === "End"
        ? buttons.length - 1
        : (i + (e.key === "ArrowRight" ? 1 : -1) + buttons.length) %
          buttons.length;
  buttons[next].focus();
  buttons[next].click();
};
document.addEventListener("keydown", (e) => {
  if ($("drawer").hidden) return;
  if (e.key === "Escape") {
    closeDetail();
    return;
  }
  if (e.key === "Tab") {
    const focusable = [
      ...$("drawer").querySelectorAll(
        'button:not(:disabled),input,select,summary,[tabindex="0"]',
      ),
    ].filter((el) => el.getClientRects().length);
    const first = focusable[0],
      last = focusable.at(-1);
    if (
      e.shiftKey &&
      (document.activeElement === first ||
        document.activeElement === $("drawer"))
    ) {
      e.preventDefault();
      last?.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first?.focus();
    }
  }
});
window.addEventListener("popstate", () => {
  closeDetail(false);
  const query = readURL();
  fillControls();
  render();
  if (query.has("method"))
    openMethod(query.get("method"), query.get("tab"), query.get("trial"));
});
refresh();
setInterval(refresh, 15000);
