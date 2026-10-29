import "./site-runtime.js";
import katex from "/vendor/katex/katex.mjs";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const count = (n) => Number(n).toLocaleString("en-US");
const percent = (n, d) => d ? 100 * n / d : 0;
const pct = (n) => `${n.toFixed((n > 0 && n < .01) || (n > 99.99 && n < 100) ? 4 : 2)}%`;
const math = (s) => katex.renderToString(s, {throwOnError:true, strict:"error", trust:false, output:"htmlAndMathml"});
const tooltip = $("chart-tooltip");
let tooltipTarget = null;
let report, rows, selectedAP;
const controls = new Map();
const charts = [
  {id:"rank-line", type:"line", metric:"same_rank", name:"True matches with equal rank"},
  {id:"rank-bar", type:"bar", metric:"same_rank", name:"Equal-rank final MPUrge matches"},
  {id:"cross-line", type:"line", metric:"cross_any", name:"True matches crossing another true match"},
  {id:"difference-line", type:"line", metric:"difference_satisfies", name:"True matches satisfying the difference bound"},
  {id:"difference-bar", type:"bar", metric:"difference", name:"Final MPUrge matches violating the difference bound"},
  {id:"difference-accuracy", type:"accuracy", metric:"difference", name:"Identity correctness among difference violations versus the MPUrge baseline"},
  {id:"sum-line", type:"line", metric:"sum_satisfies", name:"True matches satisfying the sum bound"},
  {id:"sum-bar", type:"bar", metric:"sum", name:"Final MPUrge matches violating the sum bound"},
  {id:"sum-accuracy", type:"accuracy", metric:"sum", name:"Identity correctness among sum violations versus the MPUrge baseline"},
];

function seriesFor(chart) {
  if (chart.type === "line") return [
    {key:"global", name:"Global · all receiver pairs", color:"#2670b0"},
    {key:"neighborhood", name:"MPUrge neighborhoods", color:"#098474"},
  ];
  if (chart.type === "accuracy") return [
    {key:"violating", name:"Correct among violating matches", color:"#b56a3d"},
    {key:"overall", name:"MPUrge baseline", color:"#2670b0"},
  ];
  if (chart.metric === "same_rank") return [
    {key:"same_rank_correct", name:"Same rank · correct match", color:"#098474"},
    {key:"same_rank_incorrect", name:"Same rank · incorrect match", color:"#c16a42"},
  ];
  return [
    {key:`${chart.metric}_counterpart_violates`, name:"Expected · true pair also violates", color:"#098474"},
    {key:`${chart.metric}_counterpart_satisfies`, name:"Unnecessary · true pair satisfies", color:"#c16a42"},
    {key:`${chart.metric}_counterpart_unavailable`, name:"True counterpart unavailable", color:"#87949e"},
  ];
}

function fraction(chart, row, series) {
  if (chart.type === "line") return [row[series.key][chart.metric], row[series.key].available];
  if (chart.type === "accuracy") {
    const c = row.identity_accuracy[chart.metric][series.key];
    return [c.correct, c.total];
  }
  return [row.matches[series.key], row.matches.total * (chart.metric === "same_rank" ? 1 : 2)];
}

function fractionLabel(chart, series, numerator, denominator) {
  if (denominator) return pct(percent(numerator, denominator));
  return chart.type === "accuracy" && series.key === "violating"
    ? "N/A — no violating matches" : "N/A — no matches";
}

function contextText(row) {
  return `AP ${esc(row.ap)} · ${row.repetitions.length} saved noise realizations per AP` +
    (row.noise === 0 ? " (identical at σ = 0)" : "") +
    `<br>Native repetition IDs ${row.repetitions.join(", ")} · counts pooled, not percentages averaged`;
}

function tipFor(chart, row, series) {
  const [n, d] = fraction(chart, row, series);
  let body = `<strong>${esc(series.name)}</strong><div class="tip-value">${fractionLabel(chart, series, n, d)}</div>`;
  body += `<div>Range noise σ = ${row.noise} m</div>`;
  if (chart.type === "line") {
    const c = row[series.key];
    body += `<div>${count(n)} / ${count(d)} true correspondence edges</div>` +
      `<div>${count(c.pairs)} fingerprint-pair records</div>`;
    if (chart.metric === "cross_any") body += `<div>Each crossing edge counted once, even if it crosses several others.</div>`;
    if (chart.metric === "same_rank") body += `<div>Ranks include every retained observed peak, including unshared sources.</div>`;
  } else if (chart.type === "accuracy") {
    const c = row.identity_accuracy[chart.metric][series.key];
    body += d ? `<div>${count(c.correct)} correct + ${count(c.incorrect)} incorrect = ${count(d)} ${series.key === "violating" ? "violating" : "final"} output edges</div>` +
      `<div>Incorrect share: ${pct(percent(c.incorrect,d))}</div>` : `<div>Violating output matches: 0. There are no matches in this group from which to calculate a percentage.</div>`;
    const all = row.identity_accuracy[chart.metric].overall;
    body += `<div>MPUrge baseline: ${count(all.correct)} / ${count(all.total)} final matches correct (${pct(percent(all.correct,all.total))})</div>` +
      `<div>The baseline includes all final MPUrge matches. Correct means the selected peaks have exactly the same source identity. No endpoint doubling.</div>`;
  } else if (chart.metric === "same_rank") {
    const subset = row.matches.same_rank_correct + row.matches.same_rank_incorrect;
    body += `<div>${count(n)} / ${count(d)} final MPUrge matches</div>` +
      `<div>${pct(percent(n, subset))} of the ${count(subset)} equal-rank output matches</div>` +
      `<div>Total bar: ${pct(percent(subset, d))} of all final matches</div>` +
      `<div>Correct means identical physical/virtual source, not merely the same AP.</div>`;
  } else {
    const total = row.matches[`${chart.metric}_violation_edges`];
    body += `<div>${count(n)} / ${count(d)} output-edge endpoints</div>` +
      `<div>${pct(percent(n, 2 * total))} of the ${count(2 * total)} violating-edge endpoints</div>` +
      `<div>Total bar: ${count(total)} / ${count(row.matches.total)} output matches (${pct(percent(total, row.matches.total))}) violate.</div>` +
      `<div>${count(row.matches[`${chart.metric}_endpoint_disagreement_edges`])} violating edges have different classifications at their two endpoints.</div>` +
      `<div>Each endpoint is compared with its own true partner. This is not a correct/incorrect output-edge split.</div>`;
  }
  body += `<hr><div class="tip-sub">${contextText(row)}<br>` +
    `${count(row.observations.discarded_negative)} negative-range observations discarded across these records.<br>` +
    `Standard MPUrge · α = ${report.selection.parameters.alpha} · half-window = ${report.selection.parameters.half_window} · no peak unification</div>`;
  return body;
}

function showTip(html, event, target) {
  tooltipTarget = target;
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  const rect = target.getBoundingClientRect();
  const x = event.type === "focus" ? rect.left + rect.width / 2 : event.clientX;
  const y = event.type === "focus" ? rect.top + rect.height / 2 : event.clientY;
  const width = tooltip.offsetWidth, height = tooltip.offsetHeight;
  tooltip.style.left = `${Math.max(8, Math.min(x + 15, window.innerWidth - width - 10))}px`;
  tooltip.style.top = `${Math.max(8, y + height + 22 < window.innerHeight ? y + 15 : y - height - 12)}px`;
}

function drawChart(chart) {
  const host = $(chart.id), series = seriesFor(chart);
  if (!controls.has(chart.id)) controls.set(chart.id, {hidden:new Set(), scale:"full"});
  const control = controls.get(chart.id), visible = series.filter(s => !control.hidden.has(s.key));
  const values = rows.flatMap(r => visible.map(s => fraction(chart,r,s)).filter(([,d])=>d).map(([n,d])=>percent(n,d)));
  let ymin = 0, ymax = 100;
  if (control.scale === "focus" && values.length) {
    if (chart.type === "bar") {
      ymax = Math.min(100, Math.max(.01, 1.15 * Math.max(...rows.map(r => visible.reduce(
        (sum, s) => sum + percent(...fraction(chart, r, s)), 0)))));
    } else if (chart.type === "accuracy") {
      ymax = Math.min(100, Math.max(1, 1.15 * Math.max(...values)));
    } else {
      const lo = Math.min(...values), hi = Math.max(...values), pad = Math.max(.05, (hi - lo) * .12);
      ymin = Math.max(0, lo - pad); ymax = Math.min(100, hi + pad);
    }
  }
  const W = 620, H = 330, L = 55, R = 18, T = 29, B = 58;
  const ph = H - T - B, pw = W - L - R;
  const x = i => L + (i + .5) * pw / rows.length;
  const y = v => T + ph * (1 - (v - ymin) / (ymax - ymin));
  const ticks = Array.from({length:6}, (_, i) => ymin + (ymax - ymin) * i / 5);
  let svg = `<svg class="truth-plot" viewBox="0 0 ${W} ${H}" role="group" aria-label="${esc(chart.name)}. Select a point or bar segment for exact counts.">`;
  svg += `<text class="axis-title" x="${L}" y="13">${chart.type === "line" ? "True matches (%)" : chart.type === "accuracy" ? "Correct source identity (%)" : "All final MPUrge matches (%)"}</text>`;
  for (const tick of ticks) {
    const label = ymax - ymin < 1 ? tick.toFixed(3) : (Number.isInteger(tick) ? tick.toFixed(0) : tick.toFixed(1));
    svg += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${y(tick)}" y2="${y(tick)}"/><text x="${L - 9}" y="${y(tick) + 4}" text-anchor="end">${label}</text>`;
  }
  rows.forEach((r, i) => { svg += `<text x="${x(i)}" y="${H - B + 20}" text-anchor="middle">${r.noise}</text>`; });
  svg += `<text class="axis-title" x="${L + pw / 2}" y="${H - 8}" text-anchor="middle">Range noise σ (m) · saved levels</text>`;
  const tips = [];
  const hit = (shape, attributes, row, s) => {
    const tip = tips.length, [n, d] = fraction(chart, row, s);
    tips.push(tipFor(chart, row, s));
    return `<${shape} ${attributes} class="data-hit" tabindex="0" role="img" aria-describedby="chart-tooltip" data-tip="${tip}" aria-label="${esc(`${s.name}; sigma ${row.noise} metres; ${fractionLabel(chart,s,n,d)}; ${count(n)} out of ${count(d)}`)}"/>`;
  };
  if (chart.type === "line") {
    for (const s of visible) {
      const path = rows.map((r, i) => `${i ? "L" : "M"}${x(i)},${y(percent(...fraction(chart, r, s)))}`).join(" ");
      svg += `<path class="series-line" d="${path}" stroke="${s.color}" ${s.key === "neighborhood" ? 'stroke-dasharray="6 3"' : ""}/>`;
      rows.forEach((r, i) => {
        const cy = y(percent(...fraction(chart, r, s)));
        svg += `<circle cx="${x(i)}" cy="${cy}" r="3.6" fill="${s.color}" stroke="white" stroke-width="1"/>`;
        svg += hit("circle", `cx="${x(i)}" cy="${cy}" r="11" fill="${s.color}" fill-opacity="0"`, r, s);
      });
    }
  } else if (chart.type === "accuracy") {
    const bw = Math.min(17, pw / rows.length * .3);
    rows.forEach((r, i) => visible.forEach((s, j) => {
      const [n,d] = fraction(chart,r,s), v = percent(n,d);
      const bx = x(i) + (j - (visible.length - 1) / 2) * (bw + 3) - bw / 2;
      if (!d) {
        svg += `<text x="${bx+bw/2}" y="${y(0)-7}" text-anchor="middle" fill="${s.color}">N/A</text>`;
        svg += hit("rect",`x="${bx-2}" y="${y(0)-22}" width="${bw+4}" height="25" fill="transparent"`,r,s);
      } else if (v === 0) {
        svg += hit("rect",`x="${bx}" y="${y(0)-2}" width="${bw}" height="2" fill="${s.color}"`,r,s);
      } else {
        svg += hit("rect",`x="${bx}" y="${y(v)}" width="${bw}" height="${y(0)-y(v)}" fill="${s.color}"`,r,s);
      }
    }));
  } else {
    const bw = Math.min(37, pw / rows.length * .66);
    rows.forEach((r, i) => {
      let base = 0;
      for (const s of visible) {
        const v = percent(...fraction(chart, r, s)), top = base + v;
        if (v > 0) {
          svg += hit("rect", `x="${x(i) - bw / 2}" y="${y(top)}" width="${bw}" height="${Math.max(.12, y(base) - y(top))}" fill="${s.color}"`, r, s);
        } else {
          // Zero-valued segments remain keyboard accessible and discoverable at the baseline.
          svg += hit("circle", `cx="${x(i)}" cy="${y(base)}" r="3" fill="${s.color}" fill-opacity=".28"`, r, s);
        }
        base = top;
      }
    });
  }
  svg += "</svg>";
  const legend = `<div class="chart-legend">${series.map(s => `<label style="--series:${s.color}"><input type="checkbox" data-series="${s.key}" ${control.hidden.has(s.key) ? "" : "checked"}/><span class="legend-dot" aria-hidden="true"></span>${esc(s.name)}</label>`).join("")}</div>`;
  const scale = `<label class="chart-scale">Y-axis<select aria-label="Y-axis range for ${esc(chart.name)}"><option value="full" ${control.scale === "full" ? "selected" : ""}>0–100%</option><option value="focus" ${control.scale === "focus" ? "selected" : ""}>Focus on data</option></select></label>`;
  let table = `<details class="truth-data"><summary>Exact counts and percentages</summary><div class="table-wrap"><table><caption>AP ${esc(selectedAP)}. ${chart.type === "line" ? "Denominator: true correspondences present at both receivers." : chart.type === "accuracy" ? "Separate edge denominators: violating outputs versus all outputs in the MPUrge baseline. N/A — no violating matches means the violating group contains zero edges." : chart.metric === "same_rank" ? "Denominator: all final MPUrge output edges." : "Segment denominator: twice all final output edges; total bar: violating edges / all edges."} Both saved evaluation realizations are pooled; σ = 0 repeats identical observations.</caption><thead><tr><th>σ (m)</th>${series.map(s => `<th>${esc(s.name)}</th>`).join("")}<th>Denominator${chart.type !== "bar" ? "s" : ""}</th></tr></thead><tbody>`;
  for (const r of rows) {
    table += `<tr><td>${r.noise}</td>${series.map(s => { const [n,d] = fraction(chart,r,s); return `<td>${fractionLabel(chart,s,n,d)} · ${count(n)}</td>`; }).join("")}<td>${chart.type !== "bar" ? series.map(s => `${s.key === "overall" ? "MPUrge baseline" : s.key}: ${count(fraction(chart,r,s)[1])}`).join(" / ") : count(fraction(chart, r, series[0])[1])}</td></tr>`;
  }
  table += "</tbody></table></div></details>";
  host.innerHTML = `<div class="chart-tools">${legend}${scale}</div>${svg}<p class="truth-chart-note">Noise levels are equally spaced on the horizontal axis; distances between ticks are not proportional to σ. ${chart.type === "bar" && chart.metric !== "same_rank" ? "Segments partition endpoints; total bar height is the fraction of violating output matches." : chart.type === "accuracy" ? "Grouped bars have different denominators. N/A means no violating matches at that noise level; a short colored mark at zero means matches exist but none are correct. MPUrge baseline = correctness among all final matches." : "Hover or keyboard-focus a mark for the counts and denominator."}</p>${table}`;
  host.querySelectorAll("[data-tip]").forEach(el => {
    for (const event of ["pointerenter", "pointermove", "focus"]) el.addEventListener(event, e => showTip(tips[Number(el.dataset.tip)], e, el));
    el.addEventListener("pointerleave", () => {
      if (tooltipTarget === el && document.activeElement !== el) tooltip.hidden = true;
    });
    el.addEventListener("blur", () => { if (tooltipTarget === el) tooltip.hidden = true; });
  });
  host.querySelectorAll("[data-series]").forEach(el => el.addEventListener("change", () => {
    el.checked ? control.hidden.delete(el.dataset.series) : control.hidden.add(el.dataset.series);
    tooltip.hidden = true; drawChart(chart);
  }));
  host.querySelector(".chart-scale select").addEventListener("change", e => {control.scale = e.target.value; drawChart(chart);});
}

function render() {
  selectedAP = $("ap").value;
  rows = report.pooled[selectedAP];
  const c = report.coverage, p = report.selection.parameters;
  $("room-title").textContent = `${report.protocol.room} · ${Number(c.room_width_m.toFixed(3))} × ${Number(c.room_depth_m.toFixed(3))} m`;
  $("side-room").textContent = report.protocol.room;
  $("coverage").innerHTML = `<strong>${c.survey_positions} identical empty-room survey positions.</strong> ${count(c.global_pairs_per_ap_realization)} global pairs and ${count(c.neighborhood_pairs_per_ap_realization)} neighborhood pairs per AP per realization. ${report.protocol.noise_levels.length} noise levels, ${c.noise_realizations_per_ap} saved evaluation realizations per AP. AP ${esc(selectedAP)}: counts pooled across ${rows[0].realization_records} AP × realization records per noise level.`;
  $("calibration").innerHTML = `<strong>Standard MPUrge, selected on noise-free empty-room OSPA.</strong> ${math(String.raw`\alpha=${p.alpha}`)} (range weight), half-window ${p.half_window}, neighborhood size ${p.neighbors} including boundary ties; no peak unification and no triangle gate. Reconstruction radius ${p.radius_m} m and support multiplier ${p.beta} are frozen but do not affect these pairwise matches.`;
  $("calibration").innerHTML += `<pre>${esc(JSON.stringify({...p, pattern_weight: 1-p.alpha, unification: false, triangle_gate: false}, null, 2))}</pre><a href="/construction?trial=${report.selection.id}&component=sources" target="_blank" rel="noopener">Concrete MPUrge construction ↗</a>`;
  $("protocol-detail").innerHTML = `
    <p><strong>Truth:</strong> a pair is correct only when both observed peaks have the exact same physical/virtual source ID. Native directions and delays identify the source within 0.01 m, with uniqueness checked. The ${c.native_source_counts.A} sources per AP include the direct path and primary/secondary reflected paths. Source IDs are analysis-only inputs.</p>
    <p><strong>Global:</strong> all ${count(c.global_pairs_per_ap_realization)} unordered distinct survey-position pairs within each AP. No AP A ↔ B comparisons and no sampling. <strong>Neighborhood:</strong> the actual standard MPUrge pair selector, with ${p.neighbors}-nearest-position scopes in 3D, boundary ties included, all unordered pairs within each scope, then deduplicated. Both use the lower survey index as the first endpoint.</p>
    <p><strong>Observations:</strong> the saved native empty-room survey, original 280 positions, seed ${report.protocol.seed}, direct-range conversion, evaluation repetition IDs ${report.protocol.repetitions.join(", ")}. Each peak receives independent Gaussian range noise with the plotted σ. Each observation is shared across every pair using that fingerprint. Noise is not resampled per pair. The two σ = 0 realizations are identical; their pooled counts represent repeated records, not extra independent samples.</p>
    <p><strong>Missing peaks and ranks:</strong> negative noisy ranges are discarded exactly as in the saved observation conversion. A true correspondence exists only if its source survives at both endpoints. Rank uses the entire retained fingerprint, including unshared peaks. Ties use stable range sorting in original native path order; the dataset contains ${count(report.rows.reduce((n,r)=>n+r.observations.adjacent_equal_range_pairs,0))} adjacent exact-range ties across all analysis records. No source IDs break ties.</p>
    <p><strong>Final matches:</strong> all accepted rounds from the production MPUrge pairwise matcher after its mutual-choice and crossing filters. They are not first-round candidates or reconstructed VT associations. Equal-rank bars classify their source-identity correctness. Geometric violation bars instead classify each endpoint against its own true partner's geometric predicate; missing partners have a separate segment.</p>
    <p><strong>Percentages:</strong> integer counts are added before division, across the selected APs and both saved evaluation realizations. Every true edge has equal weight within a curve. Violating-output bars classify both endpoints with half an edge's weight each, so their total height still equals violating edges / all final edges. All bounds use a numerical tolerance of ${report.protocol.tolerance_m} m, independent of σ.</p>
    <p><a href="/source?file=execution/presets.py#L14" target="_blank" rel="noopener">Standard MPUrge constructor ↗</a> · <a href="/source?file=science/matching.py" target="_blank" rel="noopener">Pairwise matcher ↗</a> · <a href="/source?file=science/grouping.py#L10" target="_blank" rel="noopener">Neighborhood construction ↗</a> · <a href="/source?file=execution/true_matching.py" target="_blank" rel="noopener">Analysis formulas and counting code ↗</a></p>
    <p>Frozen selection receipt: <code>${esc(report.selection.id)}</code>. Analysis: <code>${esc(report.schema)}</code>. Created ${esc(report.created_utc)}. Cache key: <code>${esc(report.cache_key)}</code>. The cache depends on actual inputs, selected parameters and analysis semantics; source-file hashes are recorded as provenance, not whole-repository validity gates.</p>`;
  charts.forEach(drawChart);
  const url = new URL(location.href); url.searchParams.set("ap", selectedAP); history.replaceState(null,"",url);
}

async function init() {
  try {
    document.querySelectorAll("[data-math]").forEach(el => { el.innerHTML = math(el.dataset.math); });
    const response = await fetch("/api/true-matching", {cache:"no-store"});
    report = await response.json();
    if (!response.ok || report.error) throw new Error(report.error || `Analysis request failed (${response.status})`);
    if (report.status.state !== "complete") {
      $("load-status").textContent = `Analysis ${report.status.state.replaceAll("_"," ")}${report.status.target ? ` · ${report.status.completed}/${report.status.target} observation conditions complete` : ""}. Reload after computation finishes.`;
      return;
    }
    const ap = new URL(location.href).searchParams.get("ap");
    if (["A","B","A+B"].includes(ap)) $("ap").value = ap;
    $("ap").addEventListener("change", render);
    $("export").addEventListener("click", () => {
      const blob = new Blob([JSON.stringify(report,null,2)], {type:"application/json"});
      const url = URL.createObjectURL(blob), a = document.createElement("a");
      a.href = url; a.download = "true-matching-analysis.json"; a.click(); URL.revokeObjectURL(url);
    });
    $("copy-link").addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(location.href); $("copy-link").textContent = "Link copied"; }
      catch { $("copy-link").textContent = "Copy this page’s address"; }
    });
    render(); $("analysis").hidden = false; $("load-status").hidden = true;
  } catch (error) {
    $("load-status").classList.add("error");
    $("load-status").textContent = error.message;
    console.error(error);
  }
}
window.addEventListener("scroll", () => {
  const focused = document.activeElement;
  if (tooltipTarget === focused && focused?.matches("[data-tip]")) {
    const rect = focused.getBoundingClientRect();
    if (rect.bottom >= 0 && rect.top <= window.innerHeight) {
      showTip(tooltip.innerHTML, {type:"focus"}, focused);
      return;
    }
  }
  tooltip.hidden = true;
}, {passive:true});
window.addEventListener("keydown", e => {if (e.key === "Escape") {tooltip.hidden = true; tooltipTarget = null;}});
init();
