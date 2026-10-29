import katex from "/vendor/katex/katex.mjs";

// Presentation of production algorithms only. No experimental ablations here.
export const finisherNames = {
  consumption: "Support-first peak-consumption finisher",
  robust: "Robust range-refinement and peak-consumption finisher",
};

export function sourceMetricExplanation() {
  return `<section class="source-metric-definition"><h3>A+B aggregation</h3>
    <p>Assign estimates to truth <strong>separately within each AP</strong>. No match line or metric can connect A to B. For AP ${math("a")}, let ${math("n_a,m_a")} be estimated and true source counts, ${math("w_a=\\max(n_a,m_a)")}, ${math("O_a")} its OSPA, ${math("T_a")} its maximum number of one-to-one matches within 0.5 m, and ${math("E_a")} the minimum sum of squared 3D distances among assignments with that match count.</p>
    ${eq(String.raw`\operatorname{OSPA}_{A+B}=\frac{\sum_a w_a O_a}{\sum_a w_a},\qquad F1_{A+B}=\frac{2\sum_a T_a}{\sum_a(n_a+m_a)},\qquad\operatorname{RMSE}_{A+B}=\sqrt{\frac{\sum_a E_a}{\sum_a T_a}}`)}
    <p>OSPA uses p = 1 and c = 1 m. Empty estimate and truth sets give OSPA 0 and F1 1. Matched RMSE is unavailable without matches and excludes unmatched sources. Across noise realizations, average OSPA and F1; pool matched squared errors and counts for RMSE. Historical search losses remain recorded separately from these corrected display metrics.</p></section>`;
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"})[c]);
const math = (tex, displayMode = false) => katex.renderToString(tex, {
  displayMode, throwOnError: true, strict: "error", trust: false, output: "htmlAndMathml",
});
const eq = (tex) => `<div class="finisher-equation">${math(tex, true)}</div>`;
const val = (symbol, value, unit = "") => `${math(symbol)} = <strong>${esc(value)}</strong>${unit ? " " + unit : ""}`;
const settings = (...values) => `<p class="algorithm-settings">${values.join(" · ")}</p>`;
const step = (key, title, body, links = []) => ({key, title, body, links});

export function sourceParameters(method, receipt) {
  return receipt.parameters?.source || receipt.frozen?.source ||
    (method.selection === "ospa" ? receipt.parameters : null);
}
export function hasReconstructionExplanation(method) {
  return method.component !== "initial" && (Boolean(method.finisher) ||
    (method.naked && ["ransac", "zayets"].includes(method.family)));
}
function link(method, symbol, file) {
  const found = Object.values(method.code || {}).flat().find(
    (item) => typeof item === "object" && item.file === file && item.label === symbol + "()");
  return `<a href="/source?file=${encodeURIComponent(file)}${found?.line ? "#L" + found.line : ""}" target="_blank" rel="noopener">${esc(symbol)}()</a>`;
}
function ordered(method, title, start, steps, attributes = "") {
  return `<section class="algorithm-stage" ${attributes}><h3>${title}</h3><ol class="algorithm-steps" start="${start}">${steps.map((s, index) =>
    `<li id="algorithm-step-${start + index}" data-step="${s.key}"><header><h4>${s.title}</h4><span class="algorithm-code-links">${s.links.map(([symbol, file]) => link(method, symbol, file)).join(" · ")}</span></header>${s.body}</li>`
  ).join("")}</ol></section>`;
}
export function usesConnectedComponentGrouping(method) {
  return method.family === "mpurge" && method.component !== "initial" &&
    Object.values(method.code || {}).flat().some(item => typeof item === "object" &&
      item.file === "science/grouping.py" && item.label === "ConnectedComponentsGrouper()");
}
function connectedComponentGroupingBody() {
  return `<p>Within each local receiver scope, a node is one observed peak; an edge is an accepted match between two receivers. Ignore nodes with no edges and <strong>form connected components first</strong>.</p>
    <div class="component-pruning" data-grouping="connected-components">
    <p>Afterward, if a component contains multiple peaks from one receiver, keep the peak with:</p>
    <ol class="component-pruning-rule"><li><strong>Highest degree.</strong></li><li><strong>Lowest mean incident-edge dissimilarity.</strong></li><li><strong>Lowest index</strong> to break remaining ties.</li></ol>
    <p>Apply these priorities in order: only ties proceed to the next rule. For a peak ${math("v")}, let ${math("E(v)")} be all accepted matching edges incident to it in this scope, before pruning. Its degree counts those edges; its mean dissimilarity averages their stored MPUrge costs ${math(String.raw`\Delta_e`)}:</p>
    ${eq(String.raw`d(v)=|E(v)|,\qquad\overline\Delta(v)=\frac{\sum_{e\in E(v)}\Delta_e}{d(v)}`)}
    <p>The final tie-break uses the stable original peak index within that receiver. <strong>Do not recompute components afterward</strong>, even if removing a peak disconnects a component.</p>
    </div><p>Each pruned component is one track, with at most one peak per receiver. Remove identical tracks produced by different scopes.</p>`;
}
export function connectedComponentSourceOverview(method) {
  if (!usesConnectedComponentGrouping(method) || method.finisher) return "";
  return `<section class="reconstruction-algorithm source-grouping-overview"><h3>${esc(method.source_full_name || "MPUrge")} source reconstruction</h3>
    <ol class="method-steps"><li>Match peaks between neighboring survey fingerprints using the selected pattern window and range weight.${method.triangle ? " Before matching, require both triangle inequalities: the range difference must not exceed receiver separation plus slack, and the range sum plus slack must reach that separation." : ""}${method.allow_crossing ? `<p><strong>Crossings allowed:</strong> accept every reciprocal-minimum candidate, even if ${math(String.raw`(a_i-a_u)(b_j-b_v)<0`)} for a previously accepted pair ${math(String.raw`u\leftrightarrow v`)}. Here ${math("a_i,b_j")} are the two measured ranges being paired. Purge both endpoints as usual; keep the same windows, costs and rounds. This changes reconstruction only; MPUrge-MAP and SMART-LEA keep their standard matchers.</p><p>${link(method,"CrossingAllowedMPUrgeMatcher","science/matching.py")}</p>` : ""}</li>
    <li data-step="peak-tracks"><h4>Build the local peak graph and resolve receiver conflicts</h4>${connectedComponentGroupingBody()}<p class="algorithm-code-links">${link(method,"ConnectedComponentsGrouper","science/grouping.py")} · ${link(method,"group_within_scopes","science/grouping.py")}</p></li>
    <li>Estimate a source coordinate from each track, cluster the candidate coordinates, and retain clusters meeting the selected support threshold.</li></ol></section>`;
}
function sourceLeastSquares() {
  return step("proposal-least-squares", "Turn each track into one 3D source proposal",
    `<p>A track ${math("T")} contains fixed observed peak indices ${math("(i,j)")}. Discard it unless it spans at least four receivers and their centered coordinates have rank three.</p>
    <p class="optimization-variable"><strong>Optimize:</strong> the source coordinate ${math(String.raw`q\in\mathbb R^3`)}. <strong>Hold fixed:</strong> ${math("T")}, receiver coordinates ${math("x_i")} and measured ranges ${math("y_{ij}")}.</p>
    ${eq(String.raw`\min_{q\in\mathbb R^3}\;\frac12\sum_{(i,j)\in T}\left(\|q-x_i\|_2-y_{ij}\right)^2`)}
    <p>Start from algebraic multilateration; use Levenberg–Marquardt, at most 100 function evaluations. Keep only a successful finite result. Each retained coordinate enters the proposal list ${math(String.raw`\mathcal P`)}.</p>
    <details class="algorithm-detail"><summary>Algebraic starting point</summary>
    <p>Let ${math(String.raw`\bar x`)} be this track's mean receiver coordinate, ${math(String.raw`u_i=x_i-\bar x`)}. Solve for ${math(String.raw`v\in\mathbb R^3`)} and an auxiliary scalar ${math("c_0")}:</p>
    ${eq(String.raw`\min_{v,c_0}\sum_{(i,j)\in T}\left[-2u_i^\top v+c_0-\left(y_{ij}^2-\|u_i\|_2^2\right)\right]^2,\qquad q_{\mathrm{start}}=\bar x+v`)}
    </details>`, [["Multilateration3D", "science/estimation.py"]]);
}
function mpurgeSteps(method, receipt, p) {
  const unified = receipt.unify === true, h = p.alpha === 1 ? 0 : p.half_window;
  return [
    step("neighborhoods", "Choose which survey fingerprints to compare",
      settings(val("k", p.neighbors, "positions, including the center")) +
      `<p>For each receiver ${math("s")}, take its ${math(String.raw`\min(k,N)`)} nearest positions in 3D, including itself and <strong>all ties at the boundary</strong>. These are the local scopes ${math("H_s")}; remove duplicate scopes. Compare every unordered receiver pair occurring together in any scope. Use the same scopes for grouping in step 5.</p>`, [["nearest_neighborhoods", "science/grouping.py"]]),
    step("unification", unified ? "Equalize peak counts for each pair" : "Keep both peak counts unchanged",
      settings(`Peak-count unification: <strong>${unified ? "on" : "off"}</strong>`) +
      `<p>Sort both fingerprints by range, keeping original peak indices; ties keep input order. Call the sorted arrays ${math("a")} and ${math("b")}.</p>` +
      (unified ? `<p>If counts differ, call the shorter array ${math(String.raw`a_1,\ldots,a_m`)} and the longer ${math(String.raw`b_1,\ldots,b_n`)}. Keep the order-preserving subset minimizing:</p>
        ${eq(String.raw`\min_{1\le t_1<\cdots<t_m\le n}\sum_{i=1}^m|a_i-b_{t_i}|`)}
        <p>The dynamic program either skips the last long-array peak or takes it:</p>
        ${eq(String.raw`D_{ij}=\min\left\{D_{i,j-1},\;D_{i-1,j-1}+|a_i-b_j|\right\}`)}
        <p>${math("D_{ij}")} is the minimum cost for matching the first ${math("i")} short-array peaks into the first ${math("j")} long-array peaks. Set ${math("D_{0j}=0")}, ${math(String.raw`D_{i0}=+\infty`)} for ${math("i>0")}; ties take the peak. Trace back selected indices. Equal counts pass through unchanged.</p>`
      : `<p>No subset is selected. The two arrays may have different lengths.</p>`) +
      `<p>This runs once per pair, before the rounds below. It does not delete peaks from the original survey measurements later used by the finisher.</p>`, unified ? [["_subset", "science/matching.py"]] : []),
    step("mpurge-dissimilarity", "Compute the range-and-window dissimilarity",
      settings(val(String.raw`\alpha`, p.alpha, "range weight"), val(String.raw`1-\alpha`, Number((1-p.alpha).toPrecision(12)), "pattern weight"), val("h", h, "peaks on either side")) +
      `<p>Use peaks still present in this round. For offsets ${math(String.raw`t=-h,\ldots,h`)}, define centered windows ${math(String.raw`W^a_i(t)=\widetilde a_{i+t}-a_i`)} and likewise ${math(String.raw`W^b_j(t)`)}. Candidate match ${math(String.raw`i\leftrightarrow j`)} has cost:</p>
      ${eq(String.raw`\Delta_{ij}=\alpha|a_i-b_j|+(1-\alpha)\sum_{-h\le t<u\le h}\left|\bigl(W^a_i(u)-W^a_i(t)\bigr)-\bigl(W^b_j(u)-W^b_j(t)\bigr)\right|`)}
      <p>The pattern term compares <strong>every pair of spacings within the windows</strong>. This reconstruction matcher uses the sum, with no window-size normalization. ${math(String.raw`\Delta_{ij}`)} is in metres; smaller is preferred. At ${math(String.raw`\alpha=1`)}, the window is inactive and ${math("h=0")}.</p>
      <details class="algorithm-detail"><summary>Exact end padding</summary><p>For an array ${math("a")} of length ${math("n")}, use real values within bounds. Outside, extend the nearest endpoint by steps ${math(String.raw`\eta_a=16\epsilon_{64}\max(a_n-a_1,\max_i|a_i|,1\,\mathrm m)`)}: ${math(String.raw`\widetilde a_j=a_1+(j-1)\eta_a`)} for ${math("j<1")}, and ${math(String.raw`\widetilde a_j=a_n+(j-n)\eta_a`)} for ${math("j>n")}. ${math(String.raw`\epsilon_{64}`)} is float64 machine epsilon. Pad ${math("b")} using its own scale.</p></details>` +
      (method.triangle ? `<p>Set ${math(String.raw`\Delta_{ij}=+\infty`)} if ${math(String.raw`|a_i-b_j|>d+\varepsilon`)} or ${math(String.raw`d>a_i+b_j+\varepsilon`)}, where ${math("d")} is receiver separation and ${math(String.raw`\varepsilon`)} = ${esc(p.triangle_tolerance_m)} m.</p>` : `<p>This variant applies no triangle-inequality gate.</p>`), [["_delta", "science/matching.py"]]),
    step("reciprocal-matching", "Accept reciprocal minima; purge their endpoints",
      eq(String.raw`j=\arg\min_v\Delta_{iv},\qquad i=\arg\min_u\Delta_{uj}`) +
      `<p>Keep finite candidates satisfying both equations. Ties choose the first peak in sorted order. Process candidates by increasing ${math(String.raw`\Delta`)}, then left and right indices in this pair's initial sorted, possibly unified arrays.</p>
      ${method.allow_crossing ? `<p><strong>Accept crossings:</strong> keep the candidate even when ${math(String.raw`(a_i-a_u)(b_j-b_v)<0`)} for an already accepted pair ${math(String.raw`u\leftrightarrow v`)}. This variant changes reconstruction matching only; MPUrge-MAP and SMART-LEA keep their standard matchers.</p>` : `<p>Reject ${math(String.raw`i\leftrightarrow j`)} if it crosses any accepted match ${math(String.raw`u\leftrightarrow v`)}, including one accepted earlier in this round:</p>${eq(String.raw`(a_i-a_u)(b_j-b_v)<0`)}`}
      <p><strong>Purge both endpoints of every reciprocal candidate${method.allow_crossing ? "" : ", including rejected crossings"}.</strong> Accepted matches become graph edges with cost ${math(String.raw`\Delta_{ij}`)}. Rebuild windows and costs from remaining peaks; repeat. Stop when either array is empty or no finite reciprocal minimum exists.</p>`, [[method.allow_crossing ? "CrossingAllowedMPUrgeMatcher" : "_merge_purge_urge", "science/matching.py"]]),
    step("peak-tracks", "Build a peak graph within each local scope",
      connectedComponentGroupingBody(), [["ConnectedComponentsGrouper", "science/grouping.py"], ["group_within_scopes", "science/grouping.py"]]),
    sourceLeastSquares(),
  ];
}

function zayetsSteps(method, receipt, p) {
  return [
    step("zayets-clusters", "Make local scopes and split each range array",
      settings(val("R", p.scope_radius_m, "m: scope radius"), val("g", p.gap_m, "m: splitting gap")) +
      `<p>A scope centered at ${math("x_s")} contains every receiver within ${math("R")} metres; remove duplicate scopes. Sort each fingerprint's ranges. Split between consecutive peaks whenever their gap is at least ${math("g")}.</p>`, [["radius_neighborhoods", "science/grouping.py"], ["split_at_gaps", "science/zayets.py"]]),
    step("zayets-grouping", "Group clusters from different receivers",
      settings(`Cluster threshold <code>group_cost</code> = <strong>${esc(p.group_cost)}</strong>`) +
      `<p>For a range cluster ${math("C")}, let ${math("r(C)")} be its first range, ${math("n(C)")} its peak count and ${math("s(C)")} its last-minus-first range. Compare clusters from different receivers in the same scope using:</p>
      ${eq(String.raw`d(C,D)=0.5|r(C)-r(D)|+0.5|n(C)-n(D)|+0.5|s(C)-s(D)|`)}
      <p>This fixed heuristic mixes metre-valued range differences with a count difference. Sort cluster edges by ${math("d")}, then cluster indices. Merge components only when ${math(String.raw`d\le\texttt{group\_cost}`)} and their receiver sets are disjoint. Stop beyond the threshold. Isolated clusters remain singleton groups.</p>`, [["cluster_feature_cost", "science/zayets.py"], ["ConstrainedSingleLinkageGrouper", "science/zayets.py"]]),
    step("zayets-template", "Assign peaks to the longest-cluster template",
      `<p>Within each group, let ${math("L")} be the largest cluster length. Average all length-${math("L")} clusters rank by rank to obtain template ranges ${math(String.raw`t_1,\ldots,t_L`)}. Length-${math("L")} clusters keep rank-to-rank assignments. For each shorter cluster with ranges ${math("c_j")}, use Hungarian assignment to distinct template slots:</p>
      ${eq(String.raw`\min_{\pi\ \mathrm{injective}}\sum_j|c_j-t_{\pi(j)}|`)}
      <p>One track ${math("T")} is the set of observed peaks assigned to one template slot. Keep tracks spanning at least four receivers; remove identical tracks across scopes.</p>`, [["LongestClusterTemplateAssigner", "science/zayets.py"]]),
    sourceLeastSquares(),
  ];
}
function ransacSteps(method, receipt, p) {
  return [
    step("ransac-spheres", "Generate intersections of three measured-range spheres",
      settings(`Trials per AP = <strong>${esc(p.trials)}</strong>`) +
      `<p>In each trial, choose three distinct nonempty receiver rows uniformly, then one measured peak uniformly from each. For the three chosen ${math("(i,j)")} observations, solve:</p>
      ${eq(String.raw`\|q-x_i\|_2=y_{ij}\quad\text{for the three sampled observations}`)}
      <p>Subtracting sphere equations gives two planes; intersect their line with a sphere. Keep the zero, one or two resulting coordinates. Degenerate receiver triples and inconsistent spheres yield no candidates. Sampling uses the experiment's AP-specific seed.</p>`, [["uniform_peak_sample", "science/ransac.py"], ["ThreeSphereIntersections", "science/ransac.py"]]),
    step("ransac-retention", "Retain at most 12 candidates per 400 trials",
      settings(val(String.raw`\tau`, p.tolerance_m, "m")) +
      `<p>For every candidate ${math("q")}, let ${math(String.raw`e_i(q)=\min_j|y_{ij}-\|q-x_i\|_2|`)} using <strong>all</strong> measured peaks. Empty rows give infinite error. Rank candidates with:</p>
      ${eq(String.raw`H(q)=\sum_{i=1}^N\mathbf1[e_i(q)\le\tau]+0.35\sum_{i=1}^N e^{-\frac12(e_i(q)/\tau)^2}-0.10\operatorname{median}_{i}\min(e_i(q)/\tau,4)`)}
      <p>${math(String.raw`\mathbf1[\cdot]`)} is one when its condition holds. In each 400-trial block, including the final partial block, retain the 12 largest ${math("H")} values or all candidates if fewer. Ties favor earlier generation. These coordinates form ${math(String.raw`\mathcal P`)}.</p>
      <p>The constants 0.35, 0.10 and 4 belong to <strong>proposal retention</strong>; finisher parameters do not change them. No peak is consumed here.</p>`, [["TopPerTrialBatch", "science/ransac.py"]]),
  ];
}
function associationStep(p) {
  return step("finisher-association", "Associate available peaks and count supporting receivers",
    settings(val(String.raw`\tau`, p.tolerance_m, "m: inlier tolerance"), val("f", p.support_fraction, "minimum survey fraction")) +
    `<p>Initially ${math(String.raw`A_i=\{1,\ldots,m_i\}`)} contains all original measured peaks and accepted sources ${math(String.raw`\mathcal V=\varnothing`)}. For each remaining proposal ${math(String.raw`q\in\mathcal P`)}, choose one nearest available peak at each receiver:</p>
    ${eq(String.raw`j_i(q)=\arg\min_{j\in A_i}|y_{ij}-\|q-x_i\|_2|,\qquad e_i(q)=|y_{i,j_i(q)}-\|q-x_i\|_2|`)}
    ${eq(String.raw`I(q)=\{i:e_i(q)\le\tau\},\qquad n(q)=|I(q)|,\qquad n_{\min}=\max(5,\lceil fN\rceil)`)}
    <p>${math("e_i")} is absolute range error in metres. ${math("I")} is the supporting receiver set; ${math("n")} counts receivers. ${math("N")} includes empty rows. No available peak gives ${math(String.raw`e_i=+\infty`)}; ties choose the first stored peak. ${math(String.raw`\tau`)} is a selected tolerance, not the noise standard deviation.</p>`, [["nearest_peaks", "science/selection.py"]]);
}
function rankingStep(robust, p) {
  const a=p.smooth_weight ?? .35, b=p.median_penalty_weight ?? .10, c=p.median_clip_factor ?? 4;
  return step("finisher-ranking", "Choose one proposal; stop if its support is insufficient",
    (robust ? settings(val("a",a),val("b",b),val("c",c)) : "") +
    eq(String.raw`g(q)=\sum_{i=1}^N\exp\!\left[-\tfrac12(e_i(q)/\tau)^2\right]`) +
    (robust ? eq(String.raw`m(q)=\operatorname{median}_{1\le i\le N}\min(e_i(q)/\tau,c),\qquad S(q)=n(q)+a\,g(q)-b\,m(q)`)
      : eq(String.raw`S(q)=n(q)+\frac{g(q)}{N+1}`)) +
    `<p>${math("g")} sums smooth range agreement over all receivers.` +
    (robust ? ` ${math("m")} is the median clipped normalized error over those same receivers, including non-supporters. Unavailable rows contribute zero to ${math("g")} and ${math("c")} to the median input. ${math("a,b,c")} are <code>smooth_weight</code>, <code>median_penalty_weight</code> and <code>median_clip_factor</code>.`
      : ` Since ${math(String.raw`0\le g/(N+1)<1`)}, this orders by support count first; range agreement only breaks ties.`) + `</p>
    ${eq(String.raw`q_0=\arg\max_{q\in\mathcal P}S(q),\qquad\mathcal P\leftarrow\mathcal P\setminus\{q_0\}`)}
    <p>${math("S")} is a dimensionless ranking heuristic, not a probability. Exact ties use proposal order. <strong>If ${math(String.raw`n(q_0)<n_{\min}`)}, stop the entire finisher.</strong>${robust ? " Do not try lower-ranked proposals, even if one has more support." : ""}</p>`, [[robust ? "SupportAndResidualScorer" : "SupportFirstScorer", "science/selection.py"]]);
}
function refinementStep(p) {
  return step("finisher-optimization", "Optimize this source coordinate against selected measured ranges",
    settings(val(String.raw`2\tau`,Number((2*p.tolerance_m).toPrecision(12)),"m: association cutoff and Gaussian scale"),val(String.raw`\tau`,p.tolerance_m,"m: soft-L1 scale")) +
    `<p>At ${math("q_0")}, retain nearest-peak associations within ${math(String.raw`2\tau`)} and compute their weights:</p>
    ${eq(String.raw`J=\{i:e_i(q_0)\le2\tau\},\qquad\bar y_i=y_{i,j_i(q_0)},\qquad w_i=\exp\!\left[-\tfrac12(e_i(q_0)/(2\tau))^2\right]`)}
    <p class="optimization-variable"><strong>Optimize:</strong> one coordinate ${math(String.raw`z\in\mathbb R^3`)}. <strong>Hold fixed throughout this solve:</strong> receivers ${math("x_i")}, selected measured ranges ${math(String.raw`\bar y_i`)}, receiver set ${math("J")}, weights ${math("w_i")} and scale ${math(String.raw`\tau`)}.</p>
    ${eq(String.raw`\boxed{\min_{z\in\mathbb R^3}\;\tau^2\sum_{i\in J}\left[\sqrt{1+\frac{w_i\bigl(\|z-x_i\|_2-\bar y_i\bigr)^2}{\tau^2}}-1\right]}`)}
    <p>This is the exact soft-L1 objective in square metres. SciPy receives residuals ${math(String.raw`\sqrt{w_i}(\|z-x_i\|_2-\bar y_i)`)} with scale ${math(String.raw`\tau`)}; Gaussian weights are <strong>inside</strong> the robust loss. ${math("J")} can include peaks outside the tighter ${math(String.raw`\tau`)} support threshold.</p>
    <p>Start at ${math("z=q_0")}; use trust-region reflective least squares, at most 120 evaluations. Require at least four receivers in ${math("J")} spanning three dimensions. Set ${math(String.raw`\hat q`)} to the returned point on finite solver success; on insufficient geometry or failure, use ${math(String.raw`\hat q=q_0`)}. Keep associations and weights fixed until the next step.</p>`, [["NearestPeakRefitter","science/selection.py"],["Multilateration3D","science/estimation.py"],["soft_l1_loss","science/costs.py"]]);
}
function acceptanceStep(robust,p) {
  return step("finisher-acceptance", robust ? "Reassociate at the new point; accept or discard it" : "Accept the unchanged proposal unless its range curve is a duplicate",
    settings(val(String.raw`\delta`,p.duplicate_m,"m: duplicate threshold")) +
    (robust ? `<p>Recompute ${math(String.raw`j_i(\hat q),e_i(\hat q),I(\hat q)`)} against available peaks at the resulting coordinate. Discard it if ${math(String.raw`n(\hat q)<n_{\min}`)}.</p>` : `<p>Set ${math(String.raw`\hat q=q_0`)}; this finisher does not optimize coordinates.</p>`) +
    `<p>For any accepted source ${math(String.raw`v\in\mathcal V`)}, compare predicted ranges over <strong>all</strong> receivers:</p>
    ${eq(String.raw`D(\hat q,v)=\sqrt{\frac1N\sum_{i=1}^N\left(\|\hat q-x_i\|_2-\|v-x_i\|_2\right)^2}`)}
    <p>Discard ${math(String.raw`\hat q`)} if any ${math(String.raw`D(\hat q,v)\le\delta`)}. Otherwise append it to ${math(String.raw`\mathcal V`)}. ${math("D")} is an RMS difference of range curves in metres, not source-to-source distance. A discarded proposal consumes no peaks; continue with the next remaining proposal.${robust ? " Acceptance uses the recomputed support and duplicate checks, not a reranking by S." : ""}</p>`, [["range_curve_rms","science/selection.py"],["GreedySourceSelector","science/selection.py"]]);
}
function consumptionStep(first) {
  return step("finisher-consumption", "Remove supporting peaks and duplicate proposals; repeat",
    eq(String.raw`A_i\leftarrow A_i\setminus\{j_i(\hat q)\}\quad\text{for }i\in I(\hat q)`) +
    `<p>After acceptance, remove exactly one associated peak at each supporting receiver. Leave all other peaks available. Suppress every remaining original proposal ${math("q")} with ${math(String.raw`D(q,\hat q)\le\delta`)}.</p>
    <p>Recompute nearest available peaks and ranking values for surviving proposals; repeat from step ${first}. Stop when ${math(String.raw`\mathcal P`)} is empty or the selected proposal fails the pre-refinement support check. Return ${math(String.raw`\mathcal V`)}; no fixed source count is imposed.</p>`, [["finish_one","execution/additional.py"]]);
}
function finisherSteps(method,p,start) {
  const robust=method.finisher==="robust";
  return [associationStep(p),rankingStep(robust,p),...(robust?[refinementStep(p)]:[]),acceptanceStep(robust,p),consumptionStep(start)];
}

function localizationSteps(method,receipt) {
  if(method.component!=="localization")return [];
  const map=receipt.parameters?.map||receipt.frozen?.map;
  const lea=receipt.parameters?.lea||receipt.parameters;
  if(!map||!lea)return [];
  return [
    step("map-localization", "Initialize the query coordinate with MPUrge-MAP",
      settings(val(String.raw`k_{\rm MAP}`,map.neighbors),val("p",map.power),val(String.raw`\alpha_{\rm MAP}`,map.alpha),val(String.raw`h_{\rm MAP}`,map.half_window))+
      `<p>Let ${math(String.raw`r_j^\ell`)} be observed query ranges for AP ${math(String.raw`\ell`)}. Match them to every reference fingerprint using MPUrge's sorting, optional unification, reciprocal minima and purging. Here the window-pattern sum is divided by ${math(String.raw`\binom{2h_{\rm MAP}+1}{2}`)} when the window is active. Use the MAP settings above, independently of source-reconstruction settings.</p>
      <p>For receiver ${math("i")} and AP ${math(String.raw`\ell`)}, let ${math(String.raw`\mathcal C_{i\ell}`)} contain the accepted query/reference pairs, ${math(String.raw`M_{i\ell}=|\mathcal C_{i\ell}|`)}, and ${math(String.raw`m_q^\ell,m_i^\ell`)} be the original peak counts. Average the coverage-adjusted costs across the ${math("L")} query APs:</p>
      ${eq(String.raw`d_i=\frac1L\sum_{\ell=1}^L\frac{\max(m_q^\ell,m_i^\ell)}{M_{i\ell}^2}\sum_{(j,k)\in\mathcal C_{i\ell}}\Delta^{\rm MAP}_{jk}`)}
      <p>No matches for any AP gives ${math(String.raw`d_i=+\infty`)}. Choose up to ${math(String.raw`k_{\rm MAP}`)} receivers with smallest finite ${math("d_i")}; call their set ${math("B")} (ties use receiver order). For positive costs:</p>
      ${eq(String.raw`u_0=\frac{\sum_{i\in B}d_i^{-p}x_i}{\sum_{i\in B}d_i^{-p}}`)}
      <p>If any chosen ${math(String.raw`d_i\le\epsilon_{64}`)}, average only those effectively zero-cost receivers. If no finite cost exists, use the survey centroid. ${math("u_0")} is the initial query coordinate in metres.</p>`,[["mpurge_map","execution/presets.py"],["MatchBasedScorer","science/localization.py"],["DistanceWeightedKNN","science/localization.py"]]),
    step("lea-localization", "Refine the query coordinate with SMART-LEA",
      settings(val(String.raw`\alpha_{\rm LEA}`,lea.alpha),val(String.raw`h_{\rm LEA}`,lea.half_window),`Iterations = <strong>${esc(lea.iterations)}</strong>`, `Peak-count unification: <strong>${receipt.unify?'on':'off'}</strong>`)+
      `<p>At iteration ${math("t")}, predict ${math(String.raw`\|u_t-v_k^\ell\|_2`)} from the reconstructed source coordinates ${math(String.raw`v_k^\ell\in\mathcal V_\ell`)}. Match each AP's measured query ranges to its predicted ranges with MPUrge and the LEA settings above; its window sum is <strong>not normalized</strong>. Combine accepted associations into ${math(String.raw`\mathcal C_t=\{(\ell,j,k)\}`)}.</p>
      <p class="optimization-variable"><strong>Optimize:</strong> query coordinate ${math(String.raw`u\in\mathbb R^3`)}. <strong>Hold fixed:</strong> associations ${math(String.raw`\mathcal C_t`)}, measured query ranges and source coordinates.</p>
      ${eq(String.raw`\min_{u\in\mathbb R^3}\frac12\sum_{(\ell,j,k)\in\mathcal C_t}\left(\|u-v_k^\ell\|_2-r_j^\ell\right)^2`)}
      <p>Initialize at ${math("u_t")}; use Levenberg–Marquardt, at most 100 evaluations. Require at least four matched anchors with centered rank three. Rebuild associations at the returned coordinate and repeat for the selected iteration count. If an iteration has no matches, insufficient geometry or solver failure, return the <strong>original MAP coordinate</strong> ${math("u_0")}. Otherwise return the last coordinate. Sources stay fixed throughout localization.</p>`,[["smart_lea","execution/presets.py"],["SmartLEA","science/localization.py"],["Multilateration3D","science/estimation.py"]]),
  ];
}

export function reconstructionExplanation(method, receipt) {
  if (!hasReconstructionExplanation(method)) return "";
  const p=sourceParameters(method,receipt);
  if (!p) return "<p class='error'>The source parameter record is unavailable.</p>";
  const proposer={mpurge:mpurgeSteps,zayets:zayetsSteps,ransac:ransacSteps}[method.family];
  const proposals=proposer(method,receipt,p), first=proposals.length+1;
  const finishing=method.finisher?finisherSteps(method,p,first):[];
  const localization=localizationSteps(method,receipt);
  const steps=[...proposals,...finishing,...localization];
  const name=method.source_full_name || method.name;
  const constructors=(method.code?.["Composition constructors"]||[]).filter(l=>typeof l==="object"&&!['mpurge_map()','smart_lea()'].includes(l.label));
  return `<article class="reconstruction-algorithm" data-family="${method.family}" data-finisher="${method.finisher||'none'}">
    <h3>${esc(name)}</h3>
    <div class="algorithm-io"><p><strong>Input, per AP:</strong> ${math("N")} known receiver coordinates ${math(String.raw`x_i\in\mathbb R^3`)} and ${math("m_i")} observed peak ranges ${math("y_{ij}")} at receiver ${math("i")}.</p>
    <p><strong>Output:</strong> source coordinates ${math(String.raw`\mathcal V\subset\mathbb R^3`)}. Coordinates, ranges and Euclidean distances ${math(String.raw`\|\cdot\|_2`)} are in metres. True peak identities and source count are not inputs. Source coordinates may lie outside the room.</p></div>
    <p class="algorithm-constructor"><strong>Constructors:</strong> ${constructors.map(l=>link(method,l.label.replace(/\(\)$/,''),l.file)).join(' · ')}</p>
    <nav class="algorithm-index" aria-label="Reconstruction steps">${steps.map((s,i)=>`<a href="#algorithm-step-${i+1}">${i+1}. ${esc(s.title)}</a>`).join('')}</nav>
    ${ordered(method,`A. ${esc({mpurge:'MPUrge',zayets:'Zayets',ransac:'RANSAC'}[method.family])} source proposals`,1,proposals,'data-stage="proposals"')}
    ${method.finisher?ordered(method,`B. ${esc(finisherNames[method.finisher])}`,first,finishing,`data-stage="finisher" data-finisher="${method.finisher}"`):'<p><strong>No finisher:</strong> return every proposal in P directly as a source.</p>'}
    ${localization.length?ordered(method,"C. Query localization",first+finishing.length,localization,'data-stage="localization"'):''}
    </article>`;
}

export function finisherParameterSummary(method,receipt) {
  if (!hasReconstructionExplanation(method)) return "";
  const p=sourceParameters(method,receipt); if(!p)return "";
  const recordOrigin=receipt.frozen?.source&&!receipt.parameters?.source?"Frozen upstream":"Selected / inspected";
  const specs=method.family==='mpurge' ? [
    ['k','neighbors','positions','Receiver neighborhood, including center and boundary ties'],
    ['h','half_window','peaks','Pattern-window half width; inactive when alpha = 1'],
    [String.raw`\alpha`,'alpha','','Range weight; pattern weight is 1 − alpha'],
  ] : method.family==='zayets' ? [
    ['R','scope_radius_m','m','Receiver scope radius'],['g','gap_m','m','Split range clusters at gaps ≥ g'],
    [String.raw`d_{\max}`,'group_cost','','Maximum cluster-feature dissimilarity'],
  ] : [['T','trials','trials','Three-sphere samples per AP']];
  if (method.finisher||method.family==='ransac') specs.push([String.raw`\tau`,'tolerance_m','m','Nearest-range inlier threshold']);
  if (method.finisher) {
    specs.push(['f','support_fraction','','Required survey fraction; also require at least five receivers']);
    if(method.finisher==='robust')specs.push(['a','smooth_weight','','Smooth agreement weight',.35],['b','median_penalty_weight','','Median-error penalty weight',.10],['c','median_clip_factor','','Clipping limit for normalized errors',4]);
    specs.push([String.raw`\delta`,'duplicate_m','m','Maximum RMS range-curve difference counted as a duplicate']);
  }
  return `<section class="finisher-parameter-summary"><h3>Source parameters in algorithm order</h3><p>${recordOrigin} source settings. ${method.family==='mpurge'?`Peak-count unification is <strong>${receipt.unify?'on':'off'}</strong>.`:''}</p>
    <table class="parameter-table finisher-values"><thead><tr><th>Symbol / key</th><th>Value</th><th>Use</th></tr></thead><tbody>${specs.map(([symbol,key,unit,meaning,fallback])=>`<tr><th>${math(symbol)}<br><code>${esc(key)}</code></th><td>${esc(p[key]??fallback)} ${unit}</td><td>${meaning}${key in p?'':'; fixed constructor default in this saved run'}</td></tr>`).join('')}</tbody></table>
    ${method.finisher==='robust'?`<p>Derived values: association cutoff and Gaussian scale ${val(String.raw`2\tau`,Number((2*p.tolerance_m).toPrecision(12)),'m')}; soft-L1 scale ${val(String.raw`\tau`,p.tolerance_m,'m')}. The coordinate solve allows 120 evaluations.</p>`:''}
    </section>`;
}
