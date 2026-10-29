# Redesign experiment explorer

Run `uv run indoor-vt view` and open <http://127.0.0.1:8770>. `--data`, `--run`,
and `--port` retain their existing meanings. The original Results Explorer's
visual system and 3D renderer are reused. Data comes only from redesign runs;
the old campaign, published snapshot and method catalog are not loaded. The
base `rectangle-clean-selection-v4` run also discovers the compatible sibling
`rectangle-additional-selection-v1` and `rectangle-compositions-v1` studies. Dataset versions must agree. Each
result retains its owning run, receipt, parameters and saved outcomes.

## AP-separated source metrics

Source assignments always stay within their AP, including the combined A+B
table, noise curves, CSV and 2D/3D map lines. For each AP a, let n_a and m_a be
the estimated and true counts, w_a=max(n_a,m_a), O_a the per-AP OSPA, T_a the
maximum gated matching cardinality, and E_a the minimum matched squared error
among assignments with that cardinality. A+B reports sum(w_a O_a)/sum(w_a),
2 sum(T_a)/sum(n_a+m_a), and sqrt(sum(E_a)/sum(T_a)). Opposite cardinality errors
in different APs cannot cancel. The gate is 0.5 m and OSPA uses p=1,c=1 m.
OSPA and F1 average realizations; matched RMSE pools squared errors and counts.

`scripts/refresh_source_metrics.py` updates saved presentation metrics using
retained coordinates. It does not rerun reconstruction, change frozen choices,
or alter historical search objectives; original receipt metrics remain under
`recorded_metrics`. New evaluations and searches use separate-AP assignments.

## MPUrge with crossings allowed

`scripts/run_allow_crossing.py` runs standalone reconstruction, support-first
peak-consumption finishing, and robust range-refinement/peak-consumption
finishing. Each gets unification on/off, 512 reproducibly sampled source
configurations per mode, 87 stagewise SMART-LEA settings per mode, and 512 joint
configurations per mode: 6,666 trials in total. Robust finishing includes all
three ranking hyperparameters. All tuning is noise-free, followed by the usual
frozen noise-level evaluations. The initializer's already-selected MAP settings
are reused for stagewise tuning. Joint tuning searches all three stages.

The crossing variant accepts every reciprocal candidate in the ordinary
matcher's trace. Since the ordinary method already purges rejected candidates'
endpoints, this changes only the crossing veto, not windows, costs, rounds or
endpoint uniqueness. Standard matcher cache keys remain unchanged. The new
component has a distinct cache identity. MAP and SMART-LEA keep their ordinary
matchers; the variant changes the source reconstruction stage.

The viewer discovers `rectangle-allow-crossing-{solo,consumption,robust}-v1` as
results become available. Names, tags, scientific explanations, constructor
links, parameter/search tabs, peak diagnostics and pipeline labels distinguish
the variants. `scripts/extend_allow_crossing_obstacles.py` waits for the frozen
choices and appends their 48 noiseless matched-survey obstacle conditions without
recalibration. Existing obstacle outcomes are reused and stay visible.

## Method configurations

Method names identify the algorithm. Separate columns identify parameter
selection and, in the localization list, the VT source. Peak unification appears
as a tag beside the method name where applicable. Tags on saved Zayets/RANSAC
joint source stages explicitly refer to MAP/LEA unification in the tuning pipeline.
Noise-free selection is explained once at page level. Across both
completed studies there are 32 configurations:

| List | Naming and context |
| --- | --- |
| VT reconstruction | Named proposer + full finisher name, or MPUrge / MPUrge triangle; OSPA or Joint with MPUrge-MAP + SMART-LEA |
| Raw localization | MPUrge-MAP; Independent or Joint with the named reconstructor + SMART-LEA |
| Localization | MPUrge-MAP → SMART-LEA; VTs supplied by the named reconstructor or Ground truth; Stagewise, Joint with the named reconstructor, or LEA only |

Estimated-map pipelines are named **MPUrge-MAP → SMART-LEA**, with explicit
**Stagewise** or **Joint with [reconstructor]** selection labels. In the VT list,
**Joint with MPUrge-MAP + SMART-LEA** explicitly names the other tuned stages. The joint
label means the reconstructor, MAP initializer and SMART-LEA refiner are tuned
together on mean 3D error after refinement, using noise-free calibration queries
pooled across all obstacle layouts. It does not select separately for each obstacle
selection. The partners appear in tables, graph labels, details and CSV; the
objective remains in the explanatory tooltip and method details.
They use the same algorithm
chain: the listed reconstructor supplies the source map, MPUrge-MAP initializes
the query position, and SMART-LEA refines it. Stagewise tuning selects the reconstructor on OSPA,
selects MPUrge-MAP on initial localization error, then freezes both while tuning
SMART-LEA. Joint tuning selects parameters of all three together on final
localization error. Both use noise-free calibration data. The **Ground truth** row
instead supplies ground-truth source coordinates with the separately tuned
MPUrge-MAP initializer.

Raw localization always measures MPUrge-MAP before refinement. For example,
**Joint with Zayets + finisher + SMART-LEA** describes which pipeline selected the MAP
parameters; MAP itself matches query and survey fingerprints and does not use a
VT map. Raw rows included in the Localization list are named **MPUrge-MAP only**,
with **Not used** in the VT-source column. The stage filter can show only complete
pipelines. Graph labels abbreviate MPUrge-MAP to **MAP** and SMART-LEA to **LEA**,
with a visible key; source, tuning and unification still distinguish every curve.
Detail badges and CSV exports retain the same context. Method IDs and saved
experiment records are unchanged by these presentation labels.

The full-pipeline-selected MPUrge and MPUrge-MAP rows are intermediate outputs
already saved in joint experiment outcomes. Source metrics are computed from the
saved source coordinates with the existing OSPA/F1 evaluator; initializer errors
use `initial_xyz` and the recorded query IDs. Neither row represents another
search, and neither substitutes final pipeline loss for a component metric.
Their search histories refer to the same joint trial log. Missing saved component
outputs do not produce rows. Scalar summaries are held in viewer memory; the
viewer does not rewrite the run's results, selected configurations or outcomes.

## Browsing

- The room header uses saved width, depth and ceiling height, and shows distinct
  calibration and validation query counts. Expand **Where do these queries come
  from?** for sampling, the per-layout split and how noise repetitions are counted.
  The current room is 7.121984 × 5.167174 m with a 3.189195 m ceiling. Its four
  layouts (0, 1, 3 and 5 obstacles) each have 100 independently sampled query
  positions: the first 60 calibrate parameters and the remaining 40 are held out.
  Thus there are 240 calibration and 160 validation positions, separate from the
  280 survey positions. At each nonzero noise level, two realizations yield 320
  validation estimates; clean observations need only one realization.
- Switch between noise-free calibration and fixed-parameter noise evaluation.
  Query evaluation uses held-out positions in the same room. Source reconstruction
  uses that room's survey, not unseen rooms or a held-out survey.
- Filter by noise, peak unification, parameter selection, localization stage,
  or method text. Table headings sort the current rows. CSV retains full numeric
  precision, receipt identities, parameters, and frozen upstream settings.
- All metrics appear together: OSPA (c = 1 m), F1 @ 0.5 m and matched RMSE @ 0.5 m
  for VT reconstruction, and Mean, RMSE,
  Median and P90 for localization. Click a column heading to sort. **Obstacles**
  chooses all layouts, the empty room, or 1, 3 or 5 obstacles. All metric columns
  and CSV values follow this selection; map cases do too. Noise curves show OSPA
  for reconstruction and mean error for localization. Metrics use saved predictions and pool
  noise repetitions at query level. Overview tabs retain the detailed breakdown.
  Selection and saved outcomes are unchanged, including calibration on all four
  layouts at zero noise.
- In VT reconstruction, **AP** selects A, B or A+B (the default). Counts, OSPA,
  F1, ranking, graphs, CSV and the 3D source map all use that selection. There is
  no average across A, B and A+B in these displayed values. True counts in this
  room are 25 for A, 25 for B and 50 for A+B. Scores average repetitions within
  the chosen AP view; varying counts are shown separately per saved repetition,
  never as fractional sources. The selector reads saved source coordinates and
  does not rerun reconstruction, tuning or evaluation. Search-history losses
  retain the experiment's recorded objective.
- Copy view links to preserve filters, sorting, an open method/tab, and an
  inspected trial. The local server must be running to open them.
- The noise chart always shows evaluation results with parameters held fixed.
  Choose curves with the checkbox list; Select all and Clear all operate on the
  current list. Curve visibility is retained in copied view links.
  The experiment-status section shows acquisition, search and cache progress.

## Method tabs

- **Overview:** metrics, selection protocol, noise curve and result provenance.
- **Map:** the original interactive 3D source map, including rotation, zoom, top
  and side views, AP selection, reflection visibility and F1 links. Choose a
  saved repetition/case. Localization offers an immediate **2D top view / Interactive
  3D** switch. Both use the same saved query and estimated coordinates; 2D also
  shows the CDF of 3D errors. Localization links join each query with its own
  estimate, without the source-map F1 distance cutoff.
  Saved case labels include obstacle counts. In 2D, amber polygons show the actual
  rotated footprints with height labels; 3D shows translucent boxes at their saved
  heights, with an Obstacles layer toggle. Purple survey markers identify historical
  empty-room reference positions inside the selected layout's obstacle volumes.
  A point above a box is not classified as inside it. The map reports both covered
  references and estimates inside boxes without moving or removing any predictions.
- **Scientific method:** the implemented operations, objective and metric
  definitions, with links to scientific algorithms and observation/metric code.
  Source links open the current local Python implementation with line anchors.
- **Code & caching:** execution/experiment configuration, cache identity,
  storage limits, saved evidence, and links to the implementation of each.
- **Parameters:** the current receipt's tuned settings and frozen upstream
  settings, including separate MPUrge, MPUrge-MAP and SMART-LEA joint settings.
  Range weight is **α**, and pattern weight is **1 − α**. Finished Zayets/RANSAC
  source reconstruction does not use MPUrge size unification; its pipeline's
  fingerprint initializer and SMART-LEA matcher do.
- **Search history:** all trials, 50 per page, filterable by parameter text or
  trial number, sortable by loss or trial order, and exportable. The chart shows
  trial losses and best-so-far loss. Inspecting a trial changes the parameter and
  evidence context; Return to selected result restores the original receipt.
- **Evidence:** typed intermediate batches, array pagination, on-demand replay,
  and survey/survey or query/survey matching with decision rounds and costs.
  The peak matrix uses the current configuration's first-round scores and can
  overlay first-round or final matches. Inspect individual cells or export their
  scores, decision rounds and exact parameters as CSV.
  Matching follows the chosen case's room and noise repetition. Query matching
  uses the fingerprint initializer, not the SMART-LEA refinement matcher.
  Execution and caching code links also appear here.

## Full peak-matching explorer

The obstacle audit found 0, 2, 3 and 5 of the 280 survey reference positions inside
the 0-, 1-, 3- and 5-obstacle layouts respectively. These fingerprints were acquired
before objects were added, so they remain valid historical measurements. The current
matcher keeps them eligible and neither MAP nor SMART-LEA imposes obstacle-volume
constraints on its estimates. No true query position lies inside an obstacle.
Query measurements were verified against their layout-specific native path records
and simulator inputs; they do include the corresponding obstacle geometry. This
viewer change reports the existing experiment and does not filter its reference bank.

The Peak matching navigation item and the matching method's Evidence tab open
the old viewer's full peak interface. Its data adapter uses the actual current
scientific matcher and selected parameters. Choose independently OSPA-tuned or
jointly tuned MPUrge, with or without size unification, and an AP/noise condition.
An inspected trial can also be opened from its Evidence tab.

Pair diagnostics use frozen parameters and native observations with the current
matcher. They do not replay reconstruction, optimization or finishers, so their
availability does not depend on whole-run code hashes. Dataset identity/schema,
observation reproduction and native source-identity checks still apply. The
scores and match traces are generated together; saved trial outcomes remain
unchanged. Full trial replay has a separate compatibility check. Obstacle receipts
use their inline pair diagnostic; the room-wide truth overlay is limited to the
empty-room survey until it supports each layout's own source catalogue.

The explorer retains native-gain stems, a first-round cost matrix, true-source
squares, first-round/final match circles, cell score breakdowns, room/pair
precision and recall, F1, pair ordering, correlation summaries, CSV and deep
links. Source identities are reconstructed from native path geometry only to
score/display the result; they never enter the matcher. Diagnostic ranges must
exactly equal the experiment observation conversion. The diagnostic uses the
first repetition of the receipt's split, identified in the sidebar and CSV.

## Finished reconstructors and SMART-LEA experiments

Run `uv run python scripts/run_additional.py` to run or resume the extension.
It preserves the original study and reuses its dataset, observation seeds,
calibration/evaluation split, noise levels, independently tuned MPUrge-MAP
initializer, and SMART-LEA search space. Scientific implementations remain
unchanged. The extension's code hash is recorded alongside the scientific and
dataset versions; version drift requires a fresh run directory.

The execution layer maintains nearest-peak residuals incrementally: consuming a
peak only invalidates candidates whose nearest available peak was that peak.
For support-first selection, the smooth tie-break is strictly smaller than one,
so only candidates with maximal integer support need its full calculation.
These optimizations preserve the reference greedy order, scores and selected
coordinates; clean and noisy parity checks cover both finished reconstructors.

- **Zayets + support-consumption finisher:** local range-cluster proposals,
  support-first greedy selection, duplicate suppression and consumed peaks.
- **RANSAC + robust support finisher:** deterministic sphere-intersection
  proposals, batched retention, support/residual scoring, robust nearest-peak
  refitting, duplicate suppression and consumed peaks.

The reconstructors select their parameters on noise-free OSPA. Each then has
stage-wise and jointly tuned **MPUrge-MAP + SMART-LEA** pipelines, with size
unification on and off. The stage-wise search freezes the independently selected
source and initializer settings. The joint search includes that baseline and
512 deterministic configurations per family/mode, selecting on final mean 3D
localization error. No evaluation conditions run until all parameter selection
has finished; selected parameters remain fixed across noise levels.

| Search | Grid / budget |
| --- | --- |
| Zayets neighborhood radius | 0.75, 1 m |
| Zayets cluster gap | 0.4, 0.6, 0.8 m |
| Zayets grouping threshold | 0.25, 0.5, 1 |
| RANSAC proposal trials | 10,000; 30,000 |
| Both finishers' inlier tolerance | 0.1, 0.18, 0.3 m |
| Both finishers' minimum support fraction | 0.04, 0.08, 0.16 |
| Both finishers' duplicate range-curve RMS threshold | 0.06, 0.12, 0.24 m |
| Source-only exhaustive selection | 486 Zayets + 54 RANSAC trials |
| SMART-LEA with frozen upstream stages | 87 effective configurations × 4 |
| Joint selection | 512 configurations × 4 |

This is 2,936 additional selection trials and 110 retained selected-condition
outcomes. Once complete, saved joint source and initializer stages produce 18
additional viewer configurations, for 32 configurations / 352 measured
conditions across both studies. Partial searches appear only in calibration;
evaluation curves appear as their actual measurements finish. Source-only
finished reconstructors carry a `Not used` unification filter tag because
that operation is not applicable to them. The grouped checkbox filter combines
choices with OR within a group and AND across groups; leaving a group unchecked
does not restrict it. Family, finishing, triangle matching and unification have
separate groups inside the same filter menu.

Zayets feature weights remain 0.5 each. Both finishers require at least five
supporting positions, consume support, and have no source-count cap. RANSAC
retains 12 proposals per 400-trial batch with seed 9173; its refitter uses twice
the inlier tolerance for association/Gaussian weighting, soft-L1 loss, a 0.03 m
scale floor and at most 120 evaluations. These fixed constructor choices are
linked from each method's tabs.

Only supported pair types are offered for a method. Replay retains the existing
implementation/dataset version check and bounded numerical/audit caches. It can
recreate evicted intermediate computations but does not launch another parameter
search. Old preset peak-matching matrices are intentionally not served as evidence
for the redesigned, tuned configurations.

## Map controls and code navigation

The map has one control row: room, saved noise level, obstacle layout (localization)
or AP selection (reconstruction), and noise realization. Changing noise selects
the corresponding measured receipt; it never relabels the current coordinates.
The chosen layout and realization survive noise changes when available. Calibration
offers zero noise only. Evaluation offers levels that have actually finished.
Realizations are numbered 1/2 within their split; the original seed IDs 0/1 for
calibration and 2/3 for evaluation remain in the evidence. Identical clean
realizations are retained once. The 2D/3D switch follows the data controls.

Scientific method, Code & caching, and Parameters link directly to composition
constructor lines. Those constructors show the injected algorithms; lower-level
scientific and cache implementations remain available too. Parameters are grouped
by neighborhoods, matching, proposals, clustering, finishing and localization.

## Finisher swaps and triangle comparisons

`uv run python scripts/run_compositions_parallel.py --workers 2` runs the new
suite with bounded independent worker processes. The serial entrypoint remains
`scripts/run_compositions.py`. Worker logs are private; the coordinator publishes
atomic snapshots and completed receipts/outcomes. Existing serial trials can be
resumed. The scientific implementation and dataset hashes remain unchanged;
the new composition code and the scheduling script have separate recorded hashes.

The new study compares Zayets + robust support finisher, RANSAC + support-consumption
finisher, MPUrge + each consuming finisher, and MPUrge triangle. For MPUrge, a consuming
finisher **replaces** candidate clustering and relative-support filtering. Triangle
matching applies both range triangle inequalities and keeps the usual cluster/support
stages. Its slack grid is 0, 0.05, 0.2 m. All MPUrge variants have unification on/off.

The completed v1 selection used the noise-free calibration protocol and parameter ranges.
The two swapped Zayets/RANSAC source grids exhaust 486/54 configurations. Each larger
MPUrge source space samples 512 distinct effective configurations reproducibly;
inactive windows at alpha=1 are collapsed. The same sampled set is used for both
unification modes. This is a budgeted comparison, not an exhaustive MPUrge search.
Each of ten pipeline modes has an exhaustive 87-setting stagewise SMART-LEA search
and a 512-setting joint search including the independently selected baseline.
The latter freezes nothing during tuning; stagewise tuning freezes the selected
reconstructor and independently selected MPUrge-MAP initializer.

The complete schedule is 9,602 trials, eight source-only selections and twenty
pipeline selections. Each selection is evaluated at calibration zero noise and
all ten evaluation noise levels with fixed parameters: 308 retained outcomes.
Saved joint source/initializer stages add twenty derived rows, giving 48 new viewer
configurations (80 across all three studies) when complete. Partial results are
shown as they finish; no future condition is populated with a placeholder score.
Each independent selection may be evaluated before another search finishes;
evaluation results are never fed back into selection.

The current search spaces also expose the robust finisher's `smooth_weight`
(`0, 0.35, 0.7`), `median_penalty_weight` (`0, 0.1, 0.2`), and
`median_clip_factor` (`2, 4, 8`). These control
`S = n + smooth_weight * g - median_penalty_weight * median(min(e / tolerance, median_clip_factor))`,
where `n` counts survey positions whose nearest available peak has range error
`e <= tolerance`, and `g = sum(exp(-0.5 * (e / tolerance)^2))` over survey positions.
Clipping choices collapse to one effective setting when the median penalty is zero.
The parameters apply to RANSAC, Zayets and MPUrge with the robust support finisher;
they enter source selection, are frozen for stagewise SMART-LEA, and are searched
jointly with MAP/LEA parameters in joint selection. RANSAC's earlier proposal-retention
ranking keeps its separate original defaults. The support-consumption finisher has
no use for these parameters, so its search spaces omit them.

Existing results retain their original fixed values (`0.35, 0.1, 4`); the Parameters
tab identifies those as not optimized. The expanded search requires a fresh study
and records the earlier initializer receipts it reuses; adding the grid does not rerun completed studies.
Saved outcomes remain readable after implementation changes, while recomputation
still requires the recorded implementation and dataset versions.

## Naked Zayets and RANSAC

`scripts/run_naked.py` writes `rectangle-naked-selection-v1`, reusing the saved
MPUrge-MAP initializer selections. It runs 18 Zayets proposal configurations and
6 RANSAC configurations, followed by 87 stagewise SMART-LEA configurations and
512 joint configurations for each method and each unification mode. The schedule
has 2,420 calibration trials and 110 retained calibration/evaluation outcomes.
The selected configurations are evaluated at all existing noise levels.

"No finisher" returns every proposal produced by the existing proposer. Zayets
still constructs and estimates its local tracks; RANSAC still retains the best
12 hypotheses per 400 sampling trials. There is no final coordinate refinement,
support threshold, duplicate suppression or peak consumption. Repeated candidate
coordinates remain separate predictions in the metrics. The viewer labels the
methods "Zayets (no finisher)" and "RANSAC (no finisher)" and offers a "No finisher" tag.

## Matched source RMSE

The viewer derives matched RMSE from retained source coordinates for the selected
AP scope. A one-to-one assignment first maximizes the number of pairs whose 3D
distance is at most 0.5 m, then minimizes their total squared distance. If `M_r`
is that set for noise realization `r`, the displayed value in metres is
`sqrt(sum_r sum_(i,j in M_r) ||prediction_i - truth_j||² / sum_r |M_r|)`.
No matches produces a dash, not zero. Unmatched sources affect OSPA and F1 but
do not enter matched RMSE. OSPA and F1 retain their existing per-realization
averaging and numerical values. The map draws the same pairs used by matched
RMSE. Table sorting, method details and CSV include the new metric. This is a
viewer calculation; saved outcomes, tuning objectives and trials are unchanged.

## Peak matching by distance and survey position

The peak matching page now plots precision, recall and F1 against the full 3D
separation of survey receivers. The default population is every unordered survey
pair; a selector restricts it to the unique pairs used inside the selected
MPUrge neighborhoods. All-pairs evaluation is a matching diagnostic only and
does not change reconstruction inputs or recalibrate parameters. AP, noise and
first/final matching controls apply to the graph, map and existing inspector.
Final matches are the default unless a saved view explicitly selects round one.

Distance bins pool correct matches C, selected matches M and available true
correspondences T before computing C/M, C/T and 2C/(M+T). Bin intervals are
left-inclusive and right-exclusive; hovering exposes the counts. Zero
denominators remain undefined. Truth removed by size unification remains in T,
but paths absent from the noisy observed fingerprint do not.

The 3D map colors every comparison position relative to the selected reference
using a fixed 0–100% scale. Click a point or use the reference selector; hover for
counts and use the inspect button or double-click to open the peak matrix.
Precision, recall, F1 and explicitly defined all-candidate pairwise accuracy are
available. Self-pairs have no metric. Undefined values and positions outside the
chosen pair population are grey. Reference, metric, population and bin width
are included in copied view links. Distance-bin CSV export preserves the counts.

The compact spatial API caches both matching stages per saved trial/AP/noise
realization. It carries native source identities through the same observation
conversion as the pair inspector and uses the current selected matcher,
including its unification, triangle and crossing settings.

## Checks

`uv run pytest -q` covers the experiment contracts and the viewer's separation of
pipeline objectives, component metrics, calibration winners and saved query stages.
`uv run ruff check src scripts tests` checks Python code. The frontend has no
build step or runtime npm dependencies.
