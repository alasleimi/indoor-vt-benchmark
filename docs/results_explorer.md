# Local results explorer

The **Peak matching** navigation link opens a first-round diagnostic for
standard MPUrge on **Rectangle 01** (`rectangle_calibration_00`), Direct NIST
only. Choose AP A or B, any existing noise level, and two survey fingerprints.
The horizontal and vertical headers show ordered peaks, ranges and native gain.
Each matrix cell always shows **first-round** dissimilarity (lower is better).
The **Match overlay** selector switches circles between first-round matches and
final matches from all rounds; squares mark shared true source identity. Both
shapes indicate a correct match. Pair metrics follow the selected overlay;
room summaries, reliability charts and pair ordering remain first-round.
Grey cells were excluded by size unification and retain their ground-truth
markers. Clicking a final match reports its acceptance round while retaining
the first-round score breakdown. The overlay is preserved in copied view links.
Pair CSV includes the score round, selected scope, both acceptance flags, and
the acceptance round.

The room summary pools TP/FP/FN over the 4,167 distinct fingerprint pairs in
the nearest-10-reference neighborhoods, separately for each AP/noise condition.
Precision, recall and F1 refer to peak identity, not the map's 0.5 m tolerance.
Custom pairs outside those neighborhoods can be inspected but do not enter the
room summary. Pair browsing can order by errors or F1. The score-reliability
chart uses all first-round matrix cells; accepted-match correlation is reported
separately. Matrices and summaries can be exported as CSV.

The diagnostic uses the unchanged standard preset (`p=2`, `alpha=0.5`, summed
pattern discrepancy). Native clean range/AoA identifies the source; saved RNG
seeds reproduce noisy-peak rejection and sorting exactly. Source labels and
gains are evaluation/display data only. Circles use the existing reciprocal
matching and order-check interpretation. The browser implementation is checked
against Python outputs; it runs in its own worker and caches at most three
decoded packs. Only the selected condition is loaded.

The additional published snapshot is about **1.57 MiB**, with all compressed
packs in Git LFS. It stores fingerprints once, not all pairwise matrices.
Rebuild it with `python scripts/build_peak_matching.py` in the scientific
environment; verify the first-round scores with `node scripts/check_peak_matching.mjs`.
Run `python scripts/check_peak_matching.py` in the scientific environment to
also compare every final match list with production Python MPUrge. Verification
fixtures stay under ignored `runs/`; final overlays require no extra published data.
These commands do not change the estimator or regenerate NIST data.

The same viewer is published at
**https://alasleimi.github.io/indoor-vt-benchmark/**. GitHub Pages runs a static
version: lossless metric columns are loaded on demand from the LFS-backed
snapshot, verified and aggregated in a browser worker. Every method row links
to its commit-pinned implementation. Its detail panel has a **Method** tab for
code, papers and algorithmic descriptions; the first complete walkthrough is
the Zayets–Steinbach 2018 reconstruction implementation. Its labels distinguish
**Paper** (specified operations), **Ambiguity resolved** (implementation choices
left open by the paper), and **Added** (extra operations absent from the paper’s
procedure). Parameter changes are identified separately. Mixed steps explain
which parts belong to each category; implementation lineage is mentioned once.

Each VT result also has a **3D map** tab. Select one room, noise level and AP
configuration to inspect its saved reconstruction against ground truth, room
geometry and the survey points. Drag to rotate, Shift-drag to pan, and scroll
to zoom; top/side views and fit-all/room framing are available. Hover or tap a
source for its coordinates and reflection labels. Match links reproduce the
saved F1 assignment, including combined-bag matching for A+B. A world-aligned
background grid shows metre spacing in 3D, top and side views. Turn off
**Second-order truth** to hide sources requiring two reflections and their
match links. Direct and first-order sources remain visible, including any
shared coordinate with a direct or first-order path. This is a display filter:
the saved OSPA/F1 scores and predictions do not change. **Fit all** frames the
currently visible layers. Query obstacle
layouts do not alter these survey-time reconstructions. The viewer draws every
prediction, including large failed reconstructions; it reports off-screen
counts when zoomed in. The renderer and coordinates load only on opening this tab.

Unfiltered tables and detail panels use precomputed exact summaries. Filtered
VTs share one compact pack per track; filtered localization uses room bundles
when a room or family is selected. A single worker handles requests serially,
cancels old requests, and uses bounded caches. Numeric outputs keep their
original precision, with no sampling or histogram-based quantile estimates.

To preview the published version after fetching LFS data:

```powershell
python scripts/build_pages.py
python -m http.server 8766 --bind 127.0.0.1 --directory build/pages
```

The Pages workflow checks out LFS objects, verifies snapshot checksums and
deploys the static viewer, metric columns and lazy 3D payloads. The calibration input ZIP
remains separately downloadable from the repository. Updating descriptions uses
`python scripts/build_method_catalog.py`; refreshing data uses
`python scripts/export_results_site.py --include-inputs` from the original
workspace. No dataset generation is triggered by the Pages workflow.
To rebuild only the summary, room and 3D export from existing local results,
run `python scripts/build_explorer_assets.py`.

Run from the repository root, with the existing environment:

```powershell
.venv/Scripts/python.exe scripts/results_explorer.py
```

Open **http://127.0.0.1:8765**. The server binds only to loopback. Stop it with
Ctrl+C. A different port can be selected with `--port 8766`.

The explorer reads completed **v6 initial preset receipts**. It does not start
campaign workers, run methods, change scientific artifacts, or open test-room
results. These are calibration presets using published settings and disclosed
adapter defaults, not hyperparameter winners or held-out performance estimates.
Historical search trials are preserved in their existing files and are not mixed
into these preset comparisons.

## The three lists

- **VT reconstruction:** 72 presets, 36 per track. Defaults to ascending OSPA.
- **Raw localization:** 8 presets, 4 per track, before SMART-LEA.
- **All localization:** the 8 raw baselines and 30 complete pipelines. Filter
  by stage to show just the 15 SMART-LEA pipelines per track. These are the
  evaluated pairings, not every possible combination of components.

SAPP receiver and direct NIST results are shown separately. Receiver noise is
SNR in dB; direct noise is Gaussian range standard deviation in meters.
Filter by room family, room, noise and A/B/A+B. Localization also supports
obstacle count and layout. VT maps are evaluated on surveys and have no query
obstacle-layout dimension. The VT list can filter grouping scope and merger.

Click a method to inspect its exact parameters, per-room errors, cumulative
distribution and artifact provenance. Table headings sort the current selection.
Search matches method, variant and merger names. **Export CSV** exports all
filtered rows in their current sort order, at full numerical precision, with
the track, calibration label and filter state. **Copy view link** preserves the
view and filters in the local URL; the server must still be running to open it.

## Metric definitions

- VT OSPA uses `p = 1`, `c = 1 m`, averaged uniformly over source-map cases.
- F1 uses the saved one-to-one matching at **0.5 m** and is averaged over maps
  (macro F1). The source bag includes real APs and reflected VTs. Predicted and
  true source counts are displayed as means over the selected maps.
- Localization statistics are calculated directly from selected **3D query
  errors**, including fallback outputs. Median and P90 are not averages of
  room-level quantiles. Each query case has equal weight.
- Localization **RMSE (m)** is `sqrt(mean(error_3d_m²))` over the selected query
  cases. It is available in both localization tables, sorting, details and CSV export.
- Pipeline gain is `100 × (initial mean − final mean) / initial mean` using
  the same pipeline's saved initial estimates. Positive means improvement.
- The CDF draws up to 201 exact order statistics for display. All table and
  detail summary metrics use every selected case without subsampling.
- Cached/historical runtime measurements are not ranked because their timings
  are not a controlled method-speed comparison.

## Derived cache and verification

The first launch streams each referenced gzip objective into compact float64
NumPy columns under `runs/development/explorer/v6/`. It verifies the objective's
SHA-256 against its receipt, the identity and case count, and reconstructed
summary metrics against the original scores (and median/P90/fallback counts).
References to earlier revision artifacts are followed rather than discarded;
their recorded compatibility evidence is visible in the detail panel.

Later launches reuse the index if its importer, receipts, design and artifact
size/modification-time fingerprint have not changed. A forced rebuild rechecks
every objective checksum and streams the cases again:

```powershell
.venv/Scripts/python.exe scripts/results_explorer.py --build-only --rebuild
```

The server memory-maps compact columns and processes one method at a time. It
serves the viewer assets, manifest-listed publication files and read-only JSON
endpoints, not arbitrary repository paths. No frontend build, npm dependencies or new scientific dependencies
are required. UI/backend code lives outside the frozen scientific source hash.
