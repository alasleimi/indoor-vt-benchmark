# Local results explorer

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
serves only three UI assets and read-only JSON endpoints, not the repository
filesystem. No frontend build, npm dependencies or new scientific dependencies
are required. UI/backend code lives outside the frozen scientific source hash.
