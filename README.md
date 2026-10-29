# Indoor VT benchmark

**[Open the published experiment viewer](https://alasleimi.github.io/indoor-vt-benchmark/)**

The publication contains the saved redesign results, searches, 2D/3D maps,
scientific explanations, concrete dependency-injected constructions, peak
matching and true-matching analysis. Matching runs in a browser worker on the
saved noisy fingerprints. Its implementation is checked against Python,
including crossing, unification, triangle constraints and empty inputs.
The local Python viewer additionally supports numerical replay and full cached
array paging; the static publication includes retained evidence previews.

To rebuild the published site from its saved snapshot, run
`python scripts/build_pages.py`. To refresh that snapshot from local completed
experiments, run `.venv/Scripts/python.exe scripts/export_viewer_snapshot.py`.
The export reads saved outcomes and does not rerun calibration. Pushes to the
`alasleimi/indoor-vt-benchmark` main branch deploy through GitHub Pages.

[Local stars and edge-weighting ablations](docs/star-weight-ablation/README.md)
compare grouping, similarity/distance weights and both finishers without
recalibration. The two unweighted star-plus-finisher variants are included in
the viewer as frozen exploratory methods; the full report retains every arm.

The redesign separates scientific components from caching, audit storage, and
experiment execution. Start with the **[redesign brief](docs/redesign.md)**; it
links the three detailed design documents. [Implementation status](docs/implementation_status.md)
records the completed priority study and subsequent work.

The initial experiment uses one new seeded rectangular room, native NIST
measurements, and a 0.5 m survey selected from a 0.1 m lattice at two heights.
APs have at least 5 cm wall clearance. The data model supports multiple rooms
and retains native channel fields for later receiver experiments.

## Run

Use Python 3.11 and the committed environment:

```powershell
uv sync --frozen
uv run indoor-vt prepare
uv run indoor-vt acquire
uv run indoor-vt optimize
uv run indoor-vt view
```

Open **[the new local viewer](http://127.0.0.1:8770)**. Acquisition and optimization
can run in separate terminals: acquisition prepares the 0.5 m survey and query
positions first, then completes the remaining dense grid. Optimization waits
for its input chunks. Resume either command after interruption. Run only one
acquisition driver and one optimization driver for a given dataset/run directory.

The defaults are in [configs/redesign.json](configs/redesign.json). The acquisition
commands accept `--qd-root` and `--matlab`; defaults point to the sibling NIST
checkout and MATLAB R2026a. `--data`, `--run`, and `--config` select independent
datasets and experiments. Input changes require a fresh dataset directory;
search configuration changes require a fresh run directory.

The viewer uses the previous Results Explorer's layout and interactive 3D source
maps, connected exclusively to the redesign studies. Short method names identify
the algorithm; separate columns show parameter selection and the localization
pipeline's VT source. Peak unification appears as a tag where applicable.
Raw localization labels explain which
pipeline selected jointly tuned MPUrge-MAP parameters. The reconstruction
list includes MPUrge selected on OSPA and the saved MPUrge component selected on
full-pipeline localization error. Raw localization similarly includes the saved
joint initializer; these component rows summarize existing outcomes, not new
experiments. Each method has Overview, Map, Scientific method, Code & caching, Parameters, Search history,
and Evidence tabs. Search history includes every completed trial, with filtering,
pagination, CSV export, and inspection of that trial's own parameters and replay.
Method tabs link to the scientific implementation, execution and caching code.
Localization switches between 2D and interactive 3D, and checkbox lists control
the noise curves. The full peak-matching explorer uses the selected redesign
parameters and native observations, including truth overlays and match metrics.
See [the redesign explorer guide](docs/redesign_explorer.md).

Run or resume the finished Zayets/RANSAC study with
`uv run python scripts/run_additional.py`. It adds **Zayets + support-consumption
finisher**, **RANSAC + robust support finisher**, and their stage-wise/joint
MPUrge-MAP + SMART-LEA pipelines. Its 2,936 selection trials follow the same
noise-free selection and frozen-parameter evaluation protocol. The default
viewer discovers this sibling study and keeps each result's run provenance.

## Experiment order

1. Standard MPUrge reconstruction, independently optimized with size unification
   on and off.
2. MPUrge-MAP fingerprint localization.
3. SMART-LEA using true VT coordinates and the selected MPUrge-MAP initializer.
4. SMART-LEA using the frozen independently tuned source map and initializer.
5. Joint tuning of all three components for final localization error.
6. Remaining methods and observation tracks after the priority group.

Selection now uses **noise-free rectangle calibration data only**. Test frozen
choices on the declared noise levels and held-out query positions. Both workflows
run with unification on and off, and all matcher grids include half-windows 1-7.
Independent stages exhaust their configured grids; the joint search samples 512
pipelines per mode. There are 6,540 effective selection trials in total.

The default run is `runs/redesign/rectangle-clean-selection-v4`; the frozen dataset
remains `data/redesign/rectangle-20261025`. This is a same-room experiment. At zero
noise, duplicate survey noise repetitions are identical and evaluated once.

## Code layout

- [`science/`](src/indoor_vt/science): matchers, grouping, estimators, MPUrge,
  Zayets and RANSAC proposals, source selection, clustering, and localization.
- [`execution/`](src/indoor_vt/execution): instrumented orchestration, shared
  computation store, optimization, and the local evidence viewer.
- [`acquisition/`](src/indoor_vt/acquisition): room sampling, lattice subsampling,
  NIST acquisition, channel conversion, and evaluator-only source truth.

Numerical caching uses packed, computation-addressed NPZ batches with a 2 GiB
disk limit and 256 MiB RAM limit. Every trial keeps a compact scalar summary;
selected detailed outcomes are typed arrays. The viewer can recreate evicted
algorithm evidence and individual matching traces. The frozen native acquisition
is read from its existing store without adding numerical cache records to it.
See the [dependency and storage analysis](docs/computation_cache_design.md).

```powershell
uv run pytest -q
uv run ruff check src scripts tests
```

After a study completes, generate its report and verify saved outcomes:

```powershell
uv run python scripts/report_redesign.py
uv run python scripts/audit_redesign.py
```

The report includes all selected settings, noise-evaluation metrics, and plots.
The audit independently recomputes metrics from saved coordinates and checks
the data split, frozen upstream settings, complete trial/condition coverage,
runtime limit, and storage budgets.

These checks cover the new contracts, not equivalence with the old study.
The old campaign implementation, compatibility tests, and agglomerative
complete-link backend have been removed from this branch. The previous study
remains in Git history and existing `published/`, `data/`, and `runs/` artifacts;
it is not loaded by the new execution path. Earlier documents describing that
study are historical references. This branch does not publish to GitHub Pages.
