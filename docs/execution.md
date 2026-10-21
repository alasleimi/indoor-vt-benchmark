# Execution record

This is a local-only repository. The original plan remains at
`../../DATASET_PIPELINE_PLAN.md`; it was not moved into this repository.
The publication namespace remains unused.

## Implemented

- Frozen scene sampler, semantic seeds, native NIST runner and full rich-channel
  archive; versioned direct and SAPP-receiver observation tracks.
- Signed 3D source catalogue, review plot, native geometry audit and observation
  integrity checks.
- Four initial localizers, twelve VT entries, reconstructed-anchor SMART-LEA,
  both calibration regimes, frozen test evaluation and injectable components.
- Separate shadow workspace with eight grouping prototypes and a full-calibration
  5% promotion gate. None has been promoted.
- Persistent generation finisher, with one namespace lock, native retry logs,
  incremental frontend derivation and final integrity audit.

## Validation performed

The pilot uses a separately seeded T room: a full 256-position two-layer survey
and twelve queries from its five-box layout, AP A. Native generation, all
thirteen frontend conditions, every primary reconstructor and all four initial
localizers plus LEA have run on pilot data. These are functional checks, not
calibrated rankings. Empty maps and fallback cases remain recorded.

Tests cover geometry/clearances, stable seeds, native field preservation,
SAPP receiver and matcher reference fixtures, signed 3D fitting, full-room pair
coverage, association conflicts, DiffAssign's analytic gradient/fixed count,
OSPA/F1, interrupted calibration replay, estimator information isolation and
versioned frontend replay. See the recorded validation report for the current
test count and source hash.

## Completed development generation

Required native acquisitions: **352** (16 rooms x 2 APs x 11 acquisitions).
These contain 32,000 query-channel frames and 8,888 survey-channel frames:
**40,888 native measurements** in total.
Each acquisition produces ten direct conditions and three receiver conditions:
**4,576 acquisition/condition artifacts**. A+B reuses A and B.

All 352 acquisitions and all 4,576 condition artifacts are complete. The final
audit found no missing acquisitions or observations. All 32 survey channels
passed the reference-source geometry check; maximum native image discrepancy
was 0.001204 m. An independent five-box pilot replay reproduced the exact native
JSON hash with the same simulator. The NIST workers stopped after completion.

`runs/development/generation/status.json` records `finished: true` and the replay
hashes. `data/development/archive_audit.json` records the complete archive audit.

To resume after a machine shutdown, run this once from the repository:

```powershell
uv run python -m indoor_vt.finish --namespace development --qd-root ../simulator/nist-qd-realization --matlab "C:/Program Files/MATLAB/R2026a/bin/matlab.exe" --workers 4
```

Do not run a second overlapping simulation command. When attaching to an
already running driver, add `--wait-pid <driver-pid>` so the finisher waits for
that driver instead of duplicating its jobs. The finisher's own namespace lock
prevents duplicate finisher instances. Processes cannot continue while the
computer is shut down or asleep.

## Calibration and evaluation campaign

Full scientific calibration started on 2026-10-21. The persistent queue runs
two jobs concurrently, with one numerical-library thread per newly launched
worker. It attached to the first receiver job already in progress. No test
evaluation runs until all primary calibration and shadow selections finish.

The comparison matrix swaps each of the twelve VT entries into MPUrge-MAP plus
SMART-LEA, and swaps the other three initial localizers into MPUrge-local plus
SMART-LEA. This gives fifteen distinct pairings. Each runs in two tracks and two
regimes: **60 primary calibrations and 60 frozen test evaluations**. Shared
component-mode upstream searches are reused across pairings.

The eight shadow methods run in both spatial scopes, tracks and regimes:
**64 calibration comparisons** against the canonical calibrated MPUrge result.
Only candidates passing the 5% gate receive a final frozen pipeline and test
evaluation. The queue has 64 corresponding conditional evaluation jobs; a
nonqualifier records a skip without loading test observations. Qualifying code
remains external through the campaign so registry changes cannot invalidate
existing freezes. Importing eligible implementations follows the campaign audit.

All **248 queue jobs** are recorded in
`runs/development/campaign/specification.json`. The queue records its own source
hash and external shadow hashes separately from the unchanged estimator hash.
It locks shared searches, retries a failed process once, and blocks later phases
if a failure remains unresolved. It can reattach to surviving child processes
using PID plus creation identity after a runner interruption.

Live progress and completed results are in
`runs/development/campaign/report.md` and `report.json`. The detailed status and
per-job logs are alongside them. Completed candidate counts are distinct from
completed pipeline jobs. Full searches can take much longer than propagation.

```powershell
uv run python scripts/run_campaign.py --workers 2
uv run python scripts/report_campaign.py --watch
```

These commands resume the recorded campaign. Do not edit estimator settings or
shadow implementations during it: the queue checks those hashes and stops
launching new work if they change. No pilot score is a publication result, and
the 2019 adapter retains its assumption label in every comparison.

Both complete calibration/freeze/test code paths have passed a small synthetic
integration fixture. That fixture validates orchestration and data-change
rejection; it is not a scientific calibration campaign.
