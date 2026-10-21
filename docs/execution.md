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

## Live development generation

Required native acquisitions: **352** (16 rooms x 2 APs x 11 acquisitions).
These contain 32,000 query-channel frames plus the realized survey frames.
Each acquisition produces ten direct conditions and three receiver conditions:
**4,576 acquisition/condition artifacts**. A+B reuses A and B.

The machine runs four native workers. A hidden finisher is attached to the
initial simulator driver; it starts a resume pass when that driver exits and
acquisitions remain. It also derives observations as native receipts arrive.
Once native generation is complete, it independently replays the five-box pilot
with its recorded seed and compares native bytes before the final archive audit.
The current authoritative state is
`runs/development/generation/status.json`, with logs alongside it. Final native
and observation validation is written to `data/development/archive_audit.json`.
`finished: true` requires all acquisitions, all observations and a successful
audit. An unexpected frontend/audit failure is recorded and requires diagnosis;
it is never converted into a successful completion.

To resume after a machine shutdown, run this once from the repository:

```powershell
uv run python -m indoor_vt.finish --namespace development --qd-root ../simulator/nist-qd-realization --matlab "C:/Program Files/MATLAB/R2026a/bin/matlab.exe" --workers 4
```

Do not run a second overlapping simulation command. When attaching to an
already running driver, add `--wait-pid <driver-pid>` so the finisher waits for
that driver instead of duplicating its jobs. The finisher's own namespace lock
prevents duplicate finisher instances. Processes cannot continue while the
computer is shut down or asleep.

## Still to execute

Full scientific calibration searches, shadow comparisons and frozen test comparisons have
not yet run. They require the complete verified observation dataset and may
take much longer than propagation. No pilot score is a publication result, and
the 2019 adapter must retain its assumption label in any comparison.

Both complete calibration/freeze/test code paths have passed a small synthetic
integration fixture. That fixture validates orchestration and data-change
rejection; it is not a scientific calibration campaign.
