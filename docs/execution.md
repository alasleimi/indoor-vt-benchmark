# Execution record

This is a local-only repository. The original plan remains at
`../../DATASET_PIPELINE_PLAN.md`; it was not moved into this repository.
The publication namespace remains unused.

## Active execution revision v4

The [external plan](../../DATASET_PIPELINE_PLAN.md) now runs initial
published/disclosed presets before deeper main calibration. Greedy median and
mean, agglomerative complete-link and DBSCAN are interchangeable MPUrge mergers.
All shadow work is delayed until every main hyperparameter search and freeze
finishes. Future candidates exclude p=0 together with alpha=0; original trials
remain intact. Other audited fixed policies remain fixed. No dataset
regeneration, observation reduction or search-budget reduction is involved.

The transition preserves the original source snapshot and 27 completed
MPUrge-MAP trials (25 receiver, 2 direct), plus six completed interim presets.
Partial per-case caches remain reusable. Compatibility receipts link old and
new code, data and artifact hashes; old files are not relabelled in place.

Revision v3 adds exact nearest-delay geometry reuse, cached grouping, lightweight
source-coordinate summaries and LEA iteration-prefix reuse. The v2 source and
campaign are preserved. Completed v2 objective receipts reference their original
artifacts; compatible per-case and intermediate caches are reused explicitly.
The scientific protocol and 450-job schedule are unchanged. See the
[performance record](performance.md) and [validation receipt](cache_optimization_validation.json).

At 21:40 Europe/Berlin on 2026-10-21, the v3 runner stopped with `MemoryError`
while checking source hashes. Both workers exited; the last status still listed
their interrupted jobs as running. The campaign had completed 92 of 110 initial
presets: all eight coarse presets, 54 of 72 VT presets and all 30 pipeline presets.
A recent direct star/local agglomerative case contained 39,742 candidates and
required 6.32 GB for its condensed distance array alone; concurrent worker memory
is consistent with the failure, but the traceback does not identify its allocator.

Revision v4 resumed at approximately 22:45 Europe/Berlin with one worker to reduce
concurrent memory pressure. Estimator code, runner code, jobs, parameters and search
budgets are identical to v3. All 92 completed jobs and 94 objective receipts are
retained; unchanged full objective artifacts remain in their original directories.
The migration and source identities are recorded in
`runs/development/revisions/v4/migration.json`. This is a concurrency reduction,
not a claim that the largest individual clustering job has a proven memory bound.

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

Full scientific calibration started on 2026-10-21. Revision v4 runs one
persistent worker with one numerical-library thread. During the code
transition two useful preset workers continued on an immutable old source
snapshot. The revised workers use verified caches and preserve all original
artifacts. This is ongoing calibration, not a completed benchmark.

The comparison matrix swaps each of the twelve VT entries into MPUrge-MAP plus
SMART-LEA, and swaps the other three initial localizers into MPUrge-local plus
SMART-LEA. This gives fifteen distinct pairings. Each runs in two tracks and two
regimes: **60 primary pipeline freezes and 60 frozen test evaluations**. Shared
component-mode upstream searches are reused across pairings. There are 92
distinct main searches: 8 coarse, 24 VT, 30 component LEA and 30 end-to-end.

The eight shadow methods run in both spatial scopes, tracks and regimes:
**64 calibration comparisons** against the canonical calibrated MPUrge result.
Only candidates passing the 5% gate receive a final frozen pipeline and test
evaluation. The queue has 64 corresponding conditional evaluation jobs; a
nonqualifier records a skip without loading test observations. Qualifying code
remains external through the campaign so registry changes cannot invalidate
existing freezes. Importing eligible implementations follows the campaign audit.

The revised graph has **450 jobs**, recorded in
`runs/development/revisions/v4/specification.json`:

1. 110 initial main presets: 8 coarse, 72 VT including merger alternatives,
   and 30 pipeline presets. The default merger of every VT method is covered
   before its extra merger alternatives receive priority.
2. 92 main searches and 60 pipeline freezes. Each eligible search receives at
   most one new configuration per scheduling turn. Component LEA waits for its
   upstream coarse and VT freezes. Search budgets remain 32 broad draws, up to
   two expansions of 8 draws plus boundary candidates, and 16 refinement draws.
3. 64 shadow selections, including their own presets; none starts earlier.
4. 60 primary test evaluations and 64 conditional shadow test evaluations.

The original 248-job specification/status remains in `runs/development/campaign`.
It and revisions v2/v3 are historical and must not run concurrently with v4. The runner locks
its revision, checks source and shadow identities, retries an error once, and
blocks later phases after unresolved failures. If the parent exits, workers
stop too; restart resumes saved objectives and searches. Interrupted in-flight
case computations may repeat, while completed artifacts remain available.

Live progress and completed results are in
`runs/development/revisions/v4/report.md` and `status.json`. The runner updates
them automatically. Completed candidate counts are distinct from completed
pipeline jobs. Full searches can take much longer than propagation.

```powershell
uv run python scripts/run_revision_campaign.py --revision v4 --workers 1
```

This command resumes the recorded campaign. Do not edit estimator settings or
shadow implementations during it: the queue checks those hashes and stops
launching new work if they change. No pilot score is a publication result, and
the 2019 adapter retains its assumption label in every comparison.

Both complete calibration/freeze/test code paths have passed a small synthetic
integration fixture. That fixture validates orchestration and data-change
rejection; it is not a scientific calibration campaign.

## Cache changes and validation

Per-AP query/reference score matrices are shared across A, B, A+B and k values
for all four initial localizers. Matrix computation retains the original
floating-point scoring and averaging order. Reconstruction caches matched
graphs separately from grouping and raw fitted candidates; merger/radius/beta
and applicable DiffAssign changes reuse their upstream work. Greedy mean and
median share memberships; agglomerative cuts share a hierarchy. LEA caches
predictions by observed inputs, initial predictions, source coordinates and
LEA parameters. Loading/hash checks are memoized, with a bounded query cache.

Custom injected matchers/fitting implementations bypass intermediate caches
unless their identities are explicitly covered. Atomic writes use separate
process temporary files. Standard complete-link clustering retains quadratic
distance storage; it does not impose a source cap or approximate clustering.

Validation: **53 tests passed**, Ruff clean, and an old/new implementation audit
matched **288 coarse predictions exactly** across both observation tracks and
all four localizers. All **12 VT methods** matched exactly on a controlled
multipath fixture (consensus trials reduced only for that verification fixture).
Tests also cover cache reuse across A/B/A+B and k, historical cache imports,
cluster memberships, refinement reuse, conditional DBSCAN parameters, resumed
searches, stage dependencies and the main-before-shadow barrier.
The v3 checks also matched 1,144,000 MCA/Chamfer scores exactly on full calibration
cases and cover LEA prefix reuse, later-iteration failure, unbounded peak counts,
grouping reuse and source-summary invalidation.

Original timings remain original timings. Cached score receipts record matrix
compute time separately; new per-query coarse timings cover ranking only.
Warm-cache scheduling time is not a cold end-to-end runtime benchmark. No full
campaign speedup factor is claimed from the small equivalence/performance checks.
