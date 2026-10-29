# VT benchmark redesign

Current brief, updated 2026-10-25. Read this first when resuming work. The latest
decisions here supersede earlier study queues and tuning policies.

Implementation is on `redesign/components-and-experiments`.
[Implementation status](implementation_status.md) records measurements and the
completed priority study: 6,540 selections and 100 noise evaluations. All 68
native acquisition chunks are complete.

## Detailed designs

| Document | Owns |
|---|---|
| [Scientific components and APIs](mpurge_api_design.md) | MPUrge, Zayets, RANSAC, interchangeable grouping/selection, localization and dependency injection |
| [Computation identity, caching and auditing](computation_cache_design.md) | Parameter dependencies, packed storage, bounded reuse and audit replay |
| [Dataset generation and evaluation](dataset_generation_design.md) | Room/AP generation, dense survey subsampling, observation tracks and true-VT SMART-LEA |

## Implementation policy

Build clear scientific components with separate readable and instrumented
orchestration. Reuse clean helpers; do not preserve old APIs, numerical quirks,
study outputs, or caches through adapters. Do not write migration or old-output
equivalence tests. Validate the new contracts and an end-to-end smoke run.

Keep existing historical artifacts where they are. Remove agglomerative
complete linkage; retain greedy complete-link and DBSCAN. Do not spend study
resources on remaining methods before the priority experiments.

The current runtime requirement is **under 20 seconds for a complete standard
MPUrge reconstruction on the 280-position rectangle and configured search
space**, including numerical cache writes. Measure whole runs, not only the
DSU kernel. A runtime violation stops the experiment for investigation.

## Dataset and neighborhoods

Use one seeded random rectangular room, without global rotation, with at least
5 cm AP-to-wall clearance. Acquire the survey at 0.1 m horizontal spacing and
select the default 0.5 m survey by lattice indices. Keep both height layers.
Initially use Direct NIST; retain native fields and interfaces for later
SAPP-style observations and multiple rooms.

Nearest neighborhoods use **3D Euclidean distance**, count the center, and
include all positions tied at the requested boundary. Thus size=40 can produce
40, 42 or 43 positions in this room. This avoids arbitrary grid-direction choices
and gives nested scopes. Exact-size, index-tie-broken neighborhoods remain an
explicit option. Match every pair within each scope; group each scope separately.
The pair-selector and group-scope-selector remain independently injectable.

## Selection now uses noise-free data only

The former per-noise and average-noise tuning plan is superseded for this
experiment. Select at noise level **0**, on calibration queries in the rectangle.
Freeze choices, then evaluate them at all declared noise levels using held-out
query positions and evaluation noise repetitions. Do not use noisy or held-out
results to select parameters.

At each test level, perturb both survey and query ranges. Recompute source maps
and fingerprint initializations with the selected fixed settings. Freezing the
upstream stages means freezing their parameters, not reusing clean numerical
outputs across noisy conditions.

The survey and room are shared with calibration: this measures same-room
performance, not generalization to unseen rooms. At zero noise, duplicate
direct-observation repetitions are identical and evaluated once.

Every matcher search includes half-windows **1 through 7**. Exhaust the configured
finite grids for independent stages; collapse half-window duplicates when
alpha=1 because the pattern term is inactive.

## Stage-wise and joint workflows

Run both workflows with size unification fixed **on** and fixed **off** across
their components. Each mode gets its own selection; equal computations can
still share cached batches.

1. Tune standard MPUrge for mean VT **OSPA** over AP A, AP B, and A+B.
2. Independently tune MPUrge-MAP for mean **3D localization error**.
3. Retain the priority oracle: tune SMART-LEA using ground-truth VTs and the
   frozen selected MPUrge-MAP initializer.
4. Tune SMART-LEA using the frozen OSPA-selected MPUrge map and frozen
   localization-selected MPUrge-MAP initializer.
5. Jointly tune all three components for final SMART-LEA localization error.
   Start from the stage-wise winner and use 512 reproducibly sampled pipeline
   configurations per unification mode. This is a joint search budget, not an
   exhaustive Cartesian search or a claim of the global optimum.

The configured independent grids have 2,320 effective MPUrge settings, 264 MAP
settings, and 87 SMART-LEA settings per unification mode. Including the oracle
and 512 joint configurations gives **6,540 selection trials**. All half-windows
are covered where active. Budget and spaces are explicit in
[redesign.json](../configs/redesign.json).

The true-VT oracle receives source coordinates and AP identities, never true
query coordinates or true peak associations. Report F1 and localization
RMSE/median/p90/fallback rates alongside the selection objectives.

## Storage and viewer

Use a bounded computation-addressed NPZ cache, not millions of tiny database
objects. Save every trial summary in compact stage logs and selected detailed
outcomes in typed files. Recreate other evidence when requested. The
[cache design](computation_cache_design.md) specifies dependencies, measured
codec costs, quotas, concurrency, and scientific/execution ownership.

The viewer compares methods in this room, distinguishes noise-free selection
from fixed-parameter noise evaluation, exposes frozen versus joint parameters,
and can inspect/replay a completed trial. Remaining Zayets/RANSAC experiments,
other room/resolution profiles, and additional receiver tracks come afterward.
