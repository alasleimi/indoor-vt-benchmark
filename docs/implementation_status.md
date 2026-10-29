# Redesign implementation status

Updated 2026-10-25. Branch: `redesign/components-and-experiments`.
The governing brief is [redesign.md](redesign.md).

## Implemented

- Readable scientific orchestration and separate execution for MPUrge, MAP and
  SMART-LEA. Injected components remain ordinary scientific objects.
- Packed pair blocks and shared compiled DSU. Local grouping reads existing
  edge blocks directly; conflicts use degree, mean cost, then peak index.
- Deterministic 3D neighborhoods, including boundary ties and the center.
  All scopes at the four configured sizes have full 3D rank on this rectangle.
- Faster pattern-distance calculation through sorted differences and weighted
  gaps. Alpha=1 skips the unused pattern calculation.
- Bounded computation-addressed NPZ cache, fixed-shard OS locks, atomic writes,
  incremental quota inventory and bounded RAM. No per-pair/peak SQL records,
  heartbeat threads, or full-directory scan per batch.
- Pair-bank reuse across neighborhood sizes; candidate reuse across radius and
  beta; persistent cluster batches; MAP statistic reuse across k/power;
  SMART-LEA iteration-prefix reuse.
- Noise-free selection, independently frozen upstream stages, joint pipeline
  tuning, both unification modes, and a true-VT SMART-LEA oracle.
- Compact stage trial logs, selected typed outcomes, and on-demand replay.
  Viewer comparisons distinguish calibration from fixed-parameter noise tests,
  show nested/frozen settings, and expose paginated numerical evidence and
  individual pair-matching traces.
- Native acquisition is read-only during these experiments. Direct observation
  conversion uses bounded RAM; receiver repetitions remain distinct even at
  0 dB, which is not the same as noise-free Direct NIST.
- Windows status publication retries transient reader/antivirus locks.
- Existing Zayets/RANSAC scientific components and external selectors remain;
  their additional execution profiles follow the priority experiment.

## Dataset and completed experiment

Configuration: [configs/redesign.json](../configs/redesign.json).
Dataset: `data/redesign/rectangle-20261025`.
Final-build run: `runs/redesign/rectangle-clean-selection-v4`.

The room is approximately 7.122 m by 5.167 m. The dense 0.1 m acquisition has
7,384 positions at two heights; the 0.5 m view has 280. All 68 native NIST chunks
completed. The clean empty-room survey has 25 paths per position, arising from
the configured ray tracing; 25 is not a manually imposed peak limit.

The final build completed all **6,540 selection configurations**, followed by
100 fixed-parameter noise evaluations and 10 retained calibration conditions.
Independent stage grids are exhaustive within the configured finite values;
joint tuning samples 512 pipelines per mode. All matcher windows 1-7 are covered
when active. The worker exited successfully.

The earlier `rectangle-20261025` optimization stopped at 200/1,056 trials and is
historical. The first batch-cache search, `rectangle-clean-selection-v3`, was
superseded when the shared pattern-cost kernel was accelerated. It is not the
final-build result set. No old-study equivalence or migration tests were added.

Calibration uses the first 60 of each layout's 100 query positions at zero noise.
Evaluation uses the remaining 40 and noise repetitions 2 and 3. Clean direct
repetitions are identical and evaluated once. All noise tests retain the
zero-noise choices; neither noisy nor held-out outcomes select parameters.
These are same-room results, not unseen-room generalization.
Noise perturbs both survey and query ranges; source maps and initializations
are recomputed with the frozen parameters.

## Selected results

Mean held-out 3D localization error in meters:

| Entry | Clean, unification on | Clean, off | 0.1 m noise, on | 0.1 m noise, off |
|---|---:|---:|---:|---:|
| MPUrge-MAP | 0.1266 | 0.1450 | 0.1609 | 0.1805 |
| SMART-LEA, true VTs | 0.0074 | 0.0085 | 0.0635 | 0.0794 |
| SMART-LEA, frozen independent stages | 0.0826 | 0.0854 | 0.8834 | 0.3116 |
| Joint pipeline | 0.0636 | 0.0854 | 0.1577 | 0.3116 |

The independently selected MPUrge uses half-window 2, alpha 0.7, neighborhood
size 5 (including boundary ties), cluster radius 0.1 m and beta 2 in both modes.
Its clean OSPA is 0.3584 m. The clean-selected reconstruction is fragile to noise:
OSPA reaches 0.8673 m at 0.01 m noise and 0.9248 m at 0.1 m. At 1 m and above,
it retains no sources, and the stage-wise SMART-LEA results fall back entirely
to their MAP initializations. Equal high-noise errors therefore do not mean
SMART-LEA refinement succeeded.

Joint tuning with unification selects half-window 5, alpha 0.25 and radius 0.5 m
for MPUrge, retaining size 5 and beta 2. Its MAP uses half-window 3, alpha 0.25,
k=5 and power 2; SMART-LEA uses half-window 1, alpha 0.7 and two iterations.
The unification-off joint search retains the stage-wise baseline. These are
finite-search results, not claims of global optimality or noise robustness.

Full metrics, selected/frozen settings and plots are in the run's `summary.md`,
`analysis.json`, `window-selection.png` and `noise-evaluation.png`.
`reconstruction-noise-diagnostics.json` records the retained source counts and
support thresholds across noise levels. Regenerate the report with
`scripts/report_redesign.py`.
The run also retains `source-snapshot.zip` and its SHA-256 manifest, containing
the exact package, viewer, run/report/audit scripts, configuration and dependency
lockfile. These preserve the implementation needed for replay after future edits.

## Measured performance and storage

The original pure pipeline spent about 11.9 seconds rebuilding and traversing
scope graphs. Direct packed DSU reduced a comparable 40-neighbor grouping pass
to about 0.31 seconds (actual scopes now include boundary ties).

The final 16-case runtime matrix varied 5/40-neighbor scopes, alpha 0/0.7,
unification on/off, and noise 0/5 m, using half-window 7. Complete executions,
including cold calculations and cache writes, all stayed below **7.67 seconds**.
Warm repeated executions were roughly 4-15 ms in this check. Run:

```powershell
.venv/Scripts/python.exe scripts/benchmark_redesign.py
```

The runtime report is
`runs/redesign/runtime-investigation/final-runtime.json`.
The experiment also enforces the 20-second budget on every standard MPUrge fit.
After all 4,640 independent MPUrge selection trials, its recorded maximum was
9.82 seconds; this includes configurations outside the 16-case stress matrix.
Across the entire completed search and noise evaluation, the maximum was
**13.065 seconds**. Selected MPUrge noise evaluations stayed below 4.98 seconds.
These measurements concern the current 280-position rectangle and configured
parameters, not arbitrary survey sizes.

A deliberately large source map (2,769 A sources and 2,784 B sources) exposed
SMART-LEA's asymmetric matching cost. Ten-query measurements with half-window 7
and four iterations improved from 0.888 to 0.148 seconds/query without unification.
This is a stress benchmark, not an accuracy result or typical selected-map size.

The numerical cache is limited to 2 GiB on disk and 256 MiB in RAM. Outcomes and
viewer replay have separate 256/128 MiB caps; writes preserve 5 GiB free space.
At completion the cache held 2,147,381,980 bytes of NPZ data (below 2 GiB),
plus 2,053,736 bytes of inventory metadata. All 6,540 trial logs used
2,436,765 bytes; the 110 detailed outcomes used 10,305,498 bytes. Together those
retained logs and outcomes are **12.15 MiB**. Viewer replay used 233,816 bytes.
The previous 15.68 GiB SQLite database is retained because it contains native
acquisition. New numerical trials do not add records to it. Automatic approval
review blocked removal of the temporary diagnostic
`runtime-investigation/batch-cache`; it remains separate from the production cache.

See [the dependency/storage analysis](computation_cache_design.md) for the
measured codec tradeoff and the reason for each cache boundary.

## Validation

- 24 focused scientific/storage/search checks pass; lint and JavaScript syntax
  checks pass.
- Complete smoke runs exercise all five entries, both unification modes,
  calibration selection and held-out noise testing. The final smoke is
  `runs/redesign/clean-selection-smoke-final`.
- Viewer HTTP overview/replay, map data, match tracing, pagination and JSON
  serialization have been checked against actual runs. A saved noisy result loaded
  in 0.33 s; a non-persisted intermediate was recreated in 4.08 s. Browser visual
  inspection was unavailable after launch rejection by automatic review; the
  exported scientific plots were inspected.
- `scripts/audit_redesign.py` passed on the completed run. It independently
  checked 26,240 saved query predictions, recomputed source/localization metrics,
  verified all 6,540 trial identities and 110 conditions, checked frozen choices
  and calibration-loss reproduction, and reconciled the cache inventory with
  actual files. See `completion-audit.json` for exact counts and implementation IDs.
- New contracts are tested directly. There is no historical-output comparison,
  cache migration test suite, or old-study API adapter.

## Subsequent experiments

The priority MPUrge/MAP/SMART-LEA optimization and noise evaluation are complete.
Zayets/RANSAC experiments and additional room/resolution/receiver profiles remain
subsequent work. The current measurements do not establish accuracy on those
profiles.
