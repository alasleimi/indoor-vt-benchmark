# Chamfer-kNN localization study

This is a new experiment on the current rectangle benchmark. Archived development-study Chamfer results are not reused.

## Method

For a query fingerprint with ranges $Q$ and a reference fingerprint with ranges $R$, within one AP,

$$
c(Q,R)=\frac12\left[\frac1{|Q|}\sum_{q\in Q}\min_{r\in R}|q-r|
+\frac1{|R|}\sum_{r\in R}\min_{q\in Q}|r-q|\right].
$$

Average these distances over AP A and AP B. Select the $k$ reference receivers with the smallest finite average distances $d_i$. For their known coordinates $x_i$, return

$$
u_0=\frac{\sum_i d_i^{-p}x_i}{\sum_i d_i^{-p}}.
$$

These are nearest-range distances, not one-to-one peak assignments. A range can be the closest counterpart of several other ranges. An empty fingerprint gives infinite distance. Selected distances at most float64 epsilon receive equal weights, with all positive-distance references receiving zero weight. If every reference has infinite distance, use the reference-position centroid.

The concrete dependency injection is `FingerprintLocalizer(scorer=ChamferScorer(), estimator=DistanceWeightedKNN(neighbors=k, power=p))`, in `src/indoor_vt/execution/chamfer_experiment.py`. There is no Chamfer pattern half-window, alpha, epsilon threshold, coverage penalty, or peak unification.

## Calibration and evaluation

The protocol exactly preserves `rectangle-clean-selection-v4`:

- Empty-room reference/VT survey: 280 positions, with 0.5 m horizontal spacing and the existing two survey heights.
- Calibration: noise-free queries, 60 positions in each of the 0-, 1-, 3- and 5-obstacle layouts, 240 total. The source reconstructor still receives the empty-room survey. Calling this an entirely empty-room calibration would be incorrect.
- Evaluation: the remaining 40 query positions per layout. Noise levels are `[0, .01, .02, .05, .1, .2, .5, 1, 2, 5]` m. Positive noise uses saved repetition IDs 2 and 3; the deterministic zero-noise case is evaluated once.
- Parameters freeze after calibration. Neither held-out queries nor noisy evaluation outcomes choose settings.
- Joint selection minimizes mean 3D localization error **after** SMART-LEA. The saved source, initial-coordinate and final-coordinate stages remain separately inspectable. Source OSPA/F1/matched-RMSE compare AP A and AP B independently before aggregation.

## Search

Standalone searches the Cartesian grid `neighbors=[1,3,5,10] × power=[.5,1,2]`: 12 recorded configurations. As with the existing kNN grid, power is inactive when k=1; those three rows have identical predictions.

Joint studies use three suppliers:

1. Standard MPUrge.
2. Zayets + robust range-refinement and peak-consumption finisher.
3. RANSAC + robust range-refinement and peak-consumption finisher.

The two extra families had the smallest existing MPUrge-MAP → SMART-LEA joint calibration errors: 0.0065902887 m and 0.0067743844 m respectively. Supplier choice does not use held-out outcomes. No oracle count or true source coordinates enter a reconstruction/localization algorithm.

Each supplier searches 512 unique joint configurations for unification on and 512 for off. Each configuration contains the supplier's full recorded grid, the 12 Chamfer settings, and the existing SMART-LEA grid (`half_window=[1..7] × alpha=[0,.25,.5,.7,1] × iterations=[1,2,4]`). At alpha=1 inactive half-window duplicates collapse. In the robust finisher, inactive clipping values collapse when the median penalty is zero. Exact supplier grids, seed and strategy appear in each viewer's Search history tab and each run's `config.json`.

The first configuration combines the independently selected Chamfer initializer with that supplier's previously calibration-selected joint source/SMART-LEA settings. Remaining configurations use seeded sampling without replacement from the full grid. `baseline_origins.json` records the source receipt IDs. Old receipts' implicit robust defaults `(smooth_weight=.35, median_penalty_weight=.1, median_clip_factor=4)` are made explicit in this new baseline.

Unification never affects Chamfer distances. It affects standard MPUrge reconstruction and SMART-LEA in the standard supplier study, and only SMART-LEA for the two other suppliers.

## Results and progress

Standalone completed on 2026-10-29, selecting **k=3, p=2**. Mean calibration error is **0.174094 m**. Held-out results pool all four obstacle layouts:

| Range noise σ (m) | Mean 3D error (m) | 3D RMSE (m) |
|---:|---:|---:|
| 0 | 0.162277 | 0.192109 |
| .01 | 0.162949 | 0.192191 |
| .02 | 0.159054 | 0.186858 |
| .05 | 0.166129 | 0.191991 |
| .1 | 0.195754 | 0.223147 |
| .2 | 0.284729 | 0.326725 |
| .5 | 0.588435 | 0.666401 |
| 1 | 1.057395 | 1.240201 |
| 2 | 1.866251 | 2.062289 |
| 5 | 2.575940 | 2.816421 |

All four studies are complete. The saved status files, selected receipts and retained outcomes were verified on 2026-11-02:

| Study | Completed calibration configurations | Selected configurations | Saved calibration/evaluation outcomes |
|---|---:|---:|---:|
| Standalone Chamfer-kNN | 12 | 1 | 11 |
| Chamfer-kNN + SMART-LEA, standard MPUrge supplier | 1,024 | 2 | 22 |
| Chamfer-kNN + SMART-LEA, Zayets + robust finisher supplier | 1,024 | 2 | 22 |
| Chamfer-kNN + SMART-LEA, RANSAC + robust finisher supplier | 1,024 | 2 | 22 |

Each joint study selected one configuration for each unification mode. The viewer exposes standalone Chamfer-kNN in Raw localization, each joint initializer in Raw localization, each completed pipeline in VT localization, and its saved source stage in VT reconstruction. Select the **Chamfer-kNN initializer** tag to find these entries. The source-stage rows use the parameters selected by final localization error; they are not separately optimized on OSPA.

The 77 saved outcomes are exported for the website without replaying reconstruction or repeating calibration. Their authoritative status, append-only completed trials, selected receipts and retained outcomes are under:

```
runs/redesign/rectangle-chamfer-standalone-v1/
runs/redesign/rectangle-chamfer-mpurge-v1/
runs/redesign/rectangle-chamfer-zayets_robust-v1/
runs/redesign/rectangle-chamfer-ransac-v1/
```

Each has `status.json`, `stdout.log`, `stderr.log`, `pid.txt`, `trials/`, and later `selected/`, `evaluations/`, `outcomes/`, `results.json`. Run or resume with `.venv/Scripts/python.exe scripts/run_chamfer.py --job <standalone|mpurge|zayets_robust|ransac>`. Completed trial identities and outcomes are reused. Existing studies are not recalibrated or invalidated.
