# PointNet 4M reconstruction and SMART-LEA

Six reconstruction variants use the **same frozen 4,000,000-update network**:
local connected components or local stars, each returning raw candidates or
using the support-first or robust range-refinement/peak-consumption finisher.
The raw variant returns every valid multilateration proposal; it has no
candidate clustering, relative-support filtering, or finisher.

For fingerprints A and B, Hungarian assignment maximizes the sum of logits
`log(p / (1 - p))`, with exactly `min(len(A), len(B))` matches. There is no
probability threshold, unmatched dummy choice, crossing veto, or peak-count
unification. APs are processed separately. Grouping uses `1 - p` as the edge
dissimilarity; coordinate estimation uses ordinary, unweighted range least
squares. Components keep at most one peak per receiver using highest degree,
then lowest mean incident dissimilarity, then lowest original index. Stars
retain only a center and its immediate neighbors within a local scope.

The fixed checkpoint was trained on archived R1/R2/L rooms with noise and
obstacles, excluding the held-out benchmark rectangle. Its SHA-256 is
`acf6d4d7a50219e69b3e9cf5f609dcb7387f685c0e9b0bc27f78e7656bab07a5`.
This campaign does not train the network. Previously recorded 4M logits are
reused exactly for covered receiver pairs; new pairs needed for larger scopes
are inferred from the same checkpoint. Matching parity was verified across
all 46 archived AP/conditions; both groupings also reproduced the saved robust
reconstructions on a noisy empty-room draw and a three-obstacle layout.

## Selection protocol

| Stage | Search | Objective |
|---|---|---|
| Raw reconstruction, each grouping | 4 neighborhood sizes | Empty noise-free survey OSPA, c=1 m, p=1 |
| Support-first reconstruction, each grouping | Exhaustive 108 settings | Same OSPA objective |
| Robust reconstruction, each grouping | 512 sampled distinct settings from 2,268 effective settings | Same OSPA objective |
| Stagewise SMART-LEA, each source variant and MAP/LEA unification mode | Exhaustive 87 effective settings | Mean 3D calibration query error |
| Joint source + MPUrge-MAP + SMART-LEA, each variant and mode | 512 distinct settings, including the independently selected stagewise baseline | Mean 3D calibration query error |

Total: **8,436 calibration trials**. The source neighborhood grid is
`[5, 10, 20, 40]`, including each center and all distance ties. Finisher grids
are tolerance `[0.1, 0.18, 0.3]` m, support fraction `[0.04, 0.08, 0.16]`, and
duplicate range-curve RMS `[0.06, 0.12, 0.24]` m. Robust variants additionally
search smooth weight `[0, 0.35, 0.7]`, median penalty `[0, 0.1, 0.2]`, and
median clipping factor `[2, 4, 8]`. Inactive clipping choices collapse when
the median penalty is zero. The network has no MPUrge alpha/window parameters.

MPUrge-MAP and SMART-LEA retain the existing benchmark grids. Stagewise
searches freeze the OSPA-selected source and independently selected MAP
initializer. Joint searches tune all three stages together. Pipeline peak
unification applies only to MPUrge-MAP and SMART-LEA, never to PointNet.
The viewer shows the actual grids, seeds, completed trials, frozen settings,
and concrete dependency-injected construction for each selection.

Selection uses the empty noise-free survey and **60 noise-free calibration
queries per layout** (240 total across 0/1/3/5 obstacles). Evaluation freezes
the selected parameters and uses the other **40 queries per layout**. The
reference/VT survey for localization stays the empty-room survey, as in the
existing benchmark. No evaluation noise or query error selects parameters.

## Evaluation and publication

Empty-room reconstruction and localization cover sigma
`[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]` m, with one zero-noise
realization and two saved draws per positive level. Obstacle reconstruction
additionally uses the same 270 positions outside every saved obstacle layout,
at zero noise, without recalibration. Its truth catalog is the existing union
of visible direct and reflected sources per layout/AP.

OSPA, F1 at 0.1/0.3/0.5 m, and matched RMSE at 0.5 m always match within each
AP before aggregation. Saved coordinates supply the viewer maps and metrics.
The joint pipeline also exposes its retained source map and MAP initialization
as intermediate rows, as other benchmark pipelines do.

Run/resume with the benchmark environment and the existing PyTorch interpreter:

```powershell
.venv/Scripts/python.exe scripts/run_pointnet4m_viewer.py --workers 2 --torch-python PATH_TO_PYTORCH_PYTHON
.venv/Scripts/python.exe scripts/audit_pointnet4m_viewer.py
.venv/Scripts/python.exe scripts/finish_pointnet4m_viewer.py --wait
```

The last command audits all selections and saved outcomes, evaluates the
72 frozen obstacle conditions, and exports/builds the viewer. Publication is
an explicit `--publish --expected-head COMMIT` option and verifies the
`alasleimi` account, reviewed commit, deployment, and live method count.
Campaign and publication progress are stored in
`runs/redesign/pointnet4m-viewer-status.json` and
`runs/redesign/pointnet4m-publication-status.json`.
