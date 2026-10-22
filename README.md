# Indoor VT benchmark

Local, reproducible NIST data generation and a modular delay-only localization
pipeline. The planning document remains in the parent workspace at
`../DATASET_PIPELINE_PLAN.md`. No remote repository is configured.

The dataset contains four room families, two calibration and two test instances
per family, two AP placements, two survey layers on a 0.5 m XY grid, and ten
query layouts with 100 positions each. A, B and A+B reuse the same channel
acquisitions. Native delays, gains, phase, AoA/AoD and path geometry are retained.
The primary estimator view exposes only ranges, AP labels and survey positions.

## Run locally

Use Python 3.11 and the committed `uv.lock`. The tested native toolchain is MATLAB
R2026a Update 4 and NIST Q-D revision
`1ccf0cb61c1741cc3471db8ec0373c5383da58be`. The NIST checkout is an external local
dependency; its source tree and local compatibility overlays are hashed in each
simulation receipt. The overlays do not modify that checkout.

```powershell
uv sync --frozen
uv run indoor-vt design --namespace development
uv run indoor-vt simulate --namespace development --qd-root ../simulator/nist-qd-realization --workers 4
uv run indoor-vt truth --namespace development
uv run indoor-vt observations --namespace development
uv run indoor-vt verify --namespace development
uv run pytest -q
```

Generation resumes from verified receipts. Do not start two simulation commands
for overlapping scenarios at the same time. Every attempt has its own inputs,
effective MATLAB settings, log and output directory. Failed attempts remain
available for diagnosis. Completed native channels have deterministic gzip
headers and their original JSON bytes are recoverable exactly.

For a single background job that resumes native failures, derives completed
channels and performs the final audit, use `python -m indoor_vt.finish` as
documented in [the execution record](docs/execution.md). It keeps up to four
native workers active while acquisitions remain. Its status file reports actual
progress; it does not mark the dataset finished before verification.

The separate `pilot` namespace permits `simulate --frames 12` and never changes
the development or publication geometries. The publication seed is reserved;
do not use it for development debugging.

## Calibration and evaluation

```powershell
uv run indoor-vt calibrate --track direct --regime component --initial mpurge_map --vt mpurge_local
uv run indoor-vt calibrate --track receiver --regime end_to_end --initial mpurge_map --vt mpurge_local
uv run indoor-vt evaluate --frozen configs/frozen/development/direct/component/mpurge_map__mpurge_local.json
```

`component` tunes coarse localization on its own 3D error and VT reconstruction
on OSPA, freezes both, then tunes LEA. `end_to_end` selects the whole named
pipeline by final 3D error. Settings are shared across all calibration rooms,
noise levels, layouts and AP subsets. Component-mode upstream settings are also
reused across pipeline partners. LEA settings belong to the named pairing.
Every trial uses the complete declared calibration set with uniform averaging.
These full campaigns can take substantially longer than dataset generation.

The comparison campaign is **stopped by user request** for the memory fix.
Revision v6 is prepared with the 108 completed presets preserved; restart only
when requested, using `uv run python scripts/run_revision_campaign.py --revision v6 --workers 1`.
It uses fresh workers per task, enforced memory quotas, bounded complete-link
row caching and streamed audit encoding. The same scientific queue and search
budgets remain. Main calibration and freezes precede shadow work; all selection
precedes held-out testing. Old revisions must remain stopped.
See the [memory fix and validation](docs/memory_safety.md),
[execution record](docs/execution.md) and [cache performance record](docs/performance.md).

Test evaluation requires a calibration freeze and checks the code, data,
observation manifests and reference-source hashes. A missing acquisition is an
error, not a reason to silently evaluate fewer cases. Failed LEA calls return
the initial prediction and remain in the metrics.

The registered initial methods are `mpurge_map`, `mpurge`, `mca`, `chamfer`.
Reconstructors are the local/full versions of `mpurge`, `mpurge_star`,
`mpurge_triangle`, `mpurge_diffassign`, plus `pba_local`, `ransac`, `zayets2018`
and `zayets2019_assumption`. The final name explicitly identifies a disclosed
interpretation of an underspecified published method; it is not author code.

## Artifacts and extension points

- `configs/`: geometry, seed registry, conventions and search protocol.
- `data/<namespace>/design.json`: realized rooms, APs, objects and sample positions.
- `data/<namespace>/scenarios/<room>/<AP>/<survey-or-layout>/`: immutable native
  acquisitions and versioned frontend outputs. The frontend keeps older
  versions when its implementation/configuration changes.
- `data/<namespace>/evaluation/`: evaluator-only source coordinates and review plot.
- `runs/`: fit caches, raw graph/group/candidate diagnostics, calibration trials,
  predictions and execution reports. Data and runs are Git-ignored.

[`contracts.py`](src/indoor_vt/contracts.py) defines the estimator-facing types.
[`Pipeline`](src/indoor_vt/pipeline.py) receives its initial localizer,
reconstructor and refiner by injection. MPUrge additionally accepts independent
scope, matcher, graph/group builder, fitter, merger, filter and refinement
components. No estimator imports the geometry/truth/native-feature modules.

The graph experiments remain outside this repo in
`../experiments/indoor_vt_shadow`. Their runner requires a complete calibration
comparison and at least a 5% relative accuracy improvement before declaring
eligibility. Smoke checks alone do not promote code into the primary roster.

See [method definitions](docs/methods.md), [source audit](docs/provenance.md),
and [execution status](docs/execution.md). The presence of runnable calibration
commands does not mean the publication comparison has already been completed.
