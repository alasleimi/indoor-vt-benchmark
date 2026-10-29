# Dataset generation, survey resolution, and evaluation

[Start with the redesign brief](redesign.md) for the implementation policy,
first experiment priorities, noise-free selection, and the stage-wise/joint workflows.

Design requirements agreed on 2026-10-24, updated on 2026-10-25. The initial new
acquisition is complete; see [implementation status](implementation_status.md).
Historical datasets remain unchanged.

## Initial experiment and retained capabilities

Generate a **new random rectangular room** with a recorded seed. There is no
requirement to reproduce Rectangle 01, its APs, queries, or coordinate frame.
The initial experiment uses this one room, the **Direct NIST** observation track,
and a **0.5 m survey selected from a 0.1 m acquisition**.

Keep multiple rooms, other room families, and the SAPP-style receiver observation
track supported by the data model and component interfaces. The experiment
configuration selects the active room and track; algorithms and storage must
not hardcode a single room. The viewer initially shows this room's method
comparison and diagnostics, using the same survey selection and evaluation
conditions for every method. Room/track identity remains part of its data
interface so later experiments can expose additional choices.

The SAPP-style receiver track describes how channel measurements become observed
delays. It is independent of selecting a SAPP-derived reconstruction method such
as RANSAC; that method can also run on the Direct NIST track.

## Native acquisition and parallel development

Freeze the new scene, APs, dense lattice, query sets, stable position IDs, and
simulator settings/version before launching the expensive NIST acquisition.
Generation can then run while the API, cache, and viewer are refactored. Record
the random seed and complete simulator inputs so the room is reproducible.
Changing reconstruction settings or selecting a coarser survey must not rerun
NIST. A 0.1 m horizontal lattice has roughly 25 times the positions of a 0.5 m
lattice for the same area and height layers.

Retain native delays, gains, phases, arrival/departure angles, and the mapping
from frames to stable positions in shared acquisition artifacts. Preserve native
path provenance separately for evaluation and visualization. Direct observations
and future SAPP receiver observations derive from these same acquisitions;
reconstructors still receive only survey coordinates and observed ranges.
Materialize the requested observation track, rather than eagerly computing and
storing every receiver audit during the initial Direct NIST experiment.

## AP clearance from walls

Require every AP to lie inside the room's free interior with at least
**0.05 m (5 cm) clearance from every wall**. Measure the shortest distance to
the actual wall boundary, including concave and oblique segments, rather than
the room's bounding box. If wall thickness is represented, measure clearance
from the wall volume's interior-facing surface.

The implemented sampler checks strict containment, AP-to-AP separation, and the
explicit `ap_wall_clearance_m=0.05` setting. Reject placements violating the
clearance during sampling and validate the final generated coordinates. Retain the
existing AP separation requirement and record the clearance setting in the
dataset identity.

## Remove global room rotation

Generate rooms in their canonical coordinate frame, with no random global
rotation. Remove room rotation as a generation/search dimension; new manifests
may record `rotation_deg=0` to state the coordinate convention explicitly.

This applies to the global transform of rooms and their contents. Oblique wall
geometry and independently specified obstacle orientations remain meaningful
scene properties. Generate the room, APs, survey, and queries in the same frame.

## Dense survey with reusable subsampling

Generate and acquire one base survey at **0.1 m horizontal spacing**, retaining
the configured survey-height layers. Obtain the default **0.5 m survey** through
a deterministic subsampling function. This enables later studies of resolution
using the same underlying room, APs, measurements, and query set.

Proposed interface:

```python
survey_05 = subsample_survey(dense_survey, spacing_m=0.5, phase_xy=phase)
```

- Store the base spacing, grid origin, integer lattice indices, height-layer
  IDs, and stable observation IDs.
- For 0.5 m spacing, select every fifth lattice coordinate in **both horizontal
  axes**, at a recorded phase. Do not select every fifth row of the flattened
  survey: polygon clipping and multiple heights make that incorrect.
- Use a compatible origin and phase so the coarse survey is an exact subset.
  For example, a dense origin of 0.05 m and phase `(2, 2)` yields coarse
  coordinates starting at 0.25 m. Here phase gives integer index offsets modulo
  the stride in each axis. This is an available alignment, not a requirement to
  match the old dataset's grid.
- Apply the same selected position IDs to every AP, measurement condition, and
  associated audit array. Preserve the existing observations and noise draws;
  subsampling does not resimulate or interpolate them.
- Derive measurement-noise streams from stable acquisition/position identities,
  AP, track, level, and repetition. Never seed from a position's renumbered row
  within a subset. Directly materializing a subset and selecting it from dense
  observations must give the same values for the same acquisitions and settings.
- Initially support target spacings that are integer multiples of 0.1 m;
  reject incompatible requests explicitly. Record spacing and phase so density
  and grid alignment can be studied separately.

Represent each subsampled survey as a view/index selection over the dense
artifacts, consistent with the [computation-cache design](computation_cache_design.md).
Cache identity includes the dense input identity and selected observation IDs.
Repeated requests should reuse both computation and stored data. That identity
describes the selection operation; downstream pair matches and group fits key
only the values and parameters they consume, not the entire dense survey or an
unrelated resolution label.

Create a new versioned dataset for these generation changes. Preserve old results
under their old dataset identity; this does not require preserving the old room
in the new experiment or reproducing its scores. Resolution comparisons within
the new dataset vary the survey selection while keeping underlying acquisitions
and evaluation cases fixed. Do not spend work checking equivalence against the
old study or writing migration tests. Validate the new geometry, subsampling,
and measurement contracts directly where useful.

## SMART-LEA with ground-truth VTs

Include an explicit **SMART-LEA + ground-truth VTs** entry in the SMART-LEA
comparison, alongside entries using reconstructed VTs. Supply the dataset's
ground-truth VT coordinates and their AP/channel identities as the source bank,
bypassing VT reconstruction.

This is part of the first experiment group in the [redesign brief](redesign.md),
alongside optimized standard MPUrge (unification on and off) and MPUrge-MAP.
Select on noise-free calibration data, freeze the settings, then evaluate at
all declared noise levels. The earlier per-noise/average tuning plan is
superseded. Preserve the oracle alongside both frozen-upstream and jointly
tuned reconstructed-VT pipelines before expanding to remaining methods.

Use the same initial localization estimates, query measurements, and evaluation
cases as the corresponding reconstructed-VT entry. SMART-LEA still performs its
own range association and position estimation; this entry supplies neither true
query positions nor true peak-to-VT associations. Label it as an oracle VT-map
reference so we can study how much localization error remains with a correct
map, including SMART-LEA's failures and fallbacks.

Current sources: [geometry.py](../src/indoor_vt/acquisition/geometry.py),
[dataset.py](../src/indoor_vt/acquisition/dataset.py), and
[redesign.json](../configs/redesign.json).
