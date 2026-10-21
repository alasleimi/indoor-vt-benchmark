# Method definitions and calibration contract

All primary methods consume sorted ranges in metres, channel identifiers and
known survey coordinates. They receive no floor plan, wall count, true AP/VT
coordinates, true query height, simulator path labels, power, angles or noise
level. The evaluator owns those fields. A and B are fitted independently;
A+B combines the same fitted banks and keeps their channel labels.

## Registered methods

| Identifier | Definition |
|---|---|
| `mpurge_local`, `mpurge_full` | MPUrge matching, connected groups, signed 3D range fitting, coordinate merging and support filtering. |
| `mpurge_star_local`, `mpurge_star_full` | Same stages, with each peak and its directly matched neighbours forming a group. |
| `mpurge_triangle_local`, `mpurge_triangle_full` | MPUrge with a known-survey-position triangle-inequality gate before matching. |
| `mpurge_diffassign_local`, `mpurge_diffassign_full` | MPUrge's predicted coordinates and count initialize the fixed-count refinement below. |
| `pba_local` | Historical PBA matcher: L2 size unification and absolute range-distance conflict priority, followed by the same disclosed 3D reconstruction stages. |
| `ransac` | SAPP range-consensus source reconstruction with signed 3D proposals and no source-count cap. |
| `zayets2018` | The cluster-matching/trilateration source-estimator replacement in the SAPP ablation, adapted to signed 3D. |
| `zayets2019_assumption` | Explicit implementation assumption for the underspecified cosine-clustering comparator; see the source audit. |
| `mpurge_map` | Normalized MPUrge-MAP fingerprint score with coverage penalty and weighted nearest neighbours. |
| `mpurge` (initial localization) | Original unnormalized MPUrge match dissimilarity used in weighted fingerprint nearest neighbours. |
| `mca` | Multipath coincidence score; published candidate uses the best reference. |
| `chamfer` | Bidirectional mean nearest-range distance with weighted nearest neighbours, as in SAPP. |

The initial-localization and reconstruction registries are distinct. A method
name is never inferred from a result filename or a privileged old wrapper.

## MPUrge stages and interpretation choices

The matcher unifies window lengths with ordered subset matching, computes peak
and pairwise-pattern discrepancies, finds reciprocal minimum-cost candidates,
applies the selected order check and removes all potential reciprocal matches
during Urge, including candidates rejected by the check. The pattern uses every
unordered pair in the window. MPUrge's pattern term remains unnormalized;
MPUrge-MAP uses its normalized pattern and divides the match cost by coverage.

The order-based check is the chosen interpretation of the paper ambiguity.
The old completed 24-block comparison reports mean F1 0.13134 for corrected
order/components versus 0.11028 for printed/components, and 0.04751 for
order/star versus 0.03215 for printed/star. Those artifacts support choosing an
interpretation; their tuned numeric settings and privileged preprocessing are
not imported. The SAPP supplement supports normalized MPUrge-MAP, half-window
2, alpha 0.7 and k=3 as an explicit published candidate. Exact evidence files
and hashes are listed in `upstream_sources.json`.

Local scope builds nearest-reference neighbourhoods and matches all pairs
inside each neighbourhood, deduplicating pair work. Each neighbourhood then
forms its own groups. Full scope explicitly uses every survey reference in one
set and all its pairs. Full scope ignores the neighbourhood-count parameter.
Local distance is ordinary 3D Euclidean distance; both survey layers participate.

Connected-group conflicts retain one peak per reference: largest graph degree,
then smallest mean incident dissimilarity, then stable observation ID. This
tie rule and the merging rule are disclosed implementation choices. Star groups
contain a centre and its immediate neighbours, with duplicate groups removed.
The triangle variant requires
`abs(r1-r2) <= distance + tolerance` and
`distance <= r1+r2+tolerance`. It uses only known survey coordinates.

Each usable group has at least four observations and noncoplanar survey
geometry. Linearized multilateration initializes unconstrained nonlinear least
squares on signed x,y,z coordinates. There is no room bounding box or imposed
positive VT height. Invalid/degenerate groups are recorded and rejected.
Candidates merge by deterministic complete-link insertion within the calibrated
radius. The representative is their coordinate-wise median, with deduplicated
support. The filter retains support at least beta times mean candidate support.
These stages are independently injectable; raw memberships, edges, costs,
residuals and rejection reasons remain in the fit artifact.

## MPUrge-initialized DiffAssign

Let survey point i have ranges r_ij and let x_k be the K source positions
predicted by MPUrge. K is that prediction's actual size, independently for each
AP. It is never selected from truth or a fixed room-specific count.

The pair cost is

```text
a_ijk = sqrt((||x_k - s_i|| - r_ij)^2 + 1e-12) / scale
softmin_T(a_1,...,a_n) = -T * log(mean(exp(-a/T)))
L_i = mean_j min(clutter_cost, softmin_k(a_ijk))
    + reverse_weight * mean_k min(clutter_cost, softmin_j(a_ijk))
L = mean_i L_i over nonempty profiles
```

The forward term lets a measured range be explained softly by source candidates;
the reverse term rewards candidates that explain observations. This is soft
many-to-many range association, not a one-to-one assignment or an EM claim.
The fixed clutter cap limits an unmatched observation's contribution. An
analytic gradient and unbounded L-BFGS-B update source coordinates only.
Nonconvergence, nonfinite loss or increased loss retains the initial bank.
An empty initial bank remains empty. No source birth/death, K loop, query label
or uncertainty estimator is added. Scale, temperature, reverse weight and
clutter cost are ordinary global calibration parameters.

## SAPP source-estimator adaptations

RANSAC retains SAPP's three-range proposal mechanism: three sphere surfaces
intersect in up to two points in 3D. Both signed solutions are emitted; all
survey observations decide support. This removes the old single receiver plane
and unsigned-height convention. Robust refinement needs at least four spatially
usable inliers. The existing consensus scoring, duplicate-source separation and
evidence consumption remain; the previous fixed maximum source count is removed.
The finite proposal budget is a recorded computational parameter, not K.

The 2018 adapter clusters sorted ranges by gaps, compares cluster features
(first range, count, span), joins compatible clusters without duplicate
reference owners, assigns within-cluster range tracks and fits 3D candidates.
It then uses the SAPP consensus-selection stage. It is the reconstruction
ablation, not the full published fingerprint-interpolation localizer.

## SMART-LEA and failures

SMART-LEA's paper is already formulated in 3D. This implementation replaces its
known anchors with reconstructed source coordinates and preserves channel IDs.
It predicts source-to-query ranges at the initial estimate, associates them with
query ranges and solves for a refined 3D query coordinate. Iteration count and
matcher settings are calibrated. No true physical AP is appended secretly.

An empty bank, unusable association or failed fit returns the original coarse
prediction and records the fallback. It stays in the final localization metric.
The actual predicted source bank, including an empty bank, remains the VT score
input. A no-match initial localizer returns the survey centroid with status.

## Selection and reporting

- Component regime: calibrate coarse error and VT OSPA separately, freeze each,
  then calibrate LEA with both upstream components frozen.
- End-to-end regime: jointly select the named pipeline by final mean 3D error.
- Configurations are shared across room families, rooms, noise, layouts and
  A/B/A+B. Component-mode upstream selections are also reused across partners.
  LEA and end-to-end selections belong to a named initial/VT pairing.
- Use standard OSPA p=1,c=1 m. F1 uses one 0.5 m maximum-cardinality one-to-one
  coordinate match on pooled real-plus-virtual source bags. No AP label enters
  that bag match, although labels remain essential inside the pipeline.
- Uniform case averaging; all queries, failures and fallback predictions remain.
  Test evaluation requires the exact frozen code, observation and truth hashes.
- Search budgets are fixed at 32 broad draws, at most two 8-draw boundary
  expansions, then 16 local draws, plus defaults and published candidates.
  Draws are uniform over declared lists, with deterministic tie handling and
  resumable histories. `search_spaces.py` records per-method ranges/citations.
- Unpublished numeric winners are excluded. Published values remain explicit
  candidates; other initial ranges use round values. One method-initialization
  seed is used per case, separately from the one measurement-noise realization.

The eight graph experiments live outside this repository. A full-calibration
comparison against MPUrge in the same scope, track and regime must show at least
5% relative OSPA/error reduction before promotion. Synthetic/pilot checks do not
qualify. No shadow method has been promoted.
