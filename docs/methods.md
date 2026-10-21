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
The default merges candidates by deterministic complete-link insertion within
the calibrated radius, with coordinate-wise median representatives and deduplicated
support. The merger alternatives below are calibrated for MPUrge variants.
The filter retains support at least beta times mean candidate support.
These stages are independently injectable; raw memberships, edges, costs,
residuals and rejection reasons remain in the fit artifact.

## MPUrge parameter audit (2026-10-21)

This inventory describes the current implementation, not a claim that every
choice below was specified by the paper or experimentally validated. The
original configuration/search inventory covered the main numerical knobs but
did not fully expose solver settings and fixed modeling policies. The running
campaign has not been changed by this documentation audit.

### Parameters included in calibration

| Component | Parameter | Default | Initial candidate values |
|---|---|---|---|
| MPUrge matching | Half-window `p`; full width is `2p+1` | 2 | 0, 1, 2, 3, 4 |
| MPUrge matching | Distance/pattern mixture `alpha` | 0.5 | 0, 0.25, 0.5, 0.75, 1; default is also evaluated |
| Local VT reconstruction | Number of survey points per neighborhood, including its center | 10 | 4, 5, 10, 20, 50, 100 |
| VT merging | Radius in metres | 0.5 | 0.1, 0.2, 0.5, 1, 2, 5 |
| VT filtering | `beta` multiplier of mean candidate support | 1 | 0, 0.5, 1, 1.5, 2, 4 |
| Triangle variant only | Range feasibility tolerance in metres | 0.5 | 0, 0.1, 0.2, 0.5, 1, 2, 5, 10 |
| MPUrge-MAP | Half-window, mixture and localization neighbor count | 2, 0.7, 3 | Same window/mixture lists; k = 1, 3, 5, 10, 20; default also evaluated |
| DiffAssign variant only | Range scale, temperature, reverse weight, unmatched cost | 0.5 m, 0.1, 0.5, 2 | Explicit lists in `search_spaces.py` |

The VT neighborhood count and localization k are different parameters.
Full-room reconstruction ignores the neighborhood count. Bounds, expansion
flags and refinement rules are in `search_spaces.py` and `calibration.py`;
the table lists initial values, not every possible adaptive trial.

### Fixed choices that also determine the result

| Stage | Current choice | Where it is recorded |
|---|---|---|
| Size unification | Ordered L1-optimal subset; DP ties choose inclusion of the later candidate | Matcher config plus code; PBA uses L2 instead |
| Pattern | Every unordered pair in the local window; unnormalized for MPUrge, divided by `p(2p+1)` for MAP when p > 0 | Matcher config plus code |
| Window ends | Near-repeated endpoint padding; increment `16 * float64 epsilon * max(span, max(abs(range)), 1)` | Code only |
| Reciprocal matching | First argmin on ties; candidates sorted by dissimilarity then original indices | Code only |
| Cross-check / Urge | Order-based crossing test; remove all reciprocal candidates, including rejected ones; recompute patterns on survivors | Matcher config and method definition; Urge policy is code only |
| Scope | 3D Euclidean nearest survey positions; all pairs within each neighborhood; repeated pair work deduplicated; full scope uses all survey positions | Reconstruction config and method definition |
| Grouping | Components or stars are distinct named variants, not a hidden per-room choice | Reconstruction config |
| Conflicting component owners | Retain one peak per survey position by greatest degree, then lowest mean incident cost, then stable node ID; do not split again after removal | Code and method definition |
| Minimum usable group | At least four distinct survey positions, centered position matrix of rank 3 | Code and method definition |
| 3D fitting | Centered linear least-squares initialization, equally weighted range residuals, unconstrained signed coordinates, linear loss, analytic Jacobian, LM solver | Code and method definition; not fully expanded in the configuration receipt |
| Fit limits / failure | `max_nfev=100`; reject nonfinite coordinates or solver nonconvergence; no residual-quality cutoff | Code only for MPUrge reconstruction |
| Fit numerical defaults | Pinned SciPy 1.15.3: `ftol=xtol=gtol=1e-8`, `x_scale=1`; NumPy default rank and least-squares cutoffs | Dependency lock and code, not expanded in method config |
| Candidate merging | Greedy complete-link insertion, ascending x then y then z (original index breaks exact ties), first created eligible cluster, coordinate-wise median representative | Code and method definition |
| Support | Number of unique survey-position owners after union; retain support >= beta times the arithmetic mean over merged candidates | Code and method definition |
| MAP coverage | Mean match cost divided by matches / max(original fingerprint lengths) | Code and method definition; selected interpretation of prose/equation conflict |
| Multi-AP localization | Arithmetic mean of per-AP scores before neighbor selection; a nonfinite channel score makes the combined score nonfinite | Code; a multi-AP adaptation |
| Localization weighting | Inverse score with fixed exponent 1; stable top-k; scores <= float64 epsilon share the weight, other selected scores get zero weight | Config plus code |
| Localization failure | Survey centroid when no reference has a finite score | Code |
| DiffAssign numerics | Fixed predicted K; L-BFGS-B, maxiter 100, ftol 1e-9; pinned defaults gtol 1e-5, maxcor 10, maxls 20, maxfun 15000; absolute-value smoothing adds 1e-12 m squared | Config, code and dependency lock |
| DiffAssign acceptance | Keep initial bank unless solver succeeds with finite objective no greater than its initial value | Code and method definition |

`epsilon_m` in the shared initial-localizer config is used by MCA, not by
MPUrge-MAP. `f_scale=1` in the shared fitter has no effect for MPUrge's linear
loss. These are inactive fields, not omitted MPUrge tuning dimensions. There
is no hidden peak cap, true-VT-count input, room-bound fitting constraint or
extra match-dissimilarity threshold in these implementations.

The user reviewed these fixed policies and requested that they remain fixed,
not become additional calibration dimensions. The inventory documents their
behavior, including the fact that coordinate-wise medians and lexicographic
greedy insertion are not guaranteed invariant to rotation. Source hashes
preserve their implementation. The specifically requested merger/representative
comparison below is the exception to keeping these choices fixed.

### Findings and corrections in revision v2

- Published candidates include both the SAPP MAP setting p=2, alpha=0.7, k=3
  and the MAP paper setting p=6, alpha=0.7, k=3. Initial preset coverage uses
  the latter. The paper's original-MPUrge initial-localization p=5, alpha=0.5,
  k=3 and MCA epsilon=1.6 m, k=1 are also explicit candidates.
- Future searches exclude p=0 together with alpha=0, including nested joint
  configurations, boundary expansion and refinement. Historical rows are
  preserved and marked ineligible; they are neither deleted nor rerun.
- Fixed policies and resolved numerical defaults are not all serialized into
  a single method manifest, although source and dependency hashes pin them.
- Solver-limit sufficiency and rotation sensitivity have not been established
  by the matcher agreement tests. Those tests compare against the prior SAPP
  adapter, not author code.
- Per-AP score matrices now survive changes in k and A/B/A+B. Graphs, raw
  source fits, cluster memberships/hierarchies and final LEA predictions also
  have separate caches keyed by their actual dependencies.

### Implemented merger and execution revision

The external [plan](../../DATASET_PIPELINE_PLAN.md) now specifies greedy
complete-link with median **and mean** representatives, standard agglomerative
complete-link with a median representative, and DBSCAN with a median
representative. Their distance/density parameters are calibrated; greedy
mean/median changes only the representative, not cluster membership. Other
fixed modeling and numerical choices remain fixed. No true VT count is used.

These alternatives are components of the existing MPUrge entries, not extra
high-level methods. The merger field has four choices: `greedy_median`,
`greedy_mean`, `agglomerative`, `dbscan`. All use the existing calibrated radius;
DBSCAN additionally searches min_samples in 2, 3, 5, 10, 20, with expansion
bounds 1 to 100. That parameter is inactive for other mergers.

Greedy insertion uses spatial cells and checks every member of each eligible
cluster. Mean and median reuse identical memberships. Agglomerative complete
link uses an in-place nearest-neighbor chain matching SciPy 1.15.3's full
hierarchy exactly, with large distance arrays backed by temporary files. The
hierarchy is cached once per raw candidate bank and
cut at each radius; its distance storage is quadratic, and the estimated
condensed-array bytes are recorded. DBSCAN uses on-demand cKDTree neighborhoods
and records noise candidates separately. DBSCAN density counts candidates;
the unchanged subsequent support rule counts distinct survey positions.

Revision v2 obtains initial main-method presets before deeper calibration,
advances each eligible search one new configuration per turn, and waits for
all main calibration/freeze jobs before starting any shadow work. Existing
trial artifacts retain their original provenance. Dataset coverage, search
budgets and the held-out test barrier remain unchanged. Cached timings are
labelled: historical runtime remains historical; ranking-only timings must
be combined with channel-score work before making runtime comparisons.

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
