# VT reconstruction and fingerprint localization component design

[Start with the redesign brief](redesign.md) for implementation policy, experiment
priorities, noise-free selection, and stage-wise versus joint pipeline tuning.

Design direction agreed on 2026-10-24, reconciled on 2026-10-25. Names and examples
below describe the component design. The implemented priority pipeline follows it;
remaining experiment adapters are tracked in implementation_status.md.
The initial experiment uses one newly generated rectangular room and Direct NIST
observations, as specified in the [dataset design](dataset_generation_design.md).
Room and observation-track selection belong to experiment configuration; these
scientific components remain usable across rooms and delay frontends. Each fit
receives one room's survey; campaign orchestration handles multiple rooms.

MPUrge's core is a pairwise peak matcher. The full VT reconstruction task
combines matching, grouping, source estimation, clustering, and support filtering.
Component names should describe their operation; `VTReconstructor` names the
whole task rather than an individual stage.

## Scientific API

`MPUrge` represents the complete VT reconstruction pipeline, with five
replaceable scientific stages and two injected selection callables. Each
component owns its own settings:

```python
from functools import partial

select_neighborhoods = partial(nearest_neighborhoods, size=10)

def select_local_pairs(xyz):
    return all_pairs_within(select_neighborhoods(xyz))

mpurge = MPUrge(
    pair_selector=select_local_pairs,
    group_scope_selector=select_neighborhoods,
    matcher=MPUrgeMatcher(
        half_window=2,
        alpha=0.5,
        size_unification_norm=absolute_difference,
        triangle_tolerance_m=None,
    ),
    grouper=ConnectedComponentsGrouper(),
    estimator=Multilateration3D(),
    clusterer=GreedyCompleteLink(radius_m=0.5, representative=median),
    support_filter=RelativeSupportFilter(beta=1.0),
)
```

The constructor contains no cache, audit-storage, or refinement dependency.

## Components and order

```text
PairwiseMatcher -> MatchGrouper -> SourceEstimator
                -> CandidateClusterer -> SupportFilter

Optional, outside MPUrge: PositionRefiner
```

| Component | Responsibility | Output |
|---|---|---|
| `PairwiseMatcher` | Match peaks between two fingerprints from the same AP. MPUrge implements the Merge-Purge-Urge loop here. | Peak correspondences with original observation identities and dissimilarities |
| `MatchGrouper` | Assemble pairwise correspondences into groups believed to share a source. | Groups of receiver-position/range observations |
| `SourceEstimator` | Fit one source position from all observations in each usable group. | Candidate VT position, supporting observation identities, and fit diagnostics |
| `CandidateClusterer` | Cluster candidate coordinates and choose representative positions. | Representative coordinates and lists of input candidate indices |
| `SupportFilter` | Count distinct supporting survey positions and filter consolidated candidates. | Retained VT map with evidence |
| `PositionRefiner` | Independently adjust an existing VT map using survey measurements. | Refined VT map |

`MatchGrouper` is an injected component with distinct scientific implementations:

- `ConnectedComponentsGrouper` uses DSU over packed pair blocks to follow transitive
  match connections. Within each component, keep one peak per fingerprint by
  highest degree, then lowest mean edge cost, then lowest peak index. Do not
  regroup after pruning. Cycles contribute to degree/cost but do not merge
  already-connected components. This shared scientific implementation contains
  no caching or audit-storage machinery.
- `StarGrouper` forms groups from a central observation and its directly matched
  neighbors.

Both consume pairwise correspondences and return observation groups, but they
produce different source hypotheses. Select an implementation explicitly instead
of exposing a generic `grouping` mode or `association_policy` flag:

```python
grouper = ConnectedComponentsGrouper()  # Alternative: StarGrouper()
observation_groups = grouper.group(matches)
```

These alternatives change the scientific method. Separate readable and
instrumented versions of the same method must preserve its scientific behavior.

Each successfully fitted group produces one candidate VT; each member supplies a
constraint. Rejected fits produce diagnostics but no candidate.
Clustering precedes support filtering. For example, candidates supported by
`{A, B, C, D}` and `{B, C, D, E}` have combined support **5**, not 8.
The current cutoff is `support >= beta * mean_cluster_support`.
Keep support counts and match dissimilarities as separate quantities. MPUrge
unions its candidates' observation memberships using the clusterer's returned
indices before applying support filtering; the spatial clusterer does not need
to interpret those memberships.

The optional triangle-inequality check belongs inside `PairwiseMatcher`, before
selecting matches. `triangle_tolerance_m=None` disables it; a numeric tolerance
enables it. The pair-matching caller supplies receiver separation computed from
survey coordinates. This check is separate from MPUrge's Purge crossing check.
Likewise, MPUrge's Merge phase pairs peaks; `CandidateClusterer` combines
estimated positions later.

## Configurable size unification

Use one optional callable inside `MPUrgeMatcher`, before the Merge-Purge-Urge
loop. The callable defines the cost of pairing two scalar ranges; `None` disables
size unification. There is no separate enable flag or norm-name string:

```python
MPUrgeMatcher(size_unification_norm=absolute_difference)  # Default: L1 costs
MPUrgeMatcher(size_unification_norm=squared_difference)   # Squared L2 costs
MPUrgeMatcher(size_unification_norm=None)                 # No unification
```

`absolute_difference(a, b)` returns `abs(a - b)`; `squared_difference(a, b)`
returns `(a - b) ** 2`. With a callable, select an order-preserving subset of
the longer sorted fingerprint matching the shorter fingerprint's length,
minimizing the sum of these pair costs. Equal-length inputs need no reduction.
With `None`, keep the original lengths and match the possibly unequal-length
fingerprints directly. This choice is independent of pattern normalization and
the subsequent peak-matching cost.

The implementation uses the optional callable directly, with L1 as its default.
Preserve original peak identities through
subset selection and original fingerprint lengths for MPUrge-MAP coverage.

Pass the function directly to the scientific implementation. Any mapping between
saved names and functions belongs to experiment configuration/loading, while
execution records the callable's stable implementation identity for caching.
A different unification algorithm can use a separate component if needed.

## Pair selection and grouping scopes inside fit

Inject `pair_selector` and `group_scope_selector` at construction. Both are
callables taking survey coordinates. `fit(survey)` invokes them to compute the
actual pairs and scopes; callers do not supply these intermediate results:

```python
vt_map = mpurge.fit(survey)
```

`pair_selector(survey.xyz)` returns survey-position pairs to compare. Compute
these geometric pairs once, then match their fingerprints separately for each
AP. `group_scope_selector(survey.xyz)` returns sets of survey-position
IDs whose available matches may be grouped together. The constructor contains
the selection rules; the fit input is the survey; pairs and scopes are computed
inside the method and retained as scientific intermediate results.

Each nearest neighborhood counts its center and uses 3D Euclidean distance.
By default, include all positions tied at the requested boundary, so the actual
size can exceed the requested size. This avoids arbitrary grid-direction choices
and gives nested scopes. `include_ties=False` provides exact-size neighborhoods
with deterministic position-index tie breaking. Neighborhoods can overlap;
identical sets and repeated pairs are deduplicated. `all_pairs_within` compares every pair inside each neighborhood,
not only the center with its neighbors.

| Configuration | Result of `pair_selector` | Result of `group_scope_selector` |
|---|---|---|
| Local matching, local grouping | Pairs within neighborhoods | Those neighborhoods |
| Local matching, global grouping | Pairs within neighborhoods | One set containing all survey positions |
| All-pairs matching, global grouping | Every survey pair | One set containing all survey positions |

Both local and global configurations use the whole survey. A shared edge list
does not imply global grouping: the grouper operates separately within each
specified scope, then identical observation groups are deduplicated. Changing
grouping scope should reuse already computed pairwise matches.

Keep the selection callables independent so that local pair comparisons can be
combined with global grouping. They may share a neighborhood-selection function,
as in the constructor example. When both need the same neighborhoods, the
execution implementation reuses that computation and artifact by identity;
this reuse does not require callers to construct scopes or pairs outside `fit`.

The paper specifies matching neighboring fingerprints and aggregating matches,
but does not specify the spatial neighbor count, overlapping-set construction,
or exact aggregation algorithm. The API must support experimenting with pair
selection, grouping scope, and grouping implementation independently, including
the configurations above. These are supported experimental choices. The paper's
peak-pattern window is a separate parameter.

## Readable and instrumented implementations

Keep two side-by-side implementations of the same method. The scientific version
expresses the algorithm directly, without cache keys, persistence, compatibility
receipts, or execution logging. Its orchestration should read approximately:

```python
def fit(self, survey):
    pairs = self.pair_selector(survey.xyz)
    scopes = self.group_scope_selector(survey.xyz)

    matches = match_pairs(survey, pairs, self.matcher)
    groups = group_within_scopes(matches, scopes, self.grouper)
    candidates = estimate_sources(survey, groups, self.estimator)
    clusters = cluster_with_evidence(candidates, self.clusterer)
    return self.support_filter.filter(clusters)
```

Here `cluster_with_evidence` applies the spatial clusterer per AP and unions
MPUrge observation memberships for each returned cluster. It is an ordinary
scientific helper, not another injected algorithm or storage layer.

Scientific stage results retain observation identities, match costs, memberships,
residuals, support, and rejection reasons. The execution version follows the
same stages and handles caching, audit persistence, timing, and concurrency:

```python
execution = MPUrgeExecution(mpurge, store=computation_store)
vt_map = execution.fit(survey)
```

The motivation is readability despite instrumentation, not a requirement for
different optimization levels. Shared scientific helpers are useful; deliberate
duplication is acceptable when it keeps the scientific version clear. Both
versions remain executable and implement the same new scientific behavior.
Use focused checks where useful; do not build a legacy-output comparison or
migration test suite. Clean code takes priority over reproducing old-study
outputs, numerical quirks, or serialization.

Persistence uses the [computation-identity design](computation_cache_design.md):
store each stage's evidence once and reference it, rather than copying all
upstream data into each result. Runtime measurements are separate execution data.

Posthoc refinement remains outside both MPUrge implementations and accepts any
compatible map. The optional refitter described below is injected into
`GreedySourceSelector`, external to `RansacReconstructor`. It runs during
selection, before acceptance and evidence consumption. Posthoc refinement runs
after selection has finished; the distinction is timing, not ownership by RANSAC.

```python
refined_map = refiner.fit(survey, vt_map)
```

## Global rule: dependency injection, conditionals, and hyperparameters

Apply this rule throughout reconstruction and localization, including future
methods. Decide by the responsibility that changes, not by a blanket preference
for classes or a ban on `if` statements.

| Variation | Mechanism | Example |
|---|---|---|
| A value changes within the same operation | Component-owned configuration; expose as a hyperparameter when studied or tuned | Matching weight, pattern window, clustering radius, trial budget |
| A small rule changes within an algorithm | Configuration plus a local conditional | Enable triangle checking; apply coverage normalization |
| An interchangeable calculation is a function | Inject the callable directly; use `None` when the operation is optional | Absolute versus squared difference for size unification |
| Input or execution state needs handling | Ordinary local conditional | Empty observations, no matches, failed fit, already-connected nodes |
| A coherent operation has interchangeable algorithms | Inject a component with a stable input/output contract | Connected-component versus star grouping; random geometric proposals versus cluster-track proposals |
| Several operations reuse the same calculation | Shared helper function | Range prediction, residual computation, numerical fitting |
| An experiment selects a named method | Factory or preset assembles components once | `MPUrgeMAP`; RANSAC and Zayets reconstruction presets |

Expose boundaries that correspond to meaningful algorithm comparisons. Keep
ordinary choices within each stage as configuration. A conditional is reasonable
when its effect stays local; a flag branching across several stages suggests
distinct algorithms are being mixed.

Hyperparameters and injection are not competing concepts: hyperparameter names
describe what an experiment varies; injection describes how algorithms are
composed. An experiment may tune a numeric tolerance or select a categorical
component implementation. Resolve the latter when constructing the pipeline,
rather than passing a `method` string through every stage. Each component owns
only the settings it actually uses.

Dependency injection can be as small as passing a function. It does not require
a class or a large algorithm. Avoid repeated `if norm == "l1"` branches and
string-to-function lookups inside scientific code. If an optional callable
expresses both activation and behavior, prefer it over a separate enable flag
and function setting. Saved experiment names can be resolved at construction;
serialization and cache identity remain responsibilities of the execution layer.

Shared helper functions provide code reuse without injection. Introduce a finer
dependency when a concrete experiment needs interchangeable behavior: several
alternative feasibility checks could justify a constraint component, while the
single triangle check currently only needs a local conditional.

An algorithm name controlling several substantial branches, unrelated settings,
or different intermediate computations is a concrete refactoring signal. Extract
the scientific operation rather than moving the same dispatcher into another
helper. Keep readable scientific implementations and their execution counterparts
separate, as described above; caching and audit storage do not justify additional
method switches inside scientific components.

Use one `SourceEstimator` boundary accepting the complete observation group,
rather than overlapping `fitter` and `group_fitter` callbacks. Graph construction
can remain internal to association; caching belongs to execution infrastructure.

## Zayets cluster grouping: constrained single linkage

Describe the grouping formerly inside `consensus.py:matched_tracks` as
`ConstrainedSingleLinkageGrouper`:

> Greedy Kruskal grouping: consider all edges with cost <= `group_threshold`,
> in increasing cost order. Merge two distinct components only if the result
> contains at most one node (range cluster) from each fingerprint.

Nodes are clusters of nearby ranges within individual fingerprints. Edges carry
pairwise cluster comparison costs. The threshold is inclusive. All eligible
edges are considered; they do not all cause merges:

- Endpoints already in the same component: ignore the edge because it changes
  nothing. This is not a rejection of the underlying similarity. The current
  implementation tracks components with union-find, not an explicit retained
  edge forest.
- Distinct components with overlapping fingerprint owners: forbid the merge
  because it would violate the one-node-per-fingerprint constraint.
- Otherwise: merge their memberships and fingerprint-owner sets.

This permits transitive grouping: `A1-B1` and `B1-C1` can join three nodes even
when `A1-C1` fails the threshold. It is a greedy constrained grouping procedure,
with no guarantee of a globally minimum-cost forest or minimum component count.
It is distinct from pairwise peak matching and from assignment of individual
peaks within the resulting cluster groups.

The 2018 paper specifies cluster comparison, preference for low costs, and at
most one cluster per fingerprint in a group. The precise sorted-edge union
procedure and deterministic tie handling are implementation choices. The stages
should be visible separately in the scientific code:

```text
Within-fingerprint range clustering -> pairwise cluster costs
-> constrained single-linkage grouping -> within-group peak assignment
-> track-based source fitting
```

The current Zayets adapter performs grouping independently within overlapping
radius neighborhoods, then deduplicates tracks across neighborhoods. The proposed
Zayets API has one `scope_selector`, evaluated inside `fit(survey)`. Radius
neighborhoods, nearest-k neighborhoods, and one global scope are alternatives.
Unlike MPUrge's API above, Zayets does not need a separate `pair_selector`.
Within a scope, comparisons between range clusters from different fingerprints
provide the edges for the grouper; this is not a pairwise peak-matching stage.

Do not recompute or duplicate those comparisons for every overlapping scope.
Cluster each fingerprint once for the selected gap rule, and reuse each required
cluster-pair cost for the same input clusters and cost parameters. Store reusable
cost results independently of the grouping threshold. Each scope reads those
costs, applies its threshold, and runs its own grouping state. Shared costs do
not imply one global grouping run. Changing the threshold should reuse costs;
changing scopes should reuse all comparisons already available by computation
identity. Exact duplicate tracks are fitted once.

Before cross-fingerprint comparisons, `split_at_gaps(sorted_values, threshold)`
splits at every consecutive difference **>= threshold**. It preserves original
peak indices. For example, `[2.0, 2.2, 2.4, 5.0, 5.1]` with threshold `0.6`
produces `[2.0, 2.2, 2.4]` and `[5.0, 5.1]`. The threshold is not a limit on a
cluster's total span. Empty input produces no clusters. This is the connectivity
rule of one-dimensional single linkage with a strict joining threshold, computed
by scanning the sorted sequence. The direct function name communicates the
operation better than `GapRangeClusterer`.

Within each resulting cluster group, `LongestClusterTemplateAssigner` reduces
the longest equal-length range vectors elementwise to a template. The default
reducer is the mean. Equal-length clusters align by index; shorter clusters use
minimum-cost one-to-one assignment into distinct template slots. A slot's
assigned original observations form one peak track. Template values guide
assignment; the original ranges are used for source fitting.

The proposed `ZayetsReconstructor` covers this candidate-generation portion of
the 2018 method, with our signed-3D adaptation and concrete grouping/assignment
choices. It is not the paper's complete fingerprint-interpolation localizer.
The additional survey-wide selection used in the benchmark comes from SAPP's
source-estimator adaptation and is composed separately below.

## Shared reconstruction interface and wrappers

Unify methods at their output boundary rather than requiring every generator to
expose the same internal evidence. All reconstructors and wrappers implement:

```python
sources = reconstructor.fit(survey)  # SourceMap: source positions by AP
```

An inner reconstructor may return provisional candidates or an already filtered
map. Positions use the survey's coordinate frame and retain AP identities.
The outer selector needs those positions and the survey, not the inner method's
groups, sampled triplets, or implementation-specific diagnostics. It evaluates
APs separately. Provenance remains available for auditing through stage results;
it is not an extra scientific input to a selector that only uses coordinates.

| Component | Responsibility |
|---|---|
| `ZayetsReconstructor` | Return positions fitted from peak tracks, before the additional survey-wide selection. |
| `RansacReconstructor` | Return retained random-hypothesis positions, before final source selection. |
| `ClusteredReconstructor` | Call an inner reconstructor, then cluster its positions and return representatives, separately per AP. |
| `FilteredReconstructor` | Call an inner reconstructor, then apply the injected selector to its output and the survey. |
| `GreedySourceSelector` | Score and select existing source candidates, optionally refit them, suppress duplicates, and optionally remove supporting peaks between selections. |

The wrapper implementations remain small:

```python
# ClusteredReconstructor.fit(self, survey)
sources = self.reconstructor.fit(survey)
return cluster_source_map(sources, self.clusterer)

# FilteredReconstructor.fit(self, survey)
sources = self.reconstructor.fit(survey)
return self.selector.select(survey, sources)
```

Use one spatial-clustering contract in both wrappers and MPUrge:
`clusterer.cluster(xyz)` takes an ordered `(n, 3)` coordinate array for one AP
and returns representative coordinates plus input-index memberships.
`cluster_source_map` applies it per AP and assembles a `SourceMap`; execution
retains membership references for auditing. MPUrge's `cluster_with_evidence`
additionally unions observation memberships for its own support rule. Neither
helper changes the clustering algorithm. This avoids requiring a generic
clusterer to accept generator-specific records or invent support evidence for
RANSAC proposals. Cache spatial memberships separately from representative
reduction so changing mean to median can reuse the same grouping.

Use names for the actual operations, not the paper that supplied a preset:

| Earlier proposal | Agreed name | Reason |
|---|---|---|
| `GapRangeClusterer` | `split_at_gaps` | A function that splits a sorted sequence at consecutive gaps. |
| `SappConsensusScorer` | `SupportAndResidualScorer` | Combines support count, Gaussian closeness, and a clipped median-error penalty. |
| `GreedyConsensusSelector` | `GreedySourceSelector` | Describes the repeated selection of sources and update of available evidence. |
| `ConsensusReconstructor` as a generic wrapper | `FilteredReconstructor` | The wrapper is independent of any particular scoring or selection algorithm. |

Here, a consensus set means observations agreeing with a candidate within a
tolerance. The selector evaluates such agreement, but does not sample hypotheses.
When enabled, removing supporting observations between selections is also known
as peeling. Keep the broader `GreedySourceSelector` name because evidence
consumption is optional. `FilteredReconstructor` permits an injected selector
with explicit optional refitting; it does not promise that every selector merely
returns unchanged input positions.

## Proposed Zayets and RANSAC constructors

These constructor examples specify the design and default scientific settings.
The named scientific classes and functions now live in `src/indoor_vt/science/`;
Zayets/RANSAC experiment scheduling and their granular execution caching remain
follow-up work after the priority experiments.
Callables may be functions, partial applications, or configured callable objects.

```python
zayets_candidates = ZayetsReconstructor(
    scope_selector=partial(radius_neighborhoods, radius_m=1.0),
    range_clusterer=partial(split_at_gaps, threshold=0.6),
    cluster_cost=partial(
        cluster_feature_cost,
        first_range_weight=0.5,
        peak_count_weight=0.5,
        range_span_weight=0.5,
        difference=absolute_difference,
    ),
    grouper=ConstrainedSingleLinkageGrouper(max_cost=0.5),
    peak_assigner=LongestClusterTemplateAssigner(
        template_reducer=mean,
        cost=absolute_difference,
    ),
    min_track_positions=4,
    estimator=Multilateration3D(
        loss=squared_loss,
        max_evaluations=100,
    ),
)

score = SupportAndResidualScorer(
    inlier_tolerance_m=0.18,
    smooth_scale_factor=1.0,
    smooth_weight=0.35,
    median_penalty_weight=0.10,
    median_clip_factor=4.0,
)

ransac_candidates = RansacReconstructor(
    sampler=uniform_peak_sample,
    hypothesis_solver=ThreeSphereIntersections(),
    scorer=score,
    retainer=TopPerTrialBatch(trials_per_batch=400, keep=12),
    trials=30_000,
    seed=0,
)
```

`cluster_feature_cost` uses the first range, number of peaks, and last-minus-first
range span. Its default cost is `0.5 * (abs(first_difference) +
abs(count_difference) + abs(span_difference))`, matching the current code.
Changing feature normalization, weights, or the difference function is explicit.

`uniform_peak_sample` selects distinct nonempty survey positions uniformly over
the whole survey, then one peak uniformly from each. The hypothesis solver
declares its required sample size: `ThreeSphereIntersections` requires three
and returns zero, one, or two candidate positions. Do not expose a conflicting
independent sample-size setting beside that solver. A different solver can use
four or more observations:

```python
hypothesis_solver = FittedSourceHypotheses(
    sample_size=4,
    estimator=Multilateration3D(loss=squared_loss, max_evaluations=100),
)

retainer = GlobalTopK(keep=900)
```

`TopPerTrialBatch` counts sampled trials, including rejected samples. Each trial
can emit up to two hypotheses; retain at most 12 per 400 trials with the current
settings. `GlobalTopK` instead retains the globally highest-scoring proposals.
These are different scientific policies, not interchangeable implementations.
Neither policy guarantees that the pool represents different physical sources.
The pool is built once; the final selector does not regenerate proposals after
each extraction. Score-evaluation chunk size is an independent execution setting.

## Scoring and greedy source selection

Shared helpers predict distances and find the nearest currently available peak
at each survey position. For one candidate, `errors` is the array of absolute
differences between predicted and nearest available ranges; a position with no
available peaks has infinite error. `N` is the number of all survey positions
for that AP, including empty fingerprints. Count at most one supporting
observation per survey position.

For `SupportAndResidualScorer`, let `tolerance` be `inlier_tolerance_m` and
`smooth_scale = smooth_scale_factor * tolerance`. The score is:

```python
support = np.count_nonzero(errors <= tolerance)
closeness = np.exp(-0.5 * (errors / smooth_scale)**2).sum()
penalty = np.median(np.minimum(errors, median_clip_factor * tolerance)) / tolerance
score = support + smooth_weight * closeness - median_penalty_weight * penalty
```

All terms use the full survey. The default coefficients are SAPP's preset;
the closeness contribution can outweigh a difference in support count. It is
not merely a tie-breaker. `SupportFirstScorer` instead computes
`support + tie_break_weight * closeness / (N + 1)`. Restrict `tie_break_weight`
to `[0, 1]`, so an extra supporting position always takes priority. Zero removes
the smooth bonus. Scales and tolerances must be positive. These are configured
range scales in metres, not estimated noise standard deviations.

The two default finishers are explicit compositions:

```python
zayets_selector = GreedySourceSelector(
    scorer=SupportFirstScorer(
        inlier_tolerance_m=0.18,
        smooth_scale_factor=1.0,
        tie_break_weight=1.0,
    ),
    refitter=None,
    min_support_positions=5,
    min_support_fraction=0.08,
    duplicate_distance=range_curve_rms,
    duplicate_tolerance_m=0.12,
    consume_support=True,
    max_sources=None,
)

ransac_selector = GreedySourceSelector(
    scorer=score,
    refitter=NearestPeakRefitter(
        association_tolerance_m=0.36,
        weights=partial(gaussian_weights, scale_m=0.36),
        estimator=Multilateration3D(
            loss=soft_l1_loss,
            loss_scale_m=0.18,
            loss_scale_floor_m=0.03,
            max_evaluations=120,
        ),
    ),
    min_support_positions=5,
    min_support_fraction=0.08,
    duplicate_distance=range_curve_rms,
    duplicate_tolerance_m=0.12,
    consume_support=True,
    max_sources=None,
)

zayets = FilteredReconstructor(
    reconstructor=zayets_candidates,
    selector=zayets_selector,
)

ransac = FilteredReconstructor(
    reconstructor=ransac_candidates,
    selector=ransac_selector,
)
```

The selector operates per AP using a fresh working copy or availability mask:

1. Score remaining candidates against currently available peaks and choose the
   best. The support requirement is
   `max(min_support_positions, ceil(min_support_fraction * N))`.
   The proposed stopping rule is: stop if the best candidate lacks that
   support, or no candidates remain. This is not a claim that every lower-ranked
   candidate has insufficient support under every possible scorer.
2. Disable that proposal. If `refitter` is supplied, find the nearest available
   peaks within its association tolerance, calculate weights, and refit. A failed
   refit retains the original position. Check support again afterward.
3. Reject candidates with insufficient support or too close to an already
   selected source under `duplicate_distance`.
4. Accept the source. If `consume_support=True`, mark its nearest peak unavailable
   at each supporting position. Other peaks remain available, and saved survey
   measurements are unchanged. Suppress remaining duplicate proposals.
5. Repeat with updated scores, until the stopping rule or an explicit
   `max_sources` limit is reached. `None` imposes no output-count cap.

`range_curve_rms` is the RMS difference between two candidates' predicted ranges
over all survey positions, not their coordinate separation. Inject another
distance function to study a different duplicate rule. Evidence consumption is
a local conditional; replacing the whole selection procedure uses another
selector, without method-name branching.

`NearestPeakRefitter` associates and weights observations, then delegates fitting
to its estimator. The robust estimator uses the larger of `loss_scale_m` and
`loss_scale_floor_m`. A SAPP-derived preset can derive the association and weight
scales as twice the support tolerance, and the loss scale from that tolerance
with the 0.03 m floor. Make those preset factors/floor explicit; the composed
API also permits studying the resulting scales independently.

With `refitter=None`, accepted positions are a subset of the supplied positions.
With refitting, accepted positions may change before evidence is removed.
`RansacReconstructor` generates and retains proposals; it does not own this
refitter or perform final selection. `FilteredReconstructor` passes its output
to `GreedySourceSelector`, which owns the optional refitter and can also select
proposals from other reconstructors. A separate posthoc refiner can adjust the
completed map after this selection finishes.

## Exchanging clustering and finishers, including MPUrge

### Remove agglomerative complete linkage

Decision on 2026-10-25: remove the `agglomerative` candidate-clustering variant
from the redesigned implementation, presets, hyperparameter search spaces, and
new-experiment viewer choices. Do not retain it as an optional experimental
backend. Do not port its former dedicated `complete_link.py`
implementation or full-hierarchy cache path.

The saved initial presets showed no mean OSPA wins over greedy median across
eight MPUrge variants and two tracks: ten losses and six ties. Those results,
together with its [resource cost](memory_safety.md), do not justify carrying the
variant into the redesign. These are preset comparisons, not a claim that the
algorithm is inferior for every possible configuration.

Retain `GreedyCompleteLink` with configurable mean/median representatives and
the separate DBSCAN alternative. Greedy complete-link insertion does not build
the removed agglomerative hierarchy. Zayets' constrained single-linkage grouping
is also a separate operation and remains part of its candidate generator.

Historical agglomerative results keep their original method labels in the old
artifacts. The redesign does not need a legacy reader or compatibility layer.
Reject requests to run the removed variant in the new implementation; never
silently substitute greedy clustering.

### Compose retained clustering and selection components

Clustering is an optional wrapper and need not be embedded in Zayets or RANSAC:

```python
clustered_candidates = ClusteredReconstructor(
    reconstructor=zayets_candidates,
    clusterer=GreedyCompleteLink(radius_m=0.5, representative=median),
)

zayets_clustered = FilteredReconstructor(
    reconstructor=clustered_candidates,
    selector=zayets_selector,
)

mpurge_consensus = FilteredReconstructor(
    reconstructor=mpurge,
    selector=zayets_selector,
)
```

The same selectors can finish outputs from any compatible reconstructor; names
such as `zayets_selector` above identify preset settings, not restrictions on
their input. For example, use `ransac_selector` with Zayets to study robust
refitting, or `zayets_selector` with RANSAC to study support-first ranking and
fixed candidate positions. Changing only one component isolates that variation.
Clustering returns per-AP representatives and preserves provenance references;
the subsequent selector finds supporting peaks from the survey itself.

MPUrge's existing relative-support rule stays inside MPUrge, alongside the
membership evidence it needs. After candidate clustering, it unions original
`(survey_position, peak_index)` memberships and counts distinct survey positions.
Per AP, it keeps support values at least `beta * mean_support`, with default
`beta=1.0`. It does not recheck those memberships against the cluster's final
representative, look for additional supporting observations across the survey,
or consume peaks. Several retained candidates can inherit evidence from the
same survey position. MPUrge's within-pair Urge removal is a separate operation.

The wrapper example above deliberately applies survey-based selection **after**
complete MPUrge, including its existing clustering and relative-support filter.
It cannot recover positions discarded by the inner method. To replace MPUrge's
filter, supply an explicitly defined inner variant returning positions before
that filter. Additional outer clustering would likewise be a separate operation
if the inner method has already clustered its candidates. Keep composition order
visible in the experiment definition.

Do not treat all provenance memberships as interchangeable support. RANSAC's
three sampled observations explain how a proposal was generated; they are not
its survey-wide support. A generic inherited-membership filter would require a
richer evidence contract. The agreed `FilteredReconstructor` boundary avoids
that requirement: selectors needing only source coordinates and survey data do
not read the inner method's groups or audit payloads.

Keep numerical safeguards such as rank tests, near-zero intersection tolerances,
and optimizer stopping tolerances in configurable numerical solver options,
separate from ordinary scientific hyperparameters. Mathematical requirements
remain invariants: the three-sphere solver needs three noncollinear positions,
and the current unrestricted 3D refitter requires at least four observations
with usable noncoplanar geometry. Selecting a different model requires a
different solver rather than disabling those requirements.

Execution owns caching and instrumentation for these compositions. Reusing a
Python object alone does not guarantee reuse of computation. Identical inner
fits should share their output by computation identity across wrappers; selectors
key the source coordinates/AP identities and survey values they actually consume,
not unrelated upstream diagnostics. Save each stage once with references to its
inputs, following the [cache design](computation_cache_design.md). Keep the
scientific and instrumented implementations readable side by side.

## Specific refactor from the former consensus.py

The scientific extraction described below is implemented in
[ransac.py](../src/indoor_vt/science/ransac.py),
[zayets.py](../src/indoor_vt/science/zayets.py), and
[selection.py](../src/indoor_vt/science/selection.py). The old module is retired.
The requirements remain here to explain the separation.

Replace the large `if c.method == "ransac": ... else: ...` proposal-generation
branch in the current `ConsensusReconstructor.fit` by extracting the proposed
`RansacReconstructor` and `ZayetsReconstructor`. Compose each with
`FilteredReconstructor` and the appropriate `GreedySourceSelector` settings.
Keep candidate coordinates, original observation identities, and scientific
fit/proposal diagnostics as stage outputs, without making selectors depend on
the generator-specific evidence. Do not force random three-sphere proposals
through an artificial matching/grouping interface.

Complete the refactor beyond this one branch: `score(..., mode)` currently uses
different consensus formulas, and the selection loop also branches on method
to refit RANSAC candidates. Extract `SupportAndResidualScorer` and
`SupportFirstScorer`, and inject the optional `NearestPeakRefitter`. Named presets
assemble the intended combinations; the selector must not inspect the inner
reconstructor's method name. Range-prediction, nearest-residual, and numerical
fitting helpers remain shared functions.

In the RANSAC preset, `GreedySourceSelector` invokes its injected refitter before
acceptance, changing the evidence consumed by later iterations. Keep this
dependency outside `RansacReconstructor` and preserve its timing within the
selector. Posthoc refinement instead runs on the completed map. The Zayets
preset uses `refitter=None`, keeping candidate coordinates fixed during selection.

Implement the scientific behavior specified here, including both signed sphere
solutions, proposal retention, scoring, and optional evidence consumption.
Choose and record deterministic seed derivation and tie handling for the new
implementation; historical output equality is not a requirement. RANSAC's
hypothesis batch size affects which proposals survive and belongs to the
computation identity. It is not interchangeable with an execution-only chunk
size used to evaluate an unchanged candidate set. Validate the new contracts
directly where useful, without comparing maps to the old study or writing
migration/equivalence tests.

Paper/implementation distinctions in the previous study remain in
[methods.md](methods.md); they are historical context, not compatibility requirements.

## TODO: clarify method explanations in the visualizer

Implement the Zayets walkthrough in the new
[viewer](../src/indoor_vt/execution/web/index.html), using the prior
[method-catalog.json](../web/results-explorer/method-catalog.json) as context, and review
the rendered Method tab. Apply these explanation rules to other methods too:

- [ ] Lead each step with a short, concrete description of what happens to its
  inputs. Put mathematical and implementation details after that description.
  Terms such as "smooth residual score" or "consuming evidence" are not
  sufficient explanations on their own.
- [ ] Explain shorter-cluster alignment directly: "Within each cluster group,
  average the longest equal-length clusters elementwise to form template A;
  match peaks from each shorter cluster to distinct positions in A, minimizing
  the sum of absolute range differences (L1 costs)." Make clear that these are
  range clusters, not whole fingerprints, and that original measurements remain
  unchanged for fitting. Use one small example only if needed.
- [ ] Define every symbol beside its first use, before asking the reader to
  interpret a formula. In consensus scoring, `v` is a candidate VT position;
  `e_i(v)` is the absolute discrepancy between its predicted range at survey
  position `i` and the nearest currently available measured range there;
  `N` counts all survey positions for that AP, including empty fingerprints;
  in the current implementation, `sigma` is `inlier_scale_m`, used for both the
  support tolerance and smooth-score scale. In the proposed configurable API,
  distinguish `inlier_tolerance_m` from
  `smooth_scale_factor * inlier_tolerance_m`; neither is an estimated noise
  standard deviation. Define additional symbols and their units locally.
- [ ] Explain each score's purpose before its formula. For `SupportFirstScorer`,
  count supporting positions first, then use closer range agreement to break
  equal-support comparisons. Explain that dividing the smooth sum by `N + 1`
  keeps the bonus below one. For `SupportAndResidualScorer`, explain the support
  count, Gaussian closeness bonus, and clipped median-error penalty separately;
  its default 0.35 closeness weight is not restricted to breaking support ties.
- [ ] Explain evidence consumption as an explicit sequence: accept a candidate,
  mark its nearest supporting peak unavailable at each supporting position,
  recompute other candidates' support from the remaining peaks, then repeat.
  Show a small before/after example. Other peaks in the same fingerprint stay
  available; the saved measurements are unchanged. Distinguish this from
  duplicate-candidate suppression and MPUrge's non-consuming support filter.
- [ ] Keep neighborhood scope, range-cluster grouping, peak assignment, and
  VT-candidate selection distinct. Retain the existing paper/implementation
  labels, while keeping the basic operation understandable without reading
  those provenance details or opening the source code.

## Fingerprint localization and MPUrge-MAP

Apply the same component design to `FingerprintLocalizer`. The current
`InitialConfig.method` branches across matcher construction, fingerprint scoring,
ranking, weighting, and fallback behavior. Replace that distributed method
selection with an explicit composition:

```text
Score query against each survey fingerprint
-> combine scores across APs
-> select neighboring references
-> estimate position from their coordinates
```

Two component boundaries capture the useful variations:

| Component | Responsibility |
|---|---|
| `FingerprintScorer` | Score two fingerprints from the same AP; declare whether higher or lower is better and define empty-input/no-match semantics. |
| `NeighborEstimator` | Select references from combined scores and estimate position from survey coordinates; own weighting, tie handling, zero-score handling, and fallbacks. |

Keep AP-score aggregation in the localizer; the current rule averages channel
scores for each reference. Validate that scorer and estimator score directions
are compatible.

Use distinct `MCAScorer`, `ChamferScorer`, and `MatchBasedScorer` implementations.
MCA produces a similarity to maximize; Chamfer and match-based scoring produce
dissimilarities to minimize. `MatchBasedScorer` reuses a `PairwiseMatcher` to
obtain peak correspondences, then reduces their costs to a fingerprint score.
Coverage normalization remains a local option within that scorer. Empty-input
checks remain ordinary conditionals.

For example, the existing SAPP baseline setting becomes:

```python
localizer = FingerprintLocalizer(
    scorer=MatchBasedScorer(
        matcher=MPUrgeMatcher(
            half_window=2,
            alpha=0.7,
            size_unification_norm=absolute_difference,
            normalized_pattern=True,
        ),
        coverage_penalty=True,
    ),
    estimator=DistanceWeightedKNN(neighbors=3, power=1),
)
```

`DistanceWeightedKNN` selects the lowest dissimilarities and weights reference
coordinates by inverse score. MCA instead composes `MCAScorer(epsilon_m=1)`
with `SimilarityWeightedKNN(neighbors=3)`, which selects the highest similarities
and weights by those scores. Extracting only the scoring branch would leave
these distinct estimation rules mixed together.

Expose `MPUrgeMAP(half_window=2, alpha=0.7, neighbors=3)` as a named preset
constructing the composition above. The recognizable method name identifies
the assembled localizer; its components remain independently usable. The
plain MPUrge localization preset uses unnormalized pattern costs and no coverage
penalty. Neither localizer constructs a VT map.

Distinguish the algorithm from its parameter preset and interpretation:

- The MPUrge-MAP paper's evaluated setting is `half_window=6`, `alpha=0.7`,
  `neighbors=3`; `half_window=2` above comes from the SAPP baseline. Inverse-score
  weighting corresponds to `power=1`.
- With `coverage = n_matches / max(n_query_peaks, n_reference_peaks)`, our
  `coverage_penalty=True` computes `mean_match_cost / coverage`. The paper's
  Equation (5) and Algorithm 1 instead print multiplication by coverage, while
  the prose says low coverage should increase dissimilarity. Division is the
  repository's documented interpretation of that stated intent. A literal
  multiplication variant must be identified separately in experiments.

Use the same readable/execution separation described above. Cache matches and
scores by their computation identities: changing neighbor count or weighting
reuses scores; changing only the coverage rule reuses matches. Changing matcher
normalization changes the matching computation. Define empty-input and fallback
behavior explicitly in the new components; both readable and execution versions
follow those contracts without requiring historical-output compatibility.

Evidence: MPUrge-MAP Sections III-B, III-C, and IV-A in the
[local paper](../../Majid_papers/MPUrge-MAP_A_White-Box_Multipath-Assisted_Positioning_for_Indoor_Localization_in_6G_Networks.pdf),
the [SAPP comparator specification](../../publication/sapp/paper/supplement.tex),
and current [localization.py](../src/indoor_vt/science/localization.py).

Historical behavior and implementation choices are documented in [methods.md](methods.md).
