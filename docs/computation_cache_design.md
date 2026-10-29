# Computation identity, caching, and auditing

[Redesign brief](redesign.md) · [Scientific API](mpurge_api_design.md) ·
[Dataset](dataset_generation_design.md). Updated 2026-10-25.

## Code edits and reuse

Whole-implementation hashes are provenance, not numerical-cache dependencies.
Batch keys contain the versioned operation name, injected component code/settings,
numerical inputs and NumPy/SciPy/Numba versions. An unrelated method, search grid,
viewer change or whole-file revision does not invalidate these batches. A change
to numerical behavior in an execution helper must bump its operation revision;
injected component changes are additionally detected by their component identity.

The migration in `execution/cache_compatibility.py` probes existing batch addresses
without copying or rewriting arrays. Its estimator alias names the exact reviewed
before/after class hashes for removal of the loss-scale floor. It applies only when
the old floor was inactive; a previously active floor cannot reuse that result.
Unrecognized operation revisions, changed parameters/inputs, or different numerical
library versions cannot use this migration.

Saved MAP selections can be frozen upstream inputs in a new study without rerunning
the selection search. The new run records the original study version and selected
receipt IDs. Dataset and observation-protocol compatibility remain required.
Completed outcomes stay immutable and readable even after source edits.

## What failed, and the replacement

The first redesign's SQLite database grew to **15.68 GiB** after only 200
objective trials. It held 5.17 million computations and 15.26 million objects.
Only 3.44 GiB was numeric array payload. Group reference lists alone occupied
2.95 GiB; match-output links another 1.30 GiB. Per-pair parameters were copied
millions of times. Per-computation transactions, events, and heartbeat threads
also made an otherwise seconds-long calculation take minutes.

This was primarily a granularity problem, not a claim that SQL cannot store
scientific results efficiently. Numerical execution now uses
[artifacts.py](../src/indoor_vt/execution/artifacts.py): one NPZ per substantial
reusable computation, addressed by its hash on the filesystem. Arrays within a
batch use local integer indices and offsets. There is no file, SQL row, thread,
or long hash for each peak, edge, group member, or 3D point.

The existing database remains untouched because it also contains the frozen
native acquisition. Observation loading opens it **read-only**. New numerical
execution does not write to it. Historical disk use is separate from the new
cache budget.

A packed-cache snapshot measured **98.7% typed-array payload** in 1.11 GiB of
NPZ files. Manifests and array/container headers accounted for the remainder.
The snapshot included more than one implementation build; the shared disk
quota applies to their combined size.

## Dependencies and retention decisions

| Operation | Actual scientific dependencies | Reuse / retention |
|---|---|---|
| Native NIST acquisition | Scene, APs, receiver positions, simulator/settings/seeds | Frozen input; never rerun for HPO |
| Direct ranges | Native delays, noise level, stable per-position seed | Cheap vector arithmetic; bounded RAM batches, no per-frame disk cache |
| Survey pairs and scopes | Survey coordinates, injected selection rules | RAM; reused across APs and every matcher configuration |
| MPUrge accepted matches | Both range lists, half-window, alpha, normalization, optional unification cost; separation only when triangle checking is active | Disk and RAM, one packed pair bank per AP and matcher |
| Scoped groups and candidate fits | Relevant pair blocks, scope sets, grouping policy, coordinates/ranges and estimator settings | Packed batch on disk/RAM; reused across all clustering and support settings |
| Spatial memberships and representatives | Candidate coordinates, cluster radius/policy and representative callable | Packed disk/RAM batches; reused across beta, unification modes and joint trials |
| Support selection | Cluster support counts and beta | Calculate directly; no persistent cache |
| MAP match statistics | Query/reference range lists, matcher settings | Disk/RAM: mean accepted cost and count per pair, not all matching traces |
| MAP coverage and AP aggregation | Statistics, original peak counts, coverage policy, AP inclusion | Cheap array arithmetic; calculate directly |
| MAP neighbor estimates | Scores, reference coordinates, k and power | RAM; changing k/power never rematches fingerprints |
| SMART-LEA iteration | Observed ranges, source coordinates, current positions, matcher and estimator; original initializer for failure fallback | One packed iteration batch on disk/RAM; 1/2/4-iteration searches share prefixes |
| Predicted ranges inside SMART-LEA | Current position and source coordinates | Calculate directly; cheaper than separately addressing them |
| Trial metrics | Output coordinates, evaluator truth, metric settings and case weighting | All scalar summaries retained in a stage log |
| Detailed pair trace | Two selected fingerprints and matcher | Recreate on demand; accepted edges remain available in the pair bank |

MPUrge's half-window cannot generally be applied as a cheap postprocessing
parameter: changing it changes the matches and subsequent purge/urge rounds.
Likewise alpha changes matching, not just a final weighting of an existing
score table. Local windows are rebuilt on the remaining peaks in each round.

Window size and alpha also interact scientifically. The raw pattern term sums
3, 10, 21, 36, 55, 78 or 105 terms for half-windows 1-7. Standard MPUrge and
SMART-LEA use that raw sum; MAP normalizes by the number of terms. Thus the same
alpha does not give the same relative pattern/range influence at every window
size. Current selection is optimal only within the declared finite grid; wider
windows may justify denser alpha values near 1 in a later extended search.

The pattern-distance kernel uses a shared scientific identity: for sorted windows
A and B, put d[k] = A[k] - B[k]. The original sum of discrepancies between every
pair of within-window distances equals the sum of |d[i] - d[j]| over i < j.
After sorting d, gap k contributes k * (width-k) times that gap. This removes
repeated pairwise window-distance arithmetic, with one reused scratch array.
Focused checks cover the defining cost for every half-window 1-7. Floating-point
summation order changes, so the optimized implementation has a new version and
selection is rerun under that version.

Two proven inactive dependencies are normalized in execution:

- If alpha is 1, the pattern term is unused: half-window and pattern
  normalization do not distinguish computations.
- If every comparison in a batch has equal input lengths, size unification is
  unused. Its callable does not distinguish those computations.

The clean survey currently has 25 peaks at every position. Thus MPUrge
unification on/off shares its clean computation. Query fingerprints can have
different lengths, so that conclusion does **not** automatically apply to MAP
or SMART-LEA.

## Packed matches and scope processing

A pair bank stores fingerprint pairs, per-pair offsets, peak-node endpoints,
costs, and offsets mapping peak-node IDs back to fingerprint/peak indices.
Node endpoints use int32 when possible; costs and geometry remain float64.

The HPO executor builds the union of comparisons needed by the configured
neighborhood sizes. Each scope reads only the blocks permitted by its own pair
selector. A shared bank does not turn local grouping into global grouping.

DSU reads those blocks directly. It does not rebuild adjacency dictionaries,
expand fingerprint IDs onto every edge, or convert the same scope through
Python row lists and NumPy arrays repeatedly. Conflict resolution still keeps
one peak per fingerprint by degree, mean edge cost, then peak index, without
regrouping after pruning. Identical observation groups are deduplicated before
fitting.

Measured clustering cost averaged roughly 0.1 seconds in the broad grid, while
its packed output is small. It is therefore persisted as well as retained in
RAM; limiting it to RAM caused avoidable work in the second unification search
and joint search.

Batching is intentional: persisting a separate identity for each sub-millisecond
fit costs more than it saves here. The expensive repeated work across radius,
beta, neighbor-estimation settings, and iteration limits is shared. A future
resolution study can introduce chunked cross-survey pair banks if measurements
justify their indexing cost; it must not restore one durable object per match.

## Identity and concurrency

Computation keys include the operation, effective component settings/callables,
consumed input arrays or upstream computation IDs, and implementation/numerical
library versions. Run labels, AP display names, tuning objectives, and trial
numbers are not computational inputs. Seeds enter through the observed values
or explicitly in stochastic operations.

The current implementation revision is deliberately conservative across shared
scientific helpers. It prevents stale reuse after helper changes, but a code edit
can invalidate more batches than mathematically necessary. There is no old-study
compatibility mechanism.

Cache I/O housekeeping is excluded from that numerical revision. During the
current search, replacing quota-directory scans with an incremental inventory
left the numerical source, serializers, experiment, data and configuration
unchanged. Their hashes and the 5,913-trial checkpoint are recorded in
`rectangle-clean-selection-v4/cache-io-revision.json`; completed scalar trials
were retained. Selected calibration outcomes are recomputed and checked against
their original selection losses in the completion audit.

Fixed lock shards provide single-flight computation across threads/processes.
OS locks release when a worker exits. NPZ publication uses a temporary file and
atomic replacement. A separate short lock enforces the disk quota; there is no
heartbeat thread or durable event row per cache hit. One run lock protects trial
log appends and resume.

Quota accounting uses an incremental size/recency inventory. Writers append a
small batch-level delta; other processes read only new deltas. A dirty marker
triggers one recovery scan after an interrupted publication. The inventory is
occasionally compacted, and access times are refreshed lazily during eviction.
No directory walk or sorting of every file occurs for each new batch or status
request. At 8,361 batches, a 256 KiB publication measured 13 ms median / 17 ms
maximum; the one initial inventory build took 1.21 s. Inventory metadata was
about 0.75 MiB for 1.69 GiB of NPZ files.

## Codec and resource budgets

Default numerical cache: **2 GiB disk**, **256 MiB RAM**, with **5 GiB free disk**
reserved. Least-recently-used disk batches are evicted under the budget; results
larger than the budget are computed without persistence. Low disk skips numerical
cache writes. Selected-outcome storage is capped separately at **256 MiB**;
replayed audit results at **128 MiB**. Limits are configuration values.

NPZ currently uses uncompressed arrays. On a representative 2.75 MB batch,
in-memory serialization/deserialization took 3.7/2.3 ms. Deflate level 1 reduced
it to 1.76 MB but took 63/14 ms; level 6 took 169/14 ms for 1.70 MB. These are
codec measurements, not disk throughput claims. The fixed quota makes the
faster codec a reasonable current choice. Compression can be reconsidered if
disk bandwidth or retention, rather than calculation, becomes the bottleneck.

## Audit policy and outcome analysis

- Retain **every trial's** parameters, scalar metrics, timing and replay identity
  in one append-only JSONL log per search stage. Frozen upstream settings are
  stored once per stage. Interrupted trailing log rows are safely discarded.
- Retain selected configurations and their query predictions, errors, source
  maps and evidence references as typed NPZ outcomes, at calibration and each
  evaluation noise level.
- Do not pin every intermediate version generated by HPO. Reusable batches
  remain while useful under the cache quota; recreate evicted batches from the
  recorded input/version/configuration.
- The viewer can inspect any completed trial, open packed match/group/fit/cluster
  batches, and recreate a particular survey/survey or query/survey pair trace.
  A historical run with a different implementation requires that version for
  exact replay; never silently replay it using changed science.

## Scientific code versus execution code

DSU, packed correspondence records, deterministic neighborhood ties, the pattern
cost identity, alpha=1 pattern skipping, and the single SMART-LEA iteration belong in shared scientific
helpers. They are useful without a cache and preserve readable composition.

Effective cache-key normalization, match-bank reuse across a search,
serialization, retention limits, timers, replay, and trial logging belong in
execution. Scientific orchestration stays runnable without a store. Focused
tests compare the two new implementations and their scientific contracts;
migration and old-study equivalence tests remain out of scope.
