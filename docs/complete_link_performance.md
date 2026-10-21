# Exact complete-link clustering with less private memory

The former implementation called SciPy 1.15.3 `linkage(points, method="complete")`.
This is a standard compiled nearest-neighbor-chain implementation, but it first
computes condensed pairwise distances and then copies them for mutable cluster
distances. For 39,742 candidates, each float64 array needs 6,317,547,288 bytes.
Two arrays therefore require about 12.64 GB before other worker state.

Revision v5 retains the same algorithm and full hierarchy while removing that
copy. The implementation specializes the update to complete linkage and walks
active slots in ascending order using a linked list, avoiding scans over removed
slots. It retains SciPy's strict comparison, preference for the preceding chain
member on ties, stable merge-height sort and union-find label correction.

The Euclidean distance kernel is still SciPy's `pdist`, writing directly into
the workspace. Workspaces up to 256 MiB use one in-memory array. Larger ones use
one float64 temporary file mapping. This threshold controls storage only: no
candidate cap, rounding, sampling, radius pruning or method substitution occurs.
The full hierarchy remains cached and reusable across distance thresholds.

File-backed pages can occupy substantial physical RAM when available, but the
OS can reclaim them against the file. They do not require a second large private
memory commitment. Distance storage on disk and worst-case computation remain
quadratic; the change does not make complete linkage linear-memory in total
storage. The temporary file is deleted on normal completion, exceptions and
Windows process termination, which are tested.

## Verification and measurements

The saved real calibration case has 39,742 candidates. The revised full
39,741-by-4 hierarchy matched the previously saved SciPy hierarchy exactly,
including merge ordering and labels. Thus all cuts, not just the selected
0.5 m cut, are identical. No held-out observations were used.

| Measurement | Result |
| --- | ---: |
| Full real case, file-backed workspace | 195.43 s |
| Peak private committed memory of validation process | 93.3 MB |
| Peak resident working set, including mapped file pages | 6.32 GB |
| Temporary distance file | 6.32 GB |
| 3,000-candidate sample, SciPy | 0.166 s |
| Same sample, revised in-memory implementation | 0.104 s |
| Same sample, forced file-backed implementation | 0.141 s |

Numba kernels were warmed before timings. The large-case reference was already
cached; no same-time cold SciPy run was attempted because it would recreate the
memory pressure. The small-case timings are individual checks, not a campaign
speedup claim. The scientific worker continued separately during validation.

All 58 tests passed, with Ruff clean. Tests cover full hierarchy equality for random points, duplicates, grids, tied
distances, shuffled input and both storage backends, as well as exact-threshold
cuts and adjacent float64 values. The existing 288-coarse-prediction and
12-VT-method audit also matched. Machine-readable evidence is in
`complete_link_validation.json`; detailed local artifacts are in
`runs/development/diagnostics/complete_link/`.

## Source and provenance

The chain and label routines are adapted from the BSD-licensed
[SciPy 1.15.3 source](https://github.com/scipy/scipy/blob/v1.15.3/scipy/cluster/_hierarchy.pyx).
Its copyright and license are retained in
[`third_party/scipy/LICENSE.txt`](../third_party/scipy/LICENSE.txt).
SciPy documents the original time/memory costs in its
[linkage reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).

Old objective, source and stage artifacts remain preserved. Revision v5 uses
explicit equivalence receipts to reuse them. It retains the single-worker
schedule from v4 while resuming the same scientific job queue.
