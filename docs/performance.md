# Exact cache reuse, revision v3

This revision removes repeated computation and loading without changing the
observations, search domains, budgets, grouping rules or numerical objectives.
It follows v2's reuse of per-AP score matrices across A, B, A+B and k values.

## Additional reuse

- MCA and Chamfer share nearest-delay geometry. MCA tolerance changes reuse it
  too. References are batched by length to retain the scalar reduction order.
  The disk cache stores nearest-peak indices and original float64 delays, then
  reconstructs distances with the original subtraction and absolute value.
  Indices widen as needed; this introduces no peak cap or quantization.
- MPUrge group memberships are cached separately. Changing merger, radius or
  support filtering no longer rebuilds the groups before finding cached fits.
  Custom injected grouping implementations bypass this cache.
- Repeated source-map loads read a small coordinate summary instead of inflating
  the entire graph/group/fit audit. Full audits remain intact. A bounded memory
  cache and file size/mtime checks invalidate summaries when artifacts change.
- LEA evaluations reuse earlier iterations with the same inputs and settings.
  A four-iteration evaluation saves the one-, two- and three-iteration results.
  Failure stays terminal and returns the original coarse position, including
  when it happens after a successful iteration.
- Explicit compatibility receipts permit reuse of earlier score and stage caches.
  `scripts/warm_source_stages.py` also recovers matched graphs, groups and raw
  fits already stored in compatible full source audits. It copies recorded
  intermediates; it does not refit or introduce new scientific results.

The default greedy complete-link merger still processes fitted candidates in
ascending x, then y, then z order, breaking exact ties by original index. It
uses the first previously created cluster whose every member is within the
threshold. This order can affect memberships and is unchanged here.

## Verification and measured costs

All 53 tests passed, with Ruff clean. The old/new audit matched 288 coarse
predictions across both tracks and all four localizers, and source coordinates
for all 12 VT entries on a controlled fixture. Full calibration cases matched
1,144,000 MCA/Chamfer scores exactly. Tests also cover empty and ragged profiles,
duplicate peaks, more than 255 peaks, LEA iteration order and later failure,
group reuse, summary invalidation and explicit historical-cache compatibility.
No held-out data was used for these checks.

The following check used 1,000 queries and 286 references for one room/AP/noise
condition per track. Original times come from preserved score receipts; they
are not simultaneous controlled benchmarks. New times include cache I/O.

| Track | Method | Original score computation (s) | New computation and cache I/O (s) | Geometry state |
| --- | --- | ---: | ---: | --- |
| Receiver | MCA | 3.19 | 1.59 | New |
| Receiver | Chamfer | 4.97 | 0.29 | Reused from MCA |
| Direct | MCA | 3.48 | 3.39 | New |
| Direct | Chamfer | 5.51 | 0.36 | Reused from MCA |

Warm final-score matrix reads were approximately 0.04 s; that cache already
existed in v2. These measurements do not establish an overall campaign speedup.
Original historical timings remain unchanged. Prefix timings accumulate the
actual recorded pass computations, not the time to serve a warm cache hit.

A matcher window-precomputation prototype showed only small inconsistent gains
and was left out. A full-distance geometry cache made the first direct MCA run
slower because compression dominated; compact nearest-index storage replaced it.

Machine-readable evidence and source identities are in
[cache_optimization_validation.json](cache_optimization_validation.json). Detailed
local artifacts are retained under
`runs/development/diagnostics/cache_optimization/`.

## Preserving ongoing work

The v2 workers remained active during implementation and validation in a
separate checkout. Revision v3 retains the original 27 MPUrge-MAP trials, prior
presets and all complete per-case artifacts. Its objective summary aliases link
to original v2 files with hashes, rather than rewriting those results. The v2
source and compatibility receipt are archived under
`runs/development/legacy/082d1fcad6ceae22f808e49605c320d4cf421b572efc75286e8631bb49ce2bb5/`.

The migration receipt is `runs/development/revisions/v3/migration.json`.
Stage recovery records its progress in that directory's `recovered_stages.json`.
Revision v3 initially ran two scientific workers. Revision v4 later resumed the
same main-first queue with one worker after a memory failure; see the
[execution record](execution.md). Shadow work still waits until all main
calibration and freezes finish.
