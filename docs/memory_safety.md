# Memory fix and stopped campaign, 2026-10-22

The campaign is **stopped at the user's request**. Revision v6 is prepared for
an explicit restart; no production worker was launched to validate these changes.
All 108 completed presets, original 27 calibration trials and cached artifacts
remain available. The two outstanding presets and the scientific queue are unchanged.

## What caused the pressure

The stop-time snapshot alone does not establish an unbounded memory leak. The
scientific worker then had 721 MB resident and 866 MB private memory. However,
the v5 complete-link backend mapped a 6.32 GB distance file for 39,742 candidates.
Its prior isolated validation reached 6.32 GB resident memory. A subsequent
controlled probe showed that a hard process working-set limit still allowed
substantial Windows file-cache pressure. File-backed allocation was therefore
insufficient to bound the machine's memory use.

Long-lived workers also retained evaluator caches and native/Python allocation
arenas across tasks. Audit serialization created a complete JSON string and
encoded byte buffer alongside the already populated audit structure.

## Changes

- Complete-link uses the existing condensed-array implementation for banks
  fitting within 256 MiB. Larger banks use a bounded cache of distance rows.
  Evicted rows are recomputed from original cluster leaves. No distance file
  is created. The budget is a storage policy, not a scientific hyperparameter.
- The large backend preserves ascending active-slot scans, strict comparisons,
  predecessor tie preference, stable height sorting and SciPy's label correction.
  Complete-link distances are maxima of original leaf-pair Euclidean distances;
  recomputation and resident updates give identical distances. Bookkeeping is
  linear in the number of candidates. Distance storage is at most 256 MiB or
  two rows, whichever is larger. Eviction can increase computation; no universal
  quadratic-time bound is claimed for the bounded-cache backend.
- One fresh process handles each preset, search turn or other scheduled job.
  Process exit releases all its caches and heaps. Persistent verified disk
  caches still permit reuse across tasks. The default concurrency remains one.
- Every worker installs Windows limits of **2 GiB resident working set** and
  **4 GiB private commitment**. Failure to install limits aborts the task.
  Allocation-limit failures are recorded without automatically retrying the
  same job under the same limit. Inputs and candidates are never truncated.
- Audit JSON is encoded into bounded chunks before gzip compression, preserving
  the exact decompressed JSON bytes. The audit structure itself remains in memory.
- Status and `memory.jsonl` record live and peak worker memory. The runner closes
  finished process handles and clears running status when it exits on an error.

The resident cap alone does not constrain Windows' filesystem cache. Removing
the quadratic mapped file addresses the observed source of that pressure.

## Validation

The saved 39,742-candidate calibration bank reproduced all 39,741 hierarchy rows
**exactly**, including labels, ties and heights. No test-room data were needed.

| Isolated full-case measurement | v5 mapped file | v6 bounded rows |
| --- | ---: | ---: |
| Runtime, kernels warmed | 195.43 s | 22.14 s |
| Peak resident working set | 6.32 GB | 391.84 MB |
| Peak private commitment | 93.29 MB | 1.36 GB |
| Temporary distance file | 6.32 GB | 0 |

Private commitment includes interpreter/JIT/native allocations and is distinct
from resident physical memory. These are individual runs on the same saved bank,
not an end-to-end campaign speedup measurement. The v6 measurement used the
worker's 2 GiB/4 GiB limits.

All **63 tests pass**, with Ruff clean. Tests compare full hierarchies to SciPy
on random data, duplicates, grids, permutations, several dimensions and numerical
scales, including forced eviction with only two cached rows. They check exact
threshold cuts, overflow rejection, bounded cache capacity, streamed JSON bytes,
actual Windows quota enforcement, and worker replacement between two synthetic
jobs. The implementation audit also matched 288 coarse predictions and all 12
VT methods. Machine-readable evidence is in
[memory_safety_validation.json](memory_safety_validation.json); complete local
records are in `runs/development/diagnostics/memory_fix/`.

## Resume policy

Revision v5 and its source snapshot remain preserved. Revision v6 carries
completed outputs through explicit compatibility receipts and keeps the same
450 jobs, parameters, seeds, phase barriers and search budgets. Shadow work
continues to wait for all main calibration and freezes. The interrupted job is
pending; completed work does not need to be discarded or rerun.

Only after the user requests a restart, run from the main repository:

```powershell
uv run python scripts/run_revision_campaign.py --revision v6 --workers 1
```

The original plan remains outside the repository. No remote repository was created.
