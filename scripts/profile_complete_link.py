"""Verify full hierarchy equality and record storage/time on saved candidates."""

import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import time

import numpy as np
from scipy.cluster.hierarchy import linkage

from indoor_vt.complete_link import complete_linkage
from indoor_vt.provenance import write_json


def process_memory():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
                "PrivateUsage",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not query(current(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return {
        "peak_private_commit_bytes": counters.PeakPagefileUsage,
        "peak_working_set_bytes": counters.PeakWorkingSetSize,
        "current_private_bytes": counters.PrivateUsage,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--working-set-mib", type=int)
    args = parser.parse_args()
    if args.working_set_mib is not None:
        from indoor_vt.worker_memory import install_limits

        install_limits(args.working_set_mib * 1024**2, 4 * 1024**3)
    points = np.load(args.points)
    complete_linkage(points[:10])  # Compile/load kernels outside measured region.
    complete_linkage(points[:10], memory_limit_bytes=0)
    rows = []
    if args.reference:
        expected = np.load(args.reference)
        started = time.perf_counter()
        actual = complete_linkage(points, scratch_dir=args.output.parent / "scratch")
        elapsed = time.perf_counter() - started
        np.testing.assert_array_equal(actual, expected)
        rows.append(
            {
                "candidates": len(points),
                "condensed_bytes": len(points) * (len(points) - 1) // 2 * 8,
                "distance_cache_budget_bytes": 256 * 1024**2,
                "distance_file_bytes": 0,
                "seconds": elapsed,
                "exact_full_hierarchy": True,
                "reference": str(args.reference),
                **process_memory(),
            }
        )
    else:
        for n in (100, 1000, 3000):
            sample = points[:n]
            started = time.perf_counter()
            expected = linkage(sample, method="complete")
            original_s = time.perf_counter() - started
            for limit in (256 * 1024 * 1024, 0):
                started = time.perf_counter()
                actual = complete_linkage(
                    sample, memory_limit_bytes=limit, scratch_dir=args.output.parent / "scratch"
                )
                elapsed = time.perf_counter() - started
                np.testing.assert_array_equal(actual, expected)
                rows.append(
                    {
                        "candidates": n,
                        "storage": "condensed_memory" if limit else "two_row_cache",
                        "scipy_s": original_s,
                        "revised_s": elapsed,
                        "exact_full_hierarchy": True,
                    }
                )
    write_json(args.output, {"measurements": rows})
    print(rows, flush=True)


if __name__ == "__main__":
    main()
