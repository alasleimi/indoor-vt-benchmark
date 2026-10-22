"""Windows worker quotas and memory measurements; no estimator settings."""

import ctypes
from ctypes import wintypes
import os


DEFAULT_LIMITS = {"working_set_bytes": 2 * 1024**3, "private_commit_bytes": 4 * 1024**3}


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _MemoryCounters(ctypes.Structure):
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


def _kernel():
    if os.name != "nt":
        raise RuntimeError("This campaign's memory quotas require Windows")
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.GetCurrentProcess.restype = wintypes.HANDLE
    library.CloseHandle.argtypes = [wintypes.HANDLE]
    return library


def install_limits(working_set_bytes, private_commit_bytes):
    """Apply hard resident/commit limits to this worker for its lifetime.

    File-backed mappings count toward the resident working set. Their virtual
    size does not count toward private commitment. A failed quota installation
    is an error, never permission to continue with unbounded resources.
    """
    if working_set_bytes < 64 * 1024**2 or private_commit_bytes < working_set_bytes:
        raise ValueError("Require 64 MiB <= working set <= private commit quota")
    kernel = _kernel()
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.SetProcessWorkingSetSizeEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_size_t,
        ctypes.c_size_t,
        wintypes.DWORD,
    ]
    process = kernel.GetCurrentProcess()
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        settings = _ExtendedLimits()
        settings.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        settings.ProcessMemoryLimit = private_commit_bytes
        if not kernel.SetInformationJobObject(
            job, 9, ctypes.byref(settings), ctypes.sizeof(settings)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(job, process):
            raise ctypes.WinError(ctypes.get_last_error())
        # Hard maximum, with no reserved minimum under system memory pressure.
        if not kernel.SetProcessWorkingSetSizeEx(process, 64 * 1024**2, working_set_bytes, 0x6):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        # Association and limits persist until the assigned process exits.
        kernel.CloseHandle(job)
    return dict(working_set_bytes=working_set_bytes, private_commit_bytes=private_commit_bytes)


def memory_snapshot(pid=None):
    kernel = _kernel()
    current = pid is None or pid == os.getpid()
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    process = kernel.GetCurrentProcess() if current else kernel.OpenProcess(0x410, False, pid)
    if not process:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not query(process, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "pid": os.getpid() if current else pid,
            "working_set_bytes": counters.WorkingSetSize,
            "private_bytes": counters.PrivateUsage,
            "peak_working_set_bytes": counters.PeakWorkingSetSize,
            "peak_private_bytes": counters.PeakPagefileUsage,
        }
    finally:
        if not current:
            kernel.CloseHandle(process)
