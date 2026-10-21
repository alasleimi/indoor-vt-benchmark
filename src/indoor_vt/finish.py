"""Keep native generation active, derive completed channels, then audit the dataset.

Run once per namespace. --wait-pid attaches to an already running simulator
driver, avoiding overlapping native jobs. A local OS lock prevents two finishers.
"""

import argparse
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .audit import audit_dataset
from .dataset import materialize
from .provenance import SeedBook, read_json, write_json
from .replay import replay_pilot


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def existing_process(pid):
    if pid is None:
        yield lambda: False
        return
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
        if not handle:
            if ctypes.get_last_error() == 87:  # Process has already exited.
                yield lambda: False
                return
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield lambda: kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    else:

        def active():
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False

        yield active


def finish(root, namespace, qd_root, matlab, workers, wait_pid=None, interval=15):
    root = Path(root).resolve()
    base = root / "data" / namespace
    design = read_json(base / "design.json")
    seeds = SeedBook(design["seeds"]["root"], namespace)
    expected = len(design["rooms"]) * 22
    run = root / "runs" / namespace / "generation"
    run.mkdir(parents=True, exist_ok=True)
    done, errors = set(), {}
    child, child_log = None, None
    round_number = 0
    with exclusive_lock(run / "finisher.lock"), existing_process(wait_pid) as prior_active:
        while True:
            receipts = sorted((base / "scenarios").glob("*/*/*/receipt.json"))
            native_active = prior_active() or (child is not None and child.poll() is None)
            if len(receipts) < expected and not native_active:
                if child_log is not None:
                    child_log.close()
                stamp = time.strftime("%Y%m%d_%H%M%S")
                child_log = (run / f"native_{stamp}_{round_number:03d}.log").open("w")
                command = [
                    sys.executable,
                    "-m",
                    "indoor_vt.cli",
                    "--root",
                    str(root),
                    "simulate",
                    "--namespace",
                    namespace,
                    "--qd-root",
                    str(qd_root),
                    "--matlab",
                    str(matlab),
                    "--workers",
                    str(workers),
                ]
                child = subprocess.Popen(
                    command,
                    cwd=root,
                    stdout=child_log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                round_number += 1
                native_active = True
            # Only one frontend job runs at a time, leaving memory for native workers.
            pending = [p for p in receipts if str(p) not in done]
            if pending:
                path = pending[0]
                try:
                    materialize(path.parent, design["config"], seeds)
                    done.add(str(path))
                    errors.pop(str(path), None)
                except Exception:
                    errors[str(path)] = traceback.format_exc()
                    # Try the other completed channels before retrying this one.
                    done.add(str(path))
            status = {
                "namespace": namespace,
                "native_complete": len(receipts),
                "native_expected": expected,
                "native_active": native_active,
                "derived_complete": len(done) - len(errors),
                "derived_expected": expected,
                "errors": errors,
                "native_retry_rounds": round_number,
                "updated_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "finished": False,
            }
            write_json(run / "status.json", status)
            if len(receipts) == expected and len(done) == expected and not native_active:
                if child_log is not None:
                    child_log.close()
                if errors:
                    raise RuntimeError(
                        f"Frontend errors require investigation: {run / 'status.json'}"
                    )
                # Use the released native worker capacity for one independent
                # same-seed pilot replay before the final full archive audit.
                try:
                    status["native_replay"] = replay_pilot(root, qd_root, matlab)
                    result = audit_dataset(root, namespace)
                except Exception:
                    status["terminal_error"] = traceback.format_exc()
                    write_json(run / "status.json", status)
                    raise
                status["finished"] = True
                status["native_frames"] = result["frames"]
                write_json(run / "status.json", status)
                print(status, flush=True)
                return status
            if not pending:
                # Short polling interval; no long tool-side blocking wait is needed.
                time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--namespace", choices=["development", "publication"], default="development"
    )
    parser.add_argument("--qd-root", type=Path, required=True)
    parser.add_argument("--matlab", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    finish(
        args.root, args.namespace, args.qd_root.resolve(), args.matlab, args.workers, args.wait_pid
    )


if __name__ == "__main__":
    main()
