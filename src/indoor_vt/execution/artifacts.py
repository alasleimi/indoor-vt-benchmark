"""Bounded, computation-addressed numerical batches; no per-peak records.

NPZ uses uncompressed typed arrays for predictable read/write latency. Audit
artifacts are replayed from a run receipt, rather than pinning a graph per trial.
"""
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np
import numba
import scipy

from .store import canonical, component_spec, digest
from .cache_compatibility import legacy_specs


@dataclass(frozen=True)
class ArtifactRef:
    id: str


@lru_cache(maxsize=1)
def implementation_id():
    root = Path(__file__).parents[1]
    paths = sorted((root / "science").glob("*.py"))
    paths += [Path(__file__).with_name(n + ".py") for n in
              ("reconstruction", "localization", "presets")]
    paths += [root / "acquisition" / (n + ".py") for n in ("observations", "channel")]
    paths += [root / "metrics.py"]
    return digest({"source": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                   "numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__})


def identity(value):
    if isinstance(value, ArtifactRef):
        return {"computation": value.id}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        h = hashlib.sha256(canonical({"dtype": array.dtype.str, "shape": array.shape}))
        h.update(array.tobytes())
        return {"array": h.hexdigest()}
    if isinstance(value, dict):
        return {str(k): identity(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [identity(v) for v in value]
    return component_spec(value)


def _encode(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported")
        key = "a" + str(len(arrays))
        arrays[key] = value
        return {"array": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, ArtifactRef):
        return {"computation": value.id}
    if isinstance(value, dict):
        return {str(k): _encode(v, arrays) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(v, arrays) for v in value]
    return value


def _decode(value, arrays):
    if isinstance(value, dict):
        if set(value) == {"array"}:
            return arrays[value["array"]]
        return {k: _decode(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, arrays) for v in value]
    return value


def write_npz(path, value, *, metadata=None):
    arrays = {}
    tree = _encode(value, arrays)
    arrays["_manifest"] = np.frombuffer(canonical({"value": tree, "metadata": metadata or {}}), dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(.01 * (attempt + 1), .1))
    finally:
        temporary.unlink(missing_ok=True)


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(archive["_manifest"].tobytes())
        arrays = {k: archive[k] for k in archive.files if k != "_manifest"}
    for array in arrays.values():
        array.flags.writeable = False
    return _decode(manifest["value"], arrays), manifest["metadata"]


def memory_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(memory_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(memory_bytes(v) for v in value)
    return 64


_LOCKS = {}
_LOCK_GUARD = threading.Lock()


@contextmanager
def file_lock(path):
    """OS releases the lock if a worker exits; fixed shards avoid tiny lock files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK_GUARD:
        thread_lock = _LOCKS.setdefault(str(path.resolve()), threading.RLock())
    with thread_lock, path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.05)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class ArtifactCache:
    def __init__(self, directory, *, disk_bytes=2 * 1024**3, ram_bytes=256 * 1024**2,
                 free_bytes=5 * 1024**3):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.disk_bytes, self.ram_bytes, self.free_bytes = disk_bytes, ram_bytes, free_bytes
        self.memory, self.memory_size = OrderedDict(), 0
        self.memory_lock = threading.RLock()
        self.stats = Counter()
        from .inventory import DiskInventory
        self.inventory = DiskInventory(self.directory)

    def path(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid computation ID")
        return self.directory / key[:2] / (key + ".npz")

    def _remember(self, key, value):
        with self.memory_lock:
            self._remember_locked(key, value)

    def _remember_locked(self, key, value):
        if key in self.memory:
            self.memory_size -= memory_bytes(self.memory.pop(key))
        size = memory_bytes(value)
        if size <= self.ram_bytes:
            self.memory[key] = value
            self.memory_size += size
        while self.memory_size > self.ram_bytes:
            _, removed = self.memory.popitem(last=False)
            self.memory_size -= memory_bytes(removed)

    def get(self, key):
        with self.memory_lock:
            if key in self.memory:
                self.memory.move_to_end(key)
                return self.memory[key]
        path = self.path(key)
        value, _ = read_npz(path)
        try:
            os.utime(path, None)
        except (FileNotFoundError, PermissionError):
            pass  # Another worker can evict an already loaded batch.
        self._remember(key, value)
        return value

    def _publish(self, key, value, metadata):
        # The global lock protects the quota, not the calculation.
        estimate = memory_bytes(value) + 32768
        if estimate > self.disk_bytes:
            self.stats["oversize_skipped"] += 1
            return
        with file_lock(self.directory / "locks" / "budget.lock"):
            inventory = self.inventory
            inventory.sync()
            free = shutil.disk_usage(self.directory).free
            inventory.begin()
            # On an exception the dirty marker remains, causing one recovery scan.
            while inventory.total + estimate > self.disk_bytes or free - estimate < self.free_bytes:
                removed = inventory.evict_one()
                if not removed:
                    break
                free += removed
                self.stats["evicted"] += 1
            if free - estimate < self.free_bytes:
                self.stats["low_disk_skipped"] += 1
            elif inventory.total + estimate > self.disk_bytes:
                self.stats["busy_cache_skipped"] += 1
            else:
                write_npz(self.path(key), value, metadata=metadata)
                inventory.put(key)
                # Enforce the actual ZIP size, including its headers.
                if inventory.total > self.disk_bytes:
                    inventory.remove(key)
                    self.stats["oversize_skipped"] += 1
            inventory.commit()

    def compute(self, operation, parameters, inputs, function, *, persist=True):
        spec = {"operation": operation, "parameters": component_spec(parameters),
                "inputs": identity(inputs),
                "numerics": {"numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__}}
        key = digest(spec)
        if key in self.memory:
            self.stats[operation + ":ram"] += 1
            return self.get(key), key
        with file_lock(self.directory / "locks" / (key[:2] + ".lock")):
            try:
                value = self.get(key)
                self.stats[operation + ":disk"] += 1
                return value, key
            except FileNotFoundError:
                pass
            # Retain old addresses as references so unchanged downstream batches
            # also hit their old keys. No bulk rewrite or copy of stored arrays.
            for old in legacy_specs(spec):
                old_key = digest(old)
                try:
                    value = self.get(old_key)
                    self.stats[operation + ":legacy"] += 1
                    return value, old_key
                except FileNotFoundError:
                    pass
            start = time.perf_counter()
            value = function()
            self.stats[operation + ":computed"] += 1
            self.stats[operation + ":seconds"] += time.perf_counter() - start
            if persist:
                self._publish(key, value, spec)
            self._remember(key, value)
            return value, key

    def output_ref(self, key):
        return ArtifactRef(key)

    def usage(self):
        with file_lock(self.directory / "locks" / "budget.lock"):
            self.inventory.sync()
            disk = self.inventory.usage()
        return {**disk, "ram_bytes": self.memory_size, "statistics": dict(self.stats)}

    def close(self):
        with self.memory_lock:
            self.memory.clear()
            self.memory_size = 0
