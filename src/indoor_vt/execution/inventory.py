"""Incremental cache quota inventory, coordinated by the caller's budget lock.

The journal contains file size/recency deltas, never scientific inputs/results.
A dirty marker makes interrupted publication recoverable with one directory scan.
"""
import heapq
import json
import os
from pathlib import Path
import time
import uuid

from ..acquisition.records import read_json, write_json
from .store import canonical


class DiskInventory:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.state = self.directory / "inventory.json"
        self.journal = self.directory / "inventory.jsonl"
        self.dirty = self.directory / "inventory.dirty"
        self.files, self.heap = {}, []
        self.total, self.offset, self.generation = 0, 0, -1
        self.epoch, self.events = None, []
        self.scans = 0

    def path(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid inventory computation ID")
        return self.directory / key[:2] / (key + ".npz")

    def _apply(self, key, size, modified):
        previous = self.files.pop(key, None)
        if previous is not None:
            self.total -= previous[0]
        if size is not None:
            self.files[key] = (size, modified)
            self.total += size
            heapq.heappush(self.heap, (modified, key))

    def _snapshot(self):
        temporary = self.journal.with_suffix("." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("wb") as stream:
                for key, (size, modified) in self.files.items():
                    stream.write(canonical([key, size, modified]) + b"\n")
            os.replace(temporary, self.journal)
        finally:
            temporary.unlink(missing_ok=True)
        self.epoch = uuid.uuid4().hex
        self.offset = self.journal.stat().st_size
        self.heap = [(modified, key) for key, (_, modified) in self.files.items()]
        heapq.heapify(self.heap)

    def _state(self):
        write_json(self.state, {"epoch": self.epoch, "generation": self.generation,
                   "bytes": self.total, "files": len(self.files), "journal_bytes": self.offset})

    def rebuild(self):
        self.dirty.touch()
        self.files, self.heap, self.total = {}, [], 0
        for path in self.directory.glob("*/*.npz"):
            if self.path(path.stem) != path:
                raise ValueError("Cache artifact has an unexpected location")
            stat = path.stat()
            self._apply(path.stem, stat.st_size, stat.st_mtime)
        self.scans += 1
        self.generation = 0
        self._snapshot()
        self._state()
        self.dirty.unlink()

    def sync(self):
        """Call only while holding the cache's cross-process budget lock."""
        if self.dirty.exists() or not self.state.exists() or not self.journal.exists():
            self.rebuild()
            return
        state = read_json(self.state)
        if state["epoch"] != self.epoch:
            self.files, self.heap, self.total, self.offset = {}, [], 0, 0
            self.epoch = state["epoch"]
            self.generation = -1
        if state["generation"] == self.generation:
            return
        with self.journal.open("rb") as stream:
            stream.seek(self.offset)
            for line in stream:
                self._apply(*json.loads(line))
            self.offset = stream.tell()
        if (self.offset != state["journal_bytes"] or self.total != state["bytes"]
                or len(self.files) != state["files"]):
            self.rebuild()
            return
        self.generation = state["generation"]

    def begin(self):
        self.dirty.touch()
        self.events = []

    def put(self, key):
        stat = self.path(key).stat()
        event = (key, stat.st_size, stat.st_mtime)
        self._apply(*event)
        self.events.append(event)

    def remove(self, key):
        self.path(key).unlink(missing_ok=True)
        self._apply(key, None, 0)
        self.events.append((key, None, 0))

    def evict_one(self):
        # Refresh recency lazily. Cache readers need no durable per-hit journal.
        examined = 0
        while self.heap and examined < len(self.files):
            modified, key = heapq.heappop(self.heap)
            record = self.files.get(key)
            if record is None or record[1] != modified:
                continue
            examined += 1
            path = self.path(key)
            try:
                stat = path.stat()
            except FileNotFoundError:
                self.remove(key)
                return record[0]
            if stat.st_mtime > modified:
                self.files[key] = (record[0], stat.st_mtime)
                heapq.heappush(self.heap, (stat.st_mtime, key))
                continue
            try:
                self.remove(key)
                return record[0]
            except PermissionError:
                # A Windows reader holds the file. Try another victim.
                deferred = time.time() + 1
                self.files[key] = (record[0], deferred)
                heapq.heappush(self.heap, (deferred, key))
        return 0

    def commit(self):
        with self.journal.open("ab") as stream:
            for event in self.events:
                stream.write(canonical(event) + b"\n")
        self.offset = self.journal.stat().st_size
        # Bound obsolete inventory entries; compact occasionally, not per batch.
        if self.offset > max(1024**2, 4 * len(self.files) * 128):
            self._snapshot()
        self.generation += 1
        self._state()
        self.dirty.unlink()
        self.events = []

    def usage(self):
        return {"files": len(self.files), "disk_bytes": self.total,
                "inventory_bytes": self.offset + self.state.stat().st_size,
                "inventory_scans": self.scans}
