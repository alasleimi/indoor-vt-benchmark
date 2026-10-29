"""Content-addressed typed values and single-flight pure computations.

Small arrays are compressed SQLite blobs, not individual files. Large arrays
are immutable NPY files and load through memory mapping. JSON contains only
small records and references. Computation leases survive worker crashes.
"""

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache, partial
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
import zlib

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Ref:
    id: str


@lru_cache(maxsize=512)
def source_digest(function):
    try:
        source = inspect.getsource(function)
    except (TypeError, OSError):
        source = function.__module__ + "." + function.__qualname__
    return hashlib.sha256(source.encode()).hexdigest()


def component_spec(value):
    """Record effective component settings, including injected callables."""
    if isinstance(value, partial):
        return {"callable": component_spec(value.func), "args": component_spec(value.args),
                "kwargs": component_spec(value.keywords)}
    if is_dataclass(value):
        return {"type": type(value).__module__ + "." + type(value).__qualname__,
                "source": source_digest(type(value)),
                "settings": {f.name: component_spec(getattr(value, f.name)) for f in fields(value)}}
    if callable(value):
        return {"function": value.__module__ + "." + value.__qualname__,
                "source": source_digest(value)}
    if isinstance(value, dict):
        return {str(k): component_spec(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [component_spec(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Store:
    def __init__(self, directory, *, read_only=False):
        self.directory = Path(directory)
        self.read_only = read_only
        self.database = self.directory / "store.sqlite3"
        self._local = threading.local()
        if read_only:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "arrays").mkdir(exist_ok=True)
        self.database = self.directory / "store.sqlite3"
        self._local = threading.local()
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS computations (
                    id TEXT PRIMARY KEY, operation TEXT NOT NULL, inputs TEXT NOT NULL,
                    parameters TEXT NOT NULL, output TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS leases (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, deadline REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, time REAL, run TEXT, computation TEXT,
                    event TEXT, seconds REAL);
            """)

    @contextmanager
    def connect(self):
        db = getattr(self._local, "db", None)
        if db is None:
            if self.read_only:
                db = sqlite3.connect(self.database.resolve().as_uri() + "?mode=ro", uri=True, timeout=60)
            else:
                db = sqlite3.connect(self.database, timeout=60)
                db.execute("PRAGMA synchronous=NORMAL")
            db.execute("PRAGMA cache_size=-2048")
            self._local.db = db
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise

    def put(self, value):
        if isinstance(value, Ref):
            return value.id
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            if array.dtype.hasobject:
                raise TypeError("Object arrays cannot be stored")
            dtype = array.dtype.descr if array.dtype.fields else array.dtype.str
            header = canonical({"dtype": dtype, "shape": array.shape})
            hasher = hashlib.sha256(header)
            if array.nbytes:
                hasher.update(memoryview(array).cast("B"))
            key = hasher.hexdigest()
            with self.connect() as db:
                if db.execute("SELECT 1 FROM objects WHERE id=?", (key,)).fetchone():
                    return key
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            if array.nbytes >= 1024 * 1024:
                target = self.directory / "arrays" / (key + ".npy")
                temporary = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
                temporary.write_bytes(buffer.getbuffer())
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass
                finally:
                    temporary.unlink()
                kind, payload = "file", target.name.encode()
            else:
                kind, payload = "array", zlib.compress(buffer.getbuffer(), 3)
        else:
            def encode(item):
                if isinstance(item, Ref):
                    return {"$value": item.id}
                if isinstance(item, np.ndarray):
                    return {"$array": self.put(item)}
                if isinstance(item, np.generic):
                    return item.item()
                if isinstance(item, dict):
                    return {str(k): encode(v) for k, v in item.items()}
                if isinstance(item, (tuple, list)):
                    return [encode(v) for v in item]
                return item
            payload = canonical(encode(value))
            key, kind = hashlib.sha256(payload).hexdigest(), "record"
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO objects VALUES (?,?,?)", (key, kind, payload))
        return key

    def close(self):
        """Release this thread's connection (worker threads own their connections)."""
        db = getattr(self._local, "db", None)
        if db is not None:
            db.close()
            del self._local.db

    def describe(self, key):
        """Read one record without following its references; audit views stay lazy."""
        with self.connect() as db:
            row = db.execute("SELECT kind,payload FROM objects WHERE id=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        kind, payload = row
        if kind == "record":
            return {"id": key, "kind": kind, "value": json.loads(payload)}
        array = self.get(key)
        return {"id": key, "kind": kind, "dtype": str(array.dtype),
                "shape": list(array.shape), "bytes": array.nbytes}

    def get(self, key):
        with self.connect() as db:
            row = db.execute("SELECT kind,payload FROM objects WHERE id=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        kind, payload = row
        if kind == "file":
            return np.load(self.directory / "arrays" / payload.decode(), mmap_mode="r",
                           allow_pickle=False)
        if kind == "array":
            result = np.load(io.BytesIO(zlib.decompress(payload)), allow_pickle=False)
            result.flags.writeable = False
            return result
        def decode(item):
            if isinstance(item, dict):
                if set(item) == {"$array"}:
                    return self.get(item["$array"])
                if set(item) == {"$value"}:
                    return self.get(item["$value"])
                return {k: decode(v) for k, v in item.items()}
            if isinstance(item, list):
                return [decode(v) for v in item]
            return item
        return decode(json.loads(payload))

    def output_ref(self, computation):
        with self.connect() as db:
            row = db.execute("SELECT output FROM computations WHERE id=?", (computation,)).fetchone()
        if row is None:
            raise KeyError(computation)
        return Ref(row[0])

    def compute(self, operation, parameters, inputs, function, *, run=""):
        """Return (value, computation ID); run labels never enter the key."""
        input_ids = {k: self.put(v) for k, v in inputs.items()}
        params = component_spec(parameters)
        key = digest({"operation": operation, "parameters": params, "inputs": input_ids})
        owner = uuid.uuid4().hex
        while True:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT output FROM computations WHERE id=?", (key,)).fetchone()
                if row:
                    db.execute("INSERT INTO events(time,run,computation,event) VALUES (?,?,?,?)",
                               (time.time(), run, key, "hit"))
                    output_id = row[0]
                    acquired = False
                else:
                    db.execute("DELETE FROM leases WHERE deadline < ?", (time.time(),))
                    acquired = db.execute("INSERT OR IGNORE INTO leases VALUES (?,?,?)",
                                          (key, owner, time.time() + 120)).rowcount == 1
                    output_id = None
            if output_id is not None:
                return self.get(output_id), key
            if acquired:
                break
            time.sleep(0.1)
        stop = threading.Event()
        def heartbeat():
            try:
                while not stop.wait(20):
                    with self.connect() as db:
                        db.execute("UPDATE leases SET deadline=? WHERE id=? AND owner=?",
                                   (time.time() + 120, key, owner))
            finally:
                self.close()
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        started = time.perf_counter()
        try:
            value = function()
            output = self.put(value)
            with self.connect() as db:
                db.execute("INSERT OR IGNORE INTO computations VALUES (?,?,?,?,?)",
                           (key, operation, canonical(input_ids).decode(),
                            canonical(params).decode(), output))
                db.execute("INSERT INTO events(time,run,computation,event,seconds) VALUES (?,?,?,?,?)",
                           (time.time(), run, key, "computed", time.perf_counter() - started))
            return value, key
        finally:
            stop.set()
            thread.join(timeout=1)
            with self.connect() as db:
                db.execute("DELETE FROM leases WHERE id=? AND owner=?", (key, owner))
