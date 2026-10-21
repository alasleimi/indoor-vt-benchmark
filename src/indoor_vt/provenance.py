"""Stable independent RNG streams, immutable identities and atomic artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_bytes(json.dumps(value, indent=2, allow_nan=False).encode() + b"\n")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class SeedBook:
    def __init__(self, root: int, namespace: str):
        self.root = int(root)
        self.namespace = namespace
        self.records: dict[str, int] = {}

    def seed(self, *keys) -> int:
        token = canonical([self.root, self.namespace, list(keys)])
        value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
        self.records[canonical(list(keys)).decode()] = value
        return value

    def rng(self, *keys) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.seed(*keys)))

    def manifest(self):
        return {"root": self.root, "namespace": self.namespace,
                "algorithm": "SHA256/uint64le -> PCG64", "streams": dict(sorted(self.records.items()))}


def code_provenance(root: Path) -> dict:
    files = sorted(p for base in ("src", "configs") for p in (root / base).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in files}
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return {"git_commit": result.stdout.strip() if result.returncode == 0 else None,
            "source_hash": digest(hashes), "files": hashes, "python": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__}
