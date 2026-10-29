"""Small immutable acquisition manifests and reproducible independent seeds."""

import hashlib
import json
import os
from pathlib import Path
import uuid
import time

import numpy as np

from ..execution.store import canonical, digest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(canonical(value) + b"\n")
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows readers/antivirus can briefly prevent replacement.
                if attempt == 19:
                    raise
                time.sleep(min(.01 * (attempt + 1), .1))
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Seeds:
    def __init__(self, root):
        self.root = int(root)

    def seed(self, *keys):
        return int(digest([self.root, keys])[:16], 16)

    def rng(self, *keys):
        return np.random.default_rng(self.seed(*keys))
