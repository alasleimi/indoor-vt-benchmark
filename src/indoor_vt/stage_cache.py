"""Atomic disk caches of estimator-only intermediate results."""

import gzip
import json
import os
from pathlib import Path

from .provenance import digest, json_default, replace_file


class StageCache:
    def __init__(self, directory, implementation):
        self.directory = Path(directory)
        self.implementation = implementation
        self.hits = 0
        self.misses = 0

    def get(self, stage, identity, compute):
        key = digest([self.implementation, stage, identity])
        path = self.directory / stage / (key + ".json.gz")
        if path.exists():
            self.hits += 1
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        self.misses += 1
        result = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
            json.dump(result, handle, default=json_default, allow_nan=False, separators=(",", ":"))
        replace_file(temporary, path)
        return result


def cached(cache, stage, identity, compute):
    return compute() if cache is None else cache.get(stage, identity, compute)
