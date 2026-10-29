"""Compact append-only scalar trial logs; one context per search stage."""
import json
from pathlib import Path

from ..acquisition.records import write_json
from .store import canonical


class TrialLog:
    def __init__(self, directory, label, context):
        self.path = Path(directory) / (label + ".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path.with_suffix(".json"), context)
        self.context = context
        data = self.path.read_bytes() if self.path.exists() else b""
        end = data.rfind(b"\n") + 1
        # An interrupted append can leave one unfinished trailing row.
        if end != len(data):
            with self.path.open("r+b") as stream:
                stream.truncate(end)
        self.rows = {row["id"]: row for row in
                     (json.loads(line) for line in data[:end].splitlines() if line)}

    def append(self, row):
        minimal = {k: v for k, v in row.items() if k not in self.context}
        with self.path.open("ab") as stream:
            stream.write(canonical(minimal) + b"\n")
            stream.flush()
        self.rows[row["id"]] = minimal

    def get(self, key):
        row = self.rows.get(key)
        return None if row is None else {**self.context, **row}


def read_trials(directory):
    result = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        context = json.loads(path.with_suffix(".json").read_text())
        data = path.read_bytes()
        end = data.rfind(b"\n") + 1
        result.extend({**context, **json.loads(line)} for line in data[:end].splitlines() if line)
    return result
