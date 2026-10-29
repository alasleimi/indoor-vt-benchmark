"""Small scientific records; arrays contain values, never cache handles."""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Ragged:
    values: np.ndarray
    offsets: np.ndarray

    @classmethod
    def pack(cls, rows, *, width=None, dtype=float):
        rows = [np.asarray(row, dtype=dtype) for row in rows]
        values = np.concatenate(rows) if rows else np.empty((0,) if width is None else (0, width), dtype=dtype)
        return cls(values, np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int64))

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        return self.values[self.offsets[index]:self.offsets[index + 1]]

    def select(self, indices):
        return Ragged.pack([self[int(i)] for i in indices],
                           width=self.values.shape[1] if self.values.ndim == 2 else None,
                           dtype=self.values.dtype)

    def record(self):
        return {"values": self.values, "offsets": self.offsets}


@dataclass(frozen=True)
class Survey:
    xyz: np.ndarray
    channels: dict[str, Ragged]
    position_ids: np.ndarray

    def __post_init__(self):
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or not np.isfinite(self.xyz).all():
            raise ValueError("Survey coordinates must be finite (n,3)")
        if len(self.position_ids) != len(self.xyz) or any(len(r) != len(self.xyz) for r in self.channels.values()):
            raise ValueError("Survey coordinates, identities, and fingerprints must align")

    def select(self, indices):
        return Survey(self.xyz[indices], {ap: rows.select(indices) for ap, rows in self.channels.items()},
                      self.position_ids[indices])


@dataclass
class SourceMap:
    by_ap: dict[str, np.ndarray]
    evidence: dict = field(default_factory=dict)

    def bag(self):
        return np.concatenate(list(self.by_ap.values())) if self.by_ap else np.empty((0, 3))


@dataclass
class Prediction:
    xyz: np.ndarray
    status: str = "ok"
    evidence: dict = field(default_factory=dict)
