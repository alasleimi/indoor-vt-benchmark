"""Estimator-facing records deliberately contain no evaluator or native-ray fields."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np


def readonly(array, dimensions: int | None = None):
    value = np.array(array, dtype=np.float64, copy=True)
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError(f"Expected {dimensions} dimensions, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Estimator inputs must be finite")
    value.flags.writeable = False
    return value


@dataclass(frozen=True, slots=True)
class DelayObservation:
    channels: dict[str, np.ndarray]

    def __post_init__(self):
        converted = {}
        for ap, row in self.channels.items():
            values = readonly(np.sort(row, kind="stable"), 1)
            if np.any(values < 0):
                raise ValueError("Negative observations must be discarded by the frontend")
            converted[str(ap)] = values
        object.__setattr__(self, "channels", converted)


@dataclass(frozen=True, slots=True)
class Survey:
    xyz: np.ndarray
    observations: tuple[DelayObservation, ...]

    def __post_init__(self):
        points = readonly(self.xyz, 2)
        if points.shape != (len(self.observations), 3) or not len(points):
            raise ValueError("A survey requires N three-dimensional coordinates and N observations")
        object.__setattr__(self, "xyz", points)


@dataclass(frozen=True, slots=True)
class SourceMap:
    by_ap: dict[str, np.ndarray]
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "by_ap", {ap: readonly(np.asarray(p).reshape(-1, 3), 2)
                                          for ap, p in self.by_ap.items()})

    def bag(self):
        return np.concatenate(list(self.by_ap.values()), axis=0) if self.by_ap else np.empty((0, 3))


@dataclass(frozen=True, slots=True)
class Prediction:
    xyz: np.ndarray
    status: str = "ok"
    diagnostics: dict = field(default_factory=dict)


class InitialLocalizer(Protocol):
    def fit(self, survey: Survey): ...
    def predict(self, query: DelayObservation) -> Prediction: ...


class Reconstructor(Protocol):
    def fit(self, survey: Survey) -> SourceMap: ...


class Refiner(Protocol):
    def predict(self, query: DelayObservation, initial: Prediction, sources: SourceMap) -> Prediction: ...
