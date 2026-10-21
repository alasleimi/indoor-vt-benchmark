"""Factories resolve recorded configurations; components remain independently injectable."""

from dataclasses import asdict
from .localization import FingerprintLocalizer, InitialConfig, LEAConfig, SmartLEA
from .matching import MatcherConfig, PatternMatcher
from .reconstruction import ReconstructionConfig, MPUrgeReconstructor
from .diffassign import DiffAssign, DiffAssignConfig
from .consensus import ConsensusConfig, ConsensusReconstructor
from .zayets2019 import Zayets2019, Zayets2019Config
from .pipeline import Pipeline

INITIAL_METHODS = ("mpurge_map", "mpurge", "mca", "chamfer")
VT_METHODS = tuple(
    f"{name}_{scope}"
    for name in ("mpurge", "mpurge_star", "mpurge_triangle", "mpurge_diffassign")
    for scope in ("local", "full")
) + ("pba_local", "ransac", "zayets2018", "zayets2019_assumption")


def initial_default(name):
    return asdict(
        InitialConfig(
            method=name, neighbors=1 if name == "mca" else 3, alpha=0.5 if name == "mpurge" else 0.7
        )
    )


def reconstruction_default(name):
    if name in ("ransac", "zayets2018"):
        return asdict(ConsensusConfig(method=name))
    if name == "zayets2019_assumption":
        return asdict(Zayets2019Config())
    scope = name.rsplit("_", 1)[1]
    result = asdict(
        ReconstructionConfig(
            scope=scope,
            grouping="star" if "star" in name else "components",
            triangle="triangle" in name,
            size_norm="l2" if name.startswith("pba") else "l1",
        )
    )
    if "diffassign" in name:
        result["diffassign"] = asdict(DiffAssignConfig())
    return result


def make_reconstructor(name, parameters=None, seed=0):
    if name not in VT_METHODS:
        raise ValueError(f"Unregistered primary reconstruction method: {name}")
    params = dict(reconstruction_default(name) if parameters is None else parameters)
    if name in ("ransac", "zayets2018"):
        return ConsensusReconstructor(ConsensusConfig(**params), seed)
    if name == "zayets2019_assumption":
        return Zayets2019(Zayets2019Config(**params))
    refinement = (
        DiffAssign(DiffAssignConfig(**params.pop("diffassign"))) if "diffassign" in params else None
    )
    config = ReconstructionConfig(**params)
    matcher = None
    if name.startswith("pba"):
        matcher = PatternMatcher(
            MatcherConfig(
                config.half_window, config.alpha, size_norm="l2", conflict_priority="distance"
            )
        )
    return MPUrgeReconstructor(config, matcher=matcher, refinement=refinement)


def make_pipeline(config, seed=0):
    return Pipeline(
        FingerprintLocalizer(InitialConfig(**config["initial"])),
        make_reconstructor(config["vt_method"], config["vt"], seed),
        SmartLEA(LEAConfig(**config["lea"])),
    )
