"""Published candidates plus broad, round search values; no historical tuned winners."""

from dataclasses import asdict
from .localization import LEAConfig
from .registry import initial_default, reconstruction_default


def numeric(values, *, bounds=(None, None), integer=False, expand=True):
    return {"values": values, "bounds": list(bounds), "integer": integer, "expand": expand}


def initial_space(method):
    space = {"neighbors": numeric([1, 3, 5, 10, 20], bounds=(1, 100), integer=True)}
    if method in ("mpurge", "mpurge_map"):
        space.update(
            {
                "half_window": numeric([0, 1, 2, 3, 4], bounds=(0, 10), integer=True),
                "alpha": numeric([0.0, 0.25, 0.5, 0.75, 1.0], bounds=(0, 1), expand=False),
            }
        )
    elif method == "mca":
        space = {"epsilon_m": numeric([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], bounds=(0.001, 100))}
    else:
        space["distance_power"] = numeric([1.0, 2.0], bounds=(0.25, 4.0))
    citation = {
        "mpurge_map": "MPUrge-MAP; source-checked SAPP baseline: p=2, alpha=.7, k=3",
        "mpurge": "MPUrge: total pattern width 5, alpha=.5; weighted-kNN adaptation k=3",
        "mca": "Zayets-Steinbach MCA: tolerance 1 m, best reference",
        "chamfer": "SAPP Chamfer weighted-kNN baseline",
    }[method]
    return space, [{"parameters": initial_default(method), "citation": citation}]


def vt_space(method):
    if method in ("ransac", "zayets2018"):
        space = {
            "inlier_scale_m": numeric(
                [0.05, 0.1, 0.18, 0.2, 0.5, 1.0, 2.0, 5.0], bounds=(0.001, 100)
            ),
            "support_fraction": numeric([0.01, 0.05, 0.08, 0.1, 0.2, 0.5], bounds=(0.001, 1.0)),
            "surface_separation_m": numeric([0.05, 0.1, 0.12, 0.2, 0.5, 1.0], bounds=(0.001, 100)),
        }
        if method == "ransac":
            space["trials"] = numeric(
                [1000, 5000, 10000, 30000], bounds=(100, 100000), integer=True
            )
        else:
            space.update(
                {
                    "neighborhood_radius_m": numeric([0.5, 1.0, 2.0, 5.0], bounds=(0.1, 20.0)),
                    "cluster_gap_m": numeric(
                        [0.05, 0.1, 0.2, 0.5, 0.6, 1.0, 2.0], bounds=(0.001, 100.0)
                    ),
                    "group_threshold": numeric(
                        [0.1, 0.2, 0.5, 1.0, 2.0, 5.0], bounds=(0.001, 100.0)
                    ),
                }
            )
        return space, [
            {
                "parameters": reconstruction_default(method),
                "citation": "SAPP discovery/source-fitting implementation; 3D adaptation and uncapped output",
            }
        ]
    if method == "zayets2019_assumption":
        space = {
            "embedding_scale_m": numeric([0.5, 1.0, 2.0, 5.0], bounds=(0.01, 100.0)),
            "neighborhood_radius_m": numeric([0.5, 1.0, 2.0, 5.0], bounds=(0.1, 20.0)),
            "ward_threshold": numeric([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], bounds=(0.001, 1000.0)),
            "range_tolerance_m": numeric([0.01, 0.1, 0.5, 1.0, 2.0, 5.0], bounds=(0.001, 100.0)),
            "gamma": numeric([0.0, 0.5, 1.0, 1.5, 2.0, 4.0], bounds=(0.0, 16.0)),
            "merge_radius_m": numeric([0.1, 0.2, 0.5, 1.0, 2.0, 5.0], bounds=(0.001, 100.0)),
        }
        return space, [
            {
                "parameters": reconstruction_default(method),
                "citation": "ICC 2019 gamma=1.5; remaining values are disclosed round implementation assumptions",
            }
        ]
    space = {
        "half_window": numeric([0, 1, 2, 3, 4], bounds=(0, 10), integer=True),
        "alpha": numeric([0.0, 0.25, 0.5, 0.75, 1.0], bounds=(0, 1), expand=False),
        "merge_radius_m": numeric([0.1, 0.2, 0.5, 1.0, 2.0, 5.0], bounds=(0.001, 100.0)),
        "beta": numeric([0.0, 0.5, 1.0, 1.5, 2.0, 4.0], bounds=(0.0, 16.0)),
    }
    if method.endswith("local"):
        space["neighbors"] = numeric([4, 5, 10, 20, 50, 100], bounds=(4, 500), integer=True)
    if "triangle" in method:
        space["triangle_tolerance_m"] = numeric(
            [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], bounds=(0.0, 100.0)
        )
    if "diffassign" in method:
        space.update(
            {
                "diffassign.scale_m": numeric([0.1, 0.2, 0.5, 1.0, 2.0], bounds=(0.001, 100.0)),
                "diffassign.temperature": numeric([0.05, 0.1, 0.2, 0.5, 1.0], bounds=(0.001, 10.0)),
                "diffassign.reverse_weight": numeric([0.0, 0.25, 0.5, 1.0], bounds=(0.0, 4.0)),
                "diffassign.unmatched_cost": numeric([1.0, 2.0, 5.0, 10.0], bounds=(0.1, 100.0)),
            }
        )
    return space, [
        {
            "parameters": reconstruction_default(method),
            "citation": "MPUrge/PBA: total pattern width 5 and alpha=.5; geometry/grouping values are round implementation defaults",
        }
    ]


def lea_space():
    return {
        "iterations": {"values": [1, 2, 4], "categorical": True},
        "half_window": numeric([0, 1, 2, 3, 4], bounds=(0, 10), integer=True),
        "alpha": numeric([0.0, 0.25, 0.5, 0.75, 1.0], bounds=(0, 1), expand=False),
    }, [
        {
            "parameters": asdict(LEAConfig(iterations=i)),
            "citation": "SMART-LEA: 1, 2 and 4 refinement passes",
        }
        for i in (1, 2, 4)
    ]
