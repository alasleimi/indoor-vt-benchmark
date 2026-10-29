"""Narrow migration of reviewed numerical batches; whole-file hashes are provenance.

Operation names carry their numerical revision. Bump that revision when helper
behavior changes; comments, search grids and new methods are not dependencies.
Injected component code/settings and numerical-library versions remain in keys.
"""
from copy import deepcopy


LEGACY_IMPLEMENTATION = "9d3ec02ab344465b297ff54a3d8f6bce8a6594167d2f1f905c5502086b3b7159"
LEGACY_EXTENSION = "8bb62d63fd75214aeff9e9ee4646dba99576187c28b41715f8eac43230d24697"
LEGACY_NUMERICS = {"numpy": "2.2.6", "scipy": "1.15.3", "numba": "0.62.1"}
LEGACY_OPERATIONS = {
    "comparison-geometry.2", "pair-matches.2", "grouped-candidates.2", "source-clusters.2",
    "fingerprint-match-statistics.2", "fingerprint-scores.2", "neighbor-estimation.2",
    "smart-lea-step.2", "zayets-proposals.1", "ransac-hypotheses.1", "ransac-retention.1",
    "consuming-finisher.1",
}
EXTENSION_OPERATIONS = {
    "zayets-proposals.1", "ransac-hypotheses.1", "ransac-retention.1", "consuming-finisher.1",
}
# Exact reviewed before/after class bodies: the floor was removed and a positive
# scale assertion added. This alias is inapplicable to any later estimator edit.
ESTIMATOR_WITHOUT_FLOOR = "07dc395bc046218fc9b28c67f5290b7f00a394ad07352619c78f01e9ca900394"
ESTIMATOR_WITH_FLOOR = "80b10873926bc9d48a2ac7fd0dec2b0ed90ff836daa645dc41c182c0601d0d70"


def legacy_specs(spec):
    """Probe old addresses only for unchanged operation/component contracts."""
    if spec["operation"] not in LEGACY_OPERATIONS or spec["numerics"] != LEGACY_NUMERICS:
        return []

    def estimator_alias(value, floor):
        if isinstance(value, dict):
            if (value.get("type") == "indoor_vt.science.estimation.Multilateration3D"
                    and value.get("source") == ESTIMATOR_WITHOUT_FLOOR):
                scale = value["settings"]["loss_scale_m"]
                if not scale > 0 or scale < floor:
                    raise ValueError("The old floor was active")
                value["source"] = ESTIMATOR_WITH_FLOOR
                value["settings"]["loss_scale_floor_m"] = floor
            return {k: estimator_alias(v, floor) for k, v in value.items()}
        if isinstance(value, list):
            return [estimator_alias(v, floor) for v in value]
        return value

    result = []
    for floor in (0., .03):
        try:
            parameters = estimator_alias(deepcopy(spec["parameters"]), floor)
        except ValueError:
            continue
        if spec["operation"] in EXTENSION_OPERATIONS:
            parameters["extension"] = LEGACY_EXTENSION
        old = {"operation": spec["operation"], "parameters": parameters,
               "inputs": spec["inputs"], "implementation": LEGACY_IMPLEMENTATION}
        if old not in result:
            result.append(old)
    return result
