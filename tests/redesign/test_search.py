"""Noise-free selection, frozen dependencies, and resumable outcome receipts."""
from indoor_vt.acquisition.records import read_json, write_json
from indoor_vt.execution import experiment
from indoor_vt.execution.store import digest


def test_stagewise_joint_selection_and_resume(tmp_path, monkeypatch):
    calls = []
    class Observations:
        def available(self):
            return True
    class Cache:
        def usage(self):
            return {}
    class Evaluator:
        def __init__(self, *args):
            self.observations, self.store = Observations(), Cache()
        def evaluate(self, method, parameters, **kwargs):
            calls.append((method, parameters, kwargs))
            p = parameters["lea"] if method == "joint" else parameters
            loss = abs(p["alpha"] - kwargs["level"])
            result = {"loss": loss, "metrics": {"loss": loss}, "seconds": 0}
            if kwargs.get("retain"):
                result["detail"] = {"sources": [], "queries": []}
            return result
    monkeypatch.setattr(experiment, "Evaluator", Evaluator)
    config = read_json("configs/redesign.json")
    config["observations"]["noise_levels"] = [0., 1.]
    config["optimization"]["joint_trials"] = 2
    for method in ("mpurge", "mpurge_map", "smart_lea"):
        config["optimization"][method] = {"alpha": [0., 1.], "half_window": [1]}
    data, run = tmp_path / "data", tmp_path / "run"
    write_json(data / "dataset.json", {"test": "fixed dataset"})
    result = experiment.run_experiment(data, run, config)
    assert len(result) == 30  # five methods, two modes, calibration plus two test levels
    selection = [c for c in calls if not c[2].get("retain")]
    assert all(c[2]["level"] == 0 and c[2]["split"] == "calibration" for c in selection)
    stagewise = [c for c in selection if c[0] == "smart_lea"]
    assert all(c[2]["frozen"]["mpurge"]["alpha"] == 0 for c in stagewise)
    assert all(c[2]["frozen"]["map"]["alpha"] == 0 for c in stagewise)
    joint = [c for c in selection if c[0] == "joint"]
    assert all(set(c[1]) == {"mpurge", "map", "lea"} and not c[2]["frozen"] for c in joint)
    assert all(r["parameters"].get("alpha", 0) == 0 for r in result)
    count = len(calls)
    (run / "selected" / "mpurge_unified.json").unlink()
    resumed = experiment.run_experiment(data, run, config)
    assert len(calls) == count and digest(resumed) == digest(result)
    assert read_json(run / "status.json")["state"] == "complete"


def test_grid_covers_all_active_windows():
    grid = experiment.configurations({"half_window": list(range(1, 8)), "alpha": [0., .5, 1.]})
    assert len(grid) == 15
    assert {p["half_window"] for p in grid if p["alpha"] == .5} == set(range(1, 8))
    assert sum(p["alpha"] == 1 for p in grid) == 1


def test_robust_finisher_search_is_active_only_for_robust_variants():
    from indoor_vt.execution.additional import ROBUST_FINISHER_PARAMETERS
    from indoor_vt.execution.additional_experiment import additional_config, robust_finisher_grid
    from indoor_vt.execution.composition_experiment import composition_config

    base = read_json("configs/redesign.json")
    additional = additional_config(base)["additional"]
    spaces = composition_config(base)["compositions"]["spaces"]
    keys = set(ROBUST_FINISHER_PARAMETERS)
    for space in (additional["ransac"], spaces["zayets_robust"], spaces["mpurge_robust"]):
        assert keys <= space.keys()
        assert all(space[k] == values for k, values in robust_finisher_grid().items())
    for space in (additional["zayets"], spaces["ransac_consumption"],
                  spaces["mpurge_consumption"], spaces["mpurge_triangle"]):
        assert keys.isdisjoint(space)

    grid = experiment.configurations(robust_finisher_grid())
    assert len(grid) == 21  # zero median weight makes its three clipping choices equivalent
    assert {p["median_clip_factor"] for p in grid if p["median_penalty_weight"] == 0} == {2.}
    original = {"smooth_weight": .35, "median_penalty_weight": .1, "median_clip_factor": 4.}
    assert original in grid
    baseline = {"source": original, "map": {"neighbors": 1}, "lea": {"iterations": 1}}
    joint = experiment.joint_configurations({"source": robust_finisher_grid(),
        "map": {"neighbors": [1]}, "lea": {"iterations": [1]}}, 21, 17, baseline)
    assert joint[0] == baseline and len(joint) == 21
    assert {tuple(sorted(p["source"].items())) for p in joint} == {tuple(sorted(p.items())) for p in grid}
