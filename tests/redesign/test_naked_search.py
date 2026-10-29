from types import SimpleNamespace

from indoor_vt.acquisition.records import read_json, write_json, digest
from indoor_vt.execution import composition_experiment as experiment


def test_naked_pipelines_reuse_frozen_map_choices_without_retuning_and_resume(tmp_path, monkeypatch):
    base, data, run = [tmp_path / name for name in ("base", "data", "rectangle-naked-selection-v1")]
    config = read_json("configs/redesign.json")
    config["observations"]["noise_levels"] = [0.]
    config["optimization"]["smart_lea"] = {"half_window": [1], "alpha": [.5], "iterations": [1]}
    config["optimization"]["mpurge_map"] = {
        "half_window": [1], "alpha": [.5], "neighbors": [1, 3], "power": [1.]}
    write_json(base / "config.json", config)
    write_json(data / "dataset.json", {"native_data": "already acquired"})
    write_json(base / "implementation.json", {"id": "older whole-code hash",
        "experiment": "older search code", "dataset": digest({"native_data": "already acquired"})})
    frozen_map = {"half_window": 1, "alpha": .5, "neighbors": 3, "power": 1.}
    for mode in ("unified", "ununified"):
        write_json(base / "selected" / f"mpurge_map_{mode}.json",
                   {"id": digest(mode), "parameters": frozen_map})
    original_config = experiment.composition_config
    def tiny_config(base_config):
        value = original_config(base_config)
        value["compositions"]["joint_trials"] = 2
        for method in ("zayets_naked", "ransac_naked"):
            space = value["compositions"]["spaces"][method]
            value["compositions"]["spaces"][method] = {k: v[:1] for k, v in space.items()}
        return value
    calls = []
    class Evaluator:
        def __init__(self, *args):
            self.observations = SimpleNamespace(available=lambda: True)
            self.store = SimpleNamespace(usage=lambda: {})
        def evaluate(self, method, parameters, **kwargs):
            assert method.startswith(("zayets_naked", "ransac_naked"))
            calls.append((method, parameters, kwargs))
            value = {"loss": .1, "metrics": {}, "seconds": 0.}
            if kwargs.get("retain"):
                value["detail"] = {"sources": [], "queries": []}
            return value
    monkeypatch.setattr(experiment, "composition_config", tiny_config)
    monkeypatch.setattr(experiment, "CompositionEvaluator", Evaluator)
    methods = ("zayets_naked", "ransac_naked")
    results = experiment.run_compositions(data, run, base, methods=methods)
    assert len(results) == 20
    assert {p["method"] for p in results} == {m + s for m in methods for s in ("", "_lea", "_joint")}
    assert all(k["frozen"]["map"] == frozen_map for m, p, k in calls if m.endswith("_lea"))
    status = read_json(run / "status.json")
    assert status["state"] == "complete" and status["completed_trials"] == status["target_trials"] == 14
    before = len(calls)
    assert experiment.run_compositions(data, run, base, methods=methods) == results
    assert len(calls) == before
    provenance = read_json(run / "implementation.json")["initializer_study"]
    assert provenance["selected"]["unified"] == digest("unified")
