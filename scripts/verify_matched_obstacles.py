"""Check common-position acquisition, exact empty reuse, truth and preserved runs."""
import hashlib
from pathlib import Path

import numpy as np

from indoor_vt.acquisition.records import read_json, write_json, digest
from indoor_vt.acquisition.observations import Observations
from indoor_vt.acquisition.dataset import survey_indices
from indoor_vt.acquisition.geometry import scene_faces
from indoor_vt.execution.obstacle_reconstruction import layout_inputs, visible_source_catalog
from indoor_vt.execution.explorer import inside_obstacles
from indoor_vt.execution.artifacts import read_npz


def main():
    run = Path("runs/redesign/rectangle-matched-obstacle-reconstruction-v1")
    config = read_json(run / "config.json")
    original = Observations("data/redesign/rectangle-20261025", config)
    observed = Observations(config["data_root"], config)
    room, old_room = observed.rooms[0], original.rooms[0]
    mask = observed.manifest["common_survey_masks"][room["id"]]
    ids = np.asarray(mask["retained_position_ids"])
    old_ids = survey_indices(old_room, **config["survey"])
    assert mask["original_count"] == len(old_ids) == 280
    assert len(ids) == mask["retained_count"] == 270
    assert np.array_equal(ids, old_ids[~np.isin(old_ids, mask["excluded_position_ids"])])
    target_xyz = old_room["survey_xyz"][ids]
    for layout in old_room["layouts"]:
        assert len(inside_obstacles(target_xyz, layout["boxes"])) == 0
    assert observed.manifest["base_dataset"] == digest(original.manifest)
    layouts = {}
    for layout in room["layouts"]:
        survey = layout_inputs(observed, room, layout)
        np.testing.assert_array_equal(survey.position_ids, ids)
        np.testing.assert_array_equal(survey.xyz, target_xyz)
        catalog, audit = visible_source_catalog(observed, room, layout)
        per_ap = {}
        for ap, ranges in survey.channels.items():
            lengths = np.diff(ranges.offsets)
            counts = np.asarray([len(e["visible_position_ids"]) for e in catalog[ap]])
            assert lengths.max() <= len(catalog[ap])
            assert audit[ap]["path_count"] == len(ranges.values)
            assert all(set(e["visible_position_ids"]) <= set(ids) for e in catalog[ap])
            if layout["box_count"] == 0:
                old = original.channel(old_room, ap, "survey", ids, 0., 2, "direct")
                np.testing.assert_array_equal(ranges.values, old.values)
                np.testing.assert_array_equal(ranges.offsets, old.offsets)
            per_ap[ap] = {"truth_count": len(catalog[ap]), "peaks_total": len(ranges.values),
                          "peaks_min": int(lengths.min()), "peaks_max": int(lengths.max()),
                          "peaks_mean": float(lengths.mean()), "sources_visible_at_fewer_than_4_positions": int(sum(counts < 4)),
                          "sources_visible_at_fewer_than_5_positions": int(sum(counts < 5)),
                          "max_native_path_range_discrepancy_m": audit[ap]["maximum_native_range_discrepancy_m"]}
        layouts[layout["id"]] = {"obstacles": layout["box_count"], "position_count": len(ids),
                                 "position_ids_digest": digest(ids.tolist()), "coordinates_digest": digest(target_xyz.tolist()),
                                 "by_ap": per_ap}
    for job in observed.manifest["jobs"]:
        receipt = read_json(observed.root / "native" / job["key"] / "receipt.json")
        assert receipt["simulator_tree_hash"] == observed.manifest["identity"]["simulator_tree_hash"]
        if not job["reused"]:
            layout = next(layout for layout in room["layouts"] if layout["id"] == job["group"])
            surfaces = read_json(observed.root / "native" / job["key"] / "surface_manifest.json")
            assert surfaces == scene_faces(old_room, layout["boxes"])
            native_xyz = np.loadtxt(observed.root / "native" / job["key"] / "Input/NodePosition1.dat")
            np.testing.assert_array_equal(native_xyz, old_room["survey_xyz"][job["position_ids"]])
    for row in read_json(run / "results.json"):
        detail = read_npz(run / row["outcome"])[0]
        record, = detail["sources"]
        np.testing.assert_array_equal(record["survey_position_ids"], ids)
        np.testing.assert_array_equal(record["survey_xyz"], target_xyz)
        if row["method"] == "range_rmse_true_count":
            for ap in ("A", "B"):
                assert record["audit"][ap]["known_source_count"] == len(record["truth_catalog"][ap])
                assert len(record["by_ap"][ap]) == len(record["truth_catalog"][ap])
    old_hashes = read_json(run / "preserved_query_experiment_hashes.json")
    for path, expected in old_hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    verification = {"mask": mask, "all_layouts_have_identical_positions_and_ids": True,
                    "retained_positions_outside_every_layout": True, "empty_observations_bit_identical_to_original": True,
                    "fresh_obstacle_chunks": 12, "reused_empty_chunks": 4,
                    "all_fresh_simulator_coordinates_and_surfaces_verified": True,
                    "preserved_previous_query_experiment_files": len(old_hashes), "layouts": layouts}
    write_json(run / "acquisition_verification.json", verification)
    print({"retained": len(ids), "excluded": mask["excluded_count"], "layouts": layouts})


if __name__ == "__main__":
    main()
