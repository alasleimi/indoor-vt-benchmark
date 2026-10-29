"""Real NIST acquisition followed by frozen reconstruction; no tuning."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(key, "1")
from pathlib import Path  # noqa: E402 - numerical libraries must see thread limits before import
from indoor_vt.acquisition.records import read_json, write_json  # noqa: E402 - numerical libraries must see thread limits before import
from indoor_vt.acquisition.matched_obstacles import acquire_matched_obstacles  # noqa: E402 - numerical libraries must see thread limits before import
from indoor_vt.execution.obstacle_reconstruction import run_obstacle_reconstruction  # noqa: E402 - numerical libraries must see thread limits before import

if __name__ == "__main__":
    original = Path("data/redesign/rectangle-20261025")
    destination = Path("data/redesign/rectangle-matched-obstacles-20261027")
    run = Path("runs/redesign/rectangle-matched-obstacle-reconstruction-v1")
    config = read_json("configs/redesign.json")
    try:
        manifest = acquire_matched_obstacles(original, destination, config,
            "../simulator/nist-qd-realization", "C:/Program Files/MATLAB/R2026a/bin/matlab.exe")
        mask = manifest["common_survey_masks"]["rectangle_scene_00"]
        count = mask["retained_count"]
        protocol = {
            "input": f"The same {count} original survey positions for every saved layout; real layout-specific NIST observations; sigma=0",
            "positions": "One common subset of the original 280 survey positions, outside all boxes in every layout; original IDs, order and coordinates retained; no random query positions",
            "truth": f"Union of distinct source images visible in archived NIST paths at these same {count} receivers for each layout/AP, direct plus up to 2 reflections; no never-observed sources",
            "obstacles": "Existing 0, 1, 3 and 5-box layouts, freshly simulated at the common survey subset; no new geometry",
            "inside_obstacles": mask["occupancy"],
            "control": f"Zero-box layout reuses the original empty-survey native fingerprints at exactly these same {count} positions; no interpolation or resimulation",
            "common_survey_mask": mask,
            "preserved_experiments": "Original 280-position empty-room noise evaluation and earlier 100-random-query-position obstacle experiment remain unchanged in their original directories",
            "input_protocol": "matched_survey"}
        run_obstacle_reconstruction(destination, "runs/redesign/rectangle-clean-selection-v4", run,
            selection_data_root=original, study=run.name, protocol_overrides=protocol)
    except Exception as error:
        write_json(run / "status.json", {"state": "failed", "error": str(error)})
        raise
