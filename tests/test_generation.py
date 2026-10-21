from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon

from indoor_vt.generation import generate_design, scene_faces
from indoor_vt.provenance import SeedBook

ROOT = Path(__file__).resolve().parents[1]


def test_named_streams_do_not_depend_on_execution_order():
    a, b = SeedBook(42, "test"), SeedBook(42, "test")
    first = a.rng("geometry").normal(size=10)
    b.rng("noise").normal(size=100)
    np.testing.assert_array_equal(first, b.rng("geometry").normal(size=10))
    assert a.seed("noise", 0.1) != a.seed("noise", 0.2)


def test_complete_design_constraints_and_replay(tmp_path):
    design = generate_design(ROOT/"configs/dataset.json", ROOT/"configs/seeds.json", "pilot", tmp_path/"design.json")
    repeated = generate_design(ROOT/"configs/dataset.json", ROOT/"configs/seeds.json", "pilot", tmp_path/"design.json")
    assert design == repeated
    assert len(design["rooms"]) == 16
    for room in design["rooms"]:
        polygon = Polygon(room["polygon_xy"])
        xyz = np.asarray(room["survey_xyz"])
        n = room["survey_xy_count"]
        assert len(xyz) == 2*n
        np.testing.assert_array_equal(xyz[:n, :2], xyz[n:, :2])
        assert np.all(xyz[:n, 2] == 1) and np.all(xyz[n:, 2] == 1.5)
        assert np.linalg.matrix_rank(xyz-xyz.mean(0)) == 3
        assert all(polygon.contains(Point(p[:2])) for p in xyz)
        aps = list(room["access_points_xyz"].values())
        assert np.linalg.norm(np.array(aps[0][:2])-aps[1][:2]) >= 0.4*room["diameter_m"]
        floor = [face for face in scene_faces(room) if face.get("surface_id") == "floor"]
        assert abs(sum(Polygon(np.array(f["vertices"])[:, :2]).area for f in floor)-polygon.area) < 1e-8
        for layout in room["layouts"]:
            boxes = [Polygon(b["footprint_xy"]) for b in layout["boxes"]]
            assert len(boxes) == layout["box_count"]
            for i, box in enumerate(boxes):
                assert polygon.contains(box)
                assert box.distance(polygon.boundary) >= 0.1-1e-9
                assert all(box.distance(Point(ap[:2])) >= 0.5-1e-9 for ap in aps)
                assert all(box.distance(other) >= 0.25-1e-9 for other in boxes[:i])
            assert len(layout["queries_xyz"]) == 100
            for q in layout["queries_xyz"]:
                assert polygon.contains(Point(q[:2]))
                assert 1 <= q[2] <= 1.5
                assert all(box.distance(Point(q[:2])) >= 0.25-1e-9 for box in boxes)
