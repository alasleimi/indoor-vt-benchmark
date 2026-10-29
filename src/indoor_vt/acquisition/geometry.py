"""Pure room geometry and deterministic sampling in the canonical frame."""

import math
import numpy as np
from scipy.spatial.distance import pdist
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate


def polygon_for(family: str, width: float, depth: float, rng, cfg) -> np.ndarray:
    w, d = width, depth
    if family == "rectangle":
        p = [(0, 0), (w, 0), (w, d), (0, d)]
    elif family == "L":
        a, b = rng.uniform(*cfg["L_cut_fraction"], 2) * [w, d]
        p = [(0, 0), (w, 0), (w, d - b), (w - a, d - b), (w - a, d), (0, d)]
    elif family == "T":
        stem, bar = rng.uniform(*cfg["T_fraction"], 2) * [w, d]
        left, right = (w - stem) / 2, (w + stem) / 2
        p = [
            (left, 0),
            (right, 0),
            (right, d - bar),
            (w, d - bar),
            (w, d),
            (0, d),
            (0, d - bar),
            (left, d - bar),
        ]
    elif family == "oblique":
        a, b, c, e = rng.uniform(*cfg["oblique_cut_fraction"], 4) * [w, d, w, d]
        p = [(a, 0), (w, 0), (w, d - e), (w - c, d), (0, d), (0, b)]
    else:
        raise ValueError(f"Unknown room family {family}")
    return np.asarray(p, dtype=float)


def sample_inside(polygon, rng, maximum_attempts, accept=lambda p: True):
    lower, upper = np.array(polygon.bounds[:2]), np.array(polygon.bounds[2:])
    for attempt in range(1, maximum_attempts + 1):
        p = rng.uniform(lower, upper)
        if polygon.contains(Point(p)) and accept(p):
            return p, attempt
    raise RuntimeError("Geometric rejection sampling exhausted the declared attempt budget")


def room_design(family, split, index, cfg, seeds) -> dict:
    rid = f"{family}_{split}_{index:02d}"
    rng = seeds.rng(rid, "geometry")
    w, d = rng.uniform(*cfg["width_m"]), rng.uniform(*cfg["depth_m"])
    ceiling = float(rng.uniform(*cfg["ceiling_m"]))
    local = polygon_for(family, w, d, rng, cfg)
    polygon = Polygon(local)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Invalid sampled polygon")
    diameter = float(np.max(pdist(local)))
    maximum = cfg["maximum_sampling_attempts"]
    ap_rng = seeds.rng(rid, "ap_pair")
    for pair_attempt in range(1, maximum + 1):
        def clear_ap(p):
            return polygon.boundary.distance(Point(p)) >= cfg["ap_wall_clearance_m"]
        a, _ = sample_inside(polygon, ap_rng, maximum, clear_ap)
        b, _ = sample_inside(polygon, ap_rng, maximum, clear_ap)
        if np.linalg.norm(a - b) >= cfg["ap_separation_diameter_fraction"] * diameter:
            break
    else:
        raise RuntimeError("AP pair sampling exhausted")
    aps_local = np.array([a, b])
    aps = {name: [*xy, cfg["ap_height_m"]] for name, xy in zip(("A", "B"), aps_local)}
    spacing, offset = cfg["survey_spacing_m"], cfg["survey_grid_offset_m"]
    lattice = np.array(
        [
            (ix, iy)
            for iy, y in enumerate(np.arange(offset, d, spacing))
            for ix, x in enumerate(np.arange(offset, w, spacing))
            if polygon.contains(Point(x, y))
        ], dtype=np.int64,
    )
    grid = offset + spacing * lattice
    survey = np.array([[*xy, z] for z in cfg["survey_heights_m"] for xy in grid])
    survey_lattice = np.array([[*ij, h] for h in range(len(cfg["survey_heights_m"]))
                              for ij in lattice], dtype=np.int64)
    layouts = []
    clear = cfg["clearance_m"]
    for li, count in enumerate(cfg["layouts"]):
        lid = f"layout_{li:02d}"
        box_rng = seeds.rng(rid, lid, "boxes")
        boxes, footprints, attempts = [], [], 0
        while len(boxes) < count:
            attempts += 1
            if attempts > maximum:
                raise RuntimeError(f"Box sampling exhausted for {rid}/{lid}")
            bw = float(box_rng.uniform(*cfg["box_width_m"]))
            bd = float(box_rng.uniform(*cfg["box_depth_m"]))
            bh = float(box_rng.uniform(*cfg["box_height_m"]))
            yaw = float(box_rng.uniform(*cfg["box_yaw_deg"]))
            center = box_rng.uniform([0, 0], [w, d])
            theta = math.radians(yaw)
            rot = np.array(
                [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
            )
            xy = (
                np.array(
                    [[-bw / 2, -bd / 2], [bw / 2, -bd / 2], [bw / 2, bd / 2], [-bw / 2, bd / 2]]
                )
                @ rot.T
                + center
            )
            footprint = Polygon(xy)
            if (
                not polygon.contains(footprint)
                or footprint.distance(polygon.boundary) < clear["box_wall"]
            ):
                continue
            if any(footprint.distance(Point(p)) < clear["box_ap"] for p in aps_local):
                continue
            if any(footprint.distance(previous) < clear["box_box"] for previous in footprints):
                continue
            footprints.append(footprint)
            boxes.append(
                {
                    "footprint_xy": xy.tolist(),
                    "height_m": bh,
                    "width_m": bw,
                    "depth_m": bd,
                    "yaw_local_deg": yaw,
                    "material": "MetalObstacle",
                }
            )
        qrng, hrng = seeds.rng(rid, lid, "query_xy"), seeds.rng(rid, lid, "query_height")
        queries, query_attempts = [], 0
        for _ in range(cfg["query_count_per_layout"]):
            point, n = sample_inside(
                polygon,
                qrng,
                maximum,
                lambda p: all(Point(p).distance(box) >= clear["query_box"] for box in footprints),
            )
            query_attempts += n
            queries.append([*point, float(hrng.uniform(*cfg["query_height_m"]))])
        layouts.append(
            {
                "id": lid,
                "box_count": count,
                "boxes": boxes,
                "queries_xyz": np.asarray(queries),
                "box_attempts": attempts,
                "query_attempts": query_attempts,
            }
        )
    return {
        "id": rid,
        "family": family,
        "split": split,
        "index": index,
        "width_m": float(w),
        "depth_m": float(d),
        "ceiling_m": ceiling,
        "polygon_xy": local.tolist(),
        "diameter_m": diameter,
        "access_points_xyz": aps,
        "ap_pair_attempts": pair_attempt,
        "survey_xyz": survey,
        "survey_lattice": survey_lattice,
        "survey_position_ids": np.arange(len(survey), dtype=np.int64),
        "survey_spacing_m": spacing,
        "survey_grid_origin_m": [offset, offset],
        "survey_xy_count": len(grid),
        "layouts": layouts,
    }


def scene_faces(room: dict, boxes=()) -> list[dict]:
    """Planar faces with explicit finite polygons and stable physical surface IDs."""
    p = np.asarray(room["polygon_xy"])
    h = room["ceiling_m"]
    faces = []
    for i, (a, b) in enumerate(zip(p, np.roll(p, -1, axis=0))):
        vertices = [[*a, 0], [*b, 0], [*b, h], [*a, h]]
        faces.append(
            {
                "id": f"wall_{i}",
                "material": "Walls",
                "vertices": vertices,
                "triangles": [[0, 1, 2], [0, 2, 3]],
            }
        )
    # Separate finite triangles avoid treating the concave room floor as its convex hull.
    triangles = [t for t in triangulate(Polygon(p)) if Polygon(p).covers(t)]
    for name, z, material in (("floor", 0, "Floor"), ("ceiling", h, "Ceiling")):
        for i, triangle in enumerate(triangles):
            v = [[float(x), float(y), float(z)] for x, y in list(triangle.exterior.coords)[:3]]
            faces.append(
                {
                    "id": f"{name}_{i}",
                    "surface_id": name,
                    "material": material,
                    "vertices": v,
                    "triangles": [[0, 1, 2]],
                }
            )
    for i, box in enumerate(boxes):
        q, bh = np.asarray(box["footprint_xy"]), box["height_m"]
        for j, (a, b) in enumerate(zip(q, np.roll(q, -1, axis=0))):
            faces.append(
                {
                    "id": f"box_{i}_side_{j}",
                    "material": "MetalObstacle",
                    "vertices": [[*a, 0], [*b, 0], [*b, bh], [*a, bh]],
                    "triangles": [[0, 1, 2], [0, 2, 3]],
                }
            )
        faces.append(
            {
                "id": f"box_{i}_top",
                "material": "MetalObstacle",
                "vertices": [[*v, bh] for v in q],
                "triangles": [[0, 1, 2], [0, 2, 3]],
            }
        )
    return faces
