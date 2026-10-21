"""Privileged geometric image-source catalogue for empty extruded rooms.

Visibility is established by angular intervals bounded by polygon vertices and
portal endpoints. It does not depend on received power, path counts, sampled
survey visibility or any estimator prediction. Horizontal floor/ceiling paths
are enumerated separately. This module must never be imported by estimators.
"""

from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon
from .provenance import digest, read_json, write_json


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def ray_segment(origin, direction, a, b):
    edge = b - a
    denominator = cross(direction, edge)
    if abs(denominator) < 1e-12:
        return None
    delta = a - origin
    t, u = cross(delta, edge) / denominator, cross(delta, direction) / denominator
    if t > 1e-10 and -1e-9 <= u <= 1 + 1e-9:
        return t, origin + t * direction
    return None


def reflect(point, a, b):
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-tangent[1], tangent[0]])
    return point - 2 * np.dot(point - a, normal) * normal


def angular_intervals(origin, vertices):
    delta = np.asarray(vertices) - origin
    angles = np.unique(np.arctan2(delta[:, 1], delta[:, 0]) % (2 * np.pi))
    return [
        (float(a), float(b))
        for a, b in zip(angles, np.r_[angles[1:], angles[0] + 2 * np.pi])
        if b - a > 1e-12
    ]


def illuminated_portals(origin, polygon):
    edges = list(zip(polygon, np.roll(polygon, -1, axis=0)))
    output = []
    for lower, upper in angular_intervals(origin, polygon):
        direction = np.array([np.cos((lower + upper) / 2), np.sin((lower + upper) / 2)])
        hits = [
            (hit[0], i, hit[1])
            for i, (a, b) in enumerate(edges)
            if (hit := ray_segment(origin, direction, a, b)) is not None
        ]
        if not hits:
            continue
        _, i, center = min(hits, key=lambda h: (h[0], h[1]))
        endpoints = []
        for angle in (lower, upper):
            d = np.array([np.cos(angle), np.sin(angle)])
            hit = ray_segment(origin, d, *edges[i])
            if hit is not None:
                endpoints.append(hit[1])
        if len(endpoints) == 2 and np.linalg.norm(endpoints[1] - endpoints[0]) > 1e-9:
            output.append((i, endpoints[0], endpoints[1], center))
    return output


def source_catalogue(room, ap):
    polygon = np.asarray(room["polygon_xy"])
    shape = Polygon(polygon)
    edges = list(zip(polygon, np.roll(polygon, -1, axis=0)))
    source = np.array(room["access_points_xyz"][ap])
    height = room["ceiling_m"]
    entries = []

    def add(xyz, sequence, witness):
        xyz = np.asarray(xyz)
        for entry in entries:
            if np.linalg.norm(np.array(entry["xyz"]) - xyz) <= 1e-7:
                if sequence not in entry["surface_sequences"]:
                    entry["surface_sequences"].append(sequence)
                return
        entries.append({"xyz": xyz.tolist(), "surface_sequences": [sequence], "witness": witness})

    add(source, [], {"type": "physical_ap"})
    add([*source[:2], -source[2]], ["floor"], {"type": "vertical_path_at_ap_xy"})
    add([*source[:2], 2 * height - source[2]], ["ceiling"], {"type": "vertical_path_at_ap_xy"})
    add(
        [*source[:2], 2 * height + source[2]],
        ["floor", "ceiling"],
        {"type": "vertical_path_at_ap_xy"},
    )
    add(
        [*source[:2], source[2] - 2 * height],
        ["ceiling", "floor"],
        {"type": "vertical_path_at_ap_xy"},
    )
    portals = illuminated_portals(source[:2], polygon)
    for first, left, right, center in portals:
        first_name = f"wall_{first}"
        image = reflect(source[:2], *edges[first])
        witness = {"type": "illuminated_wall_interval", "portal": [left.tolist(), right.tolist()]}
        add([*image, source[2]], [first_name], witness)
        # Vertical mirrors commute with wall mirrors. The illuminated horizontal
        # path has a valid floor/ceiling bounce by choosing the final receiver near
        # the corresponding surface; neither survey layers nor query labels enter.
        for surface, z in (("floor", -source[2]), ("ceiling", 2 * height - source[2])):
            add([*image, z], [first_name, surface], witness)
            add([*image, z], [surface, first_name], witness)
        for lower, upper in angular_intervals(image, np.vstack((polygon, left, right))):
            angle = (lower + upper) / 2
            direction = np.array([np.cos(angle), np.sin(angle)])
            hit_portal = ray_segment(image, direction, left, right)
            if hit_portal is None:
                continue
            t0, first_point = hit_portal
            hits = [
                (hit[0], j, hit[1])
                for j, edge in enumerate(edges)
                if j != first
                and (hit := ray_segment(image, direction, *edge)) is not None
                and hit[0] > t0 + 1e-9
            ]
            if not hits:
                continue
            t1, second, second_point = min(hits, key=lambda h: (h[0], h[1]))
            if not shape.contains(Point(image + ((t0 + t1) / 2) * direction)):
                continue
            final_image = reflect(image, *edges[second])
            add(
                [*final_image, source[2]],
                [first_name, f"wall_{second}"],
                {
                    "type": "two_wall_illuminated_interval",
                    "angle_interval_rad": [lower, upper],
                    "reflection_xy": [first_point.tolist(), second_point.tolist()],
                },
            )
    entries.sort(key=lambda e: (min(map(len, e["surface_sequences"])), *e["xyz"]))
    for i, entry in enumerate(entries):
        entry["id"] = f"{ap}_source_{i:03d}"
    return entries


def generate_truth(design, destination):
    destination = Path(destination)
    references = {
        room["id"]: {ap: source_catalogue(room, ap) for ap in ("A", "B")}
        for room in design["rooms"]
    }
    result = {
        "schema": "indoor-vt.source_truth.v1",
        "design_hash": design["design_hash"],
        "scope": "all geometrically valid order<=2 sources in the empty survey-time room",
        "construction": "analytic mirrors with finite-wall visibility interval witnesses",
        "sources": references,
    }
    result["truth_hash"] = digest(result)
    if destination.exists() and read_json(destination) != result:
        raise RuntimeError("Refusing to replace a frozen source-reference catalogue")
    write_json(destination, result)
    return result


def plot_truth(design, references, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 4, figsize=(18, 18))
    for room, ax in zip(design["rooms"], axes.flat):
        polygon = np.array(room["polygon_xy"])
        ax.fill(polygon[:, 0], polygon[:, 1], color="lightgray", alpha=0.6)
        for ap, color in (("A", "tab:blue"), ("B", "tab:orange")):
            sources = np.array([e["xyz"] for e in references["sources"][room["id"]][ap]])
            real = room["access_points_xyz"][ap]
            ax.scatter(sources[:, 0], sources[:, 1], c=color, s=12, alpha=0.6)
            ax.scatter(*real[:2], c=color, marker="*", s=90)
        counts = [len(references["sources"][room["id"]][a]) for a in ("A", "B")]
        ax.set_title(f"{room['id']}\nA={counts[0]}, B={counts[1]} (3D sources)", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    figure.suptitle("Survey-time source catalogue: XY projection (height-distinct sources overlap)")
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)
