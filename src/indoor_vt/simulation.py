"""Native NIST adapter. No synthetic path substitute is used by the dataset generator."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

from .generation import scene_faces
from .provenance import SeedBook, digest, file_hash, read_json, write_json


def amf(faces, materials) -> str:
    root = ET.Element("amf", unit="meter")
    ids = {name: i for i, name in enumerate(materials, 1)}
    for i, face in enumerate(faces):
        obj = ET.SubElement(root, "object", id=str(i))
        ET.SubElement(obj, "metadata", type="name").text = face["id"]
        mesh = ET.SubElement(obj, "mesh")
        vertices = ET.SubElement(mesh, "vertices")
        for v in face["vertices"]:
            coordinates = ET.SubElement(ET.SubElement(vertices, "vertex"), "coordinates")
            for axis, value in zip(("x", "y", "z"), v):
                ET.SubElement(coordinates, axis).text = format(float(value), ".17g")
        volume = ET.SubElement(mesh, "volume", materialid=str(ids[face["material"]]))
        for triangle in face["triangles"]:
            element = ET.SubElement(volume, "triangle")
            for j, vertex in enumerate(triangle, 1):
                ET.SubElement(element, f"v{j}").text = str(vertex)
    for name, identifier in ids.items():
        ET.SubElement(
            ET.SubElement(root, "material", id=str(identifier)), "metadata", type="name"
        ).text = name
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def prepare_scenario(
    root: Path,
    room: dict,
    ap: str,
    layout: dict | None,
    cfg: dict,
    seeds: SeedBook,
    qd_root: Path,
    sample_indices=None,
) -> dict:
    label = "survey" if layout is None else layout["id"]
    key = f"{room['id']}/{ap}/{label}"
    scenario = root / key
    points = np.array(room["survey_xyz"] if layout is None else layout["queries_xyz"])
    if sample_indices is not None:
        points = points[sample_indices]
    faces = scene_faces(room, () if layout is None else layout["boxes"])
    seed = seeds.seed("simulation", key) % 2**32
    identity = {
        "key": key,
        "points": points.tolist(),
        "ap_xyz": room["access_points_xyz"][ap],
        "faces": faces,
        "config": cfg,
        "matlab_seed": seed,
    }
    input_hash = digest(identity)
    if (scenario / "input_manifest.json").exists():
        old = read_json(scenario / "input_manifest.json")
        if old["input_hash"] != input_hash:
            raise RuntimeError(f"Frozen simulator inputs differ: {scenario}")
        return old
    inputs = scenario / "Input"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "scene.amf").write_text(amf(faces, cfg["materials"]), encoding="utf-8")
    np.savetxt(
        inputs / "NodePosition0.dat", np.array(room["access_points_xyz"][ap])[None], fmt="%.17g"
    )
    np.savetxt(inputs / "NodePosition1.dat", points, fmt="%.17g")
    for node in (0, 1):
        np.savetxt(inputs / f"NodeRotation{node}.dat", np.zeros((1, 3)), fmt="%.17g")
    material_file = inputs / "materials.csv"
    material_file.write_text(
        "Reflector,Material,RelativePermittivity\n"
        + "".join(f"{name},{name},{value}\n" for name, value in cfg["materials"].items()),
        encoding="ascii",
    )
    material_relative = Path(
        os.path.relpath(material_file.resolve(), (qd_root / "src").resolve())
    ).as_posix()
    settings = {
        "environmentFileName": "scene.amf",
        "indoorSwitch": 1,
        "numberOfTimeDivisions": len(points),
        "referencePoint": "[0,0,0]",
        "selectPlanesByDist": "inf",
        "switchDiffuseComponent": int(cfg["diffuse"]),
        "switchQDModel": "tgayMeasurements",
        "totalNumberOfReflections": cfg["reflections"],
        "totalTimeDuration": len(points),
        "switchSaveVisualizerFiles": 1,
        "carrierFrequency": "60e9",
        "qdFilesFloatPrecision": 10,
        "useOptimizedOutputToFile": 1,
        "outputFormat": "json",
        "enablePhaseOutput": 1,
        "reflectionLoss": 10,
        "materialLibraryPath": material_relative,
    }
    (inputs / "paraCfgCurrent.txt").write_text(
        "ParameterName\tParameterValue\n"
        + "".join(f"{key}\t{value}\n" for key, value in settings.items()),
        encoding="ascii",
    )
    manifest = {
        "schema": "indoor-vt.scenario.v1",
        "input_hash": input_hash,
        "key": key,
        "room_id": room["id"],
        "split": room["split"],
        "ap": ap,
        "layout": label,
        "frames": len(points),
        "matlab_seed": seed,
        "position_indices": list(range(len(points)))
        if sample_indices is None
        else list(map(int, sample_indices)),
        "settings": settings,
        "input_files": {p.name: file_hash(p) for p in sorted(inputs.iterdir())},
    }
    write_json(scenario / "input_manifest.json", manifest)
    write_json(scenario / "surface_manifest.json", faces)
    return manifest


def simulator_provenance(qd_root: Path):
    files = {
        p.relative_to(qd_root).as_posix(): file_hash(p)
        for p in sorted((qd_root / "src").rglob("*.m"))
    }
    files.update(
        {
            "compat/" + p.name: file_hash(p)
            for p in (Path(__file__).parent / "matlab_compat").glob("*.m")
        }
    )
    return {
        "source": "NIST Q-D realization",
        "expected_revision": "1ccf0cb61c1741cc3471db8ec0373c5383da58be",
        "files": files,
        "tree_hash": digest(files),
    }


def run_scenario(scenario: Path, qd_root: Path, matlab: str, timeout_s=7200) -> dict:
    manifest = read_json(scenario / "input_manifest.json")
    provenance = simulator_provenance(qd_root)
    receipt_path = scenario / "receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (
            receipt["input_hash"] != manifest["input_hash"]
            or receipt["simulator_tree_hash"] != provenance["tree_hash"]
        ):
            raise RuntimeError("Existing simulator receipt has incompatible provenance")
        for name, expected in receipt["outputs"].items():
            if file_hash(scenario / name) != expected:
                raise RuntimeError(f"Corrupt cached native output: {name}")
        return receipt
    for name, expected in manifest["input_files"].items():
        if file_hash(scenario / "Input" / name) != expected:
            raise RuntimeError(f"Modified frozen input: {name}")
    # Each attempt writes in a new folder. Failed native outputs are preserved for diagnosis.
    attempts = scenario / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f"attempt_{len(list(attempts.iterdir())):03d}"
    attempt.mkdir()
    shutil.copytree(scenario / "Input", attempt / "Input")

    # Material paths were relative to the simulator's source directory, not the scenario.
    def quote(p):
        return str(Path(p).resolve()).replace("\\", "/").replace("'", "''")

    script = attempt / "run.m"
    script.write_text(
        f"cd('{quote(qd_root / 'src')}');\n"
        "addpath('raytracer','utils','utils/quaternions');\n"
        f"addpath('{quote(Path(__file__).parent / 'matlab_compat')}','-begin');\n"
        f"rng({manifest['matlab_seed']}, 'twister');\n"
        f"paraCfg=parameterCfg('{quote(attempt)}');\n"
        "[paraCfg,nodeCfg]=nodeProfileCfg(paraCfg);\n"
        f"save('{quote(attempt / 'effective_configuration.mat')}','paraCfg','nodeCfg');\n"
        "outputPath=Raytracer(paraCfg,nodeCfg);\n"
        f"fid=fopen('{quote(attempt / 'output_path.txt')}','w'); fprintf(fid,'%s',outputPath); fclose(fid);\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    with (attempt / "matlab.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [matlab, "-batch", f"run('{quote(script)}')"],
            cwd=qd_root / "src",
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if result.returncode:
        raise RuntimeError(f"NIST failed ({result.returncode}); inspect {attempt / 'matlab.log'}")
    native = list(attempt.rglob("qdOutput.json"))
    if len(native) != 1:
        raise RuntimeError(f"Expected one NIST output, found {len(native)}")
    archive = scenario / "native"
    archive.mkdir(exist_ok=True)
    # Deterministic gzip headers; original JSON bytes remain exactly recoverable.
    with native[0].open("rb") as source, (archive / "qdOutput.json.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as destination:
            shutil.copyfileobj(source, destination)
    outputs = {"native/qdOutput.json.gz": file_hash(archive / "qdOutput.json.gz")}
    receipt = {
        "schema": "indoor-vt.simulation.v1",
        "input_hash": manifest["input_hash"],
        "simulator_tree_hash": provenance["tree_hash"],
        "elapsed_s": time.perf_counter() - started,
        "attempt": attempt.relative_to(scenario).as_posix(),
        "frames": manifest["frames"],
        "native_uncompressed_sha256": file_hash(native[0]),
        "outputs": outputs,
    }
    write_json(scenario / "simulator_provenance.json", provenance)
    write_json(receipt_path, receipt)
    return receipt
