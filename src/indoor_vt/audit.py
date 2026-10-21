"""Native archive integrity, rich-field preservation and geometry-reference checks."""

from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from .observations import native_frames, angle_to_unit, C_MPS
from .dataset import conditions
from .provenance import read_json, write_json, file_hash, code_provenance, SeedBook, digest


def audit_observations(scenario, config, seeds, frames):
    scenario = Path(scenario)
    acquisition = read_json(scenario / "input_manifest.json")
    records, missing = [], []
    for track, level, _ in conditions(config):
        directory = scenario / "observations" / track / level
        path = directory / "manifest.json"
        if not path.exists():
            missing.append(f"{acquisition['key']}/{track}/{level}")
            continue
        manifest = read_json(path)
        if digest(manifest["identity"]) != manifest["identity_hash"]:
            raise RuntimeError(f"Observation identity changed: {path}")
        for name, expected in manifest["files"].items():
            if file_hash(directory / name) != expected:
                raise RuntimeError(f"Observation checksum mismatch: {directory / name}")
        with np.load(directory / manifest["delay_file"], allow_pickle=False) as arrays:
            values, offsets = arrays["ranges_m"], arrays["offsets"]
            if (
                len(offsets) != len(frames) + 1
                or offsets[0] != 0
                or offsets[-1] != len(values)
                or np.any(np.diff(offsets) < 0)
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or len(values) != manifest["peak_count"]
                or manifest["frames"] != len(frames)
                or len(manifest["measurements"]) != len(frames)
            ):
                raise RuntimeError(f"Invalid observation arrays: {path}")
            for i, frame in enumerate(frames):
                row = values[offsets[i] : offsets[i + 1]]
                measurement = manifest["measurements"][i]
                frame_id = acquisition["position_indices"][i]
                seed = seeds.seed("measurement", acquisition["key"], frame_id, track, level)
                if np.any(np.diff(row) < 0) or measurement["seed"] != seed:
                    raise RuntimeError(f"Observation ordering/seed mismatch: {path}")
                if track == "direct":
                    if len(row) + measurement["discarded_negative"] != frame.path_count:
                        raise RuntimeError(f"Unexplained native peak loss: {path}")
                    if float(level) == 0:
                        np.testing.assert_array_equal(row, np.sort(frame.delay_s * C_MPS))
        records.append(
            {
                "track": track,
                "level": level,
                "peaks": manifest["peak_count"],
                "manifest_hash": file_hash(path),
            }
        )
    return records, missing


def matlab_json(value):
    if isinstance(value, dict):
        return {str(k): matlab_json(v) for k, v in value.items() if not str(k).startswith("__")}
    if isinstance(value, np.ndarray):
        return matlab_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [matlab_json(v) for v in value]
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return matlab_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def audit_dataset(root, namespace, require_complete=True):
    root = Path(root)
    base = root / "data" / namespace
    design, truth = (
        read_json(base / "design.json"),
        read_json(base / "evaluation/source_truth.json"),
    )
    records, missing, missing_observations = [], [], []
    seeds = SeedBook(design["seeds"]["root"], namespace)
    for room in design["rooms"]:
        for ap in ("A", "B"):
            for label in ["survey", *[x["id"] for x in room["layouts"]]]:
                scenario = base / "scenarios" / room["id"] / ap / label
                if not (scenario / "receipt.json").exists():
                    missing.append(f"{room['id']}/{ap}/{label}")
                    continue
                receipt = read_json(scenario / "receipt.json")
                manifest = read_json(scenario / "input_manifest.json")
                for name, expected in receipt["outputs"].items():
                    if file_hash(scenario / name) != expected:
                        raise RuntimeError(f"Native checksum mismatch: {scenario / name}")
                for name, expected in manifest["input_files"].items():
                    if file_hash(scenario / "Input" / name) != expected:
                        raise RuntimeError(f"Simulator input changed: {scenario / name}")
                attempt = scenario / receipt["attempt"]
                resolved = matlab_json(
                    loadmat(attempt / "effective_configuration.mat", simplify_cells=True)
                )
                write_json(scenario / "effective_configuration.json", resolved)
                # Archive every emitted native field/path identity plus exact executed inputs.
                files = {
                    p.relative_to(scenario).as_posix(): {
                        "sha256": file_hash(p),
                        "bytes": p.stat().st_size,
                    }
                    for p in sorted(attempt.rglob("*"))
                    if p.is_file()
                }
                files["native/qdOutput.json.gz"] = {
                    "sha256": file_hash(scenario / "native/qdOutput.json.gz"),
                    "bytes": (scenario / "native/qdOutput.json.gz").stat().st_size,
                }
                files["effective_configuration.json"] = {
                    "sha256": file_hash(scenario / "effective_configuration.json"),
                    "bytes": (scenario / "effective_configuration.json").stat().st_size,
                }
                write_json(
                    scenario / "archive_manifest.json",
                    {"files": files, "receipt_hash": file_hash(scenario / "receipt.json")},
                )
                frames = list(native_frames(scenario / "native/qdOutput.json.gz"))
                if len(frames) != manifest["frames"]:
                    raise RuntimeError("Incomplete native channel")
                observations, absent = audit_observations(scenario, design["config"], seeds, frames)
                missing_observations.extend(absent)
                record = {
                    "key": manifest["key"],
                    "frames": len(frames),
                    "paths": sum(f.path_count for f in frames),
                    "elapsed_s": receipt["elapsed_s"],
                    "bytes": sum(v["bytes"] for v in files.values()),
                    "archive_hash": file_hash(scenario / "archive_manifest.json"),
                    "observations": observations,
                }
                if label == "survey":
                    refs = np.array([x["xyz"] for x in truth["sources"][room["id"]][ap]])
                    positions = np.array(room["survey_xyz"])[manifest["position_indices"]]
                    errors, observed = [], set()
                    for position, frame in zip(positions, frames):
                        images = position + (frame.delay_s * C_MPS)[:, None] * angle_to_unit(
                            frame.aoa_az_deg, frame.aoa_el_deg
                        )
                        if len(images):
                            distances = cdist(images, refs)
                            errors.extend(distances.min(axis=1))
                            observed.update(distances.argmin(axis=1).tolist())
                    record.update(
                        {
                            "reference_sources": len(refs),
                            "native_observed_sources": len(observed),
                            "maximum_native_image_error_m": max(errors, default=0.0),
                        }
                    )
                    # This tolerance checks simulator rounding, not the evaluation F1 gate.
                    if record["maximum_native_image_error_m"] > 0.01:
                        raise RuntimeError(
                            f"Native geometry mismatch requires investigation: {manifest['key']}"
                        )
                records.append(record)
    result = {
        "schema": "indoor-vt.archive_audit.v1",
        "namespace": namespace,
        "design_hash": design["design_hash"],
        "truth_hash": truth["truth_hash"],
        "completed_scenarios": len(records),
        "expected_scenarios": len(design["rooms"]) * 22,
        "frames": sum(r["frames"] for r in records),
        "missing": missing,
        "missing_observations": missing_observations,
        "completed_observation_conditions": sum(len(r["observations"]) for r in records),
        "expected_observation_conditions": len(design["rooms"])
        * 22
        * len(conditions(design["config"])),
        "scenarios": records,
        "audit_provenance": code_provenance(root),
    }
    write_json(base / "archive_audit.json", result)
    if missing and require_complete:
        raise RuntimeError(f"Dataset is incomplete: {len(missing)} scenarios remain")
    if missing_observations and require_complete:
        raise RuntimeError(
            f"Observations are incomplete: {len(missing_observations)} conditions remain"
        )
    return result
