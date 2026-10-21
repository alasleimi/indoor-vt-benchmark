"""Versioned observation artifacts; rich native fields never enter estimator records."""

from pathlib import Path
import os
import numpy as np
from .contracts import DelayObservation, Survey
from .observations import direct_observation, native_frames, receiver_observation
from .provenance import digest, file_hash, read_json, write_json


def pack(rows):
    offsets = np.r_[0, np.cumsum([len(r) for r in rows])].astype(np.int64)
    return np.concatenate(rows) if rows else np.empty(0), offsets


def unpack(values, offsets):
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def conditions(config):
    return [("direct", str(v), v) for v in config["direct_sigma_m"]] + [
        ("receiver", str(v), v) for v in config["receiver_snr_db"]
    ]


def materialize(scenario, config, seeds):
    scenario = Path(scenario)
    manifest = read_json(scenario / "input_manifest.json")
    receipt = read_json(scenario / "receipt.json")
    native = scenario / "native/qdOutput.json.gz"
    if file_hash(native) != receipt["outputs"]["native/qdOutput.json.gz"]:
        raise RuntimeError("Native archive checksum mismatch")
    frames = list(native_frames(native))
    if len(frames) != manifest["frames"]:
        raise RuntimeError("Native frame count mismatch")
    output = []
    for track, level, value in conditions(config):
        base = scenario / "observations" / track / level
        identity = {
            "native_hash": receipt["native_uncompressed_sha256"],
            "config": config,
            "track": track,
            "level": level,
            "frontend_hash": file_hash(Path(__file__).with_name("observations.py")),
            "storage_hash": file_hash(Path(__file__)),
            "root_seed": seeds.root,
            "namespace": seeds.namespace,
        }
        identity_hash = digest(identity)
        destination = base / "versions" / identity_hash
        if (base / "manifest.json").exists():
            old = read_json(base / "manifest.json")
            for name, expected in old["files"].items():
                if file_hash(base / name) != expected:
                    raise RuntimeError("Observation checksum mismatch")
            if old["identity_hash"] == identity_hash:
                output.append(old)
                continue
            # Preserve the old manifest and every old artifact; only the active
            # pointer changes after a new version has finished atomically.
            archive = base / "versions" / old["identity_hash"] / "manifest.json"
            if not archive.exists():
                write_json(archive, old)
        if (destination / "manifest.json").exists():
            record = read_json(destination / "manifest.json")
            for name, expected in record["files"].items():
                if file_hash(base / name) != expected:
                    raise RuntimeError("Versioned observation checksum mismatch")
            write_json(base / "manifest.json", record)
            output.append(record)
            continue
        rows, audits, arrays = [], [], {}
        for i, frame in enumerate(frames):
            frame_id = manifest["position_indices"][i]
            key = ("measurement", manifest["key"], frame_id, track, level)
            rng = seeds.rng(*key)
            if track == "direct":
                row, audit = direct_observation(frame, value, rng)
            else:
                row, rich = receiver_observation(frame, value, rng, config["receiver"])
                audit = {
                    "noise_variance": rich.pop("noise_variance"),
                    "threshold": rich.pop("threshold"),
                }
                for name, array in rich.items():
                    arrays[f"frame_{frame_id}/{name}"] = array
            rows.append(row)
            audits.append({"frame_id": frame_id, "seed": seeds.seed(*key), **audit})
        values, offsets = pack(rows)
        atomic_npz(destination / "delays.npz", {"ranges_m": values, "offsets": offsets})
        prefix = destination.relative_to(base).as_posix()
        files = {prefix + "/delays.npz": file_hash(destination / "delays.npz")}
        if arrays:
            atomic_npz(destination / "receiver_audit.npz", arrays)
            files[prefix + "/receiver_audit.npz"] = file_hash(destination / "receiver_audit.npz")
        record = {
            "schema": "indoor-vt.observations.v1",
            "identity": identity,
            "identity_hash": identity_hash,
            "files": files,
            "delay_file": prefix + "/delays.npz",
            "frames": len(rows),
            "peak_count": len(values),
            "measurements": audits,
        }
        write_json(destination / "manifest.json", record)
        write_json(base / "manifest.json", record)
        output.append(record)
    return output


def delay_file(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    name = manifest.get("delay_file", "delays.npz")
    path = directory / name
    if file_hash(path) != manifest["files"][name]:
        raise RuntimeError("Estimator observations have changed")
    return path


def observation_signature(scenario_root, track="*"):
    root = Path(scenario_root)
    return digest(
        {
            p.relative_to(root).as_posix(): file_hash(p)
            for p in sorted(root.glob(f"*/*/*/observations/{track}/*/manifest.json"))
        }
    )


def load_observations(scenario_root, room_id, subset, layout, track, level):
    channels = {}
    for ap in subset:
        directory = (
            Path(scenario_root) / room_id / ap / layout / "observations" / track / str(level)
        )
        with np.load(delay_file(directory), allow_pickle=False) as z:
            channels[ap] = unpack(z["ranges_m"], z["offsets"])
    lengths = {len(rows) for rows in channels.values()}
    if len(lengths) != 1:
        raise ValueError("AP channels must use the same sample positions")
    return tuple(
        DelayObservation({ap: rows[i] for ap, rows in channels.items()})
        for i in range(next(iter(lengths)))
    )


def load_survey(scenario_root, room, subset, track, level):
    observations = load_observations(scenario_root, room["id"], subset, "survey", track, level)
    return Survey(np.asarray(room["survey_xyz"]), observations)
