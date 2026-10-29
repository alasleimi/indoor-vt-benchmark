"""Native rich-channel archive and separate delay-only observation frontends.

Receiver equations adapted from publication/sapp/src/sapp/receiver.py (local
SAPP manuscript implementation). Changes: no count cap, complete receiver audit
arrays, and explicit immutable observation conversion. See docs/provenance.md.
"""

from __future__ import annotations
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import math
import numpy as np

C_MPS = 3.0e8


@dataclass(frozen=True)
class NativeFrame:
    delay_s: np.ndarray
    gain_db: np.ndarray
    phase_rad: np.ndarray
    aoa_az_deg: np.ndarray
    aoa_el_deg: np.ndarray
    aod_az_deg: np.ndarray
    aod_el_deg: np.ndarray

    @property
    def path_count(self):
        return len(self.delay_s)


def native_frames(path: Path, tx=0, rx=1):
    opener = gzip.open if path.suffix == ".gz" else open
    keys = ["Delay", "Gain", "Phase", "AOAAZ", "AOAEL", "AODAZ", "AODEL"]
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if (record["TX"], record["RX"], record["PAA_TX"], record["PAA_RX"]) != (tx, rx, 0, 0):
                continue
            for i in range(len(record["Delay"])):
                values = [np.atleast_1d(record[key][i]).astype(float) for key in keys]
                if len({len(v) for v in values}) != 1:
                    raise ValueError("Inconsistent native field lengths")
                if not all(np.isfinite(v).all() for v in values):
                    raise ValueError(
                        "Nonfinite native paths must be diagnosed, not silently removed"
                    )
                yield NativeFrame(*values)
            return
    raise ValueError("Expected NIST AP-to-receiver channel was not found")


def angle_to_unit(azimuth, polar_elevation):
    az, el = np.deg2rad(azimuth), np.deg2rad(polar_elevation)
    return np.column_stack((np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)))


def clean_receiver(frame: NativeFrame, config: dict):
    resolution = C_MPS / config["bandwidth_hz"]
    samples = np.arange(0, config["maximum_range_m"] + 0.5 * resolution, resolution)
    clean = np.zeros((8, len(samples)), dtype=np.complex128)
    ranges = frame.delay_s * C_MPS
    keep = (ranges >= 0) & (ranges <= config["maximum_range_m"])
    directions = angle_to_unit(frame.aoa_az_deg[keep], frame.aoa_el_deg[keep])
    amplitude = 10 ** (frame.gain_db[keep] / 20) * np.exp(1j * frame.phase_rad[keep])
    # Preserve SAPP's two explicitly different constants, rather than silently changing it.
    half = 299792458 / config["carrier_hz"] / 2
    positions = np.array(
        [[x * half, y * half, z * half] for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    )
    wavelength = C_MPS / config["carrier_hz"]
    for r, a, direction in zip(ranges[keep], amplitude, directions):
        steering = np.exp(-2j * np.pi * (positions @ direction) / wavelength)
        clean += a * steering[:, None] * np.sinc((samples - r) / resolution)[None, :]
    return samples, clean


def receiver_observation(frame: NativeFrame, snr: float, rng, config: dict):
    if config.get("maximum_peaks") is not None:
        raise ValueError("This benchmark forbids a receiver peak-count cap")
    samples, clean = clean_receiver(frame, config)
    variance = max(
        max(float(np.mean(abs(clean) ** 2, axis=0).max()), 1e-20) / 10 ** (snr / 10), 1e-20
    )
    observed = clean + math.sqrt(variance / 2) * (
        rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
    )
    power = np.mean(abs(observed) ** 2, axis=0)
    floor = max(float(np.median(power)) / math.log(2), variance, 1e-20)
    threshold = max(
        floor * 10 ** (config["threshold_above_noise_db"] / 10),
        max(float(power.max()), 1e-20) * 10 ** (-config["dynamic_range_db"] / 10),
    )
    local = np.zeros(len(power), bool)
    local[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
    local[0], local[-1] = power[0] > power[1], power[-1] > power[-2]
    indices = np.flatnonzero(local & (power >= threshold))
    logpower = np.log(np.maximum(power, 1e-30))
    ranges, kept = [], []
    resolution = C_MPS / config["bandwidth_hz"]
    for i in indices:
        offset = 0.0
        if config["parabolic_refinement"] and 0 < i < len(power) - 1:
            left, mid, right = logpower[i - 1 : i + 2]
            curvature = left - 2 * mid + right
            if np.isfinite(curvature) and curvature < -1e-12:
                offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
        value = (i + offset) * resolution
        if 0 < value <= config["maximum_range_m"]:
            ranges.append(value)
            kept.append(i)
    return np.asarray(ranges), {
        "sample_ranges": samples,
        "clean": clean,
        "observed": observed,
        "power": power,
        "peak_indices": np.asarray(kept, dtype=int),
        "peak_power": power[kept],
        "noise_variance": variance,
        "threshold": threshold,
    }
