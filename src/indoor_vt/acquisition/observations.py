"""Derive observation values from frozen native frames, with stable noise IDs."""

from pathlib import Path
from collections import OrderedDict
from functools import lru_cache
import numpy as np

from .records import Seeds, read_json
from .dataset import survey_indices
from ..execution.store import Store
from ..science.models import Ragged, Survey, SourceMap
from .channel import NativeFrame, receiver_observation


def direct_ranges(delays_s, sigma_m, seed):
    clean = delays_s * 3.0e8
    noisy = clean.copy() if sigma_m == 0 else clean + np.random.default_rng(seed).normal(0, sigma_m, len(clean))
    valid = np.flatnonzero(noisy >= 0)
    ordered = valid[np.argsort(noisy[valid], kind="stable")]
    return {"ranges": noisy[ordered], "native_path_indices": ordered,
            "discarded_negative": len(clean) - len(valid)}


class Observations:
    def __init__(self, root, config, *, run=""):
        self.root, self.config, self.run = Path(root), config, run
        self.store = Store(self.root / "store", read_only=True)
        self._channels = OrderedDict()
        self.manifest = read_json(self.root / "dataset.json")
        self.rooms = self.store.get(self.manifest["scene_value"])
        self.seeds = Seeds(config["seed"])

    def available(self, priorities=(0, 1)):
        return all((self.root / "native" / j["key"] / "receipt.json").exists()
                   for j in self.manifest["jobs"] if j["priority"] in priorities)

    def channel(self, room, ap, group, position_ids, level, repetition, track="direct"):
        key = (room["id"], ap, group, tuple(map(int, position_ids)), level,
               repetition if level or track != "direct" else 0, track)
        if key in self._channels:
            self._channels.move_to_end(key)
            return self._channels[key]
        wanted = set(map(int, position_ids))
        rows = {}
        for job in self.manifest["jobs"]:
            if (job["room"], job["ap"], job["group"]) != (room["id"], ap, group):
                continue
            if not wanted.intersection(job["position_ids"]):
                continue
            receipt_path = self.root / "native" / job["key"] / "receipt.json"
            if not receipt_path.exists():
                raise FileNotFoundError(f"Acquisition still pending: {job['key']}")
            receipt = read_json(receipt_path)
            native = self._native(receipt["native_value"])
            for i, position_id in enumerate(native["position_ids"]):
                position_id = int(position_id)
                if position_id not in wanted:
                    continue
                paths = native["paths"][native["offsets"][i]:native["offsets"][i + 1]]
                seed = self.seeds.seed("measurement", room["id"], ap, group, position_id,
                                       track, level, repetition)
                if track == "direct":
                    values = direct_ranges(paths[:, 0], level, seed)
                elif track == "receiver":
                    receiver = self.config["observations"]["receiver"]
                    def receive(paths=paths, seed=seed):
                        ranges, audit = receiver_observation(NativeFrame(*paths.T), level,
                                                             np.random.default_rng(seed), receiver)
                        return {"ranges": ranges, "audit": audit}
                    values = receive()
                else:
                    raise ValueError(f"Unknown observation track: {track}")
                rows[position_id] = values["ranges"]
        if wanted != set(rows):
            raise ValueError("Acquisition does not contain all requested position IDs")
        result = Ragged.pack([rows[int(i)] for i in position_ids])
        self._channels[key] = result
        while len(self._channels) > 64:
            self._channels.popitem(last=False)
        return result

    @lru_cache(maxsize=16)
    def _native(self, key):
        return self.store.get(key)

    def survey(self, room, level, repetition, aps=("A", "B")):
        indices = survey_indices(room, **self.config["survey"])
        channels = {ap: self.channel(room, ap, "survey", indices, level, repetition,
                                     self.config["observations"]["track"]) for ap in aps}
        return Survey(room["survey_xyz"][indices], channels, room["survey_position_ids"][indices])

    def queries(self, room, layout, level, repetition, split, aps=("A", "B")):
        n = len(layout["queries_xyz"])
        boundary = int(n * self.config["observations"]["query_calibration_fraction"])
        ids = np.arange(boundary) if split == "calibration" else np.arange(boundary, n)
        channels = {ap: self.channel(room, ap, layout["id"], ids, level, repetition,
                                     self.config["observations"]["track"]) for ap in aps}
        return Survey(layout["queries_xyz"][ids], channels, ids)

    def truth(self, room, aps=("A", "B")):
        record = self.store.get(self.manifest["truth_value"])[room["id"]]
        return SourceMap({ap: np.asarray([s["xyz"] for s in record[ap]]) for ap in aps})
