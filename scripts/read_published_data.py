"""Load the Git LFS calibration snapshot without extracting or regenerating data."""

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from indoor_vt.contracts import DelayObservation, Survey


DEFAULT = Path(__file__).resolve().parents[1] / "published/v6/calibration-inputs.zip"


class CalibrationSnapshot:
    def __init__(self, path=DEFAULT):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as archive:
            self.design = json.loads(archive.read("design.json"))
            self.manifest = json.loads(archive.read("manifest.json"))
        self.rooms = {r["id"]: r for r in self.design["rooms"]}
        self.entries = {
            (r["room"], r["ap"], r["layout"], r["track"], float(r["level"])): r
            for r in self.manifest["observations"]
        }

    def observations(self, room, subset, layout, track, level):
        channels = {}
        with zipfile.ZipFile(self.path) as archive:
            for ap in subset:
                entry = self.entries[(room, ap, layout, track, float(level))]
                raw = archive.read(entry["file"])
                if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Observation checksum mismatch: {entry['file']}")
                with np.load(io.BytesIO(raw), allow_pickle=False) as packed:
                    ranges, offsets = packed["ranges_m"], packed["offsets"]
                    channels[ap] = [ranges[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        lengths = {len(rows) for rows in channels.values()}
        if len(lengths) != 1:
            raise ValueError("AP channels do not have matching frame counts")
        return tuple(
            DelayObservation({ap: rows[i] for ap, rows in channels.items()})
            for i in range(next(iter(lengths)))
        )

    def survey(self, room, subset=("A",), track="receiver", level=40):
        return Survey(
            np.array(self.rooms[room]["survey_xyz"]),
            self.observations(room, subset, "survey", track, level),
        )

    def queries(self, room, layout="layout_00", subset=("A",), track="receiver", level=40):
        """Return estimator observations and evaluator-only truth as separate objects."""
        geometry = next(x for x in self.rooms[room]["layouts"] if x["id"] == layout)
        observations = self.observations(room, subset, layout, track, level)
        truth = np.array(geometry["queries_xyz"])
        if len(observations) != len(truth):
            raise ValueError("Query coordinates do not match observation frames")
        return observations, truth

    def reference_sources(self, room, subset=("A",)):
        """Evaluator-only source bag; never pass it to a reconstruction method."""
        with zipfile.ZipFile(self.path) as archive:
            truth = json.loads(archive.read("source_truth.json"))
        return np.asarray([source["xyz"] for ap in subset for source in truth["sources"][room][ap]])


if __name__ == "__main__":
    snapshot = CalibrationSnapshot()
    print(
        f"{len(snapshot.rooms)} calibration rooms; {len(snapshot.entries)} observation conditions"
    )
    for room in snapshot.rooms:
        survey = snapshot.survey(room)
        queries, _ = snapshot.queries(room)
        print(
            f"{room}: {len(survey.observations)} survey positions; {len(queries)} empty-layout queries"
        )
