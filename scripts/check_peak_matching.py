"""Compare both browser overlays with production MPUrge using existing published inputs."""

import gzip
import json
from pathlib import Path
import subprocess
import numpy as np

from indoor_vt.matching import PatternMatcher
from build_peak_matching import first_round

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "published/v6"
    index = json.loads((base / "matching/index.json").read_text(encoding="utf8"))

    def unpack(descriptor):
        return json.loads(gzip.decompress((base / descriptor["file"]).read_bytes()))

    scene = unpack(index["scene"])
    matcher = PatternMatcher()
    conditions = []
    for condition in index["conditions"]:
        profiles = unpack(condition["data"])["profiles"]
        matches = [matcher.match_array(profiles[i]["ranges"], profiles[j]["ranges"]).tolist()
                   for i, j in scene["pairs"]]
        native = unpack(index["native"][condition["ap"]])["profiles"]
        labels = [np.asarray(native[i]["source_ids"])[p["native_indices"]] for i, p in enumerate(profiles)]
        counts, xs, ys = [], [], []
        for (i, j), final in zip(scene["pairs"], matches):
            ia, ib, delta, _ = first_round(np.asarray(profiles[i]["ranges"]), np.asarray(profiles[j]["ranges"]))
            ai, bi = {int(v): k for k, v in enumerate(ia)}, {int(v): k for k, v in enumerate(ib)}
            good = [bool(labels[i][int(a)] == labels[j][int(b)]) for a, b, _ in final]
            counts.append([sum(good), len(final), len(np.intersect1d(labels[i], labels[j]))])
            xs.extend(float(delta[ai[int(a)], bi[int(b)]]) for a, b, _ in final)
            ys.extend(good)
        tp, predicted, available = np.sum(counts, axis=0).tolist()
        correlation = float(np.corrcoef(xs, ys)[0, 1]) if np.std(xs) and np.std(ys) else None
        summary = dict(tp=tp, fp=predicted-tp, fn=available-tp, precision=tp/predicted,
                       recall=tp/available, f1=2*tp/(predicted+available), pairs=len(counts),
                       accepted_correlation=correlation)
        conditions.append(dict(ap=condition["ap"], noise=condition["noise"], matches=matches,
                               pair_counts=counts, summary=summary))
    cases = [([], []), ([1], []), ([1, 1, 2], [1, 2]),
             ([4, 1, 3, 1, 8], [7, 2, 1]), ([0, 0, 0], [0, 0, 0])]
    special = [dict(a=a, b=b, matches=matcher.match_array(a, b).tolist()) for a, b in cases]
    output = ROOT / "runs/development/publication-checkpoint/peak-matching/final-validation.json.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(json.dumps(dict(conditions=conditions, special=special),
                                               separators=(",", ":")).encode(), mtime=0))
    subprocess.run(["node", str(ROOT / "scripts/check_peak_matching.mjs"),
                    "--final-fixture", str(output)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
