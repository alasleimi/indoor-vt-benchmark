"""Compare both browser overlays with production MPUrge using existing published inputs."""

import gzip
import json
from pathlib import Path
import subprocess

from indoor_vt.matching import PatternMatcher

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
        conditions.append(dict(ap=condition["ap"], noise=condition["noise"], matches=matches))
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
