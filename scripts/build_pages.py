"""Build GitHub Pages from committed compact data, with no simulator or Python packages."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output or root / "build/pages").resolve()
    if (
        output == root
        or output.is_relative_to(root / "data")
        or output.is_relative_to(root / "runs")
    ):
        raise ValueError("Pages output must be separate from generated data and experiment results")
    snapshot = root / "published/v6"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf8"))
    output.mkdir(parents=True, exist_ok=True)
    for record in manifest["files"]:
        source = (snapshot / record["file"]).resolve()
        if not source.is_relative_to(snapshot.resolve()):
            raise ValueError("Invalid snapshot path")
        with source.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != record["sha256"]:
                raise ValueError(f"Snapshot checksum mismatch: {record['file']}")
        # The input archive is downloadable from the repository; the viewer needs only metrics.
        if source.suffix == ".zip":
            continue
        destination = output / record["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in (
        "index.html",
        "style.css",
        "app.js",
        "data-client.js",
        "static-worker.js",
        "vt-map.js",
        "method-catalog.json",
    ):
        source = root / "web/results-explorer" / name
        if name == "index.html":
            output.joinpath(name).write_text(
                source.read_text(encoding="utf8").replace(
                    '<html lang="en">', '<html lang="en" data-mode="static">'
                ),
                encoding="utf8",
            )
        else:
            shutil.copyfile(source, output / name)
    (output / ".nojekyll").write_text("")
    print(f"Built Pages site: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(output=args.output)
