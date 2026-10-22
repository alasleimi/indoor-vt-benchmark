"""Exercise two real worker processes on a small, separately supplied test fixture."""

import sys
from pathlib import Path

import run_revision_campaign as runner
from indoor_vt.registry import initial_default


def main():
    root = Path(sys.argv[1]).resolve()
    jobs = [
        {
            "id": f"preset/direct/coarse/{name}",
            "kind": "preset",
            "phase": 0,
            "order": 0,
            "stage": "coarse",
            "method": name,
            "track": "direct",
            "parameters": initial_default(name),
            "dependencies": [],
        }
        for name in ("mca", "chamfer")
    ]
    runner.build_jobs = lambda: jobs
    sleep = runner.time.sleep
    runner.time.sleep = lambda _: sleep(0.05)
    runner.run(
        root,
        "development",
        "memory_fixture",
        root / "shadow",
        workers=1,
        limits={"working_set_bytes": 128 * 1024**2, "private_commit_bytes": 512 * 1024**2},
    )


if __name__ == "__main__":
    main()
