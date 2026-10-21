import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage, fcluster

from indoor_vt.complete_link import complete_linkage


@pytest.mark.parametrize("storage_bytes", [0, 256 * 1024 * 1024])
def test_full_hierarchy_bit_exact_including_ties_duplicates_and_permutations(
    storage_bytes, tmp_path
):
    rng = np.random.default_rng(47)
    fixtures = [
        np.zeros((30, 3)),
        np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1.5, 0, 0]]),
        np.array(list(np.ndindex(4, 4, 4)), dtype=float),
    ]
    for n in (2, 3, 5, 20, 100, 250):
        fixtures += [rng.normal(size=(n, 3)), rng.integers(-3, 4, size=(n, 3)).astype(float)]
    for points in fixtures:
        for order in (np.arange(len(points)), rng.permutation(len(points))):
            points = points[order]
            expected = linkage(points, method="complete")
            actual = complete_linkage(
                points, memory_limit_bytes=storage_bytes, scratch_dir=tmp_path
            )
            np.testing.assert_array_equal(actual, expected)
            for radius in (0.0, 0.5, 1.0, expected[len(expected) // 2, 2]):
                for threshold in (
                    np.nextafter(radius, -np.inf),
                    radius,
                    np.nextafter(radius, np.inf),
                ):
                    np.testing.assert_array_equal(
                        fcluster(actual, threshold, criterion="distance"),
                        fcluster(expected, threshold, criterion="distance"),
                    )
    assert not list(tmp_path.iterdir())


def test_disk_workspace_removed_after_failure(tmp_path, monkeypatch):
    from indoor_vt import complete_link

    def fail(*args):
        raise RuntimeError("simulated computation failure")

    monkeypatch.setattr(complete_link, "_calculate", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        complete_linkage(np.zeros((3, 3)), memory_limit_bytes=0, scratch_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_invalid_or_trivial_candidate_arrays():
    for n in (0, 1):
        assert complete_linkage(np.zeros((n, 3))).shape == (0, 4)
    for points in (np.array([1, 2]), np.array([[0, np.nan], [1, 2]])):
        with pytest.raises(ValueError):
            complete_linkage(points)


def test_interrupted_worker_does_not_leave_distance_file(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    if os.name != "nt":
        pytest.skip("Windows delete-on-close verification")
    scratch = tmp_path / "scratch"
    ready = tmp_path / "ready"
    script = """
import os,sys,time
from pathlib import Path
import numpy as np
from indoor_vt import complete_link
def wait(points, distances):
    Path(sys.argv[2]).write_text(str(os.getpid()))
    time.sleep(30)
complete_link._calculate=wait
complete_link.complete_linkage(np.zeros((3,3)),memory_limit_bytes=0,scratch_dir=sys.argv[1])
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(scratch), str(ready)])
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists()
        assert list(scratch.iterdir())
        # A Windows venv launcher can have a separate interpreter child. Kill
        # the process actually holding the mapping, as the runner does.
        os.kill(int(ready.read_text()), signal.SIGTERM)
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    assert not list(scratch.iterdir())
