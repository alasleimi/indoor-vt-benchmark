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


def test_bounded_storage_never_creates_a_distance_file(tmp_path, monkeypatch):
    from indoor_vt import complete_link

    original = complete_link._bounded_chain
    capacities = []

    def capture(points, capacity):
        capacities.append(capacity)
        return original(points, capacity)

    monkeypatch.setattr(complete_link, "_bounded_chain", capture)
    points = np.random.default_rng(83).normal(size=(200, 3))
    for budget, capacity in ((0, 2), (200 * 8 * 3, 3), (200 * 8 * 17, 17)):
        actual = complete_linkage(points, memory_limit_bytes=budget, scratch_dir=tmp_path)
        np.testing.assert_array_equal(actual, linkage(points, method="complete"))
        assert capacities[-1] == capacity
    assert not list(tmp_path.iterdir())


def test_invalid_or_trivial_candidate_arrays():
    for n in (0, 1):
        assert complete_linkage(np.zeros((n, 3))).shape == (0, 4)
    for points in (np.array([1, 2]), np.array([[0, np.nan], [1, 2]])):
        with pytest.raises(ValueError):
            complete_linkage(points)


@pytest.mark.parametrize("budget", [0, 256 * 1024 * 1024])
def test_distance_overflow_rejected(budget):
    with pytest.raises(ValueError, match="finite"):
        complete_linkage(np.array([[1e308, 0], [-1e308, 0]]), memory_limit_bytes=budget)


def test_bounded_distance_kernel_matches_scipy_in_multiple_dimensions():
    rng = np.random.default_rng(490)
    for dimensions in (0, 1, 2, 3, 8, 31):
        for scale in (1e-100, 1, 1e100):
            points = rng.normal(size=(40, dimensions)) * scale
            np.testing.assert_array_equal(
                complete_linkage(points, memory_limit_bytes=0),
                linkage(points, method="complete"),
            )
