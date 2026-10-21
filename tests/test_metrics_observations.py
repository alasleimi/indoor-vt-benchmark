import numpy as np
from indoor_vt.metrics import source_metrics
from indoor_vt.observations import NativeFrame, direct_observation


def test_f1_maximum_cardinality_instead_of_nearest_greedy():
    truth = [[0, 0, 0], [0.59, 0, 0]]
    predicted = [[0.1, 0, 0], [-0.3, 0, 0]]
    assert source_metrics(predicted, truth)["f1"] == 1


def test_ospa_counts_extras_and_uses_3d():
    assert source_metrics([[0, 0, 2]], [[0, 0, 0]])["ospa"] == 1
    assert source_metrics([[0, 0, 0], [3, 0, 0]], [[0, 0, 0]])["ospa"] == 0.5
    assert source_metrics([], [[0, 0, 0]])["ospa"] == 1


def test_direct_all_paths_no_window_or_count_cap_and_discards_negative():
    ranges = np.arange(1, 61, dtype=float)
    frame = NativeFrame(ranges / 3e8, *[np.zeros(60) for _ in range(6)])
    clean, audit = direct_observation(frame, 0, np.random.default_rng(42))
    np.testing.assert_allclose(clean, ranges)
    assert len(clean) == 60
    rng = np.random.default_rng(42)
    expected = ranges + rng.normal(0, 50, 60)
    noisy, audit = direct_observation(frame, 50, np.random.default_rng(42))
    np.testing.assert_allclose(noisy, np.sort(expected[expected >= 0]))
    assert audit["discarded_negative"] == (expected < 0).sum()
