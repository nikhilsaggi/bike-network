import numpy as np
import pandas as pd
import pytest

from bike_network import calibration


def test_score_removes_the_density_effect():
    # Citi Bike's share rises exactly with local density: no route-choice error left.
    counted = np.array([1000.0, 2000.0, 500.0, 800.0])
    ends = np.array([1e4, 1e5, 1e6, 3e5])
    routed = counted * 0.01 * ends**0.4
    s = calibration.score(routed, counted, ends)
    assert s["spread"] == pytest.approx(0, abs=1e-9)
    assert s["slope"] == pytest.approx(0.4)
    assert s["raw"] > 0.5


def test_score_floors_an_empty_counter_rather_than_failing():
    s = calibration.score(
        np.array([0.0, 100.0, 50.0]), np.array([100.0, 100.0, 80.0]), np.array([10.0, 20.0, 40.0])
    )
    assert np.isfinite(s["spread"])


def test_routed_at_sums_directions_and_takes_a_crossings_busiest_block():
    edges = pd.DataFrame(
        {"segment_id": ["a", "a", "b", "b", "c"], "dir": ["fwd", "rev", "fwd", "rev", "fwd"]}
    )
    trips = np.array([365.0, 730.0, 3650.0, 0.0, 365.0])  # a: 3/day, b: 10/day, c: 1/day
    got = calibration.routed_at([["a"], ["a", "b", "c"], ["missing"]], edges, trips)
    assert got.tolist() == [3.0, 10.0, 0.0]
