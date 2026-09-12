"""Tests for `stats.py`'s pure helpers (exact hand-computed values where
feasible; bootstrap CI checked against a hand-computed point estimate plus
sanity bounds, since the resampled distribution itself isn't hand-computable).
"""

import statistics

import pytest

from fabric_defect_hub.stats import bootstrap_group_ci, coefficient_of_variation, least_squares_slope


def test_coefficient_of_variation_hand_computed():
    values = [10.0, 20.0, 30.0]
    mean = 20.0
    pstdev = statistics.pstdev(values)
    assert coefficient_of_variation(values) == pytest.approx(pstdev / mean)


def test_coefficient_of_variation_needs_two_points():
    assert coefficient_of_variation([]) == 0.0
    assert coefficient_of_variation([5.0]) == 0.0


def test_coefficient_of_variation_zero_mean_is_zero():
    assert coefficient_of_variation([-1.0, 1.0]) == 0.0


def test_least_squares_slope_perfect_line():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [3.0, 5.0, 7.0, 9.0]  # y = 2x + 1
    beta, alpha = least_squares_slope(x, y)
    assert beta == pytest.approx(2.0)
    assert alpha == pytest.approx(1.0)


def test_least_squares_slope_needs_two_distinct_x():
    with pytest.raises(ValueError, match="distinct x"):
        least_squares_slope([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])


def test_least_squares_slope_needs_two_points():
    with pytest.raises(ValueError, match="at least two points"):
        least_squares_slope([1.0], [2.0])


def test_bootstrap_group_ci_point_estimate_is_exact_difference():
    group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    group_b = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = bootstrap_group_ci(
        group_a, group_b, statistics.fmean, lambda a, b: a - b, n_resamples=500, seed=0,
    )
    assert result["estimate"] == pytest.approx(statistics.fmean(group_a) - statistics.fmean(group_b))
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_bootstrap_group_ci_is_deterministic_given_a_seed():
    group_a = [1.0, 5.0, 3.0, 9.0, 2.0]
    group_b = [2.0, 4.0, 6.0, 8.0, 10.0]
    first = bootstrap_group_ci(group_a, group_b, statistics.fmean, lambda a, b: a - b, n_resamples=200, seed=42)
    second = bootstrap_group_ci(group_a, group_b, statistics.fmean, lambda a, b: a - b, n_resamples=200, seed=42)
    assert first == second


def test_bootstrap_group_ci_requires_at_least_two_points_per_group():
    with pytest.raises(ValueError, match="at least two points"):
        bootstrap_group_ci([1.0], [1.0, 2.0], statistics.fmean, lambda a, b: a - b)


def test_bootstrap_group_ci_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_group_ci([1.0, 2.0], [1.0, 2.0], statistics.fmean, lambda a, b: a - b, confidence=1.5)
