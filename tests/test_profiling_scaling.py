"""Tests for `profiling/scaling.py`'s cross-run analysis helpers."""

import pytest

from fabric_defect_hub.profiling.scaling import (
    max_concurrent_streams,
    throughput_resolution_slope,
    throughput_resolution_slope_delta,
)


def test_throughput_resolution_slope_perfect_decay():
    # throughput = -2*resolution + 200 -- a clean linear falloff
    resolutions = [10.0, 20.0, 30.0, 40.0]
    throughputs = [180.0, 160.0, 140.0, 120.0]
    result = throughput_resolution_slope(resolutions, throughputs)
    assert result["beta"] == pytest.approx(-2.0)
    assert result["alpha"] == pytest.approx(200.0)


def test_throughput_resolution_slope_delta_matches_hand_computed_slopes():
    # group_a: beta = -1 exactly; group_b: beta = -3 exactly
    group_a = [(1.0, 99.0), (2.0, 98.0), (3.0, 97.0), (4.0, 96.0)]
    group_b = [(1.0, 97.0), (2.0, 94.0), (3.0, 91.0), (4.0, 88.0)]
    result = throughput_resolution_slope_delta(group_a, group_b, n_resamples=300, seed=0)
    assert result["delta_beta"] == pytest.approx(-1.0 - (-3.0))
    assert result["ci_low"] <= result["delta_beta"] <= result["ci_high"]


def test_max_concurrent_streams_finds_the_breaking_point():
    # latency grows 10ms per stream; budget 33ms -> streams 1,2,3 fit (10,20,30), 4 doesn't (40)
    assert max_concurrent_streams(lambda n: n * 10.0, frame_budget_ms=33.0) == 3


def test_max_concurrent_streams_zero_when_even_one_stream_misses_budget():
    assert max_concurrent_streams(lambda n: 50.0, frame_budget_ms=33.0) == 0


def test_max_concurrent_streams_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="frame_budget_ms"):
        max_concurrent_streams(lambda n: 1.0, frame_budget_ms=0.0)
