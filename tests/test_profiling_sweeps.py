"""Multi-point profiling sweeps: the drivers that feed `scaling.py`.

These are what turned `throughput_resolution_slope_delta` and
`max_concurrent_streams` from unreachable pure functions into measurements
the sweep can actually take, so the tests focus on the driver behaviour —
dropping unmeasurable points, and choosing N by the worst stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fabric_defect_hub.profiling.base import ProfileConfig
from fabric_defect_hub.profiling.sweeps import (
    concurrency_capacity,
    resolution_scaling,
    resolution_scaling_delta,
    resolution_sweep_points,
    summarize_sweep,
)


@dataclass
class _Artifact:
    path: str = "fake.pt"
    target: str = "torchscript"
    metadata: dict = None


class _Profiler:
    """Throughput that falls off as area grows, like real hardware."""

    def __init__(self, fps_at=None, fails_above=None, latency_per_stream=None):
        self.fps_at = fps_at or (lambda side: 640.0 * 640.0 / (side * side) * 100.0)
        self.fails_above = fails_above
        self.latency_per_stream = latency_per_stream
        self.calls: list[tuple[int, int]] = []

    def profile(self, artifact, config: ProfileConfig) -> dict[str, float]:
        side = config.input_size[0]
        self.calls.append(config.input_size)
        if self.fails_above is not None and side > self.fails_above:
            raise RuntimeError("fixed-shape export: input dims are baked in")
        metrics = {"fps": self.fps_at(side)}
        if self.latency_per_stream is not None:
            metrics["latency_ms_mean"] = self.latency_per_stream(len(self.calls))
        return metrics


def _config() -> ProfileConfig:
    return ProfileConfig(device="cpu", engine="pytorch", measured_runs=2, warmup_runs=0)


# --------------------------------------------------------------------------- #
# Resolution sweep
# --------------------------------------------------------------------------- #
def test_sweep_measures_every_requested_side():
    profiler = _Profiler()

    points = resolution_sweep_points(profiler, _Artifact(), _config(), sides=(320, 640))

    assert [size[0] for size in profiler.calls] == [320, 640]
    # x-axis is pixel count, not side length
    assert [point[0] for point in points] == [320 * 320, 640 * 640]


def test_a_fixed_shape_export_drops_points_instead_of_raising():
    """An Ultralytics TorchScript export bakes in its imgsz — feeding it
    another resolution fails in the runtime, which is a property of the
    export rather than a broken sweep.
    """

    profiler = _Profiler(fails_above=320)

    points = resolution_sweep_points(profiler, _Artifact(), _config(), sides=(320, 448, 640))

    assert len(points) == 1


def test_too_few_points_yields_no_slope_rather_than_a_bogus_one():
    profiler = _Profiler(fails_above=320)

    assert resolution_scaling(profiler, _Artifact(), _config(), sides=(320, 448, 640)) == {}


def test_slope_is_negative_when_throughput_falls_with_area():
    profiler = _Profiler()

    metrics = resolution_scaling(profiler, _Artifact(), _config(), sides=(320, 448, 576, 640))

    assert metrics["resolution_slope_beta"] < 0
    assert metrics["resolution_sweep_points"] == 4


def test_zero_fps_points_are_not_fitted():
    profiler = _Profiler(fps_at=lambda side: 0.0 if side == 640 else 50.0)

    points = resolution_sweep_points(profiler, _Artifact(), _config(), sides=(320, 448, 640))

    assert [point[0] for point in points] == [320 * 320, 448 * 448]


# --------------------------------------------------------------------------- #
# Group delta
# --------------------------------------------------------------------------- #
def test_delta_reports_an_estimate_with_a_confidence_interval():
    steep = [(1.0, 100.0), (2.0, 60.0), (3.0, 20.0), (4.0, 5.0)]
    flat = [(1.0, 100.0), (2.0, 95.0), (3.0, 90.0), (4.0, 88.0)]

    result = resolution_scaling_delta(steep, flat, n_resamples=200)

    assert result["delta_beta"] < 0  # steep degrades faster
    assert result["ci_low"] <= result["delta_beta"] <= result["ci_high"]


def test_delta_needs_two_points_per_group():
    assert resolution_scaling_delta([(1.0, 2.0)], [(1.0, 2.0), (2.0, 3.0)]) == {}


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def test_capacity_stops_at_the_first_stream_count_missing_budget():
    # Latency grows with the number of concurrent streams.
    profiler = _Profiler(latency_per_stream=lambda call: 10.0)

    def make(streams_latency):
        prof = _Profiler()
        prof.profile = lambda artifact, config: {  # type: ignore[assignment]
            "fps": 10.0, "latency_ms_mean": streams_latency.pop(0) if streams_latency else 999.0,
        }
        return prof

    # 1 stream: 10ms (fits), 2 streams: 20+20 (fits), 3 streams: 40ms each (misses)
    prof = make([10.0, 20.0, 20.0, 40.0, 40.0, 40.0])
    metrics = concurrency_capacity(prof, _Artifact(), _config(), frame_budget_ms=33.0, max_streams_to_try=8)

    assert metrics["max_concurrent_streams"] == 2.0
    assert metrics["frame_budget_ms"] == 33.0


def test_capacity_is_zero_when_even_one_stream_misses_budget():
    prof = _Profiler()
    prof.profile = lambda artifact, config: {"fps": 1.0, "latency_ms_mean": 500.0}

    metrics = concurrency_capacity(prof, _Artifact(), _config(), frame_budget_ms=33.0)

    assert metrics["max_concurrent_streams"] == 0.0


def test_the_worst_stream_decides_not_the_average():
    """A budget every stream must meet cannot be satisfied on average."""

    from fabric_defect_hub.profiling.scaling import max_concurrent_streams

    # Two streams: one at 5ms, one at 100ms. Mean would pass a 33ms budget.
    assert max_concurrent_streams(lambda n: 5.0 if n == 1 else 100.0, frame_budget_ms=33.0) == 1


# --------------------------------------------------------------------------- #
# Summary helper
# --------------------------------------------------------------------------- #
def test_summarize_ignores_rows_missing_the_key():
    rows = [{"fps": 10.0}, {"fps": 20.0}, {"latency_ms_mean": 5.0}]

    summary = summarize_sweep(rows, "fps")

    assert summary["fps_mean"] == 15.0
    assert summary["fps_n"] == 2.0


def test_summarize_returns_nothing_when_no_row_has_the_key():
    assert summarize_sweep([{"a": 1}], "fps") == {}
