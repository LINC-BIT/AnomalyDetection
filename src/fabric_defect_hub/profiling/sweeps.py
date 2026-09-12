"""Drivers that *run* multi-point profiling sweeps, feeding the pure
analysis functions in `scaling.py`.

`scaling.py` deliberately takes plain numbers (and, for `max_concurrent_
streams`, an injected latency function) so it stays unit-testable without
hardware. That left a gap: nothing in the project actually produced those
numbers outside the Gradio Benchmark tab, so `throughput_resolution_slope_
delta` and `max_concurrent_streams` had no caller at all. This module is the
missing half — it measures, `scaling.py` reduces.

Both sweeps re-profile the *same* exported artifact rather than re-exporting
per point: the question is how one deployed model behaves as its input grows
or as streams contend, and re-exporting would fold export-time differences
into what should be a pure runtime measurement.
"""

from __future__ import annotations

import statistics
import threading
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Sequence

from fabric_defect_hub.models.base import ExportedArtifact
from fabric_defect_hub.profiling.base import BackendProfiler, ProfileConfig
from fabric_defect_hub.profiling.scaling import (
    max_concurrent_streams,
    throughput_resolution_slope,
    throughput_resolution_slope_delta,
)

# Four points, matching the Gradio Benchmark tab's sweep: enough for a
# least-squares slope, few enough that a sweep is not a whole benchmark run.
DEFAULT_SWEEP_SIDES: tuple[int, ...] = (320, 448, 576, 640)


def resolution_sweep_points(
    profiler: BackendProfiler,
    artifact: ExportedArtifact,
    config: ProfileConfig,
    sides: Sequence[int] = DEFAULT_SWEEP_SIDES,
) -> list[tuple[float, float]]:
    """`(pixel_count, fps)` for each square input side in `sides`.

    Pixel count rather than side length is the x-axis because compute grows
    with area, so a slope fitted against side length is curved by
    construction and its `beta` is not comparable between sweeps that used
    different side ranges.

    A point that cannot be measured is dropped rather than raising: many
    exported graphs are fixed-shape (an Ultralytics TorchScript export bakes
    in its `imgsz`, ONNX graphs pin their input dims), so feeding them a
    different resolution fails in the runtime rather than in this project.
    That is a property of the export, not a broken sweep — the caller sees
    it as too few points to fit a slope.
    """

    points: list[tuple[float, float]] = []
    for side in sides:
        try:
            metrics = profiler.profile(artifact, replace(config, input_size=(side, side)))
        except Exception:  # noqa: BLE001 -- fixed-shape export; see docstring
            continue
        fps = float(metrics.get("fps", 0.0))
        if fps > 0:
            points.append((float(side * side), fps))
    return points


def resolution_scaling(
    profiler: BackendProfiler,
    artifact: ExportedArtifact,
    config: ProfileConfig,
    sides: Sequence[int] = DEFAULT_SWEEP_SIDES,
) -> dict[str, float]:
    """One model's throughput-vs-resolution decay slope."""

    points = resolution_sweep_points(profiler, artifact, config, sides)
    if len(points) < 2:
        return {}
    resolutions, throughputs = zip(*points)
    slope = throughput_resolution_slope(resolutions, throughputs)
    return {
        "resolution_slope_beta": slope["beta"],
        "resolution_slope_alpha": slope["alpha"],
        "resolution_sweep_points": float(len(points)),
    }


def resolution_scaling_delta(
    points_a: Sequence[tuple[float, float]],
    points_b: Sequence[tuple[float, float]],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, float]:
    """Group-vs-group slope difference with a bootstrap CI.

    Kept as a separate call taking pre-measured points, because the two
    groups being compared are usually two *different* profiling runs (two
    models, or one model at two precisions) that the caller already has —
    forcing them through one function would mean re-profiling group A every
    time it is compared against a new group B.
    """

    if len(points_a) < 2 or len(points_b) < 2:
        return {}
    return throughput_resolution_slope_delta(
        points_a, points_b, n_resamples=n_resamples, confidence=confidence, seed=seed
    )


def concurrency_capacity(
    profiler: BackendProfiler,
    artifact: ExportedArtifact,
    config: ProfileConfig,
    *,
    frame_budget_ms: float = 33.0,
    max_streams_to_try: int = 16,
) -> dict[str, float]:
    """Largest stream count whose per-frame latency stays inside the budget.

    33 ms is the default because that is one frame at 30 fps — the usual
    hard real-time line for an inline inspection camera.

    Concurrency is simulated with threads rather than processes: the point is
    contention for one accelerator, and separate processes would each get
    their own CUDA context, measuring something else entirely. For CPU
    inference the GIL is largely released inside the framework's kernels, so
    threads do contend realistically there too.
    """

    measured: dict[int, float] = {}

    def latency_at(streams: int) -> float:
        if streams in measured:
            return measured[streams]
        per_stream = replace(config, measured_runs=max(5, config.measured_runs // 10))
        start_barrier = threading.Barrier(streams)

        def profile_one(_index: int) -> dict[str, float]:
            # BackendProfiler stores its last power report, and concrete
            # runtimes own sessions/models. Each stream therefore gets an
            # independent profiler instance rather than racing on one object.
            try:
                stream_profiler = copy.deepcopy(profiler)
            except Exception:  # noqa: BLE001 -- fallback for non-copyable runtimes
                stream_profiler = type(profiler)()
            start_barrier.wait()
            return stream_profiler.profile(artifact, per_stream)

        with ThreadPoolExecutor(max_workers=streams) as pool:
            results = list(pool.map(profile_one, range(streams)))
        # The budget applies to every stream, so the worst stream is what
        # decides whether N fits -- an average would let one starved stream
        # hide behind the others.
        latency = max(float(result.get("latency_ms_mean", 0.0)) for result in results)
        measured[streams] = latency
        return latency

    best = max_concurrent_streams(
        latency_at, frame_budget_ms=frame_budget_ms, max_streams_to_try=max_streams_to_try
    )
    metrics: dict[str, float] = {
        "max_concurrent_streams": float(best),
        "frame_budget_ms": float(frame_budget_ms),
    }
    if 1 in measured:
        metrics["single_stream_latency_ms"] = measured[1]
    if measured:
        metrics["concurrency_probe_points"] = float(len(measured))
    return metrics


def summarize_sweep(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    """Mean/stdev of `key` across sweep rows that reported it — used by the
    report writer to say "across N models" without re-reading the JSONL."""

    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return {}
    return {
        f"{key}_mean": statistics.fmean(values),
        f"{key}_stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        f"{key}_n": float(len(values)),
    }
