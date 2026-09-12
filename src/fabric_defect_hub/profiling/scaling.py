"""Cross-run analysis over a resolution sweep or a concurrency sweep --
distinct from `profiling/base.py`'s single-run summary, since these
metrics only make sense across *multiple* profiling runs:

* throughput-vs-resolution decay slope (least-squares beta), and the
  slope difference between two groups (e.g. two backends/precisions/
  devices) with a bootstrap CI, so "how much worse does resolution scaling
  get" has a number and an uncertainty, not just two separate slopes eyeballed
  side by side.
* the largest concurrent stream count that still fits a hard real-time
  frame budget (e.g. 33ms for 30fps).
"""

from __future__ import annotations

from typing import Callable, Sequence

from fabric_defect_hub.stats import bootstrap_group_ci, least_squares_slope


def throughput_resolution_slope(resolutions: Sequence[float], throughputs: Sequence[float]) -> dict[str, float]:
    """Fit throughput = beta*resolution + alpha via least squares.
    `resolutions` is whatever the sweep varied (e.g. total pixel count
    h*w, or a single side length) -- pass whichever unit the sweep used,
    consistently, since `beta`'s scale depends on it.
    """

    beta, alpha = least_squares_slope(resolutions, throughputs)
    return {"beta": beta, "alpha": alpha}


def throughput_resolution_slope_delta(
    group_a: Sequence[tuple[float, float]],
    group_b: Sequence[tuple[float, float]],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, float]:
    """Bootstrap CI for beta_a - beta_b, where each group is a sequence of
    (resolution, throughput) points from its own sweep (e.g. fp32 vs int8,
    or GPU vs edge device) -- the "does this backend degrade faster as
    resolution grows" comparison, with an uncertainty bound instead of a
    single point difference.
    """

    def _slope(points: Sequence[tuple[float, float]]) -> float:
        xs, ys = zip(*points)
        return least_squares_slope(xs, ys)[0]

    result = bootstrap_group_ci(
        group_a, group_b, _slope, lambda a, b: a - b, n_resamples, confidence, seed
    )
    return {"delta_beta": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]}


def max_concurrent_streams(
    latency_at_concurrency: Callable[[int], float],
    frame_budget_ms: float = 33.0,
    max_streams_to_try: int = 64,
) -> int:
    """Largest N (1..max_streams_to_try) for which
    `latency_at_concurrency(N)` (per-frame latency in ms with N concurrent
    streams) stays within `frame_budget_ms`. Returns 0 if even a single
    stream misses budget. Stops at the first N that misses budget rather
    than scanning every value, since latency is monotonically
    non-decreasing in concurrency for a fixed hardware budget (more
    contention never makes a single frame faster).

    `latency_at_concurrency` is injected so this is driven by a real
    multi-stream profiler at call sites and by a synthetic function in
    tests -- probing real concurrent-hardware capacity isn't something a
    unit test can do.
    """

    if frame_budget_ms <= 0:
        raise ValueError("frame_budget_ms must be positive.")
    if max_streams_to_try < 1:
        raise ValueError("max_streams_to_try must be at least 1.")

    best = 0
    for n in range(1, max_streams_to_try + 1):
        if latency_at_concurrency(n) <= frame_budget_ms:
            best = n
        else:
            break
    return best
