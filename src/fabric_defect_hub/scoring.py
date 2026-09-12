"""Combine a leaderboard row's accuracy ("technical") and runtime-cost
("overhead") metrics into one ranked score.

`benchmark.py`'s `leaderboard()` already sorts by a single raw metric name
-- useful when you know exactly which number matters, but it can't answer
"which model is the best overall trade-off" when technical metrics
(AUROC, mAP, mIoU, ...) and overhead metrics (FPS, latency, memory, model
size) pull in different directions. This module is that combination step,
used by the Gradio Benchmark tab (`web/benchmark.py`) so its leaderboard can
rank on a configurable blend of the two instead of forcing a pick of one
column.

Metric names are classified by pattern rather than a fixed enumeration --
new evaluators/profilers add metric names over time (see `evaluation/*.py`,
`profiling/base.py::summarize_latencies`), and a name this module doesn't
recognize should just be excluded from scoring, not require this file to be
updated in lockstep or raise on an unfamiliar key.
"""

from __future__ import annotations

import math
from typing import Any, Literal

MetricDirection = Literal["higher", "lower"]

# Ordered (substring, direction) rules -- the first match wins. Specific
# lower-is-better jitter patterns must precede the general "fps" rule.
_DIRECTION_RULES: tuple[tuple[str, MetricDirection], ...] = (
    ("instantaneous_fps_std", "lower"),
    ("instantaneous_fps_cv", "lower"),
    ("latency_ms_std", "lower"),
    ("latency_ms_cv", "lower"),
    ("fps", "higher"),
    ("latency", "lower"),
    ("model_size_mb", "lower"),
    ("runtime_s", "lower"),
    ("alarms_per_unit_length", "lower"),
    ("auroc", "higher"),
    ("aupro", "higher"),
    ("iap", "higher"),
    ("map", "higher"),
    ("mar", "higher"),
    ("miou", "higher"),
    ("dice", "higher"),
    ("f1", "higher"),
    ("precision", "higher"),
    ("recall", "higher"),
)

# Metrics matched by these substrings count toward the "overhead" (runtime
# cost) group; every other classified metric is "technical" (accuracy).
# Raw memory is intentionally excluded: PyTorch allocator bytes, process RSS,
# and TensorRT I/O-buffer lower bounds are different quantities. They remain
# visible in reports but cannot influence a cross-engine composite score.
_OVERHEAD_KEYWORDS = ("fps", "latency", "model_size_mb", "runtime_s")

SCORE_PRESETS: dict[str, tuple[float, float]] = {
    "accuracy_first": (0.85, 0.15),
    "balanced": (0.5, 0.5),
    "efficiency_first": (0.15, 0.85),
}


def metric_direction(name: str) -> MetricDirection | None:
    """Whether a higher or lower value of metric `name` is better, or
    `None` if this module doesn't recognize it (excluded from scoring)."""

    lowered = name.lower()
    for pattern, direction in _DIRECTION_RULES:
        if pattern in lowered:
            return direction
    return None


def metric_group(name: str) -> Literal["technical", "overhead"] | None:
    """'technical' (accuracy) or 'overhead' (runtime cost); `None` if
    `metric_direction` doesn't recognize `name` either."""

    if metric_direction(name) is None:
        return None
    lowered = name.lower()
    return "overhead" if any(keyword in lowered for keyword in _OVERHEAD_KEYWORDS) else "technical"


def normalize_metrics(
    rows: list[dict[str, Any]],
    metric_keys: set[str] | None = None,
) -> tuple[list[str], list[dict[str, float]]]:
    """Min-max normalize every recognized metric across `rows` onto a common
    0-1 "higher is better" scale, inverting the lower-is-better ones.

    Returns `(metric_names, per_row)` where `metric_names` is sorted (a
    stable axis order, which the radar chart depends on -- see
    `polygon_area`) and `per_row[i]` maps metric name -> normalized value
    for `rows[i]`, omitting metrics that row doesn't carry.

    This is the shared basis for both `score_rows` (which averages these
    per group). Normalised values were also once plotted as
    polygon radii). Putting mixed-unit metrics -- an AUROC in [0, 1] beside
    a latency in milliseconds beside a memory figure in MiB -- on one
    comparable scale is exactly the same problem in both places, so it has
    exactly one implementation.
    """

    names = sorted({
        name
        for row in rows
        for name in row
        if (metric_keys is None or name in metric_keys) and metric_group(name) is not None
    })
    normalized_by_name = {name: _normalize(rows, name) for name in names}
    per_row = [
        {
            name: per_metric[index]
            for name, per_metric in normalized_by_name.items()
            if index in per_metric
        }
        for index in range(len(rows))
    ]
    return names, per_row


def polygon_area(values: list[float]) -> float:
    """Area of the polygon a radar chart traces through `values`, treated as
    radii on `len(values)` evenly spaced axes: `0.5 * sin(2pi/n) * sum(r_i *
    r_i+1)` around the ring.

    This is the number behind "which model encloses more area" -- a single
    summary of how a model does across every plotted axis at once.

    It depends on the *order* of the axes (swapping two axes changes the
    area), so it is only meaningful for comparing models **within one
    chart**, never as an absolute score across charts with different axis
    sets. `normalize_metrics` returns its names sorted precisely so that
    order is fixed and every model on a given chart is measured the same
    way. Fewer than 3 values cannot enclose anything, so they score 0.
    """

    count = len(values)
    if count < 3:
        return 0.0
    wedge = math.sin(2 * math.pi / count)
    return 0.5 * wedge * sum(
        values[i] * values[(i + 1) % count] for i in range(count)
    )


def score_rows(
    rows: list[dict[str, Any]],
    technical_weight: float = 0.5,
    overhead_weight: float = 0.5,
    metric_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return copies of `rows` with `technical_score`/`overhead_score`/
    `composite_score` (0-100) appended.

    Each recognized metric is min-max normalized against the *other rows in
    this same call* -- these scores rank the given leaderboard against
    itself, they are not an absolute, cross-run scale. A row missing every
    metric in a group gets `None` for that group's score, and the composite
    falls back to whichever group is actually present (renormalizing the
    weights over just those groups); a row missing both groups gets a
    `None` composite.

    `metric_keys`, if given, restricts scoring to that subset of metric
    names (e.g. a "basic parameters" UI toggle that only feeds a few
    metrics into the score) -- metrics outside it are still present in the
    row's own values, just not used for scoring.
    """

    if technical_weight < 0 or overhead_weight < 0:
        raise ValueError("weights must be non-negative")
    if technical_weight + overhead_weight <= 0:
        raise ValueError("at least one of technical_weight/overhead_weight must be positive")

    _, per_row = normalize_metrics(rows, metric_keys)

    weights = {"technical": technical_weight, "overhead": overhead_weight}
    scored_rows = []
    for index, row in enumerate(rows):
        normalized = per_row[index]
        group_scores: dict[str, float | None] = {}
        for group in ("technical", "overhead"):
            values = [
                value for name, value in normalized.items() if metric_group(name) == group
            ]
            group_scores[group] = (sum(values) / len(values) * 100.0) if values else None

        present_weight = sum(
            weights[group] for group, score in group_scores.items() if score is not None
        )
        if present_weight > 0:
            composite = sum(
                weights[group] * score
                for group, score in group_scores.items()
                if score is not None
            ) / present_weight
        else:
            composite = None

        new_row = dict(row)
        new_row["technical_score"] = group_scores["technical"]
        new_row["overhead_score"] = group_scores["overhead"]
        new_row["composite_score"] = composite
        scored_rows.append(new_row)

    return scored_rows


def _normalize(rows: list[dict[str, Any]], name: str) -> dict[int, float]:
    direction = metric_direction(name)
    present = [
        (index, row[name])
        for index, row in enumerate(rows)
        if name in row and isinstance(row[name], (int, float))
    ]
    if not present:
        return {}
    values = [value for _, value in present]
    low, high = min(values), max(values)
    result: dict[int, float] = {}
    for index, value in present:
        if high == low:
            # Every row ties on this metric -- treat it as equally good
            # rather than dividing by zero or arbitrarily favoring one row.
            result[index] = 1.0
            continue
        normalized = (value - low) / (high - low)
        result[index] = 1.0 - normalized if direction == "lower" else normalized
    return result
