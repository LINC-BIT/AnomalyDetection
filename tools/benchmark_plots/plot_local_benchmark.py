#!/usr/bin/env python3
"""Plot the final local benchmark snapshot from benchmark_rows.jsonl.

Metric grouping is not decided here. It comes from
`fabric_defect_hub.metrics_taxonomy`, the project's single source of truth, so a
metric appears under the same heading the report and the UI use. The one that
matters most for these figures: Dice / mIoU / Pixel F1 are *pixel-level*
metrics. A segmentation model outputs a binary mask, so its overlap metrics
belong beside the anomaly-map pixel metrics rather than in a segmentation table
of their own.

The script never reads the 4090 Excel/Markdown files. It selects one complete
19-model local snapshot, writes an audit table, and renders figures with missing
metrics omitted rather than converted to zero.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

# Thin hatch strokes read as a light texture rather than a solid fill.
matplotlib.rcParams["hatch.linewidth"] = 0.9
# SVG keeps text as <text> instead of outlining every glyph, so a vector editor
# can reword a label and the file stays small. The trade-off is that the SVG is
# no longer self-contained: whoever opens it needs the font. PNG and PDF are
# unaffected -- they still embed or outline as before.
matplotlib.rcParams["svg.fonttype"] = "none"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmark_plots.data import DEFAULT_LOG, SNAPSHOT_ROOT, Snapshot, family, load_latest_snapshot, write_audit, write_snapshot_metadata
from fabric_defect_hub.metrics_taxonomy import OVERHEAD_TABLES, TABLES, TECHNICAL_TABLES, label_of, specs_for


# Two families only. A segmentation model predicts a binary pixel mask scored
# with pixel-level overlap metrics, so it is an anomaly-detection model here
# rather than a third colour with its own legend entry.
COLORS = {"anomaly": "#0f766e", "detection": "#2563eb"}
# One style per metric inside a model's group of bars. The greys carry the two
# styles a reader sees most often, so they stay light enough to sit under the
# grid without competing with the coloured marks; red is dots rather than
# diagonals so it cannot be confused with the blue or amber diagonal bars at
# small print sizes. The hatch is a single character, which is the sparsest
# matplotlib will draw; repeating it (`//`) is what makes a hatch look dense.
BAR_STYLES = (
    {"facecolor": "#d1d5db", "edgecolor": "#9ca3af", "hatch": ""},
    {"facecolor": "white", "edgecolor": "#c1121f", "hatch": "."},
    {"facecolor": "white", "edgecolor": "#155eef", "hatch": "/"},
    {"facecolor": "white", "edgecolor": "#9ca3af", "hatch": "/"},
    # A six-metric figure needs six distinct styles: with only four the fifth and
    # sixth bars repeated the first two and two metrics became indistinguishable.
    {"facecolor": "white", "edgecolor": "#b45309", "hatch": "\\"},
    {"facecolor": "white", "edgecolor": "#0f766e", "hatch": "\\"},
)

# Dropped from the figures entirely. DETR's instance metrics are ~0 on this
# snapshot (mAP 0.0011), so a bar of its height is invisible in some panels and
# an outlier that flattens the rest in others; keeping it made every chart it
# appeared in worse. It is still in `snapshot_audit.csv`, and still in the PR/ROC
# curves, where its AUROC (0.65) is a real, readable measurement.
EXCLUDED_MODELS = frozenset({"DETR · ZJU-Leaper"})

# Candidate label placements, tried in order. Alternating right/left first is
# what keeps two near-coincident points from stacking their labels on top of
# each other.
LABEL_OFFSETS = (
    (7, 3, "left", "bottom"),
    (-7, 3, "right", "bottom"),
    (7, -4, "left", "top"),
    (-7, -4, "right", "top"),
    (12, 9, "left", "bottom"),
    (-12, 9, "right", "bottom"),
    (16, 0, "left", "center"),
    (-16, 0, "right", "center"),
    (0, 10, "center", "bottom"),
    (0, -11, "center", "top"),
    (0, 21, "center", "bottom"),
    (0, -22, "center", "top"),
)
MARKER_BOX_PX = 4.0

# Presentation-only subsets of the wide tables. The *grouping* is always the
# taxonomy's; these keep a single figure readable when a table has twenty
# columns. Anything measured but left out stays visible in `snapshot_audit.csv`.
# Pixel level pairs the two metric families: threshold-free ranking metrics
# (AUROC/AUPRO, anomaly maps only) and thresholded overlap metrics (Pixel F1,
# mIoU). `dice` is deliberately absent because it *is* `pixel_f1` -- see the
# evaluator's `_dice` and `_pixel_f1`, which are the same formula, and the two
# columns are bit-identical in the log.
# Both panels of `01_image_level` use this same list, in this same order, so a
# metric gets the same bar style in both: the figure's own rule is "one style per
# metric", and positional styling only delivers that when the two panels' metric
# lists are identical. `map_50` is deliberately absent -- mAP is an
# instance-level metric and already has `03_instance_level` -- which is also what
# keeps the two panels structurally identical.
IMAGE_LEVEL_FIGURE_METRICS = ("image_auroc", "image_ap", "image_f1", "image_precision", "image_recall")
# `miou` is deliberately absent. Only the three segmentation models report it,
# so it was a fifth bar on three of the twelve groups and nothing on the other
# nine -- and for a single binary mask per image, IoU is a monotone function of
# F1 (`IoU = F1 / (2 - F1)`), so it repeats `pixel_f1` rather than adding a
# measurement. Same reasoning that keeps `dice` off the figure.
PIXEL_LEVEL_FIGURE_METRICS = ("pixel_auroc", "pixel_aupro", "iap", "pixel_f1")
INSTANCE_LEVEL_FIGURE_METRICS = ("map", "map_50", "map_75", "f1_at_threshold")
# Size-bucketed recall and counts, the tables the report carries under instance
# level that the four headline metrics above do not cover.
INSTANCE_BREAKDOWN_METRICS = ("map_small", "map_medium", "map_large", "mar_1", "mar_10", "mar_100")
INSTANCE_COUNT_METRICS = ("true_positives", "false_positives", "false_negatives")
INSTANCE_RATE_METRICS = ("precision_at_threshold", "recall_at_threshold", "f1_at_threshold")
COMPUTE_FIGURE_METRICS = ("fps", "latency_ms_mean", "latency_ms_p95", "runtime_s")
MEMORY_FIGURE_METRICS = ("peak_memory_mb", "params_m", "model_size_mb")
# A detection model under this mAP has no usable boxes on this snapshot, so a
# ~0 bar in the instance-level chart is noise rather than a comparison. Models
# below the floor are printed at runtime, never dropped silently.
INSTANCE_MAP_FLOOR = 0.01

# The figures carry no title: the report writes its own heading above them, and
# a title baked into the PNG cannot be reworded or translated afterwards. Flip
# this to True for a standalone look at one figure.
FIGURE_TITLES = False

# The quality axis of each granularity panel. Pixel F1 is the metric that both
# anomaly-map and segmentation models report, which is what lets them share the
# pixel-level panel instead of being split by task.
QUALITY_METRIC = {
    "image_level": ("image_auroc", "Image AUROC"),
    "pixel_level": ("pixel_f1", "Pixel F1"),
    "instance_level": ("map_50", "mAP@0.5"),
}


def _short(name: str) -> str:
    return name.replace(" · ZJU-Leaper", "").replace(" (normal only)", "").replace(" · LAION-400M zero-shot", "").replace(" · MVTec AD adapter", "")


def _panel_label(index: int, title: str) -> str:
    """`(a) Title` — the sub-caption a two-panel figure needs, since the figure
    itself carries no heading."""

    return f"({chr(ord('a') + index)}) {title}"


# Every figure is written in all three formats, and `--missing` treats a figure
# as missing when *any* of them is absent. That is what lets a format added
# after the fact -- SVG, here -- be filled in by re-running `--missing` instead
# of redrawing every figure or remembering which ones predate it. PNG is the
# raster copy for slides, PDF the vector copy for LaTeX, SVG the editable vector
# copy for a vector editor.
FIGURE_FORMATS: tuple[str, ...] = ("png", "pdf", "svg")


def missing_formats(output: Path, stem: str) -> list[str]:
    """Which of the expected formats for one figure are not on disk yet."""

    return [suffix for suffix in FIGURE_FORMATS if not (output / f"{stem}.{suffix}").exists()]


def _save(fig: plt.Figure, output: Path, name: str, layout: bool = True) -> None:
    if layout:
        fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _annotate_points(ax, items: list[tuple[float, float, str]], fontsize: int = 8) -> None:
    """Label each point, trying offsets until the text clears other text and markers.

    Overlapping annotations were the failure mode in the complexity and
    quality-vs-latency scatters, where near-identical models land on top of one
    another. Each label takes the first candidate offset that collides with
    neither an already placed label nor an unrelated marker, so a crowded pair
    ends up labelled on opposite sides. Call after the final `tight_layout`,
    because the collision test works in pixels.
    """

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    point_boxes = []
    for x, y, _ in items:
        px, py = ax.transData.transform((x, y))
        point_boxes.append(Bbox.from_bounds(px - MARKER_BOX_PX, py - MARKER_BOX_PX, 2 * MARKER_BOX_PX, 2 * MARKER_BOX_PX))
    placed: list[Bbox] = []
    for index, (x, y, text) in enumerate(items):
        blockers = [box for other, box in enumerate(point_boxes) if other != index] + placed
        label = None
        for dx, dy, ha, va in LABEL_OFFSETS:
            candidate = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize)
            box = candidate.get_window_extent(renderer).expanded(1.05, 1.20)
            inside = axes_box.x0 <= box.x0 and box.x1 <= axes_box.x1 and axes_box.y0 <= box.y0 and box.y1 <= axes_box.y1
            if inside and not any(box.overlaps(blocker) for blocker in blockers):
                label = candidate
                break
            candidate.remove()
        if label is None:
            dx, dy, ha, va = LABEL_OFFSETS[0]
            label = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fontsize)
        placed.append(label.get_window_extent(renderer).expanded(1.05, 1.20))


def figure_metrics(snapshot: Snapshot) -> dict[str, list[str]]:
    """The metric groups behind the three bar figures.

    Membership is the taxonomy's; the order is the taxonomy's reading order. A
    figure shows a presentation subset — see `PIXEL_LEVEL_FIGURE_METRICS` — while
    `audit_groups` keeps every measured cell on record.
    """

    present = {key for row in snapshot.rows for key, value in row.get("metrics", {}).items() if value is not None}

    def ordered(table: str, keys: tuple[str, ...]) -> list[str]:
        return [spec.key for spec in specs_for(table) if spec.key in keys and spec.key in present]

    return {
        "image_level": ordered("image_level", IMAGE_LEVEL_FIGURE_METRICS),
        "pixel_level": ordered("pixel_level", PIXEL_LEVEL_FIGURE_METRICS),
        "instance_level": ordered("instance_level", INSTANCE_LEVEL_FIGURE_METRICS),
        "instance_breakdown": ordered("instance_level", INSTANCE_BREAKDOWN_METRICS),
        "compute": ordered("compute", COMPUTE_FIGURE_METRICS),
        "memory": ordered("memory", MEMORY_FIGURE_METRICS),
    }


def audit_groups(snapshot: Snapshot) -> dict[str, list[str]]:
    """The audit table's groups: every declared, measured cell of every table.

    Wider than `figure_metrics` on purpose, so a metric left off a figure for
    readability is still recorded, with the reason it has no value elsewhere.
    """

    present = {key for row in snapshot.rows for key, value in row.get("metrics", {}).items() if value is not None}
    groups: dict[str, list[str]] = {}
    for table in TABLES:
        keys = [spec.key for spec in specs_for(table) if spec.key in present]
        if keys:
            groups[table] = keys
    return groups


def instance_exclusions(snapshot: Snapshot) -> dict[str, str]:
    """Detection models with no usable boxes, and the measured reason why."""

    excluded: dict[str, str] = {}
    for row in snapshot.rows:
        if family(row) != "detection":
            continue
        value = row.get("metrics", {}).get("map")
        if value is not None and float(value) < INSTANCE_MAP_FLOOR:
            excluded[row["model"]] = f"mAP@[.5:.95]={float(value):.4f}, below the {INSTANCE_MAP_FLOOR} display floor"
    return excluded


def _draw_grouped_bars(ax, records: list, metrics: list[str]) -> None:
    """Draw one model-grouped, metric-styled bar panel on a supplied axes.

    Shared by the single-panel bar figures and by `01_image_level`, whose two
    panels pass the same metric list so a metric cannot take one style in one
    panel and a different style in the other. A model that reports nothing for a
    metric simply has no bar there, so a missing measurement never reads as a bad
    score.
    """

    x = np.arange(len(records))
    width = 0.82 / len(metrics)
    for index, metric in enumerate(metrics):
        values = [float(values[metric]) if values.get(metric) is not None else np.nan for _, values in records]
        offset = (index - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, values, width=width * 0.86, linewidth=1.0, label=label_of(metric),
               **BAR_STYLES[index % len(BAR_STYLES)])
    ax.set_xticks(x, [_short(row["model"]) for row, _ in records], rotation=55, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlim(-0.5, len(records) - 0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=len(metrics), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10)


def _grouped_bars(
    snapshot: Snapshot,
    metrics: list[str],
    title: str,
    output: Path,
    filename: str,
    excluded: set[str] | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.02),
) -> None:
    """One bar per metric, grouped per model, in a single panel.

    Four separate panels made a metric look comparable across a different set of
    models than its neighbours. One panel with a legend keeps a model's whole
    profile in one column and makes the metric a matter of bar style. A model
    that reports nothing for a metric simply has no bar there, so a missing
    measurement never reads as a bad score. Why a bar can be absent is
    explained in CAPTIONS.md rather than burned into the image.
    """

    if not metrics:
        return
    records = [
        (row, {metric: row.get("metrics", {}).get(metric) for metric in metrics})
        for row in snapshot.rows
        if excluded is None or row["model"] not in excluded
    ]
    records = [item for item in records if any(value is not None and math.isfinite(float(value)) for value in item[1].values())]
    if not records:
        return
    records.sort(key=lambda item: max((float(v) for v in item[1].values() if v is not None), default=0), reverse=True)
    fig, ax = plt.subplots(figsize=(max(13, len(records) * 0.95), 7))
    _draw_grouped_bars(ax, records, metrics)
    ax.set_ylim(*(ylim if ylim is not None else ax.get_ylim()))
    if FIGURE_TITLES:
        ax.set_title(title, pad=30)
    _save(fig, output, filename)


def _image_level(snapshot: Snapshot, output: Path) -> None:
    """Image-level metrics as two panels, one per **purpose**, in one style.

    The v1 figure put ten anomaly detectors and six supervised detectors on one
    AUROC axis. That comparison is not sound: an anomaly model reports a
    continuous normality score for every image, while a detector reports
    `max(box confidence)`, which is exactly `0` on any image where it finds
    nothing. Pooling them lets a detector's AUROC (0.99 while missing 40-64 % of
    defective images at its own threshold) sit beside an anomaly model's AUROC as
    though the two measured the same thing.

    Splitting the families into two panels is what keeps them apart, so the
    panels themselves carry **no** special-casing: the same metric list, the same
    order and therefore the same bar styles in both. The score caveat lives in
    the caption and `README.md` rather than in a flagged bar and a footnote,
    which made the figure fight itself. Models are ordered by Image AUROC, the
    metric the panel is about.
    """

    present = {key for row in snapshot.rows for key, value in row.get("metrics", {}).items() if value is not None}
    metrics = [metric for metric in IMAGE_LEVEL_FIGURE_METRICS if metric in present]

    panels = (
        (metrics, [r for r in snapshot.rows if family(r) == "anomaly"], "Anomaly detection"),
        (metrics, [r for r in snapshot.rows if family(r) == "detection"], "Defect detection"),
    )
    drawn = []
    for panel_metrics, rows, title in panels:
        records = [(row, {metric: row["metrics"].get(metric) for metric in panel_metrics}) for row in rows]
        records = [item for item in records if any(v is not None and math.isfinite(float(v)) for v in item[1].values())]
        if not panel_metrics or not records:
            continue
        records.sort(key=lambda item: (item[1].get(panel_metrics[0]) is None, -(item[1].get(panel_metrics[0]) or 0.0)))
        drawn.append((panel_metrics, records, title))

    if not drawn:
        return
    fig, axes = plt.subplots(1, len(drawn), figsize=(max(9.0, 6.5) * len(drawn), 7.2), squeeze=False)
    for index, (ax, (panel_metrics, records, title)) in enumerate(zip(axes.flat, drawn)):
        _draw_grouped_bars(ax, records, panel_metrics)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(_panel_label(index, title), fontsize=10, pad=26)
    fig.tight_layout()
    _save(fig, output, "01_image_level", layout=False)


def _compute(snapshot: Snapshot, output: Path) -> None:
    rows = [(row, row["metrics"]) for row in snapshot.rows if row["metrics"].get("fps") is not None]
    if not rows:
        return
    labels = [_short(row["model"]) for row, _ in rows]
    x = np.arange(len(rows))
    panels = (
        ("fps", "Throughput (FPS)", "FPS", BAR_STYLES[0]),
        ("latency_ms_mean", "Latency, mean", "milliseconds", BAR_STYLES[1]),
        ("latency_ms_p95", "Latency, p95", "milliseconds", BAR_STYLES[2]),
        ("runtime_s", "Wall time, whole scored split", "seconds", BAR_STYLES[3]),
    )
    fig, axes = plt.subplots(2, 2, figsize=(max(15, len(rows) * 0.9), 11))
    for panel_index, (ax, (metric, title, unit, style)) in enumerate(zip(axes.flat, panels)):
        values = [float(metrics[metric]) if metrics.get(metric) is not None else np.nan for _, metrics in rows]
        ax.bar(x, values, width=0.48, linewidth=1.0, **style)
        ax.set_xticks(x, labels, rotation=55, ha="right")
        ax.set_title(_panel_label(panel_index, title))
        ax.set_ylabel(unit)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.grid(axis="y", alpha=0.25)
        # Latency and wall time span orders of magnitude across backends; FPS
        # does not need it and a log axis there would hide the slow models.
        if metric != "fps":
            ax.set_yscale("log")
    if FIGURE_TITLES:
        fig.suptitle("Compute cost: throughput, latency and wall time")
    _save(fig, output, "04_compute_throughput_latency")


def _complexity(snapshot: Snapshot, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    items = []
    for row in snapshot.rows:
        metrics = row["metrics"]
        ax.scatter(metrics["params_m"], metrics["flops_g"], s=45, color=COLORS[family(row)], alpha=0.85)
        items.append((metrics["params_m"], metrics["flops_g"], _short(row["model"])))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (M), log scale")
    ax.set_ylabel("FLOPs (G), log scale")
    if FIGURE_TITLES:
        ax.set_title("Model size and compute complexity")
    ax.grid(alpha=0.25, which="both")
    ax.margins(x=0.08, y=0.12)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=kind.title()) for kind, color in COLORS.items()]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    _annotate_points(ax, items)
    _save(fig, output, "05_parameters_vs_flops", layout=False)


def _quality_vs_latency(snapshot: Snapshot, output: Path) -> None:
    """Quality against latency, one panel per metric granularity.

    The panels follow the taxonomy's granularities, not the model families, so
    the pixel-level panel holds anomaly-map and segmentation models together.
    """

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    panels: list[tuple[object, list[tuple[float, float, str]]]] = []
    for panel_index, (ax, table) in enumerate(zip(axes, ("image_level", "pixel_level", "instance_level"))):
        metric, metric_label = QUALITY_METRIC[table]
        rows = [row for row in snapshot.rows if row["metrics"].get(metric) is not None]
        items = []
        for row in rows:
            metrics = row["metrics"]
            ax.scatter(metrics["latency_ms_mean"], metrics[metric], color=COLORS[family(row)], s=55)
            items.append((metrics["latency_ms_mean"], metrics[metric], _short(row["model"])))
        ax.set_xscale("log")
        ax.set_xlabel("Mean latency (ms), log scale")
        ax.set_ylabel(metric_label)
        ax.set_title(_panel_label(panel_index, table.replace("_", " ").title()))
        ax.grid(alpha=0.25, which="both")
        ax.margins(x=0.08, y=0.14)
        panels.append((ax, items))
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=kind.title()) for kind, color in COLORS.items()]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.91, 0.5), frameon=False)
    if FIGURE_TITLES:
        fig.suptitle("Quality versus latency, by metric granularity")
    fig.tight_layout()
    for ax, items in panels:
        _annotate_points(ax, items)
    _save(fig, output, "06_quality_vs_latency", layout=False)


def _memory(snapshot: Snapshot, output: Path) -> None:
    """Peak memory, split by the instrument that measured it.

    `process_rss` and `device_allocator` are not two views of one number: the
    first is everything the Python process held (weights, GPU context, host
    copies, framework overhead), the second is the tensor memory actually live on
    the graphics card. Comparing them across panels would rank a measurement
    artefact, so each instrument gets its own panel and its own y-axis.

    The panel titles stay in plain words -- "whole program" and "GPU" -- because
    a reader of the report is not expected to know what an RSS or a CUDA
    allocator is. The precise definitions live in `CAPTIONS.md` and in the
    report's memory section, which is where a reader goes for the detail.
    """

    memory = [(row, row["metrics"]) for row in snapshot.rows]
    # `sharey=False`: the two instruments differ by an order of magnitude, and a
    # shared axis squashed the GPU-tensor panel into an unreadable strip.
    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(memory) * 0.8), 6.5))
    for panel_index, (ax, kind, title, style) in enumerate((
        (axes[0], "process_rss", "Peak memory, whole program", BAR_STYLES[0]),
        (axes[1], "device_allocator", "Peak memory, GPU", BAR_STYLES[1]),
    )):
        group = [(row, metrics) for row, metrics in memory if metrics.get("memory_measurement_kind") == kind]
        group.sort(key=lambda item: item[1]["peak_memory_mb"])
        group_labels = [_short(row["model"]) for row, _ in group]
        ax.bar(np.arange(len(group)), [metrics["peak_memory_mb"] for _, metrics in group], width=0.52, linewidth=1.0, **style)
        ax.set_xticks(np.arange(len(group)), group_labels, rotation=70, ha="right")
        ax.set_title(_panel_label(panel_index, title))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Peak memory (MB)")
    axes[1].set_ylabel("Peak memory (MB)")
    if FIGURE_TITLES:
        fig.suptitle("Memory by measurement instrument; panels are not directly interchangeable")
    _save(fig, output, "07_memory_by_measurement_kind")


def _instance_counts(snapshot: Snapshot, output: Path) -> None:
    """Detections against ground truth at the evaluation threshold.

    The headline mAP figure says how good the boxes are; it does not say whether
    a model found them at all. TP/FP/FN and the precision/recall/F1 at that one
    threshold are what the report's instance table carries for that question.
    Panel titles are left off — the legend names the bars and the y-axis says
    what the unit is.
    """

    rows = [(row, row["metrics"]) for row in snapshot.rows if row["metrics"].get("true_positives") is not None]
    if not rows:
        return
    labels = [_short(row["model"]) for row, _ in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(rows) * 1.1), 6.5))
    for panel_index, (ax, metrics, style_offset, ylabel) in enumerate((
        (axes[0], INSTANCE_COUNT_METRICS, 0, "Boxes"),
        (axes[1], INSTANCE_RATE_METRICS, 1, "Score"),
    )):
        for index, metric in enumerate(metrics):
            values = [float(values[metric]) if values.get(metric) is not None else np.nan for _, values in rows]
            offset = (index - (len(metrics) - 1) / 2) * 0.26
            ax.bar(x + offset, values, width=0.24, linewidth=1.0, label=label_of(metric), **BAR_STYLES[(index + style_offset) % len(BAR_STYLES)])
        ax.set_xticks(x, labels, rotation=55, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncols=len(metrics), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10)
    axes[1].set_ylim(0, 1.02)
    _save(fig, output, "09_instance_counts")


# The three ranking dimensions, and the taxonomy tables each one scores. Not a
# metric anyone invents here: `technical` is the accuracy tables, `memory` and
# `compute` are the two overhead tables, and each metric's own `direction` says
# which way is better.
RANK_DIMENSIONS = (
    ("Technical", TECHNICAL_TABLES),
    ("Memory", ("memory",)),
    ("Compute", ("compute",)),
)


def _dimension_scores(rows: list[dict[str, Any]], tables: tuple[str, ...]) -> dict[str, float]:
    """One 0-1 score per model for a dimension, min-max over the models given.

    Every metric the taxonomy files under `tables` is normalized across these
    rows in the direction the taxonomy declares (`higher` / `lower`; `neutral`
    and non-numeric columns are skipped), and a model's dimension score is the
    average of the metrics it actually reports. Models are only ever compared
    with the other models in their own paradigm, because that is the only set
    the numbers were measured against.
    """

    specs = [
        spec for table in tables for spec in specs_for(table)
        if spec.direction in ("higher", "lower")
    ]
    scores: dict[str, float] = {}
    for row in rows:
        values: list[float] = []
        for spec in specs:
            value = row["metrics"].get(spec.key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            column = [
                float(other["metrics"][spec.key]) for other in rows
                if isinstance(other["metrics"].get(spec.key), (int, float))
                and math.isfinite(float(other["metrics"][spec.key]))
            ]
            if len(column) < 2:
                continue
            low, high = min(column), max(column)
            if high == low:
                values.append(1.0)
            elif spec.direction == "higher":
                values.append((float(value) - low) / (high - low))
            else:
                values.append((high - float(value)) / (high - low))
        if values:
            scores[row["model"]] = sum(values) / len(values)
    return scores


def _ranking(snapshot: Snapshot, output: Path) -> None:
    """Three dimension ranks per model, plus the combined rank as a line.

    One panel per paradigm, because an anomaly model and a detector were never
    measured on the same metrics. Within a panel each model gets one bar per
    dimension: how it ranks on the technical metrics, on memory and on compute.
    All three are ranks (1 = best) on one axis, so the bars are directly
    comparable. Models are ordered by their mean rank, but that mean is not
    drawn: a fourth series on the same axis was read as a fourth measurement
    rather than as a summary of the other three.
    """

    panels = (
        ("Anomaly detection / zero-shot", ("anomaly",)),
        ("Supervised defect detection", ("detection",)),
    )
    available = [panel for panel in panels if any(family(row) in panel[1] for row in snapshot.rows)]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(max(15, 8.5 * len(available)), 6.8), squeeze=False)
    width = 0.26

    shared_handles: list = []
    shared_labels: list[str] = []
    for panel_index, (ax, (title, families)) in enumerate(zip(axes.flat, available)):
        models = [row for row in snapshot.rows if family(row) in families]
        ranks: dict[str, dict[str, int]] = {}
        for name, tables in RANK_DIMENSIONS:
            scored = _dimension_scores(models, tables)
            # Best first: rank 1 is the highest dimension score.
            for position, model in enumerate(sorted(scored, key=lambda key: scored[key], reverse=True), start=1):
                ranks.setdefault(model, {})[name] = position
        combined = {model: sum(by_dim.values()) / len(by_dim) for model, by_dim in ranks.items() if by_dim}
        if not combined:
            ax.axis("off")
            continue
        ordered = sorted(combined, key=lambda model: combined[model])
        x = np.arange(len(ordered))
        for index, (name, _) in enumerate(RANK_DIMENSIONS):
            values = [ranks.get(model, {}).get(name, np.nan) for model in ordered]
            ax.bar(
                x + (index - (len(RANK_DIMENSIONS) - 1) / 2) * width, values,
                width=width * 0.9, linewidth=1.0, label=name, **BAR_STYLES[index],
            )
        ax.set_xticks(x, [_short(model) for model in ordered], rotation=55, ha="right")
        ax.set_ylabel("Rank (lower is better)")
        ax.set_title(_panel_label(panel_index, title))
        ax.set_xlim(-0.5, len(ordered) - 0.5)
        ax.grid(axis="y", alpha=0.25)
        # Both panels use the same three bar styles, so one legend at the top of
        # the figure replaces the two identical ones that used to sit over each
        # panel and repeat themselves.
        handles, labels = ax.get_legend_handles_labels()
        if not shared_handles:
            shared_handles, shared_labels = handles, labels

    # One legend for the figure: both panels use the same three bar styles, so a
    # legend per panel would say the same thing twice.
    if shared_handles:
        fig.legend(shared_handles, shared_labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(shared_labels), frameon=False, fontsize=10)
    _save(fig, output, "10_ranking")


def _figures(metrics: dict[str, list[str]], excluded: dict[str, str]):
    """Figure id -> (output stem, renderer).

    The id is the handle a driver (or a person) uses to ask for one figure, so
    the plot script does not have to be run whole. Kept as data rather than an
    if-chain so `--list` and `--only` cannot drift from what is rendered.
    """

    return {
        "01": ("01_image_level", _image_level),
        "02": ("02_pixel_level", lambda s, o: _grouped_bars(s, metrics["pixel_level"], "Pixel-level metrics: anomaly maps and segmentation masks", o, "02_pixel_level")),
        "03": ("03_instance_level", lambda s, o: _grouped_bars(s, metrics["instance_level"], "Instance-level detection metrics", o, "03_instance_level", set(excluded))),
        "04": ("04_compute_throughput_latency", _compute),
        "05": ("05_parameters_vs_flops", _complexity),
        "06": ("06_quality_vs_latency", _quality_vs_latency),
        "07": ("07_memory_by_measurement_kind", _memory),
        "08": ("08_instance_size_breakdown", lambda s, o: _grouped_bars(s, metrics["instance_breakdown"], "Instance-level size-bucketed AP and AR", o, "08_instance_size_breakdown", set(excluded))),
        "09": ("09_instance_counts", _instance_counts),
        "10": ("10_ranking", _ranking),
    }


def select_figures(only: list[str] | None, missing: bool, output: Path, figures: dict) -> list[str]:
    """Which figure ids to render, honouring `--only` and `--missing`."""

    ids = list(figures)
    if only:
        unknown = sorted(set(only) - set(figures))
        if unknown:
            raise SystemExit(f"unknown figure id(s): {', '.join(unknown)}. Known: {', '.join(ids)}")
        ids = [figure_id for figure_id in ids if figure_id in set(only)]
    if missing:
        ids = [figure_id for figure_id in ids if missing_formats(output, figures[figure_id][0])]
    return ids


def render(snapshot: Snapshot, output: Path, only: list[str] | None = None, missing: bool = False) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    metrics = figure_metrics(snapshot)
    excluded = instance_exclusions(snapshot)
    figures = _figures(metrics, excluded)

    # The audit and metadata describe the snapshot, not one figure, so they are
    # rewritten whenever the selected snapshot changes.
    write_audit(snapshot, output / "snapshot_audit.csv", audit_groups(snapshot))
    write_snapshot_metadata(snapshot, output / "snapshot.json")
    for model, reason in excluded.items():
        print(f"instance-level chart excludes {model}: {reason}")

    selected = select_figures(only, missing, output, figures)
    # The audit and the printed exclusions describe the whole snapshot; the
    # figures drop `EXCLUDED_MODELS` — filtered once here rather than in each
    # renderer, so a new figure cannot forget to.
    plottable = replace(snapshot, rows=[row for row in snapshot.rows if row["model"] not in EXCLUDED_MODELS])
    for figure_id in selected:
        stem, renderer = figures[figure_id]
        print(f"rendering {figure_id} -> {stem}")
        renderer(plottable, output)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--commit", help="Select an explicit git commit instead of the newest complete snapshot")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/local_benchmark_plots"))
    parser.add_argument("--only", nargs="+", metavar="ID", help="render only these figure ids (see --list)")
    parser.add_argument("--missing", action="store_true", help="render only figures missing at least one output format (png/pdf/svg)")
    parser.add_argument("--list", action="store_true", help="list the figure ids and their output state, then exit")
    args = parser.parse_args()

    if args.list:
        # No snapshot needed to answer "what can be drawn": that is the point of
        # a driver asking the script rather than hardcoding its figure list.
        metrics = {key: [] for key in ("image_level", "pixel_level", "instance_level")}
        for figure_id, (stem, _) in _figures(metrics, {}).items():
            missing = missing_formats(args.output_dir, stem)
            state = "present" if not missing else f"missing {','.join(missing)}"
            print(f"{figure_id}  {stem:<34} {state}")
        return 0
    snapshot = load_latest_snapshot(log=args.input, commit=args.commit)
    selected = render(snapshot, args.output_dir, only=args.only, missing=args.missing)
    # Name the data behind the figures: a driver that "finds the latest" has to
    # be able to say which latest it found.
    print(f"snapshot commit: {snapshot.commit}")
    print(f"snapshot source: {snapshot.source}")
    print(f"snapshot rows:   {len(snapshot.rows)} models, {snapshot.samples} samples, latest row {snapshot.timestamp}")
    print(f"rendered:        {', '.join(selected) if selected else '<none>'}")
    print(f"outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
