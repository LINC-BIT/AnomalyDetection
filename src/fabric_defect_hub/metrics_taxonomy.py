"""The one place that says which table a metric belongs in.

Two top-level parts, and the split is not cosmetic — they answer different
questions and are read by different people:

* **technical** — how well the model detects, broken out by the granularity
  of the ground truth it was scored against: `image_level` (is this image
  defective), `pixel_level` (which pixels), `instance_level` (which defect,
  as a box), plus `cross_domain` (does any of it survive an unseen fabric).
* **overhead** — what running it costs: `compute`, `memory`, and
  `communication`.

`communication` is declared here and reports `not_implemented`. It covers
event-vs-continuous bandwidth saving, which only means something once there
is a distributed deployment to measure it on. Declaring it empty rather than
omitting it is deliberate: a reader comparing this against the project's
metric plan can see the row exists and is unbuilt, instead of wondering
whether it was measured and lost.

Everything downstream — the JSONL reader, the Gradio tables, any LaTeX
export — reads its grouping from here. That is the point: the web layer used
to decide its own columns, which is how a metric could be measured and then
never shown, or shown under a heading that disagreed with the report.

This module is pure: it maps names to groups and reshapes rows. It measures
nothing and imports no framework, so `import fabric_defect_hub` stays cheap
and a front end can ask "what tables exist" without a backend installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

Category = Literal["technical", "overhead"]
TableStatus = Literal["available", "not_implemented", "empty"]

CATEGORIES: tuple[Category, ...] = ("technical", "overhead")

TECHNICAL_TABLES: tuple[str, ...] = ("image_level", "pixel_level", "instance_level", "cross_domain")
OVERHEAD_TABLES: tuple[str, ...] = ("compute", "memory", "communication")
TABLES: tuple[str, ...] = (*TECHNICAL_TABLES, *OVERHEAD_TABLES)

# Tables with no measurement behind them yet, and why. Kept as data so the
# UI renders the same explanation the docs give.
# Why a table can legitimately come back empty. An empty table with no
# explanation reads as "broken", and the three reasons below are genuinely
# different: the first two mean "these models cannot produce this", the rest
# mean "you did not ask for it". Shown by the UI whenever a table has no rows.
EMPTY_HINTS: dict[str, str] = {
    "image_level": (
        "No selected model reports image-level scores. These come from the anomaly "
        "backends (PatchCore, PaDiM, RD4AD, STFPM, GANomaly, Dinomaly, MambaAD, MoECLIP); "
        "detection models such as YOLO and Faster R-CNN are scored at instance level instead."
    ),
    "pixel_level": (
        "No selected model reports pixel-level scores. Needs a model that outputs an "
        "anomaly map or a mask, and — for the anomaly backends — somewhere to persist "
        "those maps (`output_dir` / `--anomaly-map-dir`), without which only image-level "
        "metrics can be computed."
    ),
    "instance_level": (
        "No selected model reports box-level scores. These come from the detection "
        "backends (YOLO, Faster R-CNN, Cascade R-CNN, DETR, Mask R-CNN)."
    ),
    "cross_domain": (
        "Not measured. Pick a held-out dataset in the cross-domain selector "
        "(or pass `cross_domain_patterns=` to `fdh.measure`) — it re-evaluates the same "
        "weights on fabrics the model was not trained on."
    ),
    "compute": (
        "Only wall-clock time was recorded. Tick **Include profiling** to measure FPS, "
        "latency percentiles and frame-rate jitter — profiling exports and re-runs each "
        "model, so it is opt-in rather than part of every benchmark."
    ),
    "memory": (
        "Not measured. Tick **Include profiling** — peak and average memory are sampled "
        "during the profiling pass, not during scoring."
    ),
}

UNIMPLEMENTED: dict[str, str] = {
    "communication": (
        "Bandwidth saving is not measured without a deployed transport. "
        "The sweep records model/export transfer size as a communication proxy; "
        "actual link throughput requires a configured network endpoint."
    ),
}


@dataclass(frozen=True)
class MetricSpec:
    """One metric: where it belongs, how to show it, which way is better."""

    key: str
    label: str
    category: Category
    table: str
    direction: Literal["higher", "lower", "neutral"] = "neutral"
    unit: str = ""
    precision: int = 4

    def format(self, value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "" if value is None else str(value)
        text = f"{value:.{self.precision}f}".rstrip("0").rstrip(".")
        return f"{text} {self.unit}".strip()


def _spec(key, label, category, table, direction="neutral", unit="", precision=4) -> MetricSpec:
    return MetricSpec(key, label, category, table, direction, unit, precision)


# Every key observed in a real sweep, plus the ones `benchmark.py` adds.
# Ordered within each table the way the table should read.
METRIC_SPECS: tuple[MetricSpec, ...] = (
    # -- technical / image level ------------------------------------------
    _spec("image_auroc", "Image AUROC", "technical", "image_level", "higher"),
    _spec("image_f1", "Image F1", "technical", "image_level", "higher"),
    _spec("image_precision", "Image Precision", "technical", "image_level", "higher"),
    _spec("image_recall", "Image Recall", "technical", "image_level", "higher"),
    _spec("image_threshold", "Decision threshold", "technical", "image_level"),
    _spec("auroc", "AUROC", "technical", "image_level", "higher"),
    _spec("ap", "AP", "technical", "image_level", "higher"),
    # -- technical / pixel level ------------------------------------------
    _spec("pixel_auroc", "Pixel AUROC", "technical", "pixel_level", "higher"),
    _spec("pixel_aupro", "AUPRO", "technical", "pixel_level", "higher"),
    _spec("iap", "IAP", "technical", "pixel_level", "higher"),
    _spec("pixel_f1", "Pixel F1", "technical", "pixel_level", "higher"),
    _spec("miou", "mIoU", "technical", "pixel_level", "higher"),
    _spec("dice", "Dice", "technical", "pixel_level", "higher"),
    _spec("pro_score", "PRO score", "technical", "pixel_level", "higher"),
    # -- technical / instance level ---------------------------------------
    _spec("map", "mAP@[.5:.95]", "technical", "instance_level", "higher"),
    _spec("map_50", "mAP@0.5", "technical", "instance_level", "higher"),
    _spec("map_75", "mAP@0.75", "technical", "instance_level", "higher"),
    _spec("map_small", "mAP small", "technical", "instance_level", "higher"),
    _spec("map_medium", "mAP medium", "technical", "instance_level", "higher"),
    _spec("map_large", "mAP large", "technical", "instance_level", "higher"),
    _spec("mar_1", "mAR@1", "technical", "instance_level", "higher"),
    _spec("mar_10", "mAR@10", "technical", "instance_level", "higher"),
    _spec("mar_100", "mAR@100", "technical", "instance_level", "higher"),
    _spec("mar_small", "mAR small", "technical", "instance_level", "higher"),
    _spec("mar_medium", "mAR medium", "technical", "instance_level", "higher"),
    _spec("mar_large", "mAR large", "technical", "instance_level", "higher"),
    _spec("precision_at_threshold", "Precision @0.5", "technical", "instance_level", "higher"),
    _spec("recall_at_threshold", "Recall @0.5", "technical", "instance_level", "higher"),
    _spec("f1_at_threshold", "F1 @0.5", "technical", "instance_level", "higher"),
    # Size-bucketed recall: the fabric-specific pair. Small defects (<10px
    # shorter side) are the broken-warp/skipped-pick cases an aggregate
    # recall averages away.
    _spec("recall_small", "Recall (small <10px)", "technical", "instance_level", "higher"),
    _spec("recall_normal", "Recall (normal)", "technical", "instance_level", "higher"),
    _spec("true_positives", "TP", "technical", "instance_level", precision=0),
    _spec("false_positives", "FP", "technical", "instance_level", "lower", precision=0),
    _spec("false_negatives", "FN", "technical", "instance_level", "lower", precision=0),
    # -- technical / cross domain -----------------------------------------
    _spec("acc_src", "Source accuracy", "technical", "cross_domain", "higher"),
    # The Benchmark tab's own cross-domain column. It measures the same
    # thing as `mean_degradation_pct` but comes from `web/benchmark.py`'s
    # single held-out dataset rather than the pattern sweep, so it keeps its
    # own key — omitting it meant the cross-domain table stayed empty no
    # matter what the page measured.
    _spec("cross_domain_delta_acc_pct", "Cross-domain drop", "technical", "cross_domain", "lower", "%", 2),
    _spec("top_k_mean_degradation_pct", "Top-k mean drop", "technical", "cross_domain", "lower", "%", 2),
    _spec("mean_degradation_pct", "Mean drop (all patterns)", "technical", "cross_domain", "lower", "%", 2),
    _spec("ci_low", "CI low", "technical", "cross_domain", unit="%", precision=2),
    _spec("ci_high", "CI high", "technical", "cross_domain", unit="%", precision=2),
    _spec("k_effective", "k used", "technical", "cross_domain", precision=0),
    _spec("scored_pattern_count", "Patterns scored", "technical", "cross_domain", precision=0),
    # -- overhead / compute ------------------------------------------------
    _spec("fps", "FPS (aggregate)", "overhead", "compute", "higher", precision=2),
    _spec("instantaneous_fps_mean", "Instantaneous FPS mean", "overhead", "compute", "higher", precision=2),
    _spec("instantaneous_fps_std", "Instantaneous FPS stdev", "overhead", "compute", "lower", precision=3),
    _spec("instantaneous_fps_cv", "Instantaneous FPS CV", "overhead", "compute", "lower", precision=4),
    _spec("latency_ms_mean", "Latency mean", "overhead", "compute", "lower", "ms", 2),
    _spec("latency_ms_std", "Latency stdev", "overhead", "compute", "lower", "ms", 2),
    _spec("latency_ms_cv", "Latency CV", "overhead", "compute", "lower", precision=4),
    _spec("latency_ms_p50", "Latency p50", "overhead", "compute", "lower", "ms", 2),
    _spec("latency_ms_p95", "Latency p95", "overhead", "compute", "lower", "ms", 2),
    _spec("latency_ms_p99", "Latency p99", "overhead", "compute", "lower", "ms", 2),
    _spec("flops", "FLOPs", "overhead", "compute", "lower", precision=0),
    _spec("flops_g", "FLOPs (G)", "overhead", "compute", "lower", precision=4),
    _spec("params_m", "Parameters (M)", "overhead", "memory", "lower", precision=4),
    _spec("lmei", "LMEI", "overhead", "compute", "higher", precision=4),
    _spec("max_concurrent_streams", "Max streams @budget", "overhead", "compute", "higher", precision=0),
    _spec("single_stream_latency_ms", "1-stream latency", "overhead", "compute", "lower", "ms", 2),
    _spec("resolution_slope_beta", "Throughput-resolution slope", "overhead", "compute", "higher", precision=8),
    _spec("delta_beta", "Slope difference", "overhead", "compute", precision=8),
    _spec("power_w_mean", "Power mean", "overhead", "compute", "lower", "W", 2),
    _spec("power_w_peak", "Power peak", "overhead", "compute", "lower", "W", 2),
    _spec("energy_j", "Energy", "overhead", "compute", "lower", "J", 2),
    _spec("runtime_s", "Wall time", "overhead", "compute", "lower", "s", 2),
    # -- overhead / memory -------------------------------------------------
    _spec("peak_memory_mb", "Memory peak", "overhead", "memory", "lower", "MB", 1),
    _spec("avg_memory_mb", "Memory average", "overhead", "memory", "lower", "MB", 1),
    _spec("model_size_mb", "Model size", "overhead", "memory", "lower", "MB", 2),
    _spec("power_sample_count", "Power samples", "overhead", "compute", precision=0),
    _spec("model_transfer_mb", "Model transfer", "overhead", "compute", "lower", "MB", 2),
    _spec("model_transfer_bytes", "Model transfer bytes", "overhead", "compute", "lower", "B", 0),
    _spec("export_transfer_mb", "Export transfer", "overhead", "compute", "lower", "MB", 2),
    _spec("export_transfer_bytes", "Export transfer bytes", "overhead", "compute", "lower", "B", 0),
)

_BY_KEY: dict[str, MetricSpec] = {spec.key: spec for spec in METRIC_SPECS}

# Keys that describe the run rather than measure the model. Excluded from
# every table so a "measured nothing" row cannot look populated.
BOOKKEEPING_KEYS: frozenset[str] = frozenset({
    # Row identity, not a measurement — a leaderboard row carries it inline.
    "model", "backend", "variant", "task", "dataset", "device", "timestamp_utc",
    "sample_count", "num_evaluated", "classes", "frame_budget_ms",
    "resolution_sweep_points", "resolution_slope_alpha", "concurrency_probe_points",
    "metric", "mode", "source_value", "k", "selected_patterns", "skipped_patterns",
    "per_pattern_degradation_pct", "traceback_tail",
    "memory_measurement_kind", "memory_measurement_scope", "memory_cross_engine_comparable",
    # `scoring.py`'s blended rankings. Derived *from* the metrics below
    # rather than measured, so they belong to a summary rather than to any
    # one table — listing them here keeps them out of the tables without
    # `unrecognised_metrics()` reporting them as an unplaced measurement.
    "composite_score", "technical_score", "overhead_score",
})

# The blended rankings, in display order — shown as their own summary row
# above the tables rather than mixed into them.
SUMMARY_KEYS: tuple[str, ...] = ("composite_score", "technical_score", "overhead_score", "runtime_s")


def classify(key: str) -> MetricSpec | None:
    """The spec for `key`, or `None` if it is bookkeeping or unrecognised."""

    return None if key in BOOKKEEPING_KEYS else _BY_KEY.get(key)


def table_of(key: str) -> str | None:
    spec = classify(key)
    return spec.table if spec else None


def category_of(key: str) -> Category | None:
    spec = classify(key)
    return spec.category if spec else None


def specs_for(table: str) -> tuple[MetricSpec, ...]:
    return tuple(spec for spec in METRIC_SPECS if spec.table == table)


def unrecognised(keys: Iterable[str]) -> list[str]:
    """Measured keys this taxonomy has no home for.

    Surfaced rather than dropped: a metric that is computed and then silently
    absent from every table is the failure this module exists to prevent.
    """

    return sorted({
        key for key in keys
        if key not in BOOKKEEPING_KEYS and key not in _BY_KEY
    })


# --------------------------------------------------------------------------- #
# Turning sweep rows into tables
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricTable:
    """One rendered table: a category, a name, and rows keyed by model.

    `columns` holds only metrics at least one model actually reported, so a
    table never shows a column of blanks — with eighteen heterogeneous
    models, showing every declared metric would make every table mostly
    empty.
    """

    name: str
    category: Category
    columns: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    status: TableStatus = "available"
    note: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.rows or not self.columns

    def as_matrix(self) -> list[list[Any]]:
        """`rows` as a list-of-lists with the model name first — the shape
        Gradio's `gr.Dataframe` and most report writers want."""

        return [
            [row.get("model", "")] + [
                _BY_KEY[column].format(row.get(column)) if column in _BY_KEY else row.get(column, "")
                for column in self.columns
            ]
            for row in self.rows
        ]

    def header(self) -> list[str]:
        return ["Model", *self.labels]


def build_tables(
    rows: Sequence[dict[str, Any]],
    *,
    include_unimplemented: bool = True,
) -> list[MetricTable]:
    """Group sweep rows into one `MetricTable` per declared table.

    `rows` are the JSONL records `metric_sweep.read_sweep` returns (or any
    mapping with `model` and `metrics`). Rows whose status is not `ok` carry
    no metrics and drop out naturally.

    Every declared table is returned, in `TABLES` order, even when empty —
    a front end showing "no data yet" for a table is informative, whereas a
    table that silently disappears looks like the metric was never planned.
    """

    per_table: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in TABLES}
    for row in rows:
        if row.get("status") not in (None, "ok"):
            continue
        model = row.get("model")
        metrics = row.get("metrics") or {}
        if not model or not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            spec = classify(key)
            if spec is None:
                continue
            bucket = per_table[spec.table].setdefault(model, {"model": model})
            bucket[key] = value

    tables: list[MetricTable] = []
    for name in TABLES:
        category: Category = "technical" if name in TECHNICAL_TABLES else "overhead"
        if name in UNIMPLEMENTED:
            if include_unimplemented:
                tables.append(MetricTable(
                    name=name, category=category,
                    status="not_implemented", note=UNIMPLEMENTED[name],
                ))
            continue

        by_model = per_table[name]
        present = {key for values in by_model.values() for key in values if key != "model"}
        columns = [spec.key for spec in specs_for(name) if spec.key in present]
        populated = bool(by_model and columns)
        tables.append(MetricTable(
            name=name,
            category=category,
            columns=columns,
            labels=[_BY_KEY[key].label for key in columns],
            rows=sorted(by_model.values(), key=lambda item: str(item.get("model", ""))),
            status="available" if populated else "empty",
            # An empty table without a reason reads as a broken page. The
            # hint distinguishes "these models cannot produce this" from
            # "you did not switch it on".
            note="" if populated else EMPTY_HINTS.get(name, ""),
        ))
    return tables


def tables_by_category(tables: Sequence[MetricTable]) -> dict[str, list[MetricTable]]:
    """`{"technical": [...], "overhead": [...]}` — the two-part shape the UI
    renders as two sections."""

    grouped: dict[str, list[MetricTable]] = {category: [] for category in CATEGORIES}
    for table in tables:
        grouped[table.category].append(table)
    return grouped
