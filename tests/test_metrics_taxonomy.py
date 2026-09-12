"""The metric taxonomy: two parts, seven tables, one authority.

The tests that matter are the ones about *coverage* — a metric that is
measured but belongs to no table is invisible on a page made of named
tables, which is precisely the failure this module was built to stop.
"""

from __future__ import annotations

import pytest

from fabric_defect_hub.metrics_taxonomy import (
    BOOKKEEPING_KEYS,
    CATEGORIES,
    METRIC_SPECS,
    OVERHEAD_TABLES,
    TABLES,
    TECHNICAL_TABLES,
    UNIMPLEMENTED,
    build_tables,
    category_of,
    classify,
    specs_for,
    table_of,
    tables_by_category,
    unrecognised,
)


def _row(model, metrics, status="ok"):
    return {"model": model, "status": status, "metrics": metrics}


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_the_two_parts_partition_every_table():
    assert set(TABLES) == set(TECHNICAL_TABLES) | set(OVERHEAD_TABLES)
    assert not set(TECHNICAL_TABLES) & set(OVERHEAD_TABLES)
    assert set(CATEGORIES) == {"technical", "overhead"}


def test_every_spec_lands_in_a_declared_table_of_its_own_category():
    for spec in METRIC_SPECS:
        assert spec.table in TABLES, spec.key
        expected = "technical" if spec.table in TECHNICAL_TABLES else "overhead"
        assert spec.category == expected, spec.key


def test_no_metric_key_is_declared_twice():
    keys = [spec.key for spec in METRIC_SPECS]
    assert len(keys) == len(set(keys))


def test_a_key_cannot_be_both_a_metric_and_bookkeeping():
    assert not {spec.key for spec in METRIC_SPECS} & BOOKKEEPING_KEYS


def test_communication_is_declared_unimplemented_rather_than_omitted():
    """Omitting it would read as "never planned"; declaring it empty says
    "planned, not built", which is the true state."""

    assert "communication" in OVERHEAD_TABLES
    assert "communication" in UNIMPLEMENTED
    assert specs_for("communication") == ()


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key,table", [
    ("image_auroc", "image_level"),
    ("pixel_aupro", "pixel_level"),
    ("iap", "pixel_level"),
    ("map_50", "instance_level"),
    ("recall_small", "instance_level"),
    ("top_k_mean_degradation_pct", "cross_domain"),
    ("instantaneous_fps_cv", "compute"),
    ("max_concurrent_streams", "compute"),
    ("peak_memory_mb", "memory"),
    ("avg_memory_mb", "memory"),
])
def test_known_metrics_land_where_the_plan_says(key, table):
    assert table_of(key) == table


def test_bookkeeping_keys_classify_as_nothing():
    assert classify("sample_count") is None
    assert table_of("sample_count") is None
    assert category_of("sample_count") is None


def test_an_unknown_key_is_reported_not_guessed():
    assert classify("some_new_metric") is None
    assert unrecognised(["some_new_metric", "fps", "sample_count"]) == ["some_new_metric"]


def test_every_metric_this_project_measures_has_a_home():
    """Regression guard: the keys real sweeps produced must all classify.
    A new metric added without a `MetricSpec` fails here rather than
    silently vanishing from every table.
    """

    measured = {
        # accuracy
        "image_auroc", "image_f1", "image_precision", "image_recall", "image_threshold",
        "map", "map_50", "map_75", "map_large", "map_medium", "map_small",
        "mar_1", "mar_10", "mar_100", "mar_large", "mar_medium", "mar_small",
        "miou", "dice", "pixel_f1", "recall_small", "recall_normal",
        "precision_at_threshold", "recall_at_threshold", "f1_at_threshold",
        "true_positives", "false_positives", "false_negatives",
        # runtime / scaling / concurrency
        "fps", "instantaneous_fps_mean", "instantaneous_fps_std", "instantaneous_fps_cv",
        "latency_ms_mean", "latency_ms_std", "latency_ms_cv", "latency_ms_p50",
        "latency_ms_p95", "latency_ms_p99", "peak_memory_mb", "avg_memory_mb",
        "resolution_slope_beta", "max_concurrent_streams", "single_stream_latency_ms",
        # cross domain
        "acc_src", "ci_high", "ci_low", "mean_degradation_pct",
        "top_k_mean_degradation_pct", "k_effective", "scored_pattern_count",
    }
    assert unrecognised(measured) == []


# --------------------------------------------------------------------------- #
# Building tables
# --------------------------------------------------------------------------- #
def test_metrics_split_across_their_tables():
    rows = [_row("yolov8n", {"map_50": 0.4, "fps": 30.0, "peak_memory_mb": 800.0})]

    tables = {table.name: table for table in build_tables(rows)}

    assert tables["instance_level"].columns == ["map_50"]
    assert tables["compute"].columns == ["fps"]
    assert tables["memory"].columns == ["peak_memory_mb"]
    assert tables["image_level"].status == "empty"


def test_a_column_no_model_reported_is_not_shown():
    """With heterogeneous backends, showing every declared metric would make
    every table mostly blank."""

    tables = {t.name: t for t in build_tables([_row("m", {"fps": 10.0})])}

    assert tables["compute"].columns == ["fps"]
    assert "energy_j" not in tables["compute"].columns


def test_rows_that_did_not_measure_are_excluded():
    rows = [_row("a", {"fps": 1.0}), _row("b", {"fps": 2.0}, status="failed")]

    compute = {t.name: t for t in build_tables(rows)}["compute"]

    assert [row["model"] for row in compute.rows] == ["a"]


def test_bookkeeping_alone_leaves_a_table_empty():
    tables = {t.name: t for t in build_tables([_row("m", {"sample_count": 8})])}

    assert all(table.is_empty for table in tables.values())


def test_every_declared_table_is_returned_even_when_empty():
    """A table that disappears looks like the metric was never planned."""

    names = [table.name for table in build_tables([])]

    assert names == list(TABLES)


def test_an_empty_table_explains_why_it_is_empty():
    """An empty table with no reason reads as a broken page. The reasons are
    genuinely different: a YOLO run cannot produce pixel metrics, whereas
    memory is empty only because profiling was not switched on.
    """

    from fabric_defect_hub.metrics_taxonomy import EMPTY_HINTS

    tables = {t.name: t for t in build_tables([_row("yolov8n", {"map_50": 0.4})])}

    assert tables["instance_level"].status == "available"
    assert tables["instance_level"].note == ""
    for name in ("image_level", "pixel_level", "cross_domain", "compute", "memory"):
        assert tables[name].status == "empty"
        assert tables[name].note == EMPTY_HINTS[name], name


def test_every_table_that_can_be_empty_has_a_hint():
    """Guards against a new table being added with no explanation."""

    from fabric_defect_hub.metrics_taxonomy import EMPTY_HINTS

    explainable = set(TABLES) - set(UNIMPLEMENTED)
    assert explainable <= set(EMPTY_HINTS), sorted(explainable - set(EMPTY_HINTS))


def test_the_benchmark_tabs_cross_domain_column_is_classified():
    """`web/benchmark.py` emits its own key; without a spec the cross-domain
    table stayed empty no matter what the page measured."""

    assert table_of("cross_domain_delta_acc_pct") == "cross_domain"


def test_blended_scores_are_not_treated_as_unplaced_measurements():
    """`composite_score` and friends are derived rankings, so they belong to
    a summary rather than to any metric table — but they must not trigger the
    'measured but shown nowhere' warning either."""

    scores = ["composite_score", "technical_score", "overhead_score"]

    assert unrecognised(scores) == []
    assert all(classify(key) is None for key in scores)


def test_row_identity_keys_are_not_reported_as_metrics():
    assert unrecognised(["model", "backend", "dataset", "device"]) == []


def test_the_unimplemented_table_carries_its_explanation():
    communication = {t.name: t for t in build_tables([])}["communication"]

    assert communication.status == "not_implemented"
    assert "Bandwidth saving" in communication.note


def test_unimplemented_can_be_omitted_for_a_compact_report():
    names = [table.name for table in build_tables([], include_unimplemented=False)]

    assert "communication" not in names


def test_by_category_returns_the_two_part_split():
    grouped = tables_by_category(build_tables([_row("m", {"fps": 1.0, "map_50": 0.2})]))

    assert set(grouped) == {"technical", "overhead"}
    assert [t.name for t in grouped["overhead"]] == list(OVERHEAD_TABLES)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def test_values_carry_their_unit_and_precision():
    tables = {t.name: t for t in build_tables([
        _row("m", {"latency_ms_mean": 15.6123, "peak_memory_mb": 1286.44})
    ])}

    assert tables["compute"].as_matrix()[0][1] == "15.61 ms"
    assert tables["memory"].as_matrix()[0][1] == "1286.4 MB"


def test_the_header_puts_the_model_first_with_human_labels():
    table = {t.name: t for t in build_tables([_row("m", {"pixel_aupro": 0.9})])}["pixel_level"]

    assert table.header() == ["Model", "AUPRO"]


def test_rows_are_ordered_by_model_for_a_stable_table():
    rows = [_row("z", {"fps": 1.0}), _row("a", {"fps": 2.0})]

    compute = {t.name: t for t in build_tables(rows)}["compute"]

    assert [row["model"] for row in compute.rows] == ["a", "z"]
