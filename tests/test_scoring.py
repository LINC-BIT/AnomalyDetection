"""Tests for `scoring.py`'s technical/overhead composite leaderboard score."""

import pytest

import math

from fabric_defect_hub.scoring import (
    SCORE_PRESETS,
    metric_direction,
    metric_group,
    normalize_metrics,
    polygon_area,
    score_rows,
)


def test_metric_direction_classifies_known_metrics():
    assert metric_direction("image_auroc") == "higher"
    assert metric_direction("map50") == "higher"
    assert metric_direction("miou") == "higher"
    assert metric_direction("latency_ms_mean") == "lower"
    assert metric_direction("peak_memory_mb") is None
    assert metric_direction("model_size_mb") == "lower"
    assert metric_direction("runtime_s") == "lower"
    assert metric_direction("fps") == "higher"


def test_fps_jitter_metrics_take_precedence_over_generic_fps_rule():
    assert metric_direction("instantaneous_fps_std") == "lower"
    assert metric_direction("instantaneous_fps_cv") == "lower"


def test_unrecognized_metric_is_ignored_not_errored():
    assert metric_direction("num_skipped_empty") is None
    assert metric_group("num_skipped_empty") is None


def test_metric_group_splits_technical_and_overhead():
    assert metric_group("image_auroc") == "technical"
    assert metric_group("fps") == "overhead"
    assert metric_group("latency_ms_mean") == "overhead"


def test_score_rows_ranks_by_composite_with_known_example():
    rows = [
        {"model": "a", "image_auroc": 0.9, "fps": 10.0},
        {"model": "b", "image_auroc": 0.5, "fps": 100.0},
    ]
    scored = score_rows(rows, technical_weight=0.5, overhead_weight=0.5)
    by_model = {row["model"]: row for row in scored}

    # "a" has the max (best) technical value and the min (worst) overhead
    # value; min-max normalization puts it at 100 technical / 0 overhead.
    assert by_model["a"]["technical_score"] == pytest.approx(100.0)
    assert by_model["a"]["overhead_score"] == pytest.approx(0.0)
    assert by_model["a"]["composite_score"] == pytest.approx(50.0)

    assert by_model["b"]["technical_score"] == pytest.approx(0.0)
    assert by_model["b"]["overhead_score"] == pytest.approx(100.0)
    assert by_model["b"]["composite_score"] == pytest.approx(50.0)


def test_score_rows_weighting_favors_the_heavier_group():
    rows = [
        {"model": "a", "image_auroc": 0.9, "fps": 10.0},
        {"model": "b", "image_auroc": 0.5, "fps": 100.0},
    ]
    accuracy_first = {row["model"]: row for row in score_rows(rows, *SCORE_PRESETS["accuracy_first"])}
    efficiency_first = {row["model"]: row for row in score_rows(rows, *SCORE_PRESETS["efficiency_first"])}

    assert accuracy_first["a"]["composite_score"] > accuracy_first["b"]["composite_score"]
    assert efficiency_first["b"]["composite_score"] > efficiency_first["a"]["composite_score"]


def test_score_rows_handles_missing_metric_group():
    rows = [
        {"model": "a", "image_auroc": 0.9},
        {"model": "b", "image_auroc": 0.5, "fps": 50.0},
    ]
    scored = {row["model"]: row for row in score_rows(rows)}

    assert scored["a"]["overhead_score"] is None
    # "a" has no overhead metrics at all, so its composite falls back to
    # 100% technical weight instead of averaging in a missing group.
    assert scored["a"]["composite_score"] == pytest.approx(scored["a"]["technical_score"])
    assert scored["b"]["overhead_score"] is not None


def test_score_rows_all_tied_values_do_not_divide_by_zero():
    rows = [
        {"model": "a", "image_auroc": 0.7},
        {"model": "b", "image_auroc": 0.7},
    ]
    scored = score_rows(rows)
    assert all(row["technical_score"] == pytest.approx(100.0) for row in scored)


def test_score_rows_metric_keys_restricts_which_metrics_are_scored():
    rows = [
        {"model": "a", "image_auroc": 0.9, "fps": 10.0},
        {"model": "b", "image_auroc": 0.5, "fps": 100.0},
    ]
    scored = score_rows(rows, metric_keys={"image_auroc"})
    assert all(row["overhead_score"] is None for row in scored)


def test_score_rows_rejects_non_positive_weights():
    with pytest.raises(ValueError):
        score_rows([{"model": "a"}], technical_weight=0.0, overhead_weight=0.0)


def test_score_rows_rejects_negative_weights():
    with pytest.raises(ValueError):
        score_rows([{"model": "a"}], technical_weight=-1.0, overhead_weight=1.0)


def test_normalize_metrics_returns_sorted_names_and_per_row_values():
    rows = [
        {"model": "a", "image_auroc": 0.9, "fps": 10.0},
        {"model": "b", "image_auroc": 0.5, "fps": 100.0},
    ]
    names, per_row = normalize_metrics(rows)

    assert names == ["fps", "image_auroc"]  # sorted, and "model" excluded
    assert per_row[0]["image_auroc"] == pytest.approx(1.0)
    assert per_row[1]["image_auroc"] == pytest.approx(0.0)


def test_normalize_metrics_inverts_lower_is_better_metrics():
    rows = [
        {"model": "fast", "latency_ms_mean": 5.0},
        {"model": "slow", "latency_ms_mean": 50.0},
    ]
    _, per_row = normalize_metrics(rows)

    # Lower latency is better, so the fast model must normalize to 1.0.
    assert per_row[0]["latency_ms_mean"] == pytest.approx(1.0)
    assert per_row[1]["latency_ms_mean"] == pytest.approx(0.0)


def test_normalize_metrics_omits_metrics_a_row_does_not_carry():
    rows = [
        {"model": "a", "image_auroc": 0.9},
        {"model": "b", "image_auroc": 0.5, "fps": 30.0},
    ]
    names, per_row = normalize_metrics(rows)

    assert names == ["fps", "image_auroc"]
    assert "fps" not in per_row[0]
    assert "fps" in per_row[1]


def test_normalize_metrics_respects_metric_keys():
    rows = [{"model": "a", "image_auroc": 0.9, "fps": 10.0}]
    names, _ = normalize_metrics(rows, metric_keys={"fps"})
    assert names == ["fps"]


def test_normalize_metrics_all_tied_values_normalize_to_one():
    rows = [{"model": "a", "image_auroc": 0.7}, {"model": "b", "image_auroc": 0.7}]
    _, per_row = normalize_metrics(rows)
    assert [row["image_auroc"] for row in per_row] == [1.0, 1.0]


def test_polygon_area_matches_known_regular_polygons():
    # A regular n-gon with unit circumradius has area (n/2)*sin(2*pi/n).
    for count in (3, 4, 6):
        expected = (count / 2) * math.sin(2 * math.pi / count)
        assert polygon_area([1.0] * count) == pytest.approx(expected)


def test_polygon_area_scales_with_the_square_of_the_radii():
    full = polygon_area([1.0, 1.0, 1.0, 1.0])
    half = polygon_area([0.5, 0.5, 0.5, 0.5])
    assert half == pytest.approx(full * 0.25)


def test_polygon_area_of_fewer_than_three_axes_is_zero():
    assert polygon_area([]) == 0.0
    assert polygon_area([1.0]) == 0.0
    assert polygon_area([1.0, 1.0]) == 0.0
