"""Pattern-level cross-domain protocol: train on a few fabric patterns,
evaluate on every remaining one, reduce to a top-k mean with a CI.

The sweep takes its per-pattern evaluation as a callback, so these tests
drive it with synthetic accuracies -- the aggregation rules (which patterns
get picked, what a skipped pattern does, how the interval is formed) are
exactly the part that must not drift, and none of them need staged data.
"""

import math

import pytest

from fabric_defect_hub.evaluation.cross_domain import (
    cross_domain_degradation,
    pattern_sweep_degradation,
    top_k_mean,
)


# --------------------------------------------------------------------------- #
# top_k_mean
# --------------------------------------------------------------------------- #
def test_worst_mode_picks_the_largest_degradations():
    picked = top_k_mean({"p5": 1.0, "p6": 40.0, "p7": 20.0, "p8": 3.0}, k=2, mode="worst")

    assert picked["selected_patterns"] == ["p6", "p7"]
    assert picked["top_k_mean_degradation_pct"] == pytest.approx(30.0)


def test_best_mode_picks_the_smallest_degradations():
    picked = top_k_mean({"p5": 1.0, "p6": 40.0, "p7": 20.0, "p8": 3.0}, k=2, mode="best")

    assert picked["selected_patterns"] == ["p5", "p8"]
    assert picked["top_k_mean_degradation_pct"] == pytest.approx(2.0)


def test_the_two_modes_report_opposite_ends_of_the_same_run():
    # Why `mode` is explicit rather than implied: the same sweep yields very
    # different headline numbers, so a table has to say which it used.
    degradations = {"p5": 2.0, "p6": 50.0, "p7": 4.0}

    worst = top_k_mean(degradations, k=1, mode="worst")
    best = top_k_mean(degradations, k=1, mode="best")

    assert worst["top_k_mean_degradation_pct"] == 50.0
    assert best["top_k_mean_degradation_pct"] == 2.0


def test_k_is_clamped_to_the_patterns_available():
    picked = top_k_mean({"p5": 10.0, "p6": 20.0}, k=5, mode="worst")

    assert picked["k_effective"] == 2
    assert picked["top_k_mean_degradation_pct"] == pytest.approx(15.0)


def test_mode_vocabulary_is_checked():
    with pytest.raises(ValueError, match="mode"):
        top_k_mean({"p5": 1.0}, k=1, mode="median")


def test_empty_input_is_an_error_not_a_zero():
    with pytest.raises(ValueError):
        top_k_mean({}, k=1)


# --------------------------------------------------------------------------- #
# pattern_sweep_degradation
# --------------------------------------------------------------------------- #
def _sweep(accuracies: dict[str, float | None], **kwargs):
    return pattern_sweep_degradation(
        acc_src=0.90,
        target_patterns=list(accuracies),
        evaluate_pattern=accuracies.get,
        **kwargs,
    )


def test_each_pattern_gets_the_relative_degradation_formula():
    result = _sweep({"p5": 0.45, "p6": 0.90}, k=1, with_ci=False)

    assert result["per_pattern_degradation_pct"]["p5"] == pytest.approx(50.0)
    assert result["per_pattern_degradation_pct"]["p6"] == pytest.approx(0.0)
    assert result["per_pattern_degradation_pct"]["p5"] == pytest.approx(
        cross_domain_degradation(0.90, 0.45)
    )


def test_an_unscorable_pattern_is_skipped_not_counted_as_zero():
    """A pattern that isn't staged must not enter the mean as a 0% drop --
    that would make a half-installed benchmark look like a robust model.
    """

    result = _sweep({"p5": 0.45, "p6": None, "p7": 0.45}, k=3, with_ci=False)

    assert result["skipped_patterns"] == ["p6"]
    assert result["scored_pattern_count"] == 2
    assert result["mean_degradation_pct"] == pytest.approx(50.0)


def test_top_k_is_reported_alongside_the_plain_mean():
    result = _sweep({"p5": 0.89, "p6": 0.45, "p7": 0.88}, k=1, with_ci=False)

    # The whole point of top-k: one collapsing pattern stays visible instead
    # of being averaged away by the two that were fine.
    assert result["top_k_mean_degradation_pct"] > result["mean_degradation_pct"]
    assert result["selected_patterns"] == ["p6"]


def test_bootstrap_interval_brackets_the_estimate():
    result = _sweep(
        {"p5": 0.80, "p6": 0.60, "p7": 0.75, "p8": 0.50, "p9": 0.70},
        k=2,
        n_resamples=500,
        seed=0,
    )

    assert result["ci_low"] <= result["top_k_mean_degradation_pct"] <= result["ci_high"]
    assert math.isfinite(result["ci_low"]) and math.isfinite(result["ci_high"])


def test_a_single_scored_pattern_yields_no_interval():
    result = _sweep({"p5": 0.45, "p6": None}, k=1)

    assert "ci_low" not in result
    assert result["scored_pattern_count"] == 1


def test_a_sweep_that_scored_nothing_reports_that_rather_than_a_number():
    result = _sweep({"p5": None, "p6": None})

    assert result["scored_pattern_count"] == 0
    assert result["per_pattern_degradation_pct"] == {}
    assert "top_k_mean_degradation_pct" not in result


def test_zero_source_accuracy_is_rejected_up_front():
    with pytest.raises(ValueError, match="acc_src"):
        pattern_sweep_degradation(0.0, ["p5"], lambda _: 0.5)


def test_the_sweep_is_deterministic_under_a_fixed_seed():
    accuracies = {"p5": 0.80, "p6": 0.60, "p7": 0.75, "p8": 0.50}
    first = _sweep(accuracies, k=2, n_resamples=300, seed=7)
    second = _sweep(accuracies, k=2, n_resamples=300, seed=7)

    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]
