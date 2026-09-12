"""Tests for `evaluation/cross_domain.py`'s cross-domain accuracy degradation."""

import pytest

from fabric_defect_hub.evaluation.cross_domain import (
    cross_domain_degradation,
    cross_domain_degradation_ci,
)


def test_cross_domain_degradation_hand_computed():
    # (0.90 - 0.72) / 0.90 * 100 = 20.0
    assert cross_domain_degradation(0.90, 0.72) == pytest.approx(20.0)


def test_cross_domain_degradation_negative_means_target_scores_higher():
    assert cross_domain_degradation(0.50, 0.75) == pytest.approx(-50.0)


def test_cross_domain_degradation_rejects_zero_source_accuracy():
    with pytest.raises(ValueError, match="nonzero"):
        cross_domain_degradation(0.0, 0.5)


def test_cross_domain_degradation_ci_point_estimate_matches_hand_computed():
    # src: 9/10 correct = 0.9; tgt: 6/10 correct = 0.6 -> (0.9-0.6)/0.9*100
    src_correct = [True] * 9 + [False]
    tgt_correct = [True] * 6 + [False] * 4
    result = cross_domain_degradation_ci(src_correct, tgt_correct, n_resamples=500, seed=0)
    assert result["delta_acc_pct"] == pytest.approx((0.9 - 0.6) / 0.9 * 100.0)
    assert result["ci_low"] <= result["delta_acc_pct"] <= result["ci_high"]


def test_cross_domain_degradation_ci_zero_when_domains_identical():
    correct = [True, True, False, True, False, True]
    result = cross_domain_degradation_ci(correct, correct, n_resamples=200, seed=0)
    assert result["delta_acc_pct"] == pytest.approx(0.0)
