"""Tests for `evaluation/upgrade_consistency.py`'s model-upgrade comparison helpers."""

import pytest

from fabric_defect_hub.evaluation.upgrade_consistency import (
    threshold_drift,
    upgrade_inconsistency_rate,
)


def test_threshold_drift_hand_computed():
    assert threshold_drift(tau_old=0.35, tau_new=0.42) == pytest.approx(0.07)


def test_threshold_drift_is_symmetric_absolute_value():
    assert threshold_drift(tau_old=0.5, tau_new=0.2) == pytest.approx(0.3)
    assert threshold_drift(tau_old=0.2, tau_new=0.5) == pytest.approx(0.3)


def test_upgrade_inconsistency_rate_hand_computed():
    old = [True, True, False, False, True]
    new = [True, False, False, True, True]
    # mismatches at index 1 and 3 -> 2/5
    assert upgrade_inconsistency_rate(old, new) == pytest.approx(0.4)


def test_upgrade_inconsistency_rate_zero_when_identical():
    decisions = [True, False, True, True, False]
    assert upgrade_inconsistency_rate(decisions, decisions) == 0.0


def test_upgrade_inconsistency_rate_empty_is_zero():
    assert upgrade_inconsistency_rate([], []) == 0.0


def test_upgrade_inconsistency_rate_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        upgrade_inconsistency_rate([True], [True, False])
