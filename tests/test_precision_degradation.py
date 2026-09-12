"""Precision-degradation metric selection and applicability gating."""

from __future__ import annotations

import pytest

from fabric_defect_hub.evaluation.precision_degradation import (
    headline_metric,
    precision_applicable,
    precision_degradation,
)
from fabric_defect_hub.models.base import ModelCapabilities


def _caps(**overrides) -> ModelCapabilities:
    base = dict(
        tasks=("anomaly",),
        prediction_fields=("scores", "anomaly_map"),
        supports_amp=False,
    )
    base.update(overrides)
    return ModelCapabilities(**base)


def test_anomaly_headline_is_pixel_level_when_maps_are_declared():
    caps = _caps()
    metrics = {"image_auroc": 0.9, "pixel_auroc": 0.8}
    assert headline_metric(caps, "anomaly", metrics) == ("pixel_auroc", 0.8)


def test_image_only_models_fall_back_to_image_auroc():
    caps = _caps(prediction_fields=("scores",))  # GANomaly-shaped: no anomaly_map
    metrics = {"image_auroc": 0.9, "pixel_auroc": 0.8}  # pixel key present but meaningless
    assert headline_metric(caps, "anomaly", metrics) == ("image_auroc", 0.9)


def test_detection_accepts_both_key_spellings():
    caps = _caps(tasks=("detection",), prediction_fields=("boxes", "labels", "scores"))
    assert headline_metric(caps, "detection", {"map50": 0.7}) == ("map50", 0.7)
    assert headline_metric(caps, "detection", {"map_50": 0.7}) == ("map_50", 0.7)


def test_missing_headline_metric_is_a_loud_error():
    with pytest.raises(KeyError):
        headline_metric(_caps(), "anomaly", {"map50": 0.7})
    with pytest.raises(ValueError):
        headline_metric(_caps(), "not-a-task", {})


def test_amp_precisions_gate_on_supports_amp():
    assert not precision_applicable(_caps(supports_amp=False), "fp16")
    assert precision_applicable(_caps(supports_amp=True), "fp16")
    assert precision_applicable(_caps(supports_amp=False), "fp32")


def test_int8_gates_on_export_targets_not_amp():
    assert not precision_applicable(_caps(export_targets=()), "int8")
    assert precision_applicable(_caps(export_targets=("onnx",), supports_amp=False), "int8")


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError):
        precision_applicable(_caps(), "fp8")


def test_degradation_row_carries_metric_identity():
    caps = _caps(supports_amp=True)
    row = precision_degradation(
        caps, "anomaly",
        reference_metrics={"pixel_auroc": 0.90, "image_auroc": 0.95},
        reduced_metrics={"pixel_auroc": 0.85, "image_auroc": 0.94},
    )
    assert row["metric"] == "pixel_auroc"
    assert row["degradation_pct"] == pytest.approx((0.90 - 0.85) / 0.90 * 100)


def test_degradation_refuses_unsupported_precision():
    with pytest.raises(ValueError, match="not-applicable"):
        precision_degradation(
            _caps(supports_amp=False), "anomaly",
            reference_metrics={"pixel_auroc": 0.9},
            reduced_metrics={"pixel_auroc": 0.8},
        )
