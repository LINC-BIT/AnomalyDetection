"""Accuracy degradation across numeric precisions (FP32 -> AMP/FP16/INT8)
for one model on one dataset — the deployment-side counterpart of
`cross_domain.cross_domain_degradation`, which compares datasets at one
precision.

The relative-drop formula is the same; what this module adds is the part
that could previously not "directly be reused" (the open item from the
2026-07-25 weekly): choosing *which* metric the drop is computed over, per
model output form, and refusing up front where the comparison is
meaningless:

* A detector's headline number is mAP@0.5; an anomaly model's is pixel
  AUROC — unless its `ModelCapabilities.prediction_fields` carries no
  `anomaly_map` (e.g. GANomaly), in which case only image AUROC exists.
  `headline_metric` encodes that choice from the model's declared
  capabilities instead of guessing from whichever keys happen to be
  present.
* A backend that declares `supports_amp=False` cannot produce the AMP/FP16
  run at all; `precision_applicable` lets a benchmark emit "not
  applicable" for that cell instead of running a silently-FP32 pass and
  reporting a fake 0% degradation. Post-training quantization (INT8 via
  `quantization/onnx_quant.py`) is a property of the exported artifact,
  not the training loop, so it is applicable whenever the backend can
  export at all.
"""

from __future__ import annotations

from typing import Mapping

from fabric_defect_hub.evaluation.cross_domain import cross_domain_degradation
from fabric_defect_hub.models.base import ModelCapabilities

# Precisions produced by rerunning the *model* (need `supports_amp`) versus
# by post-processing an *export* (need an export target to exist).
_AMP_PRECISIONS = frozenset({"amp", "fp16", "bf16"})
_EXPORT_PRECISIONS = frozenset({"int8"})

# Metric-key preference per output form. Detection lists both this
# project's normalised key (`map50`, ultralytics adapter) and torchmetrics'
# raw keys (`map_50`/`map`, evaluation/detection.py), because degradation
# may be computed over either source's dict.
_HEADLINE_PRIORITY: dict[str, tuple[str, ...]] = {
    "detection": ("map50", "map_50", "map"),
    "instance_segmentation": ("map50", "map_50", "map", "miou"),
    "segmentation": ("miou", "dice"),
    "anomaly": ("pixel_auroc", "image_auroc"),
    "anomaly_image_only": ("image_auroc",),
}


def precision_applicable(capabilities: ModelCapabilities, precision: str) -> bool:
    """Whether `precision` can be produced for this model at all."""

    key = precision.strip().lower()
    if key in ("fp32", "float32"):
        return True
    if key in _AMP_PRECISIONS:
        return capabilities.supports_amp
    if key in _EXPORT_PRECISIONS:
        return bool(capabilities.export_targets)
    raise ValueError(f"unknown precision {precision!r}")


def headline_metric(
    capabilities: ModelCapabilities, task: str, metrics: Mapping[str, float]
) -> tuple[str, float]:
    """Pick the metric the degradation is computed over: the task's
    headline number, downgraded to image level when the model's declared
    `prediction_fields` say it produces no anomaly map.
    """

    form = task
    if task == "anomaly" and "anomaly_map" not in capabilities.prediction_fields:
        form = "anomaly_image_only"
    priority = _HEADLINE_PRIORITY.get(form)
    if priority is None:
        raise ValueError(f"no headline metric defined for task {task!r}")
    for key in priority:
        if key in metrics:
            return key, float(metrics[key])
    raise KeyError(
        f"none of {priority} present in metrics {sorted(metrics)} — "
        "was the evaluation run with the matching task evaluator?"
    )


def precision_degradation(
    capabilities: ModelCapabilities,
    task: str,
    reference_metrics: Mapping[str, float],
    reduced_metrics: Mapping[str, float],
    reference: str = "fp32",
    reduced: str = "fp16",
) -> dict[str, float | str]:
    """Relative headline-metric drop from `reference` (usually FP32) to
    `reduced` precision, both evaluated on the same samples. Returns the
    metric name alongside the numbers so a report row can say *what*
    degraded, not just by how much.
    """

    if not precision_applicable(capabilities, reduced):
        raise ValueError(
            f"model does not support precision {reduced!r} "
            "(capabilities.supports_amp/export_targets); report this cell as "
            "not-applicable instead of computing a degradation"
        )
    key, ref_value = headline_metric(capabilities, task, reference_metrics)
    if key not in reduced_metrics:
        raise KeyError(f"reduced-precision metrics lack {key!r}: {sorted(reduced_metrics)}")
    reduced_value = float(reduced_metrics[key])
    return {
        "metric": key,
        "reference_precision": reference,
        "reduced_precision": reduced,
        "reference_value": ref_value,
        "reduced_value": reduced_value,
        "degradation_pct": cross_domain_degradation(ref_value, reduced_value),
    }
