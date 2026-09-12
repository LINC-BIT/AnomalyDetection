"""Shared data contracts (`types.py`), name-based registries (`registry.py`),
and JSON (de)serialization matching `schemas/` (`serialization.py`).
"""

from fabric_defect_hub.core.serialization import (
    experiment_result_from_dict,
    experiment_result_to_dict,
    load_experiment_result,
    load_predictions,
    load_samples,
    prediction_from_dict,
    prediction_to_dict,
    sample_from_dict,
    sample_to_dict,
    save_experiment_result,
    save_predictions,
    save_samples,
    validate_experiment_result,
)
from fabric_defect_hub.core.preflight import require_cached_weight, resolve_cached_weight

__all__ = [
    "sample_to_dict",
    "sample_from_dict",
    "save_samples",
    "load_samples",
    "prediction_to_dict",
    "prediction_from_dict",
    "save_predictions",
    "load_predictions",
    "experiment_result_to_dict",
    "experiment_result_from_dict",
    "save_experiment_result",
    "load_experiment_result",
    "validate_experiment_result",
    "resolve_cached_weight",
    "require_cached_weight",
]
