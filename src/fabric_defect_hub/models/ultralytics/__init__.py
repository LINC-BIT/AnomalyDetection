"""Ultralytics backend (YOLOv8n / YOLOv8s / YOLO11n) — Phase 1.

Full training lifecycle behind a config-driven interface:

* `adapter.py`  — `UltralyticsAdapter`, the `ModelAdapter` implementation
  (pretrained loading, train, validate, predict, register/reload, export).
  Registers itself as `@register_model("ultralytics")` on import.
* `presets.py`  — variant resolution + fabric-tailored training defaults.
* `config.py`   — declarative `UltralyticsConfig` loaded from YAML.
* `pipeline.py` — `run_from_config` / `run_from_yaml`, the end-to-end runner.
"""

from fabric_defect_hub.models.ultralytics.adapter import UltralyticsAdapter
from fabric_defect_hub.models.ultralytics.config import UltralyticsConfig
from fabric_defect_hub.models.ultralytics.pipeline import (
    UltralyticsRunResult,
    run_from_config,
    run_from_yaml,
)
from fabric_defect_hub.models.ultralytics.presets import list_supported_variants

__all__ = [
    "UltralyticsAdapter",
    "UltralyticsConfig",
    "UltralyticsRunResult",
    "run_from_config",
    "run_from_yaml",
    "list_supported_variants",
]
