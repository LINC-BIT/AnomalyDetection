"""Anomalib backend (PatchCore, PaDiM, RD4AD, EfficientAD, SuperSimpleNet) — Phase 2.

Full training lifecycle behind a config-driven interface, same shape as
the Ultralytics/torchvision backends:

* `adapter.py`  — `AnomalibAdapter`, the `ModelAdapter` implementation
  (train, predict, register/reload, export). Registers itself as
  `@register_model("anomalib")` on import.
* `presets.py`  — README/paper model-name resolution + fabric-tailored
  default hyperparameters per model.
* `config.py`   — declarative `AnomalibConfig` loaded from YAML.
* `pipeline.py` — `run_from_config` / `run_from_yaml`, the end-to-end runner.
"""

from fabric_defect_hub.models.anomalib.adapter import AnomalibAdapter
from fabric_defect_hub.models.anomalib.config import AnomalibConfig
from fabric_defect_hub.models.anomalib.pipeline import (
    AnomalibRunResult,
    run_from_config,
    run_from_yaml,
)
from fabric_defect_hub.models.anomalib.presets import list_supported_variants

__all__ = [
    "AnomalibAdapter",
    "AnomalibConfig",
    "AnomalibRunResult",
    "run_from_config",
    "run_from_yaml",
    "list_supported_variants",
]
