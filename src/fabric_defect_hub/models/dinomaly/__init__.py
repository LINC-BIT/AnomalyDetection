"""Dinomaly backend -- vendored research code, not an anomalib model.

* `adapter.py`  -- `DinomalyAdapter`, the `ModelAdapter` implementation.
  Registers itself as `@register_model("dinomaly")` on import.
* `presets.py`  -- encoder-size presets + upstream's training defaults.
* `vendor.py`   -- `sys.path` bootstrap into `components/dinomaly`.
* `config.py`   -- declarative `DinomalyConfig` loaded from YAML.
* `pipeline.py` -- `run_from_config` / `run_from_yaml`, the end-to-end runner.

Requires the `dinomaly` extra: `pip install -e ".[dinomaly]"`.
"""

from fabric_defect_hub.models.dinomaly.adapter import DinomalyAdapter
from fabric_defect_hub.models.dinomaly.config import DinomalyConfig
from fabric_defect_hub.models.dinomaly.pipeline import (
    DinomalyRunResult,
    run_from_config,
    run_from_yaml,
)
from fabric_defect_hub.models.dinomaly.presets import ENCODER_PRESETS

__all__ = [
    "DinomalyAdapter",
    "DinomalyConfig",
    "DinomalyRunResult",
    "run_from_config",
    "run_from_yaml",
    "ENCODER_PRESETS",
]
