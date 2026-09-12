"""MoECLIP backend -- vendored research code, not an anomalib model.

* `adapter.py`  -- `MoECLIPAdapter`, the `ModelAdapter` implementation.
  Registers itself as `@register_model("moeclip")` on import.
* `presets.py`  -- backbone preset, upstream's training defaults, and the
  text-prompt registry (MoECLIP is prompt-driven).
* `vendor.py`   -- isolated import of `components/moeclip` (its top-level
  `utils`/`dataset` module names collide with Dinomaly's).
* `data.py`     -- `Sample` -> MoECLIP batch dict, mirroring upstream's
  `BaseDataset` transforms.
* `config.py`   -- declarative `MoECLIPConfig` loaded from YAML.
* `pipeline.py` -- `run_from_config` / `run_from_yaml`, the end-to-end runner.

Requires the `moeclip` extra (`pip install -e ".[moeclip]"`) plus the
OpenCLIP ViT-L-14-336px backbone checkpoint under
`components/moeclip/model/` (see `presets.CHECKPOINT_DOWNLOAD_URL`).
"""

from fabric_defect_hub.models.moeclip.adapter import MoECLIPAdapter
from fabric_defect_hub.models.moeclip.config import MoECLIPConfig
from fabric_defect_hub.models.moeclip.pipeline import (
    MoECLIPRunResult,
    run_from_config,
    run_from_yaml,
)
from fabric_defect_hub.models.moeclip.presets import DEFAULT_MODEL_NAME, MODEL_PRESETS

__all__ = [
    "MoECLIPAdapter",
    "MoECLIPConfig",
    "MoECLIPRunResult",
    "run_from_config",
    "run_from_yaml",
    "MODEL_PRESETS",
    "DEFAULT_MODEL_NAME",
]
