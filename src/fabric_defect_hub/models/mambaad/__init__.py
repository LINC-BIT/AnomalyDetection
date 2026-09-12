"""MambaAD backend -- a clean-room reimplementation, not a vendored
`components/mambaad` checkout (see `adapter.py`'s module docstring for
why: the upstream repo is an ADer plugin, not a standalone project, and
its selective scan needs a CUDA-only pip package).

* `adapter.py`  -- `MambaADAdapter`, the `ModelAdapter` implementation.
  Registers itself as `@register_model("mambaad")` on import.
* `presets.py`  -- teacher-encoder presets + upstream's published recipe.
* `scan.py`     -- the five multi-directional scan orders (pure NumPy).
* `ssm.py`      -- the selective-scan recurrence + `SS2D` (pure PyTorch,
  no `mamba_ssm` dependency).
* `decoder.py`  -- `HSSBlock`/`LSSModule`/`MambaUPNet`, the Mamba decoder.
* `network.py`  -- `MultiScaleFusion` + `MambaADNet`, the teacher-fusion-
  decoder wiring.
* `data.py`     -- `Sample` -> image tensor bridge (one-class, no masks).
* `config.py`   -- declarative `MambaADConfig` loaded from YAML.
* `pipeline.py` -- `run_from_config` / `run_from_yaml`, the end-to-end runner.

No extra install step beyond this project's own dependencies (`timm`,
`einops` -- both already pulled in by the `dinomaly`/`moeclip` extras):
unlike Dinomaly/MoECLIP there is no `components/mambaad` submodule to
`git submodule update`, and no CUDA-only package to install.
"""

from fabric_defect_hub.models.mambaad.adapter import MambaADAdapter
from fabric_defect_hub.models.mambaad.config import MambaADConfig
from fabric_defect_hub.models.mambaad.pipeline import (
    MambaADRunResult,
    run_from_config,
    run_from_yaml,
)
from fabric_defect_hub.models.mambaad.presets import DEFAULT_ENCODER_NAME, ENCODER_PRESETS

__all__ = [
    "MambaADAdapter",
    "MambaADConfig",
    "MambaADRunResult",
    "run_from_config",
    "run_from_yaml",
    "ENCODER_PRESETS",
    "DEFAULT_ENCODER_NAME",
]
