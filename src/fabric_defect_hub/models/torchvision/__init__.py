"""torchvision detection backend (Faster R-CNN, Mask R-CNN) — replaces the
originally planned `mmdetection` backend (see `presets.py` module docstring
for why: `mmcv` has no macOS arm64 wheel and doesn't build on Python 3.14,
confirmed by a real install attempt; torchvision is pure PyTorch and
installs everywhere). Cascade R-CNN / DETR / DINO are out of scope here —
Faster R-CNN and Mask R-CNN cover the "classic two-stage detector"
comparison point on their own.

Full training lifecycle behind a config-driven interface, same shape as
the Ultralytics backend:

* `adapter.py`  — `TorchvisionAdapter`, the `ModelAdapter` implementation
  (pretrained/scratch loading, train, validate, predict, register/reload,
  export). Registers itself as `@register_model("torchvision")` on import.
* `engine.py`   — the actual train/eval loop (torchvision ships no Trainer).
* `dataset.py`  — `Sample` -> `torch.utils.data.Dataset` bridge (in-memory,
  no on-disk staging — torchvision consumes Python objects directly).
* `presets.py`  — variant resolution + fabric-tailored fine-tuning defaults.
* `config.py`   — declarative `TorchvisionConfig` loaded from YAML.
* `pipeline.py` — `run_from_config` / `run_from_yaml`, the end-to-end runner.
"""

from fabric_defect_hub.models.torchvision.config import TorchvisionConfig
from fabric_defect_hub.models.torchvision.presets import list_supported_variants

__all__ = [
    "TorchvisionAdapter",
    "TorchvisionConfig",
    "TorchvisionRunResult",
    "run_from_config",
    "run_from_yaml",
    "list_supported_variants",
]


def __getattr__(name: str):
    if name == "TorchvisionAdapter":
        from fabric_defect_hub.models.torchvision.adapter import TorchvisionAdapter

        return TorchvisionAdapter
    if name in {"TorchvisionRunResult", "run_from_config", "run_from_yaml"}:
        from fabric_defect_hub.models.torchvision import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
