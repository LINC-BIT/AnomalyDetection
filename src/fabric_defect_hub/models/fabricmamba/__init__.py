"""Clean-room FabricMamba (Bao et al., 2025) as an Ultralytics YOLOv8
variant: custom modules in `modules.py`, the n-scale architecture in
`fabricmamba_n.yaml`, and no adapter of its own — training, prediction,
export, and evaluation all ride the existing `models/ultralytics` backend
once `presets.variant_weights` resolves the "fabricmamba" variant to
`architecture_yaml()` (which registers the custom modules as a side
effect).
"""

from __future__ import annotations

from pathlib import Path

from fabric_defect_hub.models.fabricmamba.modules import register_with_ultralytics


def architecture_yaml() -> str:
    """Absolute path to the FabricMamba architecture spec, with the custom
    modules guaranteed registered so Ultralytics can parse it."""

    register_with_ultralytics()
    return str(Path(__file__).resolve().parent / "fabricmamba_n.yaml")
