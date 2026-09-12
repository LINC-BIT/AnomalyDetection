"""Config profile for Dinomaly (DINOv2 encoder-decoder anomaly detection).

Mirrors the Dinomaly backend's upstream-verified defaults
(`models/dinomaly/presets.py`). No modification of the method is claimed here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe


@register_recipe("dinomaly")
class DinomalyRecipe(BaseModelRecipe):
    """Config profile for Dinomaly (ViTill: DINOv2 encoder + MLP bottleneck +
    linear-attention decoder)."""

    @property
    def recipe_id(self) -> str:
        return "dinomaly"

    @property
    def target_models(self) -> List[str]:
        return ["dinomaly"]

    @property
    def paper_reference(self) -> str:
        return "Guo et al., 'Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection', CVPR 2025 (arXiv:2405.14325)."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        # Mirrors the Dinomaly backend's upstream-verified defaults
        # (`models/dinomaly/presets.py`): the ViT-Base DINOv2 encoder and the
        # published `dinomaly_mvtec_sep.py` training schedule. The bottleneck /
        # decoder depth / fuse-layer structure are fixed in ViTill construction,
        # not tunable knobs, so they are not exposed here.
        return {
            "encoder_name": "dinov2reg_vit_base_14",
            "lr": 2e-3,
            "final_lr": 2e-4,
            "weight_decay": 1e-4,
            "total_iters": 5000,
            "image_size": 448,
            "crop_size": 392,
        }
