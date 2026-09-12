"""Config profile for teacher-student distillation detectors (RD4AD, EfficientAD).

The RD4AD settings (WideResNet-50 teacher, layer1+2+3, additive anomaly map)
in anomalib's real constructor vocabulary. No modification of the method is
claimed here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe


@register_recipe("rd4ad")
class RD4ADRecipe(BaseModelRecipe):
    """Config profile for RD4AD / EfficientAD teacher-student networks."""

    @property
    def recipe_id(self) -> str:
        return "rd4ad"

    @property
    def target_models(self) -> List[str]:
        return ["rd4ad", "efficientad"]

    @property
    def paper_reference(self) -> str:
        return "Deng & Li, 'Anomaly Detection via Reverse Distillation from One-Class Embedding', CVPR 2022 (arXiv:2201.10703)."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        # Keys/values use anomalib's *real* `ReverseDistillation` constructor
        # vocabulary (introspection-verified in `models/anomalib/presets.py`),
        # so the anomalib backend consumes them directly via
        # `AnomalibConfig.resolved_model_kwargs()`. These reproduce Deng & Li's
        # RD4AD: WideResNet-50 teacher, layer1+2+3 alignment, additive anomaly
        # map. (Optimizer lr/weight_decay are owned by anomalib's own Lightning
        # module, not the model constructor, so they are not model_kwargs.)
        return {
            "backbone": "wide_resnet50_2",
            "layers": ["layer1", "layer2", "layer3"],
            "pre_trained": True,
            "anomaly_map_mode": "add",
        }
