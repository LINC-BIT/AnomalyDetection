"""Config profile for memory-bank anomaly detectors (PatchCore, PaDiM, SuperSimpleNet).

The PatchCore settings (WideResNet-50 features, 10% coreset, k=9) that
reproduce the paper's MVTec-AD result, in anomalib's real constructor
vocabulary. No modification of the method is claimed here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe


@register_recipe("patchcore")
class PatchCoreRecipe(BaseModelRecipe):
    """Config profile for PatchCore / PaDiM / SuperSimpleNet."""

    @property
    def recipe_id(self) -> str:
        return "patchcore"

    @property
    def target_models(self) -> List[str]:
        return ["patchcore", "padim", "supersimplenet"]

    @property
    def paper_reference(self) -> str:
        return "Roth et al., 'Towards Total Recall in Industrial Anomaly Detection', CVPR 2022."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        # Keys/values use anomalib's *real* `Patchcore` constructor vocabulary
        # (introspection-verified in `models/anomalib/presets.py`), so the
        # anomalib backend can consume them directly via
        # `AnomalibConfig.resolved_model_kwargs()`. These are the settings that
        # reproduce the paper's headline MVTec-AD result (mean image AUROC
        # ~99.1%): WideResNet-50 features from layer2+layer3, 10% coreset,
        # k=9 nearest neighbours.
        return {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "pre_trained": True,
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9,
        }
