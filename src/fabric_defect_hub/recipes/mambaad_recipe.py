"""Config profile for MambaAD (state-space anomaly detection).

Mirrors the MambaAD backend's upstream-verified defaults (encoder + training
schedule). No modification of the method is claimed here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe


@register_recipe("mambaad")
class MambaADRecipe(BaseModelRecipe):
    """Config profile for MambaAD."""

    @property
    def recipe_id(self) -> str:
        return "mambaad"

    @property
    def target_models(self) -> List[str]:
        return ["mambaad"]

    @property
    def paper_reference(self) -> str:
        return "He et al., 'MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection', NeurIPS 2024 (arXiv:2404.06564)."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        # Keys/values mirror the MambaAD backend's *real*, upstream-verified
        # recipe (`models/mambaad/presets.py`): `resnet34` is upstream's
        # flagship teacher (the README's 98.6 mAUROC config), and the training
        # schedule is upstream's published `DEFAULT_TRAIN_KWARGS`. The Mamba
        # internals (`d_state`, scan type/directions) are fixed in the
        # clean-room decoder construction, not tunable knobs, so they are not
        # exposed here. NOTE: the previous `lr=0.0001` contradicted upstream's
        # published `lr=0.005`; corrected.
        return {
            "encoder_name": "resnet34",
            "lr": 0.005,
            "weight_decay": 0.0001,
            "loss_lambda": 5.0,
            "total_iters": 5000,
        }
