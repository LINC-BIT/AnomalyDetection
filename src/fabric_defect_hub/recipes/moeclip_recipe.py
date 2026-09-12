"""Config profile for vision-language zero-shot models (MoECLIP, WinCLIP).

Mirrors the MoECLIP backend's real architecture knobs (LoRA rank, experts),
which are upstream's argparse defaults. No modification of the method is
claimed here; text prompts are configured via `model.prompt_class`/`prompts`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe


@register_recipe("moeclip")
class MoECLIPRecipe(BaseModelRecipe):
    """Config profile for MoECLIP / WinCLIP."""

    @property
    def recipe_id(self) -> str:
        return "moeclip"

    @property
    def target_models(self) -> List[str]:
        return ["moeclip", "winclip"]

    @property
    def paper_reference(self) -> str:
        return "Park et al., 'MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection', CVPR 2026 (arXiv:2603.03101)."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        # Keys/values use the MoECLIP backend's *real* architecture vocabulary
        # (`models/moeclip/presets.py::DEFAULT_ARCH_KWARGS`, itself mirrored
        # from upstream's argparse defaults), so they name knobs that actually
        # exist. Invented knobs (`routing_temperature`) and the misnamed
        # `lora_rank`/`lora_alpha`/`expert_num` are gone; the text-prompt
        # policy is configured via `model.prompt_class`/`model.prompts`, not a
        # recipe hyperparameter (prompts are per-inspection data, not weights).
        return {
            "moe_r": 8,              # LoRA rank (upstream `--moe_r`)
            "moe_lora_alpha": 16,
            "moe_num_experts": 4,
            "moe_top_k": 2,          # experts routed per patch
        }
