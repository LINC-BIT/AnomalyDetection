"""Per-model config profiles: named, paper-anchored bundles of run settings."""

from fabric_defect_hub.recipes.apply import (
    apply_recipe_to_training,
    attach_recipe,
    recipe_model_kwargs,
    recipe_trainer_overrides,
    resolve_recipe,
)
from fabric_defect_hub.recipes.dinomaly_recipe import DinomalyRecipe
from fabric_defect_hub.recipes.mambaad_recipe import MambaADRecipe
from fabric_defect_hub.recipes.moeclip_recipe import MoECLIPRecipe
from fabric_defect_hub.recipes.patchcore_recipe import PatchCoreRecipe
from fabric_defect_hub.recipes.rd4ad_recipe import RD4ADRecipe
from fabric_defect_hub.recipes.yolov8_recipe import YOLOv8Recipe

__all__ = [
    "YOLOv8Recipe",
    "PatchCoreRecipe",
    "RD4ADRecipe",
    "MambaADRecipe",
    "MoECLIPRecipe",
    "DinomalyRecipe",
    "attach_recipe",
    "resolve_recipe",
    "apply_recipe_to_training",
    "recipe_trainer_overrides",
    "recipe_model_kwargs",
]
