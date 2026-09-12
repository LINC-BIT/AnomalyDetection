"""Config profile for the YOLO series (YOLOv8, YOLO11) on fabric detection.

Just the training settings we run these variants with, in Ultralytics' real
argument vocabulary. No modification of the method is claimed here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fabric_defect_hub.core.base_recipe import BaseModelRecipe
from fabric_defect_hub.core.registry import register_recipe


@register_recipe("yolov8")
class YOLOv8Recipe(BaseModelRecipe):
    """Config profile for YOLOv8 & YOLO11 fabric defect detection."""

    @property
    def recipe_id(self) -> str:
        return "yolov8"

    @property
    def target_models(self) -> List[str]:
        return ["yolov8n", "yolov8s", "yolov11n", "yolov11s"]

    @property
    def paper_reference(self) -> str:
        # The values below are Ultralytics' own documented training defaults,
        # so the honest citation is Ultralytics itself, not a method paper.
        # (This profile previously cited the SPD-Conv paper (arXiv:2208.03641)
        # as an "anchor" for a small-object architecture change it never made;
        # that hook has been removed along with the citation.)
        return "Jocher et al., Ultralytics YOLO (v8/v11), default training configuration (`ultralytics/cfg/default.yaml`)."

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        # Trainer knobs use Ultralytics' *real* `YOLO.train` argument names
        # (`box`/`cls`/`dfl` are its loss gains, not `*_loss_weight`), so
        # `UltralyticsAdapter.train` folds them in directly via
        # `recipe_trainer_overrides`.
        return {
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
        }
