"""Base contract for a model *config profile*.

A profile is an honest, named bundle of run settings for one method: the
hyperparameters we run it with, anchored to the method's real upstream paper via
`paper_reference`. It is NOT a novel research contribution and carries no
invented acronym — it is the "these are the settings, and here is the paper they
come from" seam that `load_model(recipe=...)` feeds into training/inference.

Deliberately settings-only: a profile may not change a model's loss,
architecture, or augmentation pipeline. This project is a *benchmark* platform,
and a profile that silently modified the model would mean the row labelled
"YOLOv8" in a results table is not stock YOLOv8, making every cross-model
comparison unsound. Method-level modifications belong in a separate line of
work, evaluated against this benchmark rather than hidden inside it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseModelRecipe(ABC):
    """Abstract base class for a model config profile."""

    @property
    @abstractmethod
    def recipe_id(self) -> str:
        """Unique identifier for this profile (e.g., 'patchcore')."""
        pass

    @property
    @abstractmethod
    def target_models(self) -> List[str]:
        """List of model names/patterns this profile applies to (e.g., ['yolov8n', 'yolov8s'])."""
        pass

    @property
    @abstractmethod
    def paper_reference(self) -> str:
        """Citation for the real upstream method this profile's settings come from."""
        pass

    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """The default settings for this profile, expressed in the backend's real vocabulary."""
        pass

    def get_recipe_summary(self) -> Dict[str, Any]:
        """A structured summary of this profile for logging and run provenance."""
        return {
            "recipe_id": self.recipe_id,
            "target_models": self.target_models,
            "paper_reference": self.paper_reference,
            "hyperparameters": self.get_default_hyperparameters(),
        }
