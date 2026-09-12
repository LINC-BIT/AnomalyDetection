"""Evaluator: picks metrics based on task + label availability + model
capability, instead of assuming a single `accuracy` number.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fabric_defect_hub.core.types import Prediction, Sample


class Evaluator(ABC):
    """Base class for a task-specific metric computation (detection, segmentation, anomaly)."""

    task: str

    @abstractmethod
    def evaluate(self, samples: list[Sample], predictions: list[Prediction]) -> dict[str, float]:
        """Return a flat metric-name -> value dict, e.g. {'map50': 0.81}."""
