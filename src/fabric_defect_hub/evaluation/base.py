"""Evaluator: picks metrics based on task + label availability + model
capability, instead of assuming a single `accuracy` number.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fabric_defect_hub.core.types import Prediction, Sample

# The confidence floor an *evaluation* hands a detector.
#
# A detector backend prunes its own output before anyone else sees it: the
# torchvision adapter keeps `scores >= score_threshold` (default 0.5) and
# ultralytics keeps `conf` from its own predict call (default 0.25). Both
# defaults are display choices, and both destroy an evaluation, because
# mAP sweeps confidence internally and the fixed-threshold summary applies
# its own cutoff afterwards -- neither wants a candidate set that was
# already pruned at 0.5. DETR is the extreme case: it scores no fabric box
# above 0.5, so the default turned a detector with a full set of
# low-confidence boxes into "predicts nothing" (mAP 0.00, TP 0, FP 0,
# every image score 0 -> image AUROC 0.5).
#
# 0.001 rather than 0.0 so a backend with a strict `conf > threshold`
# comparison still emits near-zero candidates, and so thousands of
# meaningless boxes do not have to cross the API boundary.
EVALUATION_CONFIDENCE = 0.001


class Evaluator(ABC):
    """Base class for a task-specific metric computation (detection, segmentation, anomaly)."""

    task: str

    @abstractmethod
    def evaluate(self, samples: list[Sample], predictions: list[Prediction]) -> dict[str, float]:
        """Return a flat metric-name -> value dict, e.g. {'map50': 0.81}."""
