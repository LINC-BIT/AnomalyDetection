"""Task-specific `Evaluator` implementations: `detection.py`, `segmentation.py`,
`anomaly.py`, plus an `industrial.py` for line-level metrics (under/over
detection rate, alarms per unit length). See `base.py` for the shared contract.

Every evaluator here operates purely on `Sample` + `Prediction` — none of
them know or care which `ModelAdapter`/backend produced the predictions,
so the same evaluator scores Ultralytics, torchvision, and Anomalib runs
alike.
"""

from fabric_defect_hub.evaluation.anomaly import AnomalyEvaluator
from fabric_defect_hub.evaluation.detection import DetectionEvaluator
from fabric_defect_hub.evaluation.industrial import IndustrialEvaluator
from fabric_defect_hub.evaluation.segmentation import SegmentationEvaluator

__all__ = [
    "AnomalyEvaluator",
    "DetectionEvaluator",
    "SegmentationEvaluator",
    "IndustrialEvaluator",
    "evaluator_for_task",
    "ground_truth_task",
]


def ground_truth_task(model_task: str) -> str:
    """The ground-truth shape a model's task is scored against.

    A `ModelAdapter` may declare `instance_segmentation` (Mask R-CNN), but a
    `DatasetAdapter`'s `task` only ever needs to be one of the three shapes
    in `core.types.Task` to decide which annotations to attach, and
    `SegmentationEvaluator` scores both the same way -- over a unioned binary
    mask. So instance segmentation folds into the segmentation bucket.

    Lives beside `evaluator_for_task` because it answers the same kind of
    question and has the same single correct answer. The Benchmark tab used
    to carry its own copy, which put a scoring rule in the UI.
    """

    return "segmentation" if model_task == "instance_segmentation" else model_task


def evaluator_for_task(task: str):
    """The `Evaluator` for one of `core.types.Task`'s three ground-truth
    shapes ("anomaly"/"detection"/"segmentation") -- the one canonical
    mapping every caller that needs "which evaluator scores this task"
    (the CLI's `evaluate` command, the web Benchmark tab) should use,
    instead of hand-rolling its own copy.
    """

    if task == "anomaly":
        return AnomalyEvaluator()
    if task == "detection":
        return DetectionEvaluator()
    if task == "segmentation":
        return SegmentationEvaluator()
    raise ValueError(f"no evaluator registered for task {task!r}")
