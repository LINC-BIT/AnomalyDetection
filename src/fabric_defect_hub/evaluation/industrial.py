"""Industrial-inspection `Evaluator`: under-detection rate (missed-defect
rate), over-detection rate (false-alarm rate), and — when fabric length is
known — alarms per unit length. Works across both the anomaly and detection
tasks by reducing each sample to a single positive/negative call, so the
same evaluator scores an Anomalib PatchCore run and a YOLO/Faster R-CNN run.

The threshold-selection idea (`recall_first_threshold`) is adapted from a
pattern seen in a sibling internal project's `TargetRecallThreshold`
metric: for quality inspection, missing a real defect (under-detection) is
usually far more costly than a false alarm (over-detection, which a human
just re-checks) — so instead of the F1-optimal threshold
`evaluation.anomaly` picks, this evaluator finds the *highest* threshold
that still keeps recall at or above a target (default 99%), i.e. the
fewest false alarms achievable without under-detecting more than allowed.
That's a genuinely different selection criterion from F1, not just a
relabeling of it, which is why it lives in its own evaluator rather than
being folded into `anomaly.py`.
"""

from __future__ import annotations


from fabric_defect_hub.core.registry import register_evaluator
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.evaluation.base import Evaluator


@register_evaluator
class IndustrialEvaluator(Evaluator):
    """Under-detection rate / over-detection rate / (optional) alarms per
    unit fabric length, at a recall-first threshold.
    """

    task = "industrial"

    def __init__(
        self,
        target_recall: float = 0.99,
        score_threshold: float | None = None,
        meters_per_sample: float | None = None,
    ):
        """`target_recall`: minimum fraction of real defects that must still
        be flagged; the threshold is chosen to maximise precision (fewest
        false alarms) subject to that floor. Ignored if `score_threshold`
        is given explicitly instead of being searched for.

        `meters_per_sample`: fixed fabric length each sample represents
        (e.g. a fixed line-scan frame length in meters). If `None`, alarms-
        per-unit-length is omitted from the result rather than guessed —
        none of this project's datasets currently carry real physical
        length metadata (see `Sample.metadata`); pass this explicitly, or
        set a per-sample `metadata['fabric_length_m']`, once a dataset does.
        """

        if not 0.0 < target_recall <= 1.0:
            raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")
        self.target_recall = target_recall
        self.score_threshold = score_threshold
        self.meters_per_sample = meters_per_sample

    def evaluate(self, samples: list[Sample], predictions: list[Prediction]) -> dict[str, float]:
        pred_by_id = {p.sample_id: p for p in predictions}

        y_true: list[int] = []
        y_score: list[float] = []
        lengths: list[float] = []
        for sample in samples:
            pred = pred_by_id.get(sample.id)
            if pred is None:
                continue
            y_true.append(1 if _is_positive_ground_truth(sample) else 0)
            y_score.append(_positive_score(pred))
            lengths.append(sample.metadata.get("fabric_length_m", self.meters_per_sample) or 0.0)

        if not y_true:
            return {}

        threshold = (
            self.score_threshold
            if self.score_threshold is not None
            else recall_first_threshold(y_true, y_score, self.target_recall)
        )

        tp = fp = fn = tn = 0
        alarms = 0
        for truth, score in zip(y_true, y_score):
            fired = score >= threshold
            alarms += int(fired)
            if truth == 1 and fired:
                tp += 1
            elif truth == 1 and not fired:
                fn += 1
            elif truth == 0 and fired:
                fp += 1
            else:
                tn += 1

        metrics: dict[str, float] = {
            "under_detection_rate": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
            "over_detection_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "chosen_threshold": float(threshold),
            "num_alarms": float(alarms),
            "num_samples": float(len(y_true)),
        }

        total_length = sum(lengths)
        if total_length > 0:
            metrics["alarms_per_unit_length"] = alarms / total_length

        return metrics


def _is_positive_ground_truth(sample: Sample) -> bool:
    if sample.annotations.is_anomalous is not None:
        return bool(sample.annotations.is_anomalous)
    return bool(sample.annotations.labels)


def _positive_score(pred: Prediction) -> float:
    if pred.anomaly_score is not None:
        return pred.anomaly_score
    if pred.scores:
        return max(pred.scores)
    return 0.0


def recall_first_threshold(y_true: list[int], y_score: list[float], target_recall: float) -> float:
    """The highest score threshold that still achieves >= `target_recall`
    recall on (`y_true`, `y_score`); among thresholds meeting that floor,
    picks the one with the best precision. Falls back to the
    maximum-recall threshold if the target is unreachable (e.g. too few
    positives, or scores that don't separate the classes).
    """

    from sklearn.metrics import precision_recall_curve

    if len(set(y_true)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]  # align with `thresholds`
    if len(thresholds) == 0:
        return 0.5

    candidates = [i for i in range(len(thresholds)) if recall[i] >= target_recall]
    if not candidates:
        best_idx = int(max(range(len(recall)), key=lambda i: recall[i]))
    else:
        best_idx = max(candidates, key=lambda i: precision[i])
    return float(thresholds[best_idx])
