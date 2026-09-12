"""Detection `Evaluator`: mAP@0.5, mAP@0.5:0.95, per-threshold Precision/
Recall/F1, computed uniformly from `Sample`+`Prediction` regardless of which
`ModelAdapter` produced the predictions.

Uses `torchmetrics.detection.MeanAveragePrecision` (COCO-style matching via
the `pycocotools` backend) rather than re-deriving mAP by hand — same
"don't reimplement a correct existing implementation" principle as
`evaluation/anomaly.py`.

Why this file matters beyond just "detection metrics": before this, the
torchvision backend computed its own mAP internally
(`models/torchvision/engine.py::evaluate`) directly against torchmetrics,
bypassing the project's unified `Evaluator` abstraction entirely — a
different model backend's predictions had no shared code path for scoring.
This evaluator is backend-agnostic: it only reads the unified `Sample` /
`Prediction` contract, so the exact same code scores Ultralytics and
torchvision detections alike (`models/torchvision/engine.py` keeps its own
internal mAP call too, since it needs per-epoch validation *during*
training before a final `Prediction` list exists — this evaluator is for
the after-the-fact, backend-agnostic scoring pass that produces the
`ExperimentResult.metrics` used for cross-backend comparison).
"""

from __future__ import annotations

from fabric_defect_hub.core.registry import register_evaluator
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.evaluation.base import Evaluator


@register_evaluator
class DetectionEvaluator(Evaluator):
    """COCO-style mAP + a fixed-threshold Precision/Recall/F1 summary."""

    task = "detection"

    def __init__(
        self,
        class_names: list[str] | None = None,
        pr_score_threshold: float = 0.5,
        small_defect_max_px: float = 10.0,
    ):
        """`class_names` fixes the label set (and iteration order) used to
        build the box_format id mapping; if omitted it's inferred from the
        union of ground-truth and predicted labels seen in `evaluate()`.
        `pr_score_threshold` is the confidence cutoff used only for the
        fixed-threshold precision/recall/F1 summary (mAP itself sweeps all
        thresholds internally and ignores this).
        `small_defect_max_px` is the shorter-side cutoff separating `recall_
        small` from `recall_normal`; 10 px is the fabric convention here
        (broken warp / skipped pick defects sit around that scale).
        """

        self.class_names = class_names
        self.pr_score_threshold = pr_score_threshold
        self.small_defect_max_px = small_defect_max_px

    def evaluate(self, samples: list[Sample], predictions: list[Prediction]) -> dict[str, float]:
        from torchmetrics.detection import MeanAveragePrecision

        pred_by_id = {p.sample_id: p for p in predictions}
        pairs = [(s, pred_by_id[s.id]) for s in samples if s.id in pred_by_id]
        if not pairs:
            return {}

        class_map = self._build_class_map(pairs)
        preds_t, targets_t = _to_torchmetrics_format(pairs, class_map)

        metric = MeanAveragePrecision(iou_type="bbox")
        metric.update(preds_t, targets_t)
        result = metric.compute()

        metrics: dict[str, float] = {}
        for key, value in result.items():
            if hasattr(value, "numel") and value.numel() == 1:
                v = float(value)
                if v >= 0:  # torchmetrics uses -1 as "not computable" sentinel
                    metrics[key] = v

        metrics.update(_precision_recall_f1(pairs, class_map, self.pr_score_threshold))
        # Size-bucketed recall ships with every detection score rather than
        # waiting for a quantization comparison to ask for it: `recall_small`
        # on its own already answers "does this model see tiny defects at
        # all", and having it on both the fp32 and the quantized row is what
        # makes `quantization_recall_decay` a subtraction the caller can do
        # afterwards instead of a second evaluation pass.
        metrics.update(
            recall_by_size(
                samples, predictions,
                small_max_px=self.small_defect_max_px,
                score_threshold=self.pr_score_threshold,
            )
        )
        return metrics

    def _build_class_map(self, pairs) -> dict[str, int]:
        if self.class_names is not None:
            return {name: idx for idx, name in enumerate(self.class_names)}
        gt_labels = {label for sample, _ in pairs for label in (sample.annotations.labels or [])}
        if len(gt_labels) == 1:
            # Every dataset this project scores is single-class ("defect" for
            # ZJU-Leaper), but a checkpoint's own detection head can carry a
            # different class name than the ground truth's (e.g. a published
            # yolov8n.pt trained/exported with class name "item" instead of
            # "defect"). Building the class map from the union of gt+pred
            # labels would then produce two distinct category ids that never
            # match each other, silently zeroing every mAP/mAR column even
            # though the boxes overlap fine (`_precision_recall_f1` is
            # class-agnostic and doesn't hit this). There's only one real
            # class here, so collapse onto it and ignore the predictor's own
            # label text entirely -- see `_to_torchmetrics_format`'s
            # `ignore_pred_label`.
            return {next(iter(gt_labels)): 0}
        labels: set[str] = set()
        for sample, pred in pairs:
            labels.update(sample.annotations.labels or [])
            labels.update(pred.labels or [])
        return {name: idx for idx, name in enumerate(sorted(labels))}


def _to_torchmetrics_format(pairs, class_map: dict[str, int]):
    import torch

    ignore_pred_label = len(class_map) == 1
    preds_t = []
    targets_t = []
    for sample, pred in pairs:
        gt_boxes = sample.annotations.boxes or []
        gt_labels = sample.annotations.labels or []
        gt_ids = [class_map[label] for label in gt_labels if label in class_map]
        kept_gt_boxes = [b for b, label in zip(gt_boxes, gt_labels) if label in class_map]
        targets_t.append(
            {
                "boxes": torch.as_tensor(kept_gt_boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.as_tensor(gt_ids, dtype=torch.int64),
            }
        )

        pred_boxes = pred.boxes or []
        pred_labels = pred.labels or []
        pred_scores = pred.scores or [1.0] * len(pred_boxes)
        if ignore_pred_label:
            pred_ids = [0] * len(pred_boxes)
            kept_pred_boxes = list(pred_boxes)
            kept_scores = list(pred_scores)
        else:
            pred_ids = [class_map[label] for label in pred_labels if label in class_map]
            kept_pred_boxes = [b for b, label in zip(pred_boxes, pred_labels) if label in class_map]
            kept_scores = [s for s, label in zip(pred_scores, pred_labels) if label in class_map]
        preds_t.append(
            {
                "boxes": torch.as_tensor(kept_pred_boxes, dtype=torch.float32).reshape(-1, 4),
                "scores": torch.as_tensor(kept_scores, dtype=torch.float32),
                "labels": torch.as_tensor(pred_ids, dtype=torch.int64),
            }
        )
    return preds_t, targets_t


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _precision_recall_f1(pairs, class_map: dict[str, int], score_threshold: float, iou_threshold: float = 0.5) -> dict[str, float]:
    """Simple greedy IoU-matching at a single fixed score/IoU threshold —
    the "would this actually fire an alarm at my chosen confidence cutoff"
    number that a sweep-based mAP doesn't directly answer.
    """

    ignore_pred_label = len(class_map) == 1
    tp = fp = fn = 0
    for sample, pred in pairs:
        gt_boxes = [
            b for b, label in zip(sample.annotations.boxes or [], sample.annotations.labels or [])
            if label in class_map
        ]
        kept = [
            (b, s) for b, label, s in zip(
                pred.boxes or [], pred.labels or [], pred.scores or [1.0] * len(pred.boxes or [])
            )
            if (ignore_pred_label or label in class_map) and s >= score_threshold
        ]
        kept.sort(key=lambda x: x[1], reverse=True)

        matched_gt = set()
        for box, _score in kept:
            best_iou, best_idx = 0.0, -1
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = _box_iou(box, gt_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= iou_threshold:
                matched_gt.add(best_idx)
                tp += 1
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "f1_at_threshold": f1,
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
    }


def _shorter_side(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return min(abs(x2 - x1), abs(y2 - y1))


def recall_by_size(
    samples: list[Sample],
    predictions: list[Prediction],
    small_max_px: float = 10.0,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Recall computed separately for 'small' ground-truth boxes (shorter
    side < `small_max_px`) and 'normal' ones (>= `small_max_px`), so a
    model/precision change's effect on tiny-defect detection can be read
    off against its effect on ordinarily-sized defects instead of being
    averaged away into one aggregate `recall_at_threshold` number -- a
    quantized model that trades small-defect recall for a faster/smaller
    model would otherwise look like a wash.
    """

    pred_by_id = {p.sample_id: p for p in predictions}
    pairs = [(s, pred_by_id[s.id]) for s in samples if s.id in pred_by_id]

    counts = {"small": {"tp": 0, "fn": 0}, "normal": {"tp": 0, "fn": 0}}
    for sample, pred in pairs:
        gt_boxes = sample.annotations.boxes or []
        kept_preds = [
            box for box, score in zip(
                pred.boxes or [], pred.scores or [1.0] * len(pred.boxes or [])
            )
            if score >= score_threshold
        ]

        matched_pred: set[int] = set()
        for gt_box in gt_boxes:
            bucket = "small" if _shorter_side(gt_box) < small_max_px else "normal"
            best_iou, best_idx = 0.0, -1
            for i, pred_box in enumerate(kept_preds):
                if i in matched_pred:
                    continue
                iou = _box_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= iou_threshold:
                matched_pred.add(best_idx)
                counts[bucket]["tp"] += 1
            else:
                counts[bucket]["fn"] += 1

    def _recall(bucket: dict[str, int]) -> float:
        total = bucket["tp"] + bucket["fn"]
        return bucket["tp"] / total if total > 0 else 0.0

    return {
        "recall_small": _recall(counts["small"]),
        "recall_normal": _recall(counts["normal"]),
    }


def quantization_recall_decay(
    recall_fp32_small: float,
    recall_quant_small: float,
    recall_fp32_normal: float,
    recall_quant_normal: float,
) -> dict[str, float]:
    """DeltaRecall_small vs DeltaRecall_normal: how much more (or less) a
    precision change (e.g. INT8 quantization) hurts small-defect recall
    compared to normal-sized-defect recall -- the two numbers a single
    aggregate recall delta would hide from each other, since small defects
    typically degrade faster under quantization than normal-sized ones.
    """

    return {
        "delta_recall_small": recall_fp32_small - recall_quant_small,
        "delta_recall_normal": recall_fp32_normal - recall_quant_normal,
    }
