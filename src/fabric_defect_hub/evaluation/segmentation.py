"""Segmentation `Evaluator`: mIoU, Dice, pixel-level F1 — computed uniformly
from `Sample`+`Prediction`, independent of which `ModelAdapter` produced the
masks.

Pure numpy, no extra dependency: unlike detection mAP or anomaly AUPRO,
binary-mask overlap metrics are a handful of array ops and not worth
depending on a metrics library for.

Mask representation is deliberately permissive, since different producers
hand masks back differently:
  * a file path string (e.g. `Sample.annotations.masks = [path]`, ZJU-Leaper's
    convention — see `datasets/zju_leaper.py`) -> loaded via PIL, binarised
    `> 0`;
  * a nested list/array (e.g. `Prediction.masks` from
    `models/torchvision/adapter.py::predict`, one entry per detected
    instance) -> stacked and unioned into one binary mask.
Multiple instance masks are unioned (logical OR) into a single binary
"defect / not defect" mask per image before comparison — this project's
segmentation datasets are single-class (see `Sample.task == "segmentation"`
in `ZJULeaperDataset`), so per-instance identity doesn't change the metric.
"""

from __future__ import annotations

from typing import Any

from fabric_defect_hub.core.registry import register_evaluator
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.evaluation.base import Evaluator


@register_evaluator
class SegmentationEvaluator(Evaluator):
    """Binary mIoU / Dice / pixel-F1 over samples with meaningful masks.

    Both-empty normal-image pairs are excluded rather than counted as a
    perfect prediction, and are reported through ``num_skipped_empty`` when
    at least one meaningful pair remains.
    """

    task = "segmentation"

    def evaluate(self, samples: list[Sample], predictions: list[Prediction]) -> dict[str, float]:
        pred_by_id = {p.sample_id: p for p in predictions}

        ious, dices, f1s = [], [], []
        skipped_empty = 0
        for sample in samples:
            pred = pred_by_id.get(sample.id)
            if pred is None:
                continue

            gt_mask = _load_binary_mask(sample.annotations.masks or sample.annotations.anomaly_mask)
            pred_mask = _load_binary_mask(pred.masks)
            if gt_mask is None or pred_mask is None:
                continue
            pred_mask = _resize_like(pred_mask, gt_mask.shape)
            if not gt_mask.any() and not pred_mask.any():
                skipped_empty += 1
                continue

            ious.append(_iou(gt_mask, pred_mask))
            dices.append(_dice(gt_mask, pred_mask))
            f1s.append(_pixel_f1(gt_mask, pred_mask))

        if not ious:
            return {}

        metrics = {
            "miou": sum(ious) / len(ious),
            "dice": sum(dices) / len(dices),
            "pixel_f1": sum(f1s) / len(f1s),
            "num_evaluated": float(len(ious)),
        }
        if skipped_empty:
            metrics["num_skipped_empty"] = float(skipped_empty)
        return metrics


def _load_binary_mask(raw: Any):
    """Normalise a mask/mask-list (file path, nested list, or None) to a 2D
    numpy bool array, unioning multiple instance masks if given a list of
    them. Returns None if `raw` doesn't resolve to anything usable.
    """

    import numpy as np

    if raw is None:
        return None

    if isinstance(raw, str):
        return _load_mask_file(raw)

    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], str):
        # A list of mask file paths (e.g. `Sample.annotations.masks`).
        masks = [_load_mask_file(p) for p in raw]
        masks = [m for m in masks if m is not None]
        return _union(masks) if masks else None

    # Otherwise assume a nested list / array: either a single HxW mask, or a
    # stack of per-instance NxHxW masks.
    arr = np.asarray(raw)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        return arr > 0
    if arr.ndim == 3:
        return _union([arr[i] > 0 for i in range(arr.shape[0])])
    return None


def _load_mask_file(path: str):
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        return np.array(img.convert("L")) > 0


def _union(masks: list):
    import numpy as np

    stacked = np.stack(masks, axis=0)
    return stacked.any(axis=0)


def _resize_like(mask, target_shape: tuple[int, ...]):
    import numpy as np

    if mask.shape == tuple(target_shape):
        return mask
    from PIL import Image

    resized = Image.fromarray(mask.astype("uint8") * 255).resize(
        (target_shape[1], target_shape[0]), Image.NEAREST
    )
    return np.array(resized) > 0


def _iou(gt, pred) -> float:
    intersection = (gt & pred).sum()
    union = (gt | pred).sum()
    return float(intersection / union) if union > 0 else 1.0  # both empty = perfect agreement


def _dice(gt, pred) -> float:
    intersection = (gt & pred).sum()
    denom = gt.sum() + pred.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0


def _pixel_f1(gt, pred) -> float:
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else (1.0 if tp == 0 and fp == 0 and fn == 0 else 0.0)
