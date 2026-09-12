"""Anomaly `Evaluator`: image/pixel AUROC/AUPRO/IAP plus optional
thresholded F1/precision/recall. Thresholded metrics require a threshold
calibrated on a validation set; same-set oracle thresholds are opt-in only.
when `Prediction.anomaly_map` files are available (see
`AnomalibAdapter.predict(..., output_dir=...)`).

Uses `scikit-learn` for the standard curve/threshold math and
`scikit-image` for AUPRO's per-region connected-component labeling —
correct, maintained implementations, in keeping with the project's
"don't reimplement what a library already gets right" principle (see
`models/anomalib/presets.py`). What's ours to own is wiring `Sample` +
`Prediction` into those computations, and the memory-safety pixel
subsampling below (a full-resolution ZJU-Leaper-scale test set can easily
exceed available RAM if every pixel is compared).
"""

from __future__ import annotations

from typing import Any

from fabric_defect_hub.core.registry import register_evaluator
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.evaluation.base import Evaluator

# Cap on pixels fed to pixel AUROC/F1 (subsampled uniformly at random) and
# on images fed to AUPRO (subsampled whole, so connected components stay
# intact) — keeps evaluation memory bounded regardless of test-set size.
DEFAULT_MAX_PIXELS = 1_000_000
DEFAULT_MAX_AUPRO_IMAGES = 50


@register_evaluator
class AnomalyEvaluator(Evaluator):
    """Image-level and pixel-level anomaly metrics with explicit calibration."""

    task = "anomaly"

    def __init__(
        self,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_aupro_images: int = DEFAULT_MAX_AUPRO_IMAGES,
        seed: int = 0,
        image_threshold: float | None = None,
        pixel_threshold: float | None = None,
        allow_oracle_threshold: bool = False,
    ):
        if max_pixels < 1 or max_aupro_images < 1:
            raise ValueError("max_pixels and max_aupro_images must be positive.")
        self.max_pixels = max_pixels
        self.max_aupro_images = max_aupro_images
        self.seed = seed
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold
        self.allow_oracle_threshold = allow_oracle_threshold

    def evaluate(self, samples: list[Sample], predictions: list[Prediction]) -> dict[str, float]:
        import numpy as np

        pred_by_id = {p.sample_id: p for p in predictions}

        y_true: list[int] = []
        y_score: list[float] = []
        pixel_pairs: list[tuple[Any, Any]] = []  # (gt_mask_2d, pred_map_2d) per sample

        for sample in samples:
            pred = pred_by_id.get(sample.id)
            if pred is None or pred.anomaly_score is None:
                continue
            y_true.append(1 if sample.annotations.is_anomalous else 0)
            y_score.append(pred.anomaly_score)

            if pred.anomaly_map is not None:
                pred_map = np.load(pred.anomaly_map)
                gt_mask = _load_ground_truth_mask(sample, pred_map.shape)
                pixel_pairs.append((gt_mask, pred_map))

        if not y_true:
            return {}

        metrics = _image_level_metrics(
            np.asarray(y_true), np.asarray(y_score, dtype=float),
            threshold=self.image_threshold,
            allow_oracle_threshold=self.allow_oracle_threshold,
        )

        if pixel_pairs:
            metrics.update(
                _pixel_level_metrics(
                    pixel_pairs, self.max_pixels, self.max_aupro_images, self.seed,
                    threshold=self.pixel_threshold,
                    allow_oracle_threshold=self.allow_oracle_threshold,
                )
            )

        return metrics


def _load_ground_truth_mask(sample: Sample, target_shape: tuple[int, ...]):
    """Binary pixel ground truth for `sample`, resized to `target_shape`
    (the predicted anomaly map's resolution, which generally differs from
    the raw image/mask resolution since models resize internally).
    """

    import numpy as np

    mask_path = sample.annotations.anomaly_mask
    if mask_path is None:
        return np.zeros(target_shape, dtype=np.uint8)

    from PIL import Image

    with Image.open(mask_path) as img:
        resized = img.convert("L").resize((target_shape[1], target_shape[0]), Image.NEAREST)
        return (np.asarray(resized) > 0).astype(np.uint8)


def _best_f1_threshold(y_true, y_score) -> float:
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    if len(set(y_true.tolist())) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(thresholds[int(f1.argmax())])


def _image_level_metrics(y_true, y_score, *, threshold: float | None = None,
                         allow_oracle_threshold: bool = False) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    metrics: dict[str, float] = {}
    metrics["image_auroc"] = (
        float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) >= 2 else float("nan")
    )

    if threshold is None and not allow_oracle_threshold:
        return metrics
    threshold = _best_f1_threshold(y_true, y_score) if threshold is None else float(threshold)
    y_pred = (y_score >= threshold).astype(int)
    metrics["image_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["image_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["image_recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["image_threshold"] = threshold
    return metrics


def _pixel_level_metrics(
    pixel_pairs: list, max_pixels: int, max_aupro_images: int, seed: int,
    *, threshold: float | None = None, allow_oracle_threshold: bool = False,
) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    flat_true = np.concatenate([m.reshape(-1) for m, _ in pixel_pairs])
    flat_score = np.concatenate([s.reshape(-1) for _, s in pixel_pairs])

    rng = np.random.default_rng(seed)
    if len(flat_true) > max_pixels:
        idx = rng.choice(len(flat_true), max_pixels, replace=False)
        flat_true, flat_score = flat_true[idx], flat_score[idx]

    metrics: dict[str, float] = {}
    metrics["pixel_auroc"] = (
        float(roc_auc_score(flat_true, flat_score)) if len(set(flat_true.tolist())) >= 2 else float("nan")
    )
    if threshold is not None or allow_oracle_threshold:
        px_threshold = _best_f1_threshold(flat_true, flat_score) if threshold is None else float(threshold)
        metrics["pixel_f1"] = float(
            f1_score(flat_true, (flat_score >= px_threshold).astype(int), zero_division=0)
        )
    metrics["pixel_aupro"] = _compute_aupro(pixel_pairs, max_aupro_images, seed)
    metrics["iap"] = _compute_iap(pixel_pairs, max_aupro_images, seed)
    return metrics


def _labeled_regions(pixel_pairs: list, max_images: int, seed: int):
    """Per-region (connected-component) pixel score arrays, plus the pooled
    negative-pixel scores, subsampled to whole images (so components stay
    intact) rather than individual pixels. Shared by AUPRO and IAP, which
    both equal-weight ground-truth defect regions instead of raw pixels.
    """

    import numpy as np
    from skimage.measure import label

    if len(pixel_pairs) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pixel_pairs), max_images, replace=False)
        pixel_pairs = [pixel_pairs[i] for i in idx]

    region_preds = []
    neg_chunks = []
    for mask, score_map in pixel_pairs:
        labeled, num_regions = label(mask.astype(np.uint8), return_num=True)
        for region_id in range(1, num_regions + 1):
            region_preds.append(score_map[labeled == region_id])
        neg_chunks.append(score_map[mask == 0])

    neg_preds = np.concatenate(neg_chunks) if neg_chunks else np.array([])
    return region_preds, neg_preds


def _compute_aupro(pixel_pairs: list, max_images: int, seed: int, num_thresholds: int = 100) -> float:
    """Area under the per-region overlap (PRO) curve, integrated over FPR.

    Each ground-truth defect region (connected component of the mask)
    contributes its own recall at each threshold; PRO is the mean recall
    across regions, plotted against the false-positive rate on normal
    pixels — this rewards detecting every defect region at least partially,
    rather than letting one large region dominate a plain pixel AUROC.
    """

    import numpy as np

    region_preds, neg_preds = _labeled_regions(pixel_pairs, max_images, seed)
    if not region_preds or neg_preds.size == 0:
        return float("nan")

    all_scores = np.concatenate([*region_preds, neg_preds])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)

    pro_scores, fprs = [], []
    for th in thresholds:
        pro_scores.append(float(np.mean([(preds >= th).sum() / preds.size for preds in region_preds])))
        fprs.append(float((neg_preds >= th).sum() / neg_preds.size))

    order = np.argsort(fprs)
    fprs_sorted = np.asarray(fprs)[order]
    pro_sorted = np.asarray(pro_scores)[order]
    return _integrate_trapezoid(np, pro_sorted, fprs_sorted)


def _compute_iap(pixel_pairs: list, max_images: int, seed: int, num_thresholds: int = 100) -> float:
    """Instance Average Precision: each ground-truth defect region
    (connected component) gets its own precision/recall integral —
    *global* pixel precision (over every positive + negative pixel in the
    subsample) as the height, integrated over that one region's own recall
    axis — then regions are averaged with equal weight.

    This is the precision-side counterpart to `_compute_aupro`'s recall-side
    region weighting: a plain pixel-level AP lets one large defect region
    dominate the recall axis simply by pixel count, which is exactly what
    IAP is meant to avoid (see the module docstring / this evaluator's
    calling docstring).
    """

    import numpy as np

    region_preds, neg_preds = _labeled_regions(pixel_pairs, max_images, seed)
    if not region_preds or neg_preds.size == 0:
        return float("nan")

    pos_preds = np.concatenate(region_preds)
    all_scores = np.concatenate([pos_preds, neg_preds])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)

    tp_counts = np.array([(pos_preds >= th).sum() for th in thresholds], dtype=float)
    fp_counts = np.array([(neg_preds >= th).sum() for th in thresholds], dtype=float)
    denom = tp_counts + fp_counts
    # No positive or negative pixel yet flagged (threshold above every
    # score): undefined precision defaults to 1.0, matching
    # `sklearn.precision_recall_curve`'s convention at recall 0.
    global_precision = np.divide(tp_counts, denom, out=np.ones_like(denom), where=denom > 0)

    # Walk thresholds strictest -> most lenient (descending) so each
    # region's own recall comes out non-decreasing -- the correct x-axis
    # order for a precision/recall integral -- then prepend the implicit
    # recall=0/precision=1 anchor for "threshold above every score, nothing
    # flagged yet" (mirroring `sklearn.precision_recall_curve`'s own
    # convention). Sorting by recall *value* instead (as `_compute_aupro`
    # does for FPR) breaks down whenever a region's pixels all sit at the
    # dataset's single highest score: recall is then 1.0 at *every* sampled
    # threshold, and a value-sort's tie-breaking hides which precision the
    # recall 0->1 jump actually happened at (it can pair the transition
    # with the wrong, much-later threshold's precision instead).
    precision_desc = global_precision[::-1]
    instance_aps = []
    for preds in region_preds:
        recall_desc = np.array([(preds >= th).sum() / preds.size for th in thresholds])[::-1]
        recall_seq = np.concatenate([[0.0], recall_desc])
        precision_seq = np.concatenate([[1.0], precision_desc])
        instance_aps.append(_integrate_trapezoid(np, precision_seq, recall_seq))
    return float(np.mean(instance_aps))


def _integrate_trapezoid(np_module, values, coordinates) -> float:
    integrate = getattr(np_module, "trapezoid", None) or getattr(np_module, "trapz")
    return float(integrate(values, coordinates))
