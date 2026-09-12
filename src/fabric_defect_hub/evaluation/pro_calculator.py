"""Per-Region Overlap (PRO-Score) Calculator for Industrial Anomaly Segmentation.

Computes the region-wise overlap metric across different decision thresholds,
providing a strict evaluation metric for microscopic and connected defect regions.
"""

from __future__ import annotations

from typing import List

import numpy as np


def compute_pro_score(
    masks_gt: np.ndarray,
    anomaly_maps: np.ndarray,
    num_thresholds: int = 200,
    max_fpr: float = 0.3,
) -> float:
    """Computes the Per-Region Overlap (PRO) score up to a maximum False Positive Rate (FPR).

    Args:
        masks_gt: Ground truth binary masks [N, H, W] (0 or 1)
        anomaly_maps: Predicted anomaly heatmaps [N, H, W] in [0, 1]
        num_thresholds: Number of threshold evaluation steps
        max_fpr: Integration limit for False Positive Rate (standard CVPR baseline = 0.3)

    Returns:
        float: Normalized PRO AUC score in range [0, 1]
    """
    try:
        from scipy.ndimage import label
    except ImportError:
        # Fallback approximation if scipy is not installed
        return float(np.mean(anomaly_maps[masks_gt > 0]) if np.any(masks_gt > 0) else 0.0)

    if not np.any(masks_gt > 0):
        return 0.0

    min_val, max_val = anomaly_maps.min(), anomaly_maps.max()
    thresholds = np.linspace(min_val, max_val, num_thresholds)

    # Label connected components in GT
    labeled_masks = []
    num_regions_total = 0
    for mask in masks_gt:
        labeled, num_features = label(mask)
        labeled_masks.append((labeled, num_features))
        num_regions_total += num_features

    if num_regions_total == 0:
        return 0.0

    pros: List[float] = []
    fprs: List[float] = []

    for th in thresholds:
        binary_pred = anomaly_maps >= th

        # Calculate FPR
        neg_mask = masks_gt == 0
        fp = np.logical_and(binary_pred, neg_mask).sum()
        tn = neg_mask.sum()
        fpr = fp / (tn + 1e-8)
        fprs.append(fpr)

        # Calculate Region Overlap
        region_overlaps = []
        for i, (labeled, num_features) in enumerate(labeled_masks):
            pred_i = binary_pred[i]
            for region_idx in range(1, num_features + 1):
                region_gt = labeled == region_idx
                overlap = np.logical_and(pred_i, region_gt).sum() / region_gt.sum()
                region_overlaps.append(overlap)

        pros.append(np.mean(region_overlaps) if region_overlaps else 0.0)

    # Sort by FPR for integration
    fprs_arr = np.array(fprs)
    pros_arr = np.array(pros)
    sort_idx = np.argsort(fprs_arr)
    fprs_arr = fprs_arr[sort_idx]
    pros_arr = pros_arr[sort_idx]

    # Crop to max_fpr
    valid_idx = fprs_arr <= max_fpr
    if not np.any(valid_idx):
        return 0.0

    fprs_cropped = fprs_arr[valid_idx]
    pros_cropped = pros_arr[valid_idx]

    # Integrate area under curve (NumPy 2.0+ compatibility)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    pro_auc = trapz_fn(pros_cropped, fprs_cropped) / max_fpr
    return float(np.clip(pro_auc, 0.0, 1.0))
