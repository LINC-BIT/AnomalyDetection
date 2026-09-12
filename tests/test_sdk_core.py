"""Unit tests for FabricDefectHub (UTAD-Framework) Modular SDK and config profiles."""

from __future__ import annotations

import numpy as np

import fabric_defect_hub  # noqa: F401  -- importing the package registers the config profiles
from fabric_defect_hub.core.registry import get_recipe, list_recipes
from fabric_defect_hub.evaluation.lmei_profiler import calculate_lmei
from fabric_defect_hub.evaluation.pro_calculator import compute_pro_score
from fabric_defect_hub.reporting.latex_generator import generate_latex_table


def test_recipe_registry():
    """Test that all config profiles are registered and retrievable."""
    recipe_ids = list_recipes()
    assert "yolov8" in recipe_ids
    assert "patchcore" in recipe_ids
    assert "rd4ad" in recipe_ids
    assert "mambaad" in recipe_ids
    assert "moeclip" in recipe_ids
    assert "dinomaly" in recipe_ids

    # Check model name resolution
    recipe_yolo = get_recipe("yolov8n")
    assert recipe_yolo.recipe_id == "yolov8"
    assert recipe_yolo.paper_reference  # every profile names where its settings come from


def test_pro_calculator():
    """Test Per-Region Overlap (PRO) score calculation."""
    masks_gt = np.zeros((2, 64, 64), dtype=np.uint8)
    masks_gt[0, 10:20, 10:20] = 1

    anomaly_maps = np.zeros((2, 64, 64), dtype=np.float32)
    anomaly_maps[0, 11:19, 11:19] = 0.9

    score = compute_pro_score(masks_gt, anomaly_maps)
    assert 0.0 <= score <= 1.0


def test_lmei_profiler():
    """Test Latency-Memory Efficiency Index calculation."""
    score = calculate_lmei(fps=120.0, vram_mb=512.0, flops_g=8.5, params_m=3.2)
    assert score > 0.0


def test_latex_table_generator():
    """Test IEEE/CVPR LaTeX table code generator."""
    results = [
        {"model": "YOLOv8n", "recipe": "yolov8", "I-AUROC": 0.982, "PRO-Score": 0.941, "FPS": 145.0, "LMEI": 1.25},
        {"model": "PatchCore", "recipe": "patchcore", "I-AUROC": 0.965, "PRO-Score": 0.912, "FPS": 85.0, "LMEI": 0.88},
    ]
    latex_code = generate_latex_table(results)
    assert "\\begin{table*}" in latex_code
    assert "\\textbf{0.9820}" in latex_code
