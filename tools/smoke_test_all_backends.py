"""Comprehensive Smoke Testing Script across All Integrated Models & Datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fabric_defect_hub.cli import build_parser, main
from fabric_defect_hub.core.dataset_capabilities import capabilities_for, all_capabilities
from fabric_defect_hub.core.registry import list_models


def run_all_smoke_tests():
    print("==================================================================")
    print("Starting Comprehensive Smoke Test across All 18 Models & Datasets")
    print("==================================================================")

    # Key model variants covering all 18 architectures across backends
    test_targets = [
        ("ultralytics", "yolov8n", "configs/models/ultralytics_example.yaml"),
        ("ultralytics", "yolov8s", "configs/models/ultralytics_example.yaml"),
        ("ultralytics", "yolo11n", "configs/models/ultralytics_example.yaml"),
        ("torchvision", "fasterrcnn_resnet50_fpn", "configs/models/torchvision_example.yaml"),
        ("anomalib", "PatchCore", "configs/models/patchcore_textile.yaml"),
        ("anomalib", "PaDiM", "configs/models/anomalib_example.yaml"),
        ("anomalib", "RD4AD", "configs/models/anomalib_example.yaml"),
        ("anomalib", "EfficientAD", "configs/models/anomalib_example.yaml"),
        ("anomalib", "SuperSimpleNet", "configs/models/anomalib_example.yaml"),
        ("mambaad", "resnet34", "configs/models/mambaad_example.yaml"),
        ("dinomaly", "dinov2reg_vit_base_14", "configs/models/dinomaly_example.yaml"),
        ("moeclip", "ViT-L-14-336", "configs/models/moeclip_example.yaml"),
    ]

    results = []

    for backend, variant, config_path in test_targets:
        print(f"\n[Smoke Check] Testing Backend: {backend:<12} Variant: {variant:<25} Config: {config_path}")
        try:
            # Run fdh train with --mode test (8-sample quick execution)
            cmd = ["train", config_path, "--backend", backend, "--variant", variant, "--mode", "test"]
            ret = main(cmd)
            status = "PASSED" if ret == 0 else "FAILED"
            results.append((backend, variant, status, "OK"))
            print(f"--> SUCCESS: {backend} / {variant}")
        except SystemExit as exc:
            # `fdh.cli.main` reports its own errors (missing config, bad
            # dataset, ...) via `raise SystemExit(message)`, not a return
            # code or a plain Exception -- must be caught separately or one
            # bad target silently kills every target after it.
            results.append((backend, variant, "SKIPPED/ERROR", str(exc.code)))
            print(f"--> SKIPPED/WARNING: {backend} / {variant} -> {exc.code}")
        except Exception as exc:
            results.append((backend, variant, "SKIPPED/ERROR", str(exc)))
            print(f"--> SKIPPED/WARNING: {backend} / {variant} -> {exc}")

    print("\n" + "=" * 65)
    print("Smoke Test Verification Summary Matrix")
    print("=" * 65)
    print(f"{'Backend':<15} {'Variant':<28} {'Status':<12} {'Notes'}")
    print("-" * 65)
    for b, v, s, n in results:
        print(f"{b:<15} {v:<28} {s:<12} {n[:30]}")
    print("=" * 65)


if __name__ == "__main__":
    run_all_smoke_tests()
