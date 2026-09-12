"""Model-variant registry and fabric-tailored training defaults for the
Ultralytics backend (YOLOv8n, YOLOv8s, YOLO11n).

Mirrors `models/anomalib/presets.py`: we do not reimplement YOLO — the
`ultralytics` package ships the architectures and the training loop. What
lives here is (1) resolution of friendly variant names to the checkpoint /
architecture files Ultralytics expects, and (2) a curated set of default
hyperparameters tuned for fabric-defect detection rather than COCO.

Why the fabric defaults differ from Ultralytics' COCO defaults:

* Fabric defects are small, low-contrast, and scattered on a repeating
  texture. Aggressive COCO-style augmentation (mosaic, mixup, large scale
  jitter) tends to destroy or hide those tiny defects, so mosaic/mixup are
  toned down and closed early.
* Fabric has no meaningful "up" orientation and mirror symmetry is label-
  preserving, so full flips (and modest rotation) are safe free augmentation.
* Hue/saturation shifts are kept small because some defects (stains, color
  bleed) are themselves colour cues we don't want to wash out.
* ZJU-Leaper (see `datasets.zju_leaper.ZJULeaperDataset`) is single-class
  (`defect`), so `single_cls` defaults to True.

Every key below is a real Ultralytics training argument (validated against
`ultralytics.cfg.DEFAULT_CFG_DICT`, ultralytics 8.4.x). Anything here can be
overridden per-run from config; see `models/ultralytics/config.py`.
"""

from __future__ import annotations

from typing import Any

# Friendly variant name -> (pretrained checkpoint file, from-scratch arch file).
# Ultralytics resolves these names to downloads/bundled yamls on first use.
#
# "fabricmamba" is this project's clean-room YOLOv8n derivative (see
# `models/fabricmamba/`), not an Ultralytics release. Both entries point at
# the same architecture yaml because no pretrained checkpoint exists for a
# custom architecture — "pretrained" loading degrades to from-scratch, and
# `variant_weights` resolves the real file path (plus registers the custom
# modules) at call time.
FABRICMAMBA_VARIANT = "fabricmamba"

MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "yolov8n": {"checkpoint": "yolov8n.pt", "architecture": "yolov8n.yaml"},
    "yolov8s": {"checkpoint": "yolov8s.pt", "architecture": "yolov8s.yaml"},
    "yolo11n": {"checkpoint": "yolo11n.pt", "architecture": "yolo11n.yaml"},
    FABRICMAMBA_VARIANT: {
        "checkpoint": "fabricmamba_n.yaml",
        "architecture": "fabricmamba_n.yaml",
    },
}

# Common aliases people actually type -> canonical variant key above.
VARIANT_ALIASES: dict[str, str] = {
    "yolov8n": "yolov8n",
    "yolo8n": "yolov8n",
    "v8n": "yolov8n",
    "yolov8s": "yolov8s",
    "yolo8s": "yolov8s",
    "v8s": "yolov8s",
    "yolo11n": "yolo11n",
    "yolov11n": "yolo11n",
    "v11n": "yolo11n",
    "11n": "yolo11n",
    "fabricmamba": FABRICMAMBA_VARIANT,
    "fabric-mamba": FABRICMAMBA_VARIANT,
    "fabricmamba_n": FABRICMAMBA_VARIANT,
}

# Fabric-tailored default training hyperparameters shared by all variants.
# Sized for the low-/few-shot ZJU-Leaper regimes this project targets.
COMMON_FABRIC_TRAIN_DEFAULTS: dict[str, Any] = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "patience": 30,  # early-stopping window
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "cos_lr": True,
    "single_cls": True,  # ZJU-Leaper: one "defect" class
    # --- augmentation, retuned for small texture defects ---
    "hsv_h": 0.010,
    "hsv_s": 0.4,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.3,
    "fliplr": 0.5,
    "flipud": 0.5,
    "mosaic": 0.5,
    "close_mosaic": 15,
    "mixup": 0.0,
    # --- bookkeeping ---
    "plots": True,
    "save": True,
    "val": True,
    "verbose": True,
}

# Per-variant overrides layered on top of the common defaults. The 's' model
# has more capacity, so it can afford a slightly larger default batch.
VARIANT_TRAIN_OVERRIDES: dict[str, dict[str, Any]] = {
    "yolov8n": {},
    "yolov8s": {"batch": 12},
    "yolo11n": {},
    # FabricMamba's published recipe (paper section 5.1): SGD, lr0 0.01,
    # weight decay 5e-4 — already this file's common defaults — and mosaic
    # switched off for the last ten epochs. The paper's batch of 64 assumes
    # a 24GB card; the common batch=16 stands until tuned on real hardware.
    FABRICMAMBA_VARIANT: {"close_mosaic": 10},
}


def resolve_variant(name: str) -> str:
    """Map a friendly name/alias/checkpoint filename to a canonical variant key."""

    key = name.strip().lower()
    # Strip a trailing extension so 'yolov8n.pt' / 'yolo11n.yaml' resolve too.
    for ext in (".pt", ".yaml", ".yml"):
        if key.endswith(ext):
            key = key[: -len(ext)]
            break
    if key in MODEL_VARIANTS:
        return key
    if key in VARIANT_ALIASES:
        return VARIANT_ALIASES[key]
    known = sorted(MODEL_VARIANTS)
    raise KeyError(f"unknown YOLO variant {name!r}. Known variants: {known}")


def variant_weights(name: str, pretrained: bool = True) -> str:
    """Return the file Ultralytics should load for `name`.

    `pretrained=True`  -> the COCO-pretrained '.pt' checkpoint (transfer
    learning; recommended for small fabric datasets).
    `pretrained=False` -> the '.yaml' architecture spec (train from random
    init).
    """

    variant = resolve_variant(name)
    if variant == FABRICMAMBA_VARIANT:
        # Custom architecture: resolve the packaged yaml's real path and
        # register the custom modules Ultralytics needs to parse it. There
        # is no pretrained checkpoint, so both branches load from scratch.
        from fabric_defect_hub.models.fabricmamba import architecture_yaml

        return architecture_yaml()
    spec = MODEL_VARIANTS[variant]
    return spec["checkpoint"] if pretrained else spec["architecture"]


def default_train_kwargs(name: str) -> dict[str, Any]:
    """Fabric-tailored default training kwargs for `name` (copy, safe to mutate)."""

    variant = resolve_variant(name)
    merged = dict(COMMON_FABRIC_TRAIN_DEFAULTS)
    merged.update(VARIANT_TRAIN_OVERRIDES.get(variant, {}))
    return merged


def list_supported_variants() -> list[str]:
    return sorted(MODEL_VARIANTS)
