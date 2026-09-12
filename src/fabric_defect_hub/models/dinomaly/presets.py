"""Encoder-size presets and default hyperparameters for Dinomaly, mirrored
from upstream's `dinomaly_mvtec_sep.py` (the reference single-class
training script) rather than reimplemented from scratch.

Dinomaly is one architecture (`ViTill`: DINOv2 encoder + MLP bottleneck +
linear-attention decoder), not a model zoo, so there's no alias table like
`anomalib/presets.py` -- just the handful of knobs that vary with encoder
size, and the training defaults upstream uses.
"""

from __future__ import annotations

from typing import Any

# encoder_name -> (embed_dim, num_heads, target_layers). Values are exactly
# upstream's per-size branch in `train()` -- 'large' uses a different
# target_layers spacing than 'small'/'base' because it has twice the depth.
ENCODER_PRESETS: dict[str, dict[str, Any]] = {
    "dinov2reg_vit_small_14": {
        "embed_dim": 384,
        "num_heads": 6,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
    },
    "dinov2reg_vit_base_14": {
        "embed_dim": 768,
        "num_heads": 12,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
    },
    "dinov2reg_vit_large_14": {
        "embed_dim": 1024,
        "num_heads": 16,
        "target_layers": [4, 6, 8, 10, 12, 14, 16, 18],
    },
}

DEFAULT_ENCODER_NAME = "dinov2reg_vit_base_14"

# Fixed in upstream's ViTill construction, independent of encoder size.
FUSE_LAYER_ENCODER: list[list[int]] = [[0, 1, 2, 3], [4, 5, 6, 7]]
FUSE_LAYER_DECODER: list[list[int]] = [[0, 1, 2, 3], [4, 5, 6, 7]]
DECODER_DEPTH = 8
BOTTLENECK_HIDDEN_RATIO = 4
BOTTLENECK_DROPOUT = 0.2

# Training defaults from `dinomaly_mvtec_sep.py::train()`.
DEFAULT_TRAIN_KWARGS: dict[str, Any] = {
    "total_iters": 5000,
    "batch_size": 16,
    "image_size": 448,
    "crop_size": 392,
    "lr": 2e-3,
    "final_lr": 2e-4,
    "warmup_iters": 100,
    "weight_decay": 1e-4,
    "hm_percent_final": 0.9,
    "hm_percent_warmup_iters": 1000,
    "hm_factor": 0.1,
    "grad_clip_max_norm": 0.1,
}


def resolve_encoder_name(name: str) -> str:
    if name not in ENCODER_PRESETS:
        raise KeyError(
            f"unknown Dinomaly encoder {name!r}. Known encoders: {sorted(ENCODER_PRESETS)}"
        )
    return name


def encoder_preset(name: str) -> dict[str, Any]:
    return dict(ENCODER_PRESETS[resolve_encoder_name(name)])


def default_train_kwargs() -> dict[str, Any]:
    return dict(DEFAULT_TRAIN_KWARGS)


def list_supported_variants() -> list[str]:
    """Every encoder this backend accepts, under the name all six backends'
    preset modules share (`api.list_models` calls it on each of them).
    """

    return sorted(ENCODER_PRESETS)
