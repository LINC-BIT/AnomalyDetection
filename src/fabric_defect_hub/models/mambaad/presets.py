"""Backbone presets and upstream's training defaults for MambaAD, mirrored
from the paper's published MVTec-AD recipe (`configs/mambaad/mambaad_mvtec.py`
in the upstream repo) rather than reimplemented from scratch -- see
`adapter.py`'s module docstring for why this is a clean-room reimplementation
rather than a vendored `components/mambaad` (as originally attempted; see
`components/README.md`'s git history) in the first place.

Like Dinomaly, MambaAD is one architecture (frozen CNN teacher + fused
embedding + Mamba decoder), not a model zoo -- the "backbone" choice is
which `timm` teacher to freeze, not a family of unrelated models.
"""

from __future__ import annotations

from typing import Any

# encoder_name -> (timm model name, `features_only` out_indices, and the
# resulting channel widths at those indices -- resolved once here instead
# of instantiating a throwaway model just to read `.feature_info`).
# `dinov2reg_vit_base_14`-style presets.py: values are what upstream's own
# configs use, not invented. `resnet34` is the flagship recipe (the
# repo's README reports its MVTec-AD numbers, 98.6 mAUROC, using it); the
# base config's own default is `wide_resnet50_2` -- kept as an alternate.
ENCODER_PRESETS: dict[str, dict[str, Any]] = {
    "resnet34": {
        "timm_name": "resnet34",
        "out_indices": [1, 2, 3],
        "channels": [64, 128, 256],
    },
    "wide_resnet50_2": {
        "timm_name": "wide_resnet50_2",
        "out_indices": [1, 2, 3],
        "channels": [256, 512, 1024],
    },
}

DEFAULT_ENCODER_NAME = "resnet34"

# Fixed in upstream's `MambaUPNet` construction (the decoder architecture
# itself), independent of encoder choice -- the decoder's own stage
# widths, not the teacher's. `dims_decoder[0]` is what the fused
# embedding is projected to (see `network.MultiScaleFusion`'s `out_channels`).
DIMS_DECODER: list[int] = [512, 256, 128, 64]
DEPTHS_DECODER: list[int] = [3, 4, 6, 3]
D_STATE = 16
DROP_PATH_RATE = 0.2

# Upstream's published recipe: Hilbert scan, 8 directions -- the paper's
# ablations report this as its best-performing configuration.
DEFAULT_SCAN_TYPE = "hilbert"
DEFAULT_NUM_DIRECTION = 8

# Gaussian smoothing applied to the summed anomaly map at inference --
# `gaussian_sigma=4` in `trainer/mambaad_trainer.py::test`'s
# `cal_anomaly_map` call. Note the kernel *size* follows scipy's
# `gaussian_filter` geometry, not an arbitrary window (see
# `MambaADAdapter._gaussian_blur`).
ANOMALY_MAP_GAUSSIAN_SIGMA = 4.0

# MambaAD is a *multi-class unified* model -- that is the paper's whole
# claim ("Multi-class Unsupervised Anomaly Detection"): ONE model is
# trained across every category of a dataset at once
# (`cfg.data.cls_names = []` selects all 15 MVTec categories), and the
# published 98.6 mAUROC is that single model's average, not a per-category
# model's. This is exactly the capability dimension anomalib's one-model-
# per-category zoo lacks, so it is worth preserving rather than quietly
# collapsing to per-category training: point it at the `fabric-train`
# composite (which unions every fabric source, see
# `datasets/fabric_train.py`) or at ZJU-Leaper with all 19 patterns, and
# it is being used the way the paper intends.
#
# Training defaults from `configs/mambaad/mambaad_mvtec.py`, with one
# substitution: upstream counts an `epoch_full` of 1000 epochs over one
# dataset, while every other backend here (and this project's shot-mode
# overrides in training.py) is driven by an iteration budget, so
# `total_iters` stands in for it. 5000 is this project's own default, not
# upstream's -- a real reproduction run needs far more; see the
# `total_iters` note in configs/models/mambaad_example.yaml.
DEFAULT_TRAIN_KWARGS: dict[str, Any] = {
    "total_iters": 5000,
    "batch_size": 16,
    "image_size": 256,
    "lr": 0.005,
    "weight_decay": 1e-4,
    "warmup_iters": 0,
    "loss_lambda": 5.0,  # upstream's L2Loss `lam`
    # Upstream's scheduler: step decay to lr/10 at 80% of training
    # (`decay_epochs=int(epoch_full*0.8)`, `decay_rate=0.1`), with
    # `lr_min = lr/100`. Expressed here as fractions of `total_iters`.
    "decay_at": 0.8,
    "decay_rate": 0.1,
}


def resolve_encoder_name(name: str) -> str:
    if name not in ENCODER_PRESETS:
        raise KeyError(f"unknown MambaAD encoder {name!r}. Known encoders: {sorted(ENCODER_PRESETS)}")
    return name


def encoder_preset(name: str) -> dict[str, Any]:
    return dict(ENCODER_PRESETS[resolve_encoder_name(name)])


def default_train_kwargs() -> dict[str, Any]:
    return dict(DEFAULT_TRAIN_KWARGS)


def list_supported_variants() -> list[str]:
    """Every teacher encoder this backend accepts, under the name all six
    backends' preset modules share (`api.list_models` calls it on each of
    them).
    """

    return sorted(ENCODER_PRESETS)
