"""Config-driven experiment description for the torchvision detection
backend. Mirrors `models/ultralytics/config.py`'s design exactly (same
section names, same layering rule) so the two backends read the same way
in `configs/models/*.yaml` despite having very different training
internals underneath.

Layering of hyperparameters (lowest priority first):
    1. `presets.default_train_kwargs()` — fabric-tailored fine-tuning defaults
    2. whatever the YAML `train:` block specifies
Later layers win; see `resolved_train_kwargs()`.

Load with `TorchvisionConfig.from_yaml(path)` or `.from_dict(mapping)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from fabric_defect_hub.models.torchvision.presets import (
    default_train_kwargs,
    list_supported_variants,
    resolve_variant,
    uses_masks,
)


@dataclass
class ModelSpec:
    """Which model to build and where its initial weights come from.

    variant: one of `presets.list_supported_variants()`
        (fasterrcnn_resnet50_fpn[_v2], maskrcnn_resnet50_fpn[_v2]).
    weights: explicit checkpoint path to fine-tune from instead of the
        variant's COCO-pretrained weights (e.g. resuming a previous
        FabricDefectHub run). Takes precedence over `pretrained`.
    pretrained: if True (and `weights` is None), start from COCO-pretrained
        detection weights with the classifier head swapped for our class
        count (see `presets.build_model`). If False, start from an
        ImageNet-pretrained backbone with a random-init detection head.
    trainable_backbone_layers: how many of the 5 ResNet stages (0-5) stay
        trainable; earlier (more generic) stages are frozen. None = the
        fabric-preset default.
    """

    variant: str = "fasterrcnn_resnet50_fpn"
    weights: str | None = None
    pretrained: bool = True
    offline: bool = False
    trainable_backbone_layers: int | None = None
    min_size: int | None = None
    max_size: int | None = None

    def validate(self) -> None:
        resolve_variant(self.variant)

    def with_masks(self) -> bool:
        return uses_masks(self.variant)


@dataclass
class DataSpec:
    """Where training/validation data comes from.

    `dataset` (+ `dataset_root`, `train_selection`, `val_selection`): a
    registered `DatasetAdapter` name (e.g. 'zju-leaper'). Samples are read
    directly into `SampleDetectionDataset` — no on-disk staging, unlike the
    Ultralytics/Anomalib backends (torchvision consumes Python objects, not
    files). `*_selection` dicts are passed straight to the adapter
    constructor.

    `class_names` fixes the foreground class-id order (id 0 is always
    background) so it matches across train/val/predict and any previously
    trained checkpoint.
    """

    dataset: str = "zju-leaper"
    dataset_root: str | None = None
    train_selection: dict[str, Any] = field(default_factory=dict)
    val_selection: dict[str, Any] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=lambda: ["defect"])

    def validate(self) -> None:
        if not self.dataset_root:
            raise ValueError("DataSpec: 'dataset_root' is required.")
        if not self.class_names:
            raise ValueError("DataSpec: 'class_names' must be non-empty.")


@dataclass
class TrainSpec:
    """Training hyperparameters. Named fields with `None` fall back to the
    fabric preset (see `presets.COMMON_FABRIC_TRAIN_DEFAULTS`); anything
    else goes in `extra`.

    `amp`: mixed-precision training (`torch.autocast` + `GradScaler`); only
        actually engaged on CUDA (see `engine.run_training`), a no-op
        elsewhere.
    `resume`: continue from `checkpoint.run_dir`'s `last.pt` (optimizer/
        scheduler state, epoch count and best val-mAP included), instead of
        starting again from `weights`/`pretrained`/scratch. Silently falls
        back to a fresh run if no `last.pt` exists yet.
    """

    enabled: bool = True
    epochs: int | None = None
    batch_size: int | None = None
    optimizer: str | None = None
    lr: float | None = None
    momentum: float | None = None
    weight_decay: float | None = None
    lr_scheduler: str | None = None
    step_size: int | None = None
    gamma: float | None = None
    warmup_epochs: int | None = None
    grad_clip_norm: float | None = None
    patience: int | None = None
    num_workers: int | None = None
    device: str | None = None
    seed: int = 0
    amp: bool | None = None
    resume: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    _NON_PRESET = {"enabled", "device", "seed", "resume", "extra"}

    def as_overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name in self._NON_PRESET or f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if value is not None:
                out[f.name] = value
        return out


@dataclass
class ValSpec:
    """Validation / metric-extraction settings (native torchmetrics mAP)."""

    enabled: bool = True
    batch_size: int | None = None
    num_workers: int | None = None


@dataclass
class PredictSpec:
    """Inference defaults applied by `TorchvisionAdapter.predict`."""

    score_threshold: float = 0.5
    nms_iou_threshold: float | None = None  # None = model's own built-in NMS
    max_detections: int = 100
    device: str | None = None


@dataclass
class ExportSpec:
    """Post-training export targets.

    formats: 'exported_program' (`torch.export`, preferred for Python 3.14+),
        'torchscript' (legacy compatibility), and/or 'onnx'.
    """

    enabled: bool = False
    formats: list[str] = field(default_factory=list)
    opset: int = 17


@dataclass
class CheckpointSpec:
    """Where runs, trained models, and exports are written/kept.

    run_dir: working directory for this run — per-epoch checkpoints and a
        `history.csv` of the training log land here.
    registry_dir: a stable location the pipeline copies the final best
        checkpoint into (named `<variant>_<name>.pt`), so trained models can
        be reloaded later independent of the transient run directory.
    """

    run_dir: str = "runs/fabric_defect_hub_tv"
    name: str = "torchvision_exp"
    registry_dir: str = "artifacts/models"
    save_every_epoch: bool = False


@dataclass
class TorchvisionConfig:
    """Top-level, fully-declarative description of a torchvision detection experiment."""

    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    val: ValSpec = field(default_factory=ValSpec)
    predict: PredictSpec = field(default_factory=PredictSpec)
    export: ExportSpec = field(default_factory=ExportSpec)
    checkpoint: CheckpointSpec = field(default_factory=CheckpointSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TorchvisionConfig":
        section_types = {
            "model": ModelSpec,
            "data": DataSpec,
            "train": TrainSpec,
            "val": ValSpec,
            "predict": PredictSpec,
            "export": ExportSpec,
            "checkpoint": CheckpointSpec,
        }
        unknown = set(data) - set(section_types)
        if unknown:
            raise ValueError(
                f"TorchvisionConfig: unknown top-level keys {sorted(unknown)}; "
                f"expected some of {sorted(section_types)}."
            )
        kwargs: dict[str, Any] = {}
        for key, spec_cls in section_types.items():
            if key in data and data[key] is not None:
                kwargs[key] = _build_section(spec_cls, data[key], key)
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TorchvisionConfig":
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(_expand_environment_variables(data))

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()

    def resolved_train_kwargs(self) -> dict[str, Any]:
        merged = default_train_kwargs(self.model.variant)
        merged.update(self.train.as_overrides())
        merged.update(self.train.extra)
        return merged


def _build_section(spec_cls, raw: dict[str, Any], section: str):
    if not isinstance(raw, dict):
        raise ValueError(f"TorchvisionConfig section '{section}' must be a mapping.")
    valid = {f.name for f in fields(spec_cls) if not f.name.startswith("_")}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"TorchvisionConfig section '{section}': unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(valid)}."
        )
    return spec_cls(**raw)


def supported_variants() -> list[str]:
    return list_supported_variants()


def _expand_environment_variables(value):
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value
