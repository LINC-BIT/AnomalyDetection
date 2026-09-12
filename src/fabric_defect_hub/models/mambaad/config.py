"""Config-driven experiment description for the MambaAD backend. Mirrors
`models/dinomaly/config.py`'s shape (same section names, `from_yaml`/
`from_dict` pattern, one-class training) since MambaAD is, like Dinomaly,
a frozen-encoder/trainable-decoder anomaly model keyed by `model.name`
rather than `model.variant` -- see `adapter.py`'s module docstring for why
this is a clean-room reimplementation, not a vendored checkout.

Load with `MambaADConfig.from_yaml(path)` or `.from_dict(mapping)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from fabric_defect_hub.models.mambaad.presets import DEFAULT_TRAIN_KWARGS, resolve_encoder_name


@dataclass
class ModelSpec:
    """Which MambaAD teacher to build.

    name: an encoder preset key, 'resnet34' (default, upstream's flagship
        recipe) or 'wide_resnet50_2'. See `presets.ENCODER_PRESETS`.
    weights: path to a checkpoint produced by a previous run
        (`register_trained_model`'s output). Only consulted when
        `train.enabled` is false — it is what makes "evaluate an existing
        checkpoint" a runnable config rather than a run that silently does
        nothing. The checkpoint must have been trained with this `name`'s
        encoder; a mismatch fails the `load_state_dict` shape check.
    """

    name: str = "resnet34"
    weights: str | None = None

    def validate(self) -> None:
        resolve_encoder_name(self.name)  # raises KeyError with a helpful list


@dataclass
class DataSpec:
    """Where training/test data comes from -- a registered `DatasetAdapter`
    name plus per-split selections. `train_selection` should select
    normal-only samples (MambaAD trains one-class, like every Anomalib
    model and Dinomaly -- see `training._ONE_CLASS_BACKENDS`).

    No `data_root` mode (unlike `AnomalibConfig`/`DinomalyConfig`): this
    reimplementation reads `Sample`s directly (see `data.py`), with no
    vendored dataset loader that expects a folder layout to fall back to.
    """

    dataset: str | None = None
    dataset_root: str | None = None
    train_selection: dict[str, Any] = field(default_factory=dict)
    test_selection: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.dataset:
            raise ValueError(
                "DataSpec: 'dataset' is required -- the MambaAD backend reads Sample lists "
                "from a registered DatasetAdapter."
            )
        if not self.dataset_root:
            raise ValueError(f"DataSpec: dataset={self.dataset!r} requires 'dataset_root'.")


@dataclass
class TrainSpec:
    """Training configuration. Fields default from
    `presets.DEFAULT_TRAIN_KWARGS`, which mirrors upstream's published
    MVTec-AD recipe -- see `MambaADAdapter.train`'s docstring for what each
    does. `total_iters` restates upstream's epoch budget in iterations
    (this project's convention across every backend); `decay_at`/
    `decay_rate` are upstream's step-decay schedule expressed as a
    fraction of it.
    """

    enabled: bool = True
    total_iters: int = DEFAULT_TRAIN_KWARGS["total_iters"]
    batch_size: int = DEFAULT_TRAIN_KWARGS["batch_size"]
    image_size: int = DEFAULT_TRAIN_KWARGS["image_size"]
    lr: float = DEFAULT_TRAIN_KWARGS["lr"]
    weight_decay: float = DEFAULT_TRAIN_KWARGS["weight_decay"]
    warmup_iters: int = DEFAULT_TRAIN_KWARGS["warmup_iters"]
    loss_lambda: float = DEFAULT_TRAIN_KWARGS["loss_lambda"]
    decay_at: float = DEFAULT_TRAIN_KWARGS["decay_at"]
    decay_rate: float = DEFAULT_TRAIN_KWARGS["decay_rate"]
    num_workers: int = 0
    device: str | None = None


@dataclass
class ValSpec:
    """Post-training evaluation via `evaluation.anomaly.AnomalyEvaluator`
    over `MambaADAdapter.predict()`'s output (no native `.validate()`
    either -- same reasoning as `DinomalyConfig.ValSpec`).
    """

    enabled: bool = True
    output_dir: str | None = None
    max_pixels: int = 1_000_000
    max_aupro_images: int = 50
    seed: int = 0


@dataclass
class PredictSpec:
    """Inference defaults applied by `MambaADAdapter.predict`."""

    output_dir: str | None = None


@dataclass
class ExportSpec:
    """Post-training export. `MambaADAdapter.export` is not implemented
    (see its docstring) -- kept so a config's shape matches every other
    backend's; `enabled: true` will raise at run time.
    """

    enabled: bool = False
    formats: list[str] = field(default_factory=list)


@dataclass
class CheckpointSpec:
    """Where the trained checkpoint is written/kept.

    work_dir: passed to `MambaADAdapter.train`'s `config['work_dir']` --
        defaults to a fresh temp dir if omitted.
    registry_dir: a stable location the pipeline copies the final
        checkpoint into, mirroring `DinomalyConfig.CheckpointSpec`.
    """

    work_dir: str | None = None
    name: str = "mambaad_exp"
    registry_dir: str = "artifacts/models"


@dataclass
class MambaADConfig:
    """Top-level, fully-declarative description of a MambaAD experiment."""

    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    val: ValSpec = field(default_factory=ValSpec)
    predict: PredictSpec = field(default_factory=PredictSpec)
    export: ExportSpec = field(default_factory=ExportSpec)
    checkpoint: CheckpointSpec = field(default_factory=CheckpointSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MambaADConfig":
        section_types = {
            "model": ModelSpec, "data": DataSpec, "train": TrainSpec, "val": ValSpec,
            "predict": PredictSpec, "export": ExportSpec, "checkpoint": CheckpointSpec,
        }
        unknown = set(data) - set(section_types)
        if unknown:
            raise ValueError(
                f"MambaADConfig: unknown top-level keys {sorted(unknown)}; "
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
    def from_yaml(cls, path: str | Path) -> "MambaADConfig":
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(_expand_environment_variables(data))

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()

    def resolved_train_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "total_iters": self.train.total_iters,
            "batch_size": self.train.batch_size,
            "image_size": self.train.image_size,
            "lr": self.train.lr,
            "weight_decay": self.train.weight_decay,
            "warmup_iters": self.train.warmup_iters,
            "loss_lambda": self.train.loss_lambda,
            "decay_at": self.train.decay_at,
            "decay_rate": self.train.decay_rate,
            "num_workers": self.train.num_workers,
            "device": self.train.device,
        }
        if self.checkpoint.work_dir is not None:
            kwargs["work_dir"] = self.checkpoint.work_dir
        return kwargs


def _build_section(spec_cls, raw: dict[str, Any], section: str):
    if not isinstance(raw, dict):
        raise ValueError(f"MambaADConfig section '{section}' must be a mapping.")
    valid = {f.name for f in fields(spec_cls) if not f.name.startswith("_")}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"MambaADConfig section '{section}': unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(valid)}."
        )
    return spec_cls(**raw)


def _expand_environment_variables(value):
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value
