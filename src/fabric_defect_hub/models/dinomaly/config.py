"""Config-driven experiment description for the Dinomaly backend. Mirrors
`models/anomalib/config.py`'s shape (same section names, `from_yaml`/
`from_dict` pattern, consumed by `models/dinomaly/pipeline.py`) since
Dinomaly is, like Anomalib's models, one-class/anomaly-only and keyed by
`model.name` rather than `model.variant`.

One divergence from `AnomalibConfig`: `TrainSpec` *does* have fixed
hyperparameter fields (`total_iters`, `batch_size`, `image_size`,
`crop_size`, `lr`, ...) instead of a free-form `model_kwargs` dict —
unlike Anomalib's five differently-shaped models, Dinomaly is one
architecture with one training recipe (see `presets.DEFAULT_TRAIN_KWARGS`),
so a fixed field set is more honest than a dict that always has the same
keys anyway.

Load with `DinomalyConfig.from_yaml(path)` or `.from_dict(mapping)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from fabric_defect_hub.models.dinomaly.presets import DEFAULT_TRAIN_KWARGS, resolve_encoder_name


@dataclass
class ModelSpec:
    """Which Dinomaly encoder to build.

    name: an encoder preset key, e.g. 'dinov2reg_vit_base_14' (default),
        '..._small_14', '..._large_14'. See `presets.ENCODER_PRESETS`.
    weights: path to a checkpoint produced by a previous run
        (`register_trained_model`'s output). Only consulted when
        `train.enabled` is false — it is what makes "evaluate an existing
        checkpoint" a runnable config rather than a run that silently does
        nothing. The checkpoint must have been trained with this `name`'s
        encoder; a mismatch fails the `load_state_dict` shape check.
    """

    name: str = "dinov2reg_vit_base_14"
    weights: str | None = None

    def validate(self) -> None:
        resolve_encoder_name(self.name)  # raises KeyError with a helpful list


@dataclass
class DataSpec:
    """Where training/test data comes from. Same two mutually exclusive
    modes as `AnomalibConfig.DataSpec` (see its docstring):

    * `data_root`: an existing MVTec-style folder on disk, used as-is
      (`DinomalyAdapter.train`'s `config['data_root']`).
    * `dataset` (+ `dataset_root`, `train_selection`, `test_selection`): a
      registered `DatasetAdapter` name. `train_selection` should select
      normal-only samples (Dinomaly trains one-class, like every Anomalib
      model here); staged into a temporary MVTec-style folder on the fly.
    """

    data_root: str | None = None
    dataset: str | None = None
    dataset_root: str | None = None
    train_selection: dict[str, Any] = field(default_factory=dict)
    test_selection: dict[str, Any] = field(default_factory=dict)

    def uses_adapter(self) -> bool:
        return self.dataset is not None

    def validate(self) -> None:
        if self.data_root and self.dataset:
            raise ValueError("DataSpec: set either 'data_root' or 'dataset', not both.")
        if not self.data_root and not self.dataset:
            raise ValueError("DataSpec: one of 'data_root' or 'dataset' is required.")
        if self.dataset and not self.dataset_root:
            raise ValueError(f"DataSpec: dataset={self.dataset!r} requires 'dataset_root'.")


@dataclass
class TrainSpec:
    """Training configuration. Fields default from
    `presets.DEFAULT_TRAIN_KWARGS` (upstream's `dinomaly_mvtec_sep.py`
    reference recipe) -- see `DinomalyAdapter.train`'s docstring for what
    each one does. `target_layers` overrides the model preset's default
    (rarely needed -- see `presets.ENCODER_PRESETS`).
    """

    enabled: bool = True
    total_iters: int = DEFAULT_TRAIN_KWARGS["total_iters"]
    batch_size: int = DEFAULT_TRAIN_KWARGS["batch_size"]
    image_size: int = DEFAULT_TRAIN_KWARGS["image_size"]
    crop_size: int = DEFAULT_TRAIN_KWARGS["crop_size"]
    lr: float = DEFAULT_TRAIN_KWARGS["lr"]
    final_lr: float = DEFAULT_TRAIN_KWARGS["final_lr"]
    warmup_iters: int = DEFAULT_TRAIN_KWARGS["warmup_iters"]
    weight_decay: float = DEFAULT_TRAIN_KWARGS["weight_decay"]
    hm_percent_final: float = DEFAULT_TRAIN_KWARGS["hm_percent_final"]
    hm_percent_warmup_iters: int = DEFAULT_TRAIN_KWARGS["hm_percent_warmup_iters"]
    hm_factor: float = DEFAULT_TRAIN_KWARGS["hm_factor"]
    grad_clip_max_norm: float = DEFAULT_TRAIN_KWARGS["grad_clip_max_norm"]
    num_workers: int = 0
    device: str | None = None
    target_layers: list[int] | None = None


@dataclass
class ValSpec:
    """Post-training evaluation via `evaluation.anomaly.AnomalyEvaluator`
    over `DinomalyAdapter.predict()`'s output -- same reasoning as
    `AnomalibConfig.ValSpec` (Dinomaly has no native `.validate()` either).
    """

    enabled: bool = True
    output_dir: str | None = None
    max_pixels: int = 1_000_000
    max_aupro_images: int = 50
    seed: int = 0


@dataclass
class PredictSpec:
    """Inference defaults applied by `DinomalyAdapter.predict`."""

    output_dir: str | None = None


@dataclass
class ExportSpec:
    """Post-training export. `DinomalyAdapter.export` is not implemented
    (see its docstring) -- kept here only so a config's shape matches every
    other backend's; `enabled: true` will raise at run time.
    """

    enabled: bool = False
    formats: list[str] = field(default_factory=list)


@dataclass
class CheckpointSpec:
    """Where the trained checkpoint is written/kept.

    work_dir: passed to `DinomalyAdapter.train`'s `config['work_dir']` --
        defaults to a fresh temp dir if omitted (see its docstring).
    registry_dir: a stable location the pipeline copies the final
        checkpoint into, mirroring `AnomalibConfig.CheckpointSpec`.
    """

    work_dir: str | None = None
    name: str = "dinomaly_exp"
    registry_dir: str = "artifacts/models"


@dataclass
class DinomalyConfig:
    """Top-level, fully-declarative description of a Dinomaly experiment."""

    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    val: ValSpec = field(default_factory=ValSpec)
    predict: PredictSpec = field(default_factory=PredictSpec)
    export: ExportSpec = field(default_factory=ExportSpec)
    checkpoint: CheckpointSpec = field(default_factory=CheckpointSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DinomalyConfig":
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
                f"DinomalyConfig: unknown top-level keys {sorted(unknown)}; "
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
    def from_yaml(cls, path: str | Path) -> "DinomalyConfig":
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(_expand_environment_variables(data))

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()

    def resolved_train_kwargs(self) -> dict[str, Any]:
        """`train.*` fields as a plain dict for `DinomalyAdapter.train`,
        plus `target_layers` only when the config actually set one (`None`
        lets the adapter fall back to the model preset's default).
        """

        kwargs = {
            "total_iters": self.train.total_iters,
            "batch_size": self.train.batch_size,
            "image_size": self.train.image_size,
            "crop_size": self.train.crop_size,
            "lr": self.train.lr,
            "final_lr": self.train.final_lr,
            "warmup_iters": self.train.warmup_iters,
            "weight_decay": self.train.weight_decay,
            "hm_percent_final": self.train.hm_percent_final,
            "hm_percent_warmup_iters": self.train.hm_percent_warmup_iters,
            "hm_factor": self.train.hm_factor,
            "grad_clip_max_norm": self.train.grad_clip_max_norm,
            "num_workers": self.train.num_workers,
            "device": self.train.device,
        }
        if self.train.target_layers is not None:
            kwargs["target_layers"] = self.train.target_layers
        if self.checkpoint.work_dir is not None:
            kwargs["work_dir"] = self.checkpoint.work_dir
        return kwargs


def _build_section(spec_cls, raw: dict[str, Any], section: str):
    if not isinstance(raw, dict):
        raise ValueError(f"DinomalyConfig section '{section}' must be a mapping.")
    valid = {f.name for f in fields(spec_cls) if not f.name.startswith("_")}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"DinomalyConfig section '{section}': unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(valid)}."
        )
    return spec_cls(**raw)


def _expand_environment_variables(value):
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value
