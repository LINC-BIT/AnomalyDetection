"""Config-driven experiment description for the MoECLIP backend. Mirrors
`models/dinomaly/config.py`'s shape (same section names, `from_yaml`/
`from_dict` pattern, consumed by `models/moeclip/pipeline.py`), since
MoECLIP is likewise anomaly-only and keyed by `model.name` rather than
`model.variant`.

Three divergences from `DinomalyConfig`, all following from MoECLIP being
a *zero-shot* detector trained on labelled anomalies rather than a
one-class model (see `adapter.MoECLIPAdapter`'s module docstring):

* `DataSpec` names *two* datasets, not one: an auxiliary cross-domain
  training corpus and a separate zero-shot evaluation target (the fabric
  set). Training and testing MoECLIP on the same data would make its
  headline numbers in-domain and void the transfer claim being measured.
* `DataSpec` has no `data_root` mode — upstream reads `.jsonl` metadata
  files, not a folder layout, so there is nothing to point at on disk; the
  adapter consumes `Sample` lists directly (see `data.py`).
* `train_selection` is expected to include defective samples *with masks*
  (`use_defect: true`, a mask-bearing task), the opposite of the one-class
  backends' normal-only train split.

Load with `MoECLIPConfig.from_yaml(path)` or `.from_dict(mapping)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from fabric_defect_hub.models.moeclip.presets import (
    DEFAULT_ARCH_KWARGS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAIN_KWARGS,
    resolve_model_name,
)


@dataclass
class ModelSpec:
    """Which MoECLIP backbone + MoE architecture to build.

    name: a backbone preset key; upstream supports exactly one,
        'ViT-L-14-336' (see `presets.MODEL_PRESETS`).
    The middle fields are the architecture knobs that must match between
    training and inference; they are written into the trained `Artifact`'s
    metadata so `predict` rebuilds the same model.

    `prompt_class`/`prompts` configure the text side instead — which noun
    phrase MoECLIP is asked to look for anomalies *in*. Leave them unset
    for training (each corpus category then supplies its own prompt, as
    upstream does) and set them for fabric inference, e.g.
    `prompt_class: fabric` with
    `prompts: {fabric: "plain woven cotton fabric"}`.
    """

    name: str = DEFAULT_MODEL_NAME
    img_size: int = DEFAULT_ARCH_KWARGS["img_size"]
    moe_r: int = DEFAULT_ARCH_KWARGS["moe_r"]
    moe_lora_alpha: int = DEFAULT_ARCH_KWARGS["moe_lora_alpha"]
    moe_num_experts: int = DEFAULT_ARCH_KWARGS["moe_num_experts"]
    moe_top_k: int = DEFAULT_ARCH_KWARGS["moe_top_k"]
    moe_layers: list[int] = field(default_factory=lambda: list(DEFAULT_ARCH_KWARGS["moe_layers"]))
    use_fofs: bool = DEFAULT_ARCH_KWARGS["use_fofs"]
    use_paa: bool = DEFAULT_ARCH_KWARGS["use_paa"]
    seg_proj_sharing_strategy: str = DEFAULT_ARCH_KWARGS["seg_proj_sharing_strategy"]
    image_adapt_weight: float = DEFAULT_ARCH_KWARGS["image_adapt_weight"]
    relu: bool = DEFAULT_ARCH_KWARGS["relu"]
    # Prompt policy -- not architecture (no effect on any weight shape),
    # hence kept out of `arch_kwargs()` and out of the artifact's
    # rebuild-the-model metadata. See `MoECLIPAdapter`'s class docstring.
    prompt_class: str | None = None
    prompts: dict[str, str] = field(default_factory=dict)
    # Path to a checkpoint produced by a previous run
    # (`register_trained_model`'s output). Only consulted when
    # `train.enabled` is false — it is what makes "evaluate an existing
    # checkpoint" a runnable config rather than a run that silently does
    # nothing. The architecture knobs above must match the ones the
    # checkpoint was trained with, or `load_state_dict` fails the shape check.
    weights: str | None = None

    def validate(self) -> None:
        resolve_model_name(self.name)  # raises KeyError with a helpful list
        if self.seg_proj_sharing_strategy not in ("shared", "separate"):
            raise ValueError(
                "ModelSpec: seg_proj_sharing_strategy must be 'shared' or 'separate', "
                f"got {self.seg_proj_sharing_strategy!r}."
            )

    def arch_kwargs(self) -> dict[str, Any]:
        """Only the knobs that change the model's shape — what has to match
        between training and inference, and what lands in the artifact.
        """

        return {key: getattr(self, key) for key in DEFAULT_ARCH_KWARGS}

    def adapter_kwargs(self) -> dict[str, Any]:
        """Everything `MoECLIPAdapter.__init__` takes: architecture plus
        the prompt policy.
        """

        return {**self.arch_kwargs(), "prompt_class": self.prompt_class, "prompts": dict(self.prompts)}


@dataclass
class DataSpec:
    """Where training and evaluation samples come from. Unlike the
    one-class backends there is only one mode (a registered
    `DatasetAdapter` name plus per-split selections, see module docstring)
    — but there are *two datasets*, because MoECLIP is zero-shot:

    * `dataset` (+ `dataset_root`, `train_selection`) is the **auxiliary
      training corpus**: a cross-domain object benchmark (VisA / MVTec AD
      / MVTec LOCO — see `training.ZERO_SHOT_TRAINABLE_DATASETS`), whose
      per-category names become the text prompts the MoE learns against.
      `train_selection` should select a mask-bearing, defect-containing
      split (`use_defect: true`, `task: segmentation`) — MoECLIP learns
      from labelled anomalies, not from normal data.
    * `test_dataset` (+ `test_dataset_root`, `test_selection`) is the
      **zero-shot evaluation target**: the fabric set the model has never
      seen. Omit it and evaluation falls back to `dataset`, which measures
      in-domain performance instead — useful for reproducing upstream's
      own numbers, misleading as a fabric benchmark result.
    """

    dataset: str | None = None
    dataset_root: str | None = None
    test_dataset: str | None = None
    test_dataset_root: str | None = None
    train_selection: dict[str, Any] = field(default_factory=dict)
    test_selection: dict[str, Any] = field(default_factory=dict)

    def uses_adapter(self) -> bool:
        return self.dataset is not None

    def eval_dataset(self) -> tuple[str, str]:
        """(name, root) of the split `test_selection` is read from — the
        zero-shot target when declared, else the training corpus itself.
        """

        if self.test_dataset:
            return self.test_dataset, self.test_dataset_root  # type: ignore[return-value]
        return self.dataset, self.dataset_root  # type: ignore[return-value]

    def validate(self) -> None:
        if not self.dataset:
            raise ValueError(
                "DataSpec: 'dataset' is required — the MoECLIP backend reads Sample lists "
                "from a registered DatasetAdapter (it has no on-disk folder mode)."
            )
        if not self.dataset_root:
            raise ValueError(f"DataSpec: dataset={self.dataset!r} requires 'dataset_root'.")
        if self.test_dataset and not self.test_dataset_root:
            raise ValueError(
                f"DataSpec: test_dataset={self.test_dataset!r} requires 'test_dataset_root'."
            )
        if self.test_dataset_root and not self.test_dataset:
            raise ValueError("DataSpec: 'test_dataset_root' set without a 'test_dataset'.")


@dataclass
class TrainSpec:
    """Training configuration; fields default from
    `presets.DEFAULT_TRAIN_KWARGS` (upstream's `train.py` argparse
    defaults) — see `MoECLIPAdapter.train`'s docstring for what each does.
    """

    enabled: bool = True
    epochs: int = DEFAULT_TRAIN_KWARGS["epochs"]
    batch_size: int = DEFAULT_TRAIN_KWARGS["batch_size"]
    lr: float = DEFAULT_TRAIN_KWARGS["lr"]
    balance_loss_lambda: float = DEFAULT_TRAIN_KWARGS["balance_loss_lambda"]
    etf_loss_lambda: float = DEFAULT_TRAIN_KWARGS["etf_loss_lambda"]
    lr_milestones: list[int] = field(
        default_factory=lambda: list(DEFAULT_TRAIN_KWARGS["lr_milestones"])
    )
    lr_gamma: float = DEFAULT_TRAIN_KWARGS["lr_gamma"]
    seed: int = DEFAULT_TRAIN_KWARGS["seed"]
    num_workers: int = 0
    device: str | None = None


@dataclass
class ValSpec:
    """Post-training evaluation via `evaluation.anomaly.AnomalyEvaluator`
    over `MoECLIPAdapter.predict()`'s output — same reasoning as
    `DinomalyConfig.ValSpec` (upstream's `test.py` computes its own pandas
    leaderboard, which isn't this project's metric contract).
    """

    enabled: bool = True
    output_dir: str | None = None
    max_pixels: int = 1_000_000
    max_aupro_images: int = 50
    seed: int = 0


@dataclass
class PredictSpec:
    """Inference defaults applied by `MoECLIPAdapter.predict`."""

    output_dir: str | None = None


@dataclass
class ExportSpec:
    """Post-training export. `MoECLIPAdapter.export` is not implemented
    (see its docstring) — kept so a config's shape matches every other
    backend's; `enabled: true` will raise at run time.
    """

    enabled: bool = False
    formats: list[str] = field(default_factory=list)


@dataclass
class CheckpointSpec:
    """Where the trained checkpoint is written/kept.

    work_dir: passed to `MoECLIPAdapter.train`'s `config['work_dir']` —
        defaults to a fresh temp dir if omitted.
    registry_dir: a stable location the pipeline copies the final
        checkpoint into, mirroring `DinomalyConfig.CheckpointSpec`.
    """

    work_dir: str | None = None
    name: str = "moeclip_exp"
    registry_dir: str = "artifacts/models"


@dataclass
class MoECLIPConfig:
    """Top-level, fully-declarative description of a MoECLIP experiment."""

    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    val: ValSpec = field(default_factory=ValSpec)
    predict: PredictSpec = field(default_factory=PredictSpec)
    export: ExportSpec = field(default_factory=ExportSpec)
    checkpoint: CheckpointSpec = field(default_factory=CheckpointSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoECLIPConfig":
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
                f"MoECLIPConfig: unknown top-level keys {sorted(unknown)}; "
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
    def from_yaml(cls, path: str | Path) -> "MoECLIPConfig":
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(_expand_environment_variables(data))

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()

    def resolved_train_kwargs(self) -> dict[str, Any]:
        """`train.*` + `model.*` architecture fields as one plain dict for
        `MoECLIPAdapter.train`.
        """

        kwargs: dict[str, Any] = {
            "epochs": self.train.epochs,
            "batch_size": self.train.batch_size,
            "lr": self.train.lr,
            "balance_loss_lambda": self.train.balance_loss_lambda,
            "etf_loss_lambda": self.train.etf_loss_lambda,
            "lr_milestones": list(self.train.lr_milestones),
            "lr_gamma": self.train.lr_gamma,
            "seed": self.train.seed,
            "num_workers": self.train.num_workers,
            "device": self.train.device,
        }
        kwargs.update(self.model.arch_kwargs())
        if self.checkpoint.work_dir is not None:
            kwargs["work_dir"] = self.checkpoint.work_dir
        return kwargs


def _build_section(spec_cls, raw: dict[str, Any], section: str):
    if not isinstance(raw, dict):
        raise ValueError(f"MoECLIPConfig section '{section}' must be a mapping.")
    valid = {f.name for f in fields(spec_cls) if not f.name.startswith("_")}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"MoECLIPConfig section '{section}': unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(valid)}."
        )
    return spec_cls(**raw)


def _expand_environment_variables(value):
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value
