"""Config-driven experiment description for the Anomalib backend.

Mirrors `models/ultralytics/config.py`/`models/torchvision/config.py`'s
overall shape (same section names, same `from_yaml`/`from_dict` pattern,
consumed by `models/anomalib/pipeline.py`) — but two sections genuinely
diverge, deliberately, rather than being forced into the other backends'
shape:

* **`TrainSpec` has no fixed hyperparameter fields.** Ultralytics/torchvision
  share one training loop, so a common `lr`/`epochs`/`batch` field set makes
  sense. Anomalib's five models (PatchCore, PaDiM, RD4AD, EfficientAD,
  SuperSimpleNet) have wildly different constructors — a coreset ratio here,
  a learning rate there, an `imagenet_dir` only EfficientAD needs (see
  `presets.MODEL_PRESETS`) — so `TrainSpec.model_kwargs` stays a free-form
  dict merged over `presets.default_model_kwargs(name)`, exactly like
  `AnomalibAdapter.train()` already expects.
* **`DataSpec` splits into `train_selection`/`test_selection`, not
  `train_selection`/`val_selection`.** Anomalib's own `Folder` datamodule
  structure is `train/good` + `test/{good,defect}` (see
  `datasets.anomalib_folder`) — there is no separate "val" split in its
  vocabulary, and renaming it to match the other two backends would misname
  what the data actually is.
* **There is no native `.validate()` on `AnomalibAdapter`** (unlike
  `YOLO.val()` / torchvision's own mAP loop) — anomalib models don't compute
  our cross-backend metrics themselves. `ValSpec` therefore configures an
  `evaluation.anomaly.AnomalyEvaluator` run over `predict()`'s output
  instead of backend-native validation; see `pipeline.py`.

Load with `AnomalibConfig.from_yaml(path)` or `.from_dict(mapping)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from fabric_defect_hub.models.anomalib.presets import (
    default_model_kwargs,
    list_supported_variants,
    resolve_model_class_name,
)


@dataclass
class ModelSpec:
    """Which anomalib model to build.

    name: a README/paper alias ('PatchCore', 'RD4AD', 'EfficientAD',
        'SuperSimpleNet', 'PaDiM' — case-insensitive) or the literal
        anomalib class name ('Patchcore', 'ReverseDistillation', ...). See
        `presets.list_supported_variants()` for the full set.
    weights: path to a checkpoint produced by a previous run
        (`register_trained_model`'s output). Only consulted when
        `train.enabled` is false — it is what makes "evaluate an existing
        checkpoint" a runnable config rather than a run that silently does
        nothing.
    allow_unsafe_checkpoint: required alongside `weights`. A Lightning
        checkpoint is a pickle and can deserialize arbitrary Python objects,
        so `AnomalibAdapter.load_trained_model` refuses a bare path without
        an explicit opt-in. Keeping that opt-in visible in the config is the
        point — do not set it for a checkpoint you did not produce.
    """

    name: str = "PatchCore"
    backend: str | None = None
    recipe: str | None = None
    loss_fn: str | None = None
    task: str | None = None
    weights: str | None = None
    allow_unsafe_checkpoint: bool = False

    def validate(self) -> None:
        resolve_model_class_name(self.name)  # raises KeyError with a helpful list
        if self.weights and not self.allow_unsafe_checkpoint:
            raise ValueError(
                "ModelSpec: loading model.weights requires "
                "model.allow_unsafe_checkpoint: true — an anomalib checkpoint is a "
                "Lightning pickle and can execute arbitrary code on load. Set it only "
                "for a checkpoint you produced yourself."
            )


@dataclass
class DataSpec:
    """Where training/test data comes from.

    Two mutually exclusive modes:

    * `datamodule_kwargs`: passed straight through to
      `anomalib.data.Folder(**datamodule_kwargs)` — use this if you already
      have an MVTec-style dataset on disk.
    * `dataset` (+ `dataset_root`, `train_selection`, `test_selection`): a
      registered `DatasetAdapter` name (e.g. 'zju-leaper'). The pipeline
      loads samples for each split via the adapter (`train_selection`
      should select normal-only samples, e.g. `use_defect: false`; models
      here train one-class), then stages them into a temporary MVTec-style
      folder on the fly (see `datasets.anomalib_folder`) — no converted
      copy is persisted.
    """

    datamodule_kwargs: dict[str, Any] | None = None
    dataset: str | None = None
    dataset_root: str | None = None
    train_selection: dict[str, Any] = field(default_factory=dict)
    test_selection: dict[str, Any] = field(default_factory=dict)
    tiling: bool = False
    tile_size: list[int] | tuple[int, int] = (256, 256)
    overlap: float = 0.25

    def uses_adapter(self) -> bool:
        return self.dataset is not None

    def validate(self) -> None:
        if self.datamodule_kwargs and self.dataset:
            raise ValueError("DataSpec: set either 'datamodule_kwargs' or 'dataset', not both.")
        if not self.datamodule_kwargs and not self.dataset:
            raise ValueError("DataSpec: one of 'datamodule_kwargs' or 'dataset' is required.")
        if self.dataset and not self.dataset_root:
            raise ValueError(f"DataSpec: dataset={self.dataset!r} requires 'dataset_root'.")


@dataclass
class TrainSpec:
    """Training configuration.

    model_kwargs: merged over the fabric-tailored preset for this model
        (`presets.default_model_kwargs`) — caller keys win. See
        `presets.MODEL_PRESETS` for what each model accepts (backbone,
        layers, coreset_sampling_ratio, imagenet_dir, supervised, ...).
    engine_kwargs: passed to `anomalib.engine.Engine(**engine_kwargs)`
        (e.g. `max_epochs`, `accelerator`, `devices`, `precision`).
        `checkpoint.default_root_dir` is merged in automatically by
        `pipeline.py` — no need to repeat it here.
    num_workers: DataLoader worker count for the staged `Folder` datamodule.
        Defaults to 0 (see `AnomalibAdapter.train`'s docstring: the staged
        directory is transient symlinks, and worker subprocesses opening it
        introduce a shutdown race with no benefit at the low-/few-shot
        sample counts this path targets).
    """

    enabled: bool = True
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    num_workers: int = 0


@dataclass
class ValSpec:
    """Post-training evaluation via `evaluation.anomaly.AnomalyEvaluator`
    over `AnomalibAdapter.predict()`'s output — see module docstring for
    why this isn't a backend-native `.validate()` call.

    output_dir: where per-sample pixel `anomaly_map` `.npy` files are
        written. Required for pixel-level AUROC/AUPRO; omit (`None`) to
        skip that disk write and only get image-level metrics.
    max_pixels / max_aupro_images / seed / image_threshold / pixel_threshold:
        passed straight to
        `AnomalyEvaluator(...)` — see its docstring for the memory-safety
        tradeoffs these control.
    """

    enabled: bool = True
    output_dir: str | None = None
    max_pixels: int = 1_000_000
    max_aupro_images: int = 50
    seed: int = 0
    image_threshold: float | None = None
    pixel_threshold: float | None = None
    allow_oracle_threshold: bool = False


@dataclass
class PredictSpec:
    """Inference defaults applied by `AnomalibAdapter.predict`.

    output_dir: same meaning as `ValSpec.output_dir`, for ad-hoc predict
    calls made outside of the validation step.
    """

    output_dir: str | None = None
    calibrate_bn: bool = False
    tta_mode: str | None = None


@dataclass
class ExportSpec:
    """Post-training export targets.

    formats: `anomalib.deploy.ExportType` values — 'torch', 'onnx',
        'openvino' (verified against installed anomalib 2.5.0; see
        `models/anomalib/adapter.py::export`). Empty = skip export.
    """

    enabled: bool = False
    formats: list[str] = field(default_factory=list)


@dataclass
class CheckpointSpec:
    """Where runs, trained models, and exports are written/kept.

    default_root_dir: anomalib `Engine`'s own working directory — it
        organises checkpoints as
        `<default_root_dir>/<ModelClass>/<name>/v{N}/weights/lightning/model.ckpt`
        (a versioned, transient path, analogous to Ultralytics'
        `runs/detect/train/weights/best.pt`).
    registry_dir: a stable location the pipeline copies the final
        checkpoint into (named `<model_class>.ckpt` by default — see
        `AnomalibAdapter.register_trained_model`), so a trained model can be
        reloaded later independent of that versioned run path.
    """

    default_root_dir: str = "results"
    name: str = "anomalib_exp"
    registry_dir: str = "artifacts/models"


@dataclass
class AnomalibConfig:
    """Top-level, fully-declarative description of an Anomalib experiment."""

    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    val: ValSpec = field(default_factory=ValSpec)
    predict: PredictSpec = field(default_factory=PredictSpec)
    export: ExportSpec = field(default_factory=ExportSpec)
    checkpoint: CheckpointSpec = field(default_factory=CheckpointSpec)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnomalibConfig":
        """Build from a plain mapping (e.g. parsed YAML). Unknown top-level
        keys are rejected so typos surface immediately rather than being
        silently ignored.
        """

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
                f"AnomalibConfig: unknown top-level keys {sorted(unknown)}; "
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
    def from_yaml(cls, path: str | Path) -> "AnomalibConfig":
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(_expand_environment_variables(data))

    # ------------------------------------------------------------------ #
    # Validation & resolution
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        self.model.validate()
        self.data.validate()

    def resolved_model_kwargs(self) -> dict[str, Any]:
        """Constructor kwargs for the anomalib model, layered:

            fabric preset  <  recipe (`model.recipe`)  <  `train.model_kwargs`

        The recipe layer lets a config profile (e.g. `patchcore`)
        set the backbone/coreset/neighbours the paper reproduces with, while
        `recipe_model_kwargs` intersects the recipe's hyperparameters with this
        model's known constructor arguments (the preset keys, which are
        introspection-verified in `presets.py`) so a recipe can never inject an
        unknown kwarg that would raise inside the anomalib model's `__init__`.
        Anything set explicitly in `train.model_kwargs` still wins.
        """

        merged = default_model_kwargs(self.model.name)
        if self.model.recipe:
            from fabric_defect_hub.recipes.apply import recipe_model_kwargs, resolve_recipe

            recipe_obj = resolve_recipe(self.model.recipe)
            merged.update(recipe_model_kwargs(recipe_obj.get_default_hyperparameters(), set(merged)))
        merged.update(self.train.model_kwargs)
        return merged

    def resolved_engine_kwargs(self) -> dict[str, Any]:
        """`train.engine_kwargs`, with `checkpoint.default_root_dir` merged in."""

        merged = {"default_root_dir": self.checkpoint.default_root_dir}
        merged.update(self.train.engine_kwargs)
        return merged


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _build_section(spec_cls, raw: dict[str, Any], section: str):
    """Instantiate a dataclass section, rejecting unknown keys with context."""

    if not isinstance(raw, dict):
        raise ValueError(f"AnomalibConfig section '{section}' must be a mapping.")
    valid = {f.name for f in fields(spec_cls) if not f.name.startswith("_")}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"AnomalibConfig section '{section}': unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(valid)}."
        )
    return spec_cls(**raw)


def _expand_environment_variables(value):
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value
