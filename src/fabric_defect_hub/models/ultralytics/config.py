"""Config-driven experiment description for the Ultralytics backend.

The design goal (per project direction: configuration-driven management) is
that *what a run does is declared in a YAML file, not in command-line
flags*. A single `UltralyticsConfig` object captures the entire lifecycle —
which model variant, where the data comes from, training hyperparameters,
validation, prediction, export targets, and checkpoint/registry locations —
and is consumed by `models/ultralytics/pipeline.py`.

Layering of hyperparameters (lowest priority first):
    1. Ultralytics' own DEFAULT_CFG_DICT (implicit, inside ultralytics)
    2. fabric-tailored defaults from `presets.default_train_kwargs()`
    3. whatever the YAML `train:` block specifies
Later layers win. This keeps YAML files short — you only write the knobs
you actually want to change — while still being fully explicit about the
rest via `resolved_train_kwargs()`.

Load with `UltralyticsConfig.from_yaml(path)` or `.from_dict(mapping)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from fabric_defect_hub.models.ultralytics.presets import (
    default_train_kwargs,
    list_supported_variants,
    resolve_variant,
    variant_weights,
)


@dataclass
class ModelSpec:
    """Which model to build and where its initial weights come from.

    variant: one of `presets.list_supported_variants()` (yolov8n/v8s/yolo11n).
    weights: explicit path to a checkpoint to load instead of the variant's
        default. Use this to (a) resume/fine-tune a previously trained
        FabricDefectHub model, or (b) point at custom pretrained weights.
        When set, it takes precedence over `pretrained`.
    pretrained: if True (and `weights` is None), load the variant's
        COCO-pretrained checkpoint (transfer learning). If False, start from
        the architecture spec with random init.
    task: Ultralytics task; 'detect' for this backend.
    recipe: Optional name of the training recipe/strategy.
    """

    variant: str = "yolov8n"
    weights: str | None = None
    pretrained: bool = True
    offline: bool = False
    task: str = "detect"
    recipe: str | None = None

    def initial_weights(self) -> str:
        """Resolve the file/name Ultralytics should be initialised from."""

        if self.weights:
            return self.weights
        return variant_weights(self.variant, pretrained=self.pretrained)

    def validate(self) -> None:
        resolve_variant(self.variant)  # raises KeyError with a helpful list
        if self.task != "detect":
            raise ValueError(
                f"UltralyticsAdapter only supports task='detect', got {self.task!r}."
            )


@dataclass
class DataSpec:
    """Where training/validation data comes from.

    Two mutually exclusive modes:

    * `data_yaml`: path to an existing Ultralytics `data.yaml`. Use when you
      already have a YOLO-format dataset on disk.
    * `dataset` (+ `dataset_root`, `train_selection`, `val_selection`): a
      registered `DatasetAdapter` name (e.g. 'zju-leaper'). The pipeline
      loads samples for each split via the adapter, then stages them into a
      temporary YOLO dataset on the fly (see `datasets.yolo_bbox`) — no
      converted copy is persisted. `*_selection` dicts are passed straight
      to the adapter constructor (e.g. `{pattern, num_samples, defect_ratio,
      seed}` for ZJU-Leaper).

    `class_names` fixes the class-id order in the generated data.yaml so it
    matches across train/val/predict and any previously trained checkpoint.
    """

    data_yaml: str | None = None
    dataset: str | None = None
    dataset_root: str | None = None
    train_selection: dict[str, Any] = field(default_factory=dict)
    val_selection: dict[str, Any] = field(default_factory=dict)
    class_names: list[str] | None = None
    tiling: bool = False
    tile_size: list[int] | tuple[int, int] = (256, 256)
    overlap: float = 0.25
    require_background: bool = True

    def uses_adapter(self) -> bool:
        return self.dataset is not None

    def validate(self) -> None:
        if self.data_yaml and self.dataset:
            raise ValueError(
                "DataSpec: set either 'data_yaml' or 'dataset', not both."
            )
        if not self.data_yaml and not self.dataset:
            raise ValueError(
                "DataSpec: one of 'data_yaml' or 'dataset' is required."
            )
        if self.dataset and not self.dataset_root:
            raise ValueError(
                f"DataSpec: dataset={self.dataset!r} requires 'dataset_root'."
            )


@dataclass
class TrainSpec:
    """Training hyperparameters.

    A curated set of the knobs most worth putting in a config sit as named
    fields (with `None` meaning "fall back to the fabric preset"); anything
    else Ultralytics accepts can go in `extra` and is merged verbatim. See
    `presets.COMMON_FABRIC_TRAIN_DEFAULTS` for the fallbacks.
    """

    enabled: bool = True
    epochs: int | None = None
    imgsz: int | None = None
    batch: int | float | None = None
    patience: int | None = None
    optimizer: str | None = None
    lr0: float | None = None
    lrf: float | None = None
    weight_decay: float | None = None
    cos_lr: bool | None = None
    single_cls: bool | None = None
    freeze: int | list[int] | None = None
    device: str | int | None = None
    workers: int | None = None
    seed: int | None = None
    resume: bool = False
    augmentation: "AugmentationSpec" = field(default_factory=lambda: AugmentationSpec())
    extra: dict[str, Any] = field(default_factory=dict)

    # Fields that are pipeline-control, not Ultralytics train() kwargs.
    _NON_ULTRALYTICS = {"enabled", "resume", "augmentation", "extra"}

    def as_overrides(self) -> dict[str, Any]:
        """Explicitly-set named fields (non-None) as Ultralytics kwargs."""

        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name in self._NON_ULTRALYTICS or f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if value is not None:
                out[f.name] = value
        out.update(self.augmentation.as_overrides())
        self.augmentation.validate()
        duplicated = set(self.augmentation.as_overrides()) & set(self.extra)
        if duplicated:
            raise ValueError(
                "train.extra duplicates typed train.augmentation keys: "
                f"{sorted(duplicated)}. Set each augmentation only once."
            )
        out.update(self.extra)
        return out


@dataclass
class AugmentationSpec:
    """Typed Ultralytics image augmentations for fabric detection training."""

    hsv_h: float | None = None
    hsv_s: float | None = None
    hsv_v: float | None = None
    degrees: float | None = None
    translate: float | None = None
    scale: float | None = None
    shear: float | None = None
    perspective: float | None = None
    flipud: float | None = None
    fliplr: float | None = None
    mosaic: float | None = None
    close_mosaic: int | None = None
    mixup: float | None = None
    cutmix: float | None = None
    erasing: float | None = None

    def as_overrides(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
        }

    def validate(self) -> None:
        probabilities = {
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "translate": self.translate,
            "scale": self.scale,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "cutmix": self.cutmix,
            "erasing": self.erasing,
        }
        invalid = {name: value for name, value in probabilities.items() if value is not None and not 0 <= value <= 1}
        if invalid:
            raise ValueError(f"train.augmentation values must be in [0, 1]: {invalid}")
        if self.close_mosaic is not None and self.close_mosaic < 0:
            raise ValueError("train.augmentation.close_mosaic must be non-negative")


@dataclass
class ValSpec:
    """Validation / metric-extraction settings."""

    enabled: bool = True
    imgsz: int | None = None
    batch: int | float | None = None
    conf: float | None = None
    iou: float | None = None
    device: str | int | None = None
    split: str = "val"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {"split": self.split}
        for name in ("imgsz", "batch", "conf", "iou", "device"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out.update(self.extra)
        return out


@dataclass
class PredictSpec:
    """Inference defaults applied by `UltralyticsAdapter.predict`."""

    conf: float = 0.25
    iou: float = 0.7
    imgsz: int | None = None
    max_det: int = 300
    device: str | int | None = None
    augment: bool = False
    tta_mode: str | None = None
    tiling: bool = False
    tile_size: list[int] | tuple[int, int] = (640, 640)
    tile_overlap: float = 0.25
    calibrate_bn: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_overrides(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "augment": self.augment,
            "tiling": self.tiling,
            "tile_size": tuple(self.tile_size),
            "tile_overlap": self.tile_overlap,
        }
        for name in ("imgsz", "device"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out.update(self.extra)
        return out


@dataclass
class ExportSpec:
    """Post-training export targets, e.g. ONNX / TensorRT.

    formats: list of Ultralytics export format strings ('onnx', 'engine',
        'torchscript', 'openvino', ...). Empty = skip export.
    """

    enabled: bool = False
    formats: list[str] = field(default_factory=list)
    half: bool = False
    dynamic: bool = False
    simplify: bool = True
    opset: int | None = None
    imgsz: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_overrides(self, fmt: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "format": fmt,
            "half": self.half,
            "dynamic": self.dynamic,
            "simplify": self.simplify,
        }
        if self.opset is not None:
            out["opset"] = self.opset
        if self.imgsz is not None:
            out["imgsz"] = self.imgsz
        out.update(self.extra)
        return out


@dataclass
class CheckpointSpec:
    """Where runs, trained models, and exports are written/kept.

    project/name: Ultralytics run directory (`<project>/<name>/`), holding
        weights/best.pt, weights/last.pt, plots, and results.csv.
    registry_dir: a stable location the pipeline copies the final best.pt
        into (named `<variant>_<name>.pt`), so trained models can be loaded
        back later independent of the transient run directory.
    exist_ok: allow reusing an existing run directory (else Ultralytics
        auto-increments the name).
    """

    project: str = "runs/fabric_defect_hub"
    name: str = "yolo_exp"
    registry_dir: str = "artifacts/models"
    exist_ok: bool = False

    def as_overrides(self) -> dict[str, Any]:
        return {"project": self.project, "name": self.name, "exist_ok": self.exist_ok}


@dataclass
class UltralyticsConfig:
    """Top-level, fully-declarative description of a YOLO experiment."""

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
    def from_dict(cls, data: dict[str, Any]) -> "UltralyticsConfig":
        """Build from a plain mapping (e.g. parsed YAML). Unknown top-level
        keys are rejected so typos surface immediately rather than being
        silently ignored.
        """

        data = resolve_variant_profile(data)
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
                f"UltralyticsConfig: unknown top-level keys {sorted(unknown)}; "
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
    def from_yaml(cls, path: str | Path) -> "UltralyticsConfig":
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
        self.train.augmentation.validate()
        if len(self.predict.tile_size) != 2 or any(size <= 0 for size in self.predict.tile_size):
            raise ValueError("predict.tile_size must contain two positive integers")
        if not 0 <= self.predict.tile_overlap < 1:
            raise ValueError("predict.tile_overlap must be in [0, 1)")

    def resolved_train_kwargs(self) -> dict[str, Any]:
        """The full, explicit Ultralytics `train()` kwargs after layering
        fabric presets < config named fields < config `extra`.
        """

        merged = default_train_kwargs(self.model.variant)
        merged.update(self.train.as_overrides())
        merged.update(self.train.extra)
        merged.update(self.checkpoint.as_overrides())
        return merged


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _build_section(spec_cls, raw: dict[str, Any], section: str):
    """Instantiate a dataclass section, rejecting unknown keys with context."""

    if not isinstance(raw, dict):
        raise ValueError(f"UltralyticsConfig section '{section}' must be a mapping.")
    valid = {f.name for f in fields(spec_cls) if not f.name.startswith("_")}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"UltralyticsConfig section '{section}': unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(valid)}."
        )
    if spec_cls is TrainSpec and isinstance(raw.get("augmentation"), dict):
        raw = dict(raw)
        raw["augmentation"] = _build_section(AugmentationSpec, raw["augmentation"], "train.augmentation")
    return spec_cls(**raw)


def resolve_variant_profile(data: dict[str, Any]) -> dict[str, Any]:
    """Apply the selected YAML ``variants.<model.variant>`` profile.

    The base YAML holds shared fabric strategy. A selected profile may
    override any normal config section, and wins over the base before CLI
    overrides are applied. This keeps per-variant resource and optimization
    settings editable in one file without duplicating a complete recipe.
    """

    if not isinstance(data, dict):
        raise ValueError("UltralyticsConfig must be a mapping")
    profiles = data.get("variants")
    if profiles is None:
        return data
    if not isinstance(profiles, dict):
        raise ValueError("UltralyticsConfig 'variants' must be a mapping")

    base = {key: value for key, value in data.items() if key != "variants"}
    model = base.get("model")
    if not isinstance(model, dict) or not model.get("variant"):
        raise ValueError("UltralyticsConfig with 'variants' requires model.variant")
    variant = resolve_variant(str(model["variant"]))
    profile = profiles.get(variant)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(str(key) for key in profiles)) or "<none>"
        raise ValueError(
            f"UltralyticsConfig has no profile for model.variant={variant!r}; "
            f"available profiles: {available}"
        )
    profile_model = profile.get("model")
    if isinstance(profile_model, dict) and "variant" in profile_model:
        raise ValueError("variants.<name>.model.variant is not allowed; the profile key selects the variant")

    resolved = _deep_merge(base, profile)
    resolved_model = dict(resolved.get("model") or {})
    resolved_model["variant"] = variant
    resolved["model"] = resolved_model
    return resolved


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def supported_variants() -> list[str]:
    return list_supported_variants()


def _expand_environment_variables(value):
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value
