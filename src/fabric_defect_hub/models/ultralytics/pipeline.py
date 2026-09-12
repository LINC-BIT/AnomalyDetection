"""Config-driven end-to-end runner for the Ultralytics backend.

This is the payoff of configuration-driven management: give it an
`UltralyticsConfig` (typically
`UltralyticsConfig.from_yaml("configs/models/ultralytics_*.yaml")`) and it
executes the whole declared lifecycle — resolve data, train, validate,
register the trained model, export — with no command-line flags. Each stage is
gated by its `enabled` flag in the config, so the same file can describe "just
train", "train + export to ONNX", "validate an existing checkpoint", etc.

The lifecycle order lives in `core.pipeline.BasePipeline`; this file is only
what is specific to Ultralytics. Unlike the anomaly backends, Ultralytics has
a native `validate()` that produces metrics itself, so `evaluate` calls that
rather than predict-then-score.
"""

from __future__ import annotations

from typing import Any

from fabric_defect_hub.core.pipeline import BasePipeline, RunResult
from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact
from fabric_defect_hub.models.ultralytics.adapter import UltralyticsAdapter
from fabric_defect_hub.models.ultralytics.config import DataSpec, UltralyticsConfig

# The unified result type, under this backend's historical name.
UltralyticsRunResult = RunResult


def _load_split_samples(data: DataSpec, selection: dict[str, Any]) -> list[Sample]:
    """Resolve one split's `Sample` list via the configured DatasetAdapter."""

    dataset = load_dataset(data.dataset, root=data.dataset_root, **selection)
    return dataset.load_samples()


def _build_data_kwargs(config: UltralyticsConfig) -> dict[str, Any]:
    """Turn the DataSpec into the `data=`/`samples=` kwargs train/val expect."""

    data = config.data
    if not data.uses_adapter():
        return {"data": data.data_yaml}

    train_samples = _load_split_samples(data, data.train_selection)
    # Fall back to reusing the train selection for val only if none is given.
    if data.val_selection:
        val_samples = _load_split_samples(data, data.val_selection)
    else:
        val_samples = train_samples
    train_background = sum(not sample.annotations.boxes for sample in train_samples)
    if data.require_background and train_background == 0:
        raise ValueError(
            "YOLO detection training selected zero normal/background images. "
            "Set train_selection.defect_ratio below 1.0 (the textile configs use 0.5), "
            "or explicitly set data.require_background: false."
        )
    return {
        "samples": {"train": train_samples, "val": val_samples},
        "class_names": data.class_names,
        "sample_summary": {
            "train_total": len(train_samples),
            "train_background": train_background,
            "train_defect": len(train_samples) - train_background,
            "val_total": len(val_samples),
            "val_background": sum(not sample.annotations.boxes for sample in val_samples),
        },
        "tiling": data.tiling,
        "tile_size": tuple(data.tile_size),
        "tile_overlap": data.overlap,
    }


class UltralyticsPipeline(BasePipeline):
    """`UltralyticsConfig` -> a full train / validate / export run."""

    def build_adapter(self) -> UltralyticsAdapter:
        return UltralyticsAdapter(name=self.config.model.variant)

    def prepare(self) -> None:
        # Shared by both training and validation, so resolved once.
        self.data_kwargs = _build_data_kwargs(self.config)

    def build_train_config(self) -> dict[str, Any]:
        config = self.config
        train_config: dict[str, Any] = {}
        train_config.update(config.resolved_train_kwargs())
        train_config.update(self.data_kwargs)
        train_config["pretrained"] = config.model.pretrained
        train_config["offline"] = config.model.offline
        if config.model.weights:
            train_config["weights"] = config.model.weights
        if config.train.resume:
            train_config["resume"] = True
        return train_config

    def load_existing_artifact(self, adapter: UltralyticsAdapter) -> Artifact | None:
        # No training: load the specified checkpoint so val/export can run.
        if self.config.model.weights:
            return adapter.load_trained_model(self.config.model.weights)
        return None

    def evaluate(self, adapter: UltralyticsAdapter, artifact: Artifact) -> dict[str, float]:
        # val() only consumes samples/data + val kwargs, not class training keys.
        val_config: dict[str, Any] = dict(self.config.val.as_overrides())
        val_config.update(self.data_kwargs)
        return adapter.validate(artifact, val_config)

    def export_config(self, fmt: str) -> dict[str, Any]:
        return self.config.export.as_overrides(fmt)


def run_from_config(config: UltralyticsConfig) -> RunResult:
    """Execute the lifecycle declared in `config`."""

    return UltralyticsPipeline(config).run()


def run_from_yaml(path: str) -> RunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(UltralyticsConfig.from_yaml(path))
