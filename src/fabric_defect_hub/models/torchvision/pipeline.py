"""Config-driven end-to-end runner for the torchvision detection backend.

Give it a `TorchvisionConfig` (typically
`TorchvisionConfig.from_yaml("configs/models/torchvision_*.yaml")`) and it
executes the whole declared lifecycle — resolve data via the configured
`DatasetAdapter`, train, validate, register the trained model, export — driven
entirely by the config file, no command-line flags.

The lifecycle order lives in `core.pipeline.BasePipeline`; this file is only
what is specific to torchvision.
"""

from __future__ import annotations

from typing import Any

from fabric_defect_hub.core.pipeline import BasePipeline, RunResult
from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact
from fabric_defect_hub.models.torchvision.config import TorchvisionConfig

# The unified result type, under this backend's historical name.
TorchvisionRunResult = RunResult


def _load_split_samples(config: TorchvisionConfig, selection: dict[str, Any]) -> list[Sample]:
    dataset = load_dataset(config.data.dataset, root=config.data.dataset_root, **selection)
    return dataset.load_samples()


class TorchvisionPipeline(BasePipeline):
    """`TorchvisionConfig` -> a full train / validate / export run.

    `adapter_factory` exists so tests can drive the whole lifecycle against a
    stand-in adapter without importing torchvision.
    """

    def __init__(self, config: TorchvisionConfig, adapter_factory=None):
        super().__init__(config)
        self.adapter_factory = adapter_factory

    def build_adapter(self):
        factory = self.adapter_factory
        if factory is None:
            from fabric_defect_hub.models.torchvision.adapter import TorchvisionAdapter

            factory = TorchvisionAdapter
        return factory(name=self.config.model.variant)

    def prepare(self) -> None:
        config = self.config
        self.train_samples = _load_split_samples(config, config.data.train_selection)
        self.val_samples = (
            _load_split_samples(config, config.data.val_selection)
            if config.data.val_selection
            else self.train_samples
        )

    def build_train_config(self) -> dict[str, Any]:
        config = self.config
        train_config: dict[str, Any] = dict(config.resolved_train_kwargs())
        train_config["train_samples"] = self.train_samples
        train_config["val_samples"] = self.val_samples
        train_config["class_names"] = config.data.class_names
        train_config["pretrained"] = config.model.pretrained
        train_config["offline"] = config.model.offline
        train_config["trainable_backbone_layers"] = config.model.trainable_backbone_layers
        train_config["min_size"] = config.model.min_size
        train_config["max_size"] = config.model.max_size
        train_config["run_dir"] = config.checkpoint.run_dir
        train_config["name"] = config.checkpoint.name
        train_config["save_every_epoch"] = config.checkpoint.save_every_epoch
        # `device`/`seed` are excluded from `resolved_train_kwargs()` (they are
        # pipeline-level, not native torchvision train() kwargs — see
        # `TrainSpec._NON_PRESET`) and must be re-added explicitly here, the
        # same way `weights`/`min_size`/... already are just above.
        train_config["device"] = config.train.device
        train_config["seed"] = config.train.seed
        if config.train.resume:
            train_config["resume"] = True
        if config.model.weights:
            train_config["weights"] = config.model.weights
        return train_config

    def load_existing_artifact(self, adapter) -> Artifact | None:
        if self.config.model.weights:
            return adapter.load_trained_model(self.config.model.weights)
        return None

    def evaluate(self, adapter, artifact: Artifact) -> dict[str, float]:
        val_kwargs = {
            "batch_size": self.config.val.batch_size,
            "num_workers": self.config.val.num_workers,
        }
        val_kwargs = {k: v for k, v in val_kwargs.items() if v is not None}
        return adapter.validate(self.val_samples, artifact, val_kwargs)

    def export_config(self, fmt: str) -> dict[str, Any]:
        return {"opset": self.config.export.opset}


def run_from_config(config: TorchvisionConfig, adapter_factory=None) -> RunResult:
    """Execute the lifecycle declared in `config`."""

    return TorchvisionPipeline(config, adapter_factory=adapter_factory).run()


def run_from_yaml(path: str) -> RunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(TorchvisionConfig.from_yaml(path))
