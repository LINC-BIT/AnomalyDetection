"""Config-driven end-to-end runner for the Anomalib backend.

The lifecycle itself lives in `core.pipeline.BasePipeline`; this file is only
what is specific to anomalib: how the adapter is built, how a train config is
assembled from `AnomalibConfig`, and which data mode yields `Sample` objects.

One real divergence from the Ultralytics/torchvision pipelines (see
`config.py`'s module docstring for the other): there is no backend-native
`.validate()` here. `AnomalibAdapter` only trains and predicts; scoring is
`evaluation.anomaly.AnomalyEvaluator`'s job, same as it would be for any other
caller of `predict()` — which is exactly what `AnomalyPipeline` provides.
"""

from __future__ import annotations

from typing import Any

from fabric_defect_hub.core.pipeline import AnomalyPipeline, RunResult
from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact
from fabric_defect_hub.models.anomalib.adapter import AnomalibAdapter
from fabric_defect_hub.models.anomalib.config import AnomalibConfig

# The unified result type, under this backend's historical name.
AnomalibRunResult = RunResult


def _load_split_samples(config: AnomalibConfig, selection: dict[str, Any]) -> list[Sample]:
    dataset = load_dataset(config.data.dataset, root=config.data.dataset_root, **selection)
    return dataset.load_samples()


class AnomalibPipeline(AnomalyPipeline):
    """`AnomalibConfig` -> a full train / evaluate / export run."""

    def build_adapter(self) -> AnomalibAdapter:
        return AnomalibAdapter(name=self.config.model.name)

    def prepare(self) -> None:
        # Only `data.dataset` mode produces `Sample` objects. In
        # `data.datamodule_kwargs` mode anomalib's `Folder` is pointed at an
        # existing on-disk dataset, and `AnomalyEvaluator` — which is
        # `Sample`/`Prediction`-only, not `Folder`-aware — has nothing to
        # score against, so validation yields no metrics rather than a guess.
        if self.config.data.uses_adapter():
            self.test_samples = _load_split_samples(self.config, self.config.data.test_selection)

    def load_existing_artifact(self, adapter: AnomalibAdapter) -> Artifact | None:
        # No training: load the configured checkpoint so validation can run.
        # `allow_unsafe_checkpoint` is required by the config (see
        # `ModelSpec.validate`) and forwarded explicitly rather than defaulted,
        # so the opt-in stays visible at the call site too.
        model = self.config.model
        if model.weights:
            return adapter.load_trained_model(
                model.weights, allow_unsafe_checkpoint=model.allow_unsafe_checkpoint
            )
        return None

    def build_train_config(self) -> dict[str, Any]:
        config = self.config
        train_config: dict[str, Any] = {
            "model_kwargs": config.resolved_model_kwargs(),
            "engine_kwargs": config.resolved_engine_kwargs(),
            "num_workers": config.train.num_workers,
        }
        if config.data.uses_adapter():
            train_config["train_samples"] = _load_split_samples(config, config.data.train_selection)
            train_config["test_samples"] = self.test_samples
        else:
            train_config["datamodule_kwargs"] = config.data.datamodule_kwargs
        return train_config


def run_from_config(config: AnomalibConfig) -> RunResult:
    """Execute the lifecycle declared in `config`."""

    return AnomalibPipeline(config).run()


def run_from_yaml(path: str) -> RunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(AnomalibConfig.from_yaml(path))
