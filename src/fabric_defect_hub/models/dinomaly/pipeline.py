"""Config-driven end-to-end runner for the Dinomaly backend.

The lifecycle lives in `core.pipeline.BasePipeline`; this file is only what is
specific to Dinomaly. Export raises a clear error if enabled (see
`DinomalyAdapter.export`, and its empty `capabilities().export_targets`).
"""

from __future__ import annotations

from typing import Any

from fabric_defect_hub.core.pipeline import AnomalyPipeline, RunResult
from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact
from fabric_defect_hub.models.dinomaly.adapter import DinomalyAdapter
from fabric_defect_hub.models.dinomaly.config import DinomalyConfig

# The unified result type, under this backend's historical name.
DinomalyRunResult = RunResult


def _load_split_samples(config: DinomalyConfig, selection: dict[str, Any]) -> list[Sample]:
    dataset = load_dataset(config.data.dataset, root=config.data.dataset_root, **selection)
    return dataset.load_samples()


class DinomalyPipeline(AnomalyPipeline):
    """`DinomalyConfig` -> a full train / evaluate run."""

    def build_adapter(self) -> DinomalyAdapter:
        return DinomalyAdapter(name=self.config.model.name)

    def prepare(self) -> None:
        # `data_root` mode points at an existing on-disk folder with no
        # corresponding `Sample` objects to score against, same as
        # `AnomalibConfig`'s `datamodule_kwargs` mode.
        if self.config.data.uses_adapter():
            self.test_samples = _load_split_samples(self.config, self.config.data.test_selection)

    def load_existing_artifact(self, adapter: DinomalyAdapter) -> Artifact | None:
        # No training: load the configured checkpoint so validation can run.
        if self.config.model.weights:
            return adapter.load_trained_model(self.config.model.weights)
        return None

    def build_train_config(self) -> dict[str, Any]:
        config = self.config
        train_config: dict[str, Any] = config.resolved_train_kwargs()
        if config.data.uses_adapter():
            train_config["train_samples"] = _load_split_samples(config, config.data.train_selection)
            train_config["test_samples"] = self.test_samples
        else:
            train_config["data_root"] = config.data.data_root
        return train_config


def run_from_config(config: DinomalyConfig) -> RunResult:
    """Execute the lifecycle declared in `config`."""

    return DinomalyPipeline(config).run()


def run_from_yaml(path: str) -> RunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(DinomalyConfig.from_yaml(path))
