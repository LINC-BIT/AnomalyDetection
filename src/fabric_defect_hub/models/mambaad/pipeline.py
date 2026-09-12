"""Config-driven end-to-end runner for the MambaAD backend.

The lifecycle lives in `core.pipeline.BasePipeline`; this file is only what is
specific to MambaAD. Export raises a clear error if enabled (see
`MambaADAdapter.export`, and its empty `capabilities().export_targets`).
"""

from __future__ import annotations

from typing import Any

from fabric_defect_hub.core.pipeline import AnomalyPipeline, RunResult
from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact
from fabric_defect_hub.models.mambaad.adapter import MambaADAdapter
from fabric_defect_hub.models.mambaad.config import MambaADConfig

# The unified result type, under this backend's historical name.
MambaADRunResult = RunResult


def _load_split_samples(config: MambaADConfig, selection: dict[str, Any]) -> list[Sample]:
    dataset = load_dataset(config.data.dataset, root=config.data.dataset_root, **selection)
    return dataset.load_samples()


class MambaADPipeline(AnomalyPipeline):
    """`MambaADConfig` -> a full train / evaluate run."""

    def build_adapter(self) -> MambaADAdapter:
        return MambaADAdapter(name=self.config.model.name)

    def prepare(self) -> None:
        self.test_samples = _load_split_samples(self.config, self.config.data.test_selection)

    def load_existing_artifact(self, adapter: MambaADAdapter) -> Artifact | None:
        # No training: load the configured checkpoint so validation can run.
        if self.config.model.weights:
            return adapter.load_trained_model(self.config.model.weights)
        return None

    def build_train_config(self) -> dict[str, Any]:
        config = self.config
        train_config: dict[str, Any] = config.resolved_train_kwargs()
        train_config["train_samples"] = _load_split_samples(config, config.data.train_selection)
        return train_config


def run_from_config(config: MambaADConfig) -> RunResult:
    """Execute the lifecycle declared in `config`."""

    return MambaADPipeline(config).run()


def run_from_yaml(path: str) -> RunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(MambaADConfig.from_yaml(path))
