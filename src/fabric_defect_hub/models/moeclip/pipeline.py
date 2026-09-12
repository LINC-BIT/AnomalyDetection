"""Config-driven end-to-end runner for the MoECLIP backend.

The lifecycle lives in `core.pipeline.BasePipeline`; this file is only what is
specific to MoECLIP. Its one shape difference from the other anomaly backends
is that training and evaluation read *different* datasets (auxiliary corpus
vs. zero-shot fabric target), which is what makes the metrics zero-shot.
Export raises a clear error if enabled (see `MoECLIPAdapter.export`).
"""

from __future__ import annotations

from typing import Any

from fabric_defect_hub.core.pipeline import AnomalyPipeline, RunResult
from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact
from fabric_defect_hub.models.moeclip.adapter import MoECLIPAdapter
from fabric_defect_hub.models.moeclip.config import MoECLIPConfig

# The unified result type, under this backend's historical name.
MoECLIPRunResult = RunResult


def _load_split_samples(name: str, root: str, selection: dict[str, Any]) -> list[Sample]:
    return load_dataset(name, root=root, **selection).load_samples()


class MoECLIPPipeline(AnomalyPipeline):
    """`MoECLIPConfig` -> a full train / zero-shot evaluate run."""

    def build_adapter(self) -> MoECLIPAdapter:
        return MoECLIPAdapter(
            name=self.config.model.name, **self.config.model.adapter_kwargs()
        )

    def prepare(self) -> None:
        # Evaluation reads a *different* dataset from training whenever
        # `data.test_dataset` is set -- that separation is what makes the
        # reported numbers zero-shot (see `MoECLIPConfig.DataSpec`).
        eval_name, eval_root = self.config.data.eval_dataset()
        self.test_samples = _load_split_samples(
            eval_name, eval_root, self.config.data.test_selection
        )

    def load_existing_artifact(self, adapter: MoECLIPAdapter) -> Artifact | None:
        # No training: load the configured checkpoint so zero-shot evaluation
        # can run against it.
        if self.config.model.weights:
            return adapter.load_trained_model(self.config.model.weights)
        return None

    def build_train_config(self) -> dict[str, Any]:
        config = self.config
        train_config: dict[str, Any] = config.resolved_train_kwargs()
        train_config["train_samples"] = _load_split_samples(
            config.data.dataset, config.data.dataset_root, config.data.train_selection
        )
        return train_config


def run_from_config(config: MoECLIPConfig) -> RunResult:
    """Execute the lifecycle declared in `config`."""

    return MoECLIPPipeline(config).run()


def run_from_yaml(path: str) -> RunResult:
    """Convenience wrapper: load a YAML config and run it."""

    return run_from_config(MoECLIPConfig.from_yaml(path))
