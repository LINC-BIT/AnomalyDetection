"""BasePipeline: the one config-driven run lifecycle, shared by every backend.

Each backend used to carry its own `run_from_config` and its own
`XxxRunResult` dataclass — six copies of the same eight steps (validate, build
adapter, resolve data, train, register, evaluate, export, return), differing
only in how each step is spelled. Six copies means a fix to the lifecycle has
to be made six times, and the drift is invisible until someone diffs them: the
`elif model.weights: load_trained_model(...)` branch, for instance, existed in
two of the six and was simply missing from the rest.

Here the *order* is written once and the *differences* are hooks:

    validate -> build_adapter -> prepare -> train+register -> evaluate -> export

A backend subclasses `BasePipeline` and fills in three required hooks
(`build_adapter`, `build_train_config`, `evaluate`) plus whatever optional
ones it needs. Anomaly backends get `AnomalyPipeline`, which implements
`evaluate` as predict + `AnomalyEvaluator` — the four one-class/zero-shot
backends had that identical block copied four times.

The `run_from_config` / `run_from_yaml` module functions each backend exports
stay exactly as they were; they are now three lines around a pipeline class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from fabric_defect_hub.core.progress import note
from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter


@dataclass
class RunResult:
    """Everything a config-driven run produced.

    One type for every backend: the five fields were already identical in all
    six per-backend copies, and a caller that wanted to handle "whatever a
    pipeline returned" had to name a union of six types to say so (see
    `training.TrainRunResult.result`'s comment).
    """

    config: Any
    trained_artifact: Artifact | None = None
    registered_artifact: Artifact | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    exports: list[ExportedArtifact] = field(default_factory=list)

    @property
    def active_artifact(self) -> Artifact | None:
        """The artifact later stages act on: the registered copy in its stable
        location if training produced one, else whatever was loaded.
        """

        return self.registered_artifact or self.trained_artifact


class BasePipeline(ABC):
    """Template for one config-driven run. Subclass per backend."""

    def __init__(self, config: Any):
        self.config = config

    # ------------------------------------------------------------------ #
    # Required hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build_adapter(self) -> ModelAdapter:
        """Construct this backend's adapter from `self.config`."""

    @abstractmethod
    def build_train_config(self) -> Any:
        """The argument for `adapter.train()`. Only called when training is enabled."""

    @abstractmethod
    def evaluate(self, adapter: ModelAdapter, artifact: Artifact) -> dict[str, float]:
        """Metrics for `artifact`. Only called when validation is enabled and
        an artifact exists. Return `{}` when this run has nothing to score
        against (e.g. a data mode that yields no `Sample` list).
        """

    # ------------------------------------------------------------------ #
    # Optional hooks
    # ------------------------------------------------------------------ #
    def prepare(self) -> None:
        """Resolve datasets and any other state the later steps share.

        Runs after the adapter is built and before training, because several
        backends need the test split both to train against and to score with.
        """

    def load_existing_artifact(self, adapter: ModelAdapter) -> Artifact | None:
        """The artifact to use when training is *disabled* — typically a
        checkpoint named in the config, so validate/export can still run.
        `None` (the default) means such a run does nothing.
        """

        return None

    def export_config(self, fmt: str) -> dict[str, Any] | None:
        """Per-format export options, or `None` for the backend's defaults."""

        return None

    # ------------------------------------------------------------------ #
    # The lifecycle
    # ------------------------------------------------------------------ #
    def run(self) -> RunResult:
        """Execute the lifecycle declared in `self.config`."""

        config = self.config
        config.validate()

        # Stage announcements, written once here rather than in six
        # backends. They are the only progress signal that covers every
        # backend uniformly: Lightning (anomalib) and Ultralytics print
        # their own within-stage bars, the four loops this project owns
        # report through `core.progress`, but nothing else says which
        # *stage* a silent run is sitting in -- staging a dataset
        # directory and computing AUPRO are both long and both mute.
        adapter = self.build_adapter()
        result = RunResult(config=config)
        # Every registered adapter declares `backend`, but a label for a log
        # line is not worth being able to break a run over -- hence the
        # fallback rather than a direct attribute read.
        label = getattr(adapter, "backend", type(adapter).__name__)
        note(f"{label}: preparing data")
        self.prepare()

        if config.train.enabled:
            note(f"{label}: training")
            result.trained_artifact = adapter.train(self.build_train_config())
            result.registered_artifact = adapter.register_trained_model(
                result.trained_artifact, registry_dir=config.checkpoint.registry_dir
            )
            note(f"{label}: trained -> {result.registered_artifact.path}")
        else:
            result.trained_artifact = self.load_existing_artifact(adapter)

        artifact = result.active_artifact
        wants_artifact = config.val.enabled or (config.export.enabled and config.export.formats)
        if wants_artifact and artifact is None:
            # Previously this branch silently produced an empty result: the
            # config asked for validation or an export, training was off, and
            # nothing said so. Fail instead — a config that asks to evaluate
            # and evaluates nothing is a bug in the config, not a no-op.
            raise ValueError(
                "Nothing to validate or export: train.enabled is false and no checkpoint "
                "was loaded. Set model.weights to an existing checkpoint, or enable "
                "train.enabled."
            )

        if config.val.enabled:
            note(f"{label}: evaluating")
            result.metrics = self.evaluate(adapter, artifact)
            if result.metrics:
                note(
                    f"{label}: metrics | "
                    + " ".join(
                        f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                        for k, v in sorted(result.metrics.items())
                    )
                )

        if config.export.enabled and config.export.formats:
            for fmt in config.export.formats:
                note(f"{label}: exporting {fmt}")
                result.exports.append(
                    adapter.export(artifact, fmt, config=self.export_config(fmt))
                )

        return result


class AnomalyPipeline(BasePipeline):
    """Shared `evaluate` for the anomaly backends.

    Unlike Ultralytics and torchvision, these backends have no native
    `validate()` — scoring is `AnomalyEvaluator`'s job over the `Sample` list
    and the `Prediction`s. That predict-then-evaluate block was identical in
    the anomalib, Dinomaly, MambaAD and MoECLIP pipelines.

    A subclass sets `self.test_samples` in `prepare()`. Leaving it empty
    means "nothing to score against" — the case where a backend was pointed at
    an on-disk folder rather than a `DatasetAdapter`, so no `Sample` objects
    exist — and yields no metrics rather than a guess.
    """

    test_samples: list[Sample] | None = None

    def evaluate(self, adapter: ModelAdapter, artifact: Artifact) -> dict[str, float]:
        if not self.test_samples:
            return {}

        from fabric_defect_hub.evaluation.anomaly import AnomalyEvaluator

        val = self.config.val
        predictions = adapter.predict(
            self.test_samples, artifact, output_dir=val.output_dir
        )
        evaluator = AnomalyEvaluator(
            max_pixels=val.max_pixels,
            max_aupro_images=val.max_aupro_images,
            seed=val.seed,
            image_threshold=val.image_threshold,
            pixel_threshold=val.pixel_threshold,
            allow_oracle_threshold=val.allow_oracle_threshold,
        )
        return evaluator.evaluate(self.test_samples, predictions)
