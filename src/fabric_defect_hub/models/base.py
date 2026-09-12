"""ModelAdapter: unifies train/predict/export across Ultralytics,
torchvision and Anomalib, while letting each backend keep its native config.

The three method signatures here are the project's model-side contract and are
deliberately *identical* for every backend, including parameters a given
backend ignores. A union signature costs one unused argument; the alternative
— each family declaring only what it uses — costs every caller a runtime
`inspect.signature` check before it dares to call, which is what `loader.py`
used to do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from fabric_defect_hub.core.train_config import TrainConfig
from fabric_defect_hub.core.types import Prediction, Sample

# The `Prediction` fields a backend can fill. A backend declares the subset it
# actually produces via `ModelCapabilities.prediction_fields`, so an evaluator
# can tell — without running anything — whether a metric it wants is even
# computable (e.g. pixel-level PRO needs `anomaly_map` or `masks`).
PREDICTION_FIELDS = ("boxes", "labels", "scores", "masks", "anomaly_score", "anomaly_map")

# The `Annotations` fields a dataset can carry, i.e. the vocabulary
# `ModelCapabilities.required_annotations` draws from.
ANNOTATION_FIELDS = ("boxes", "masks", "labels", "is_anomalous", "anomaly_mask")

# Task names: the `Evaluator.task` keys registered in `evaluation/`, plus
# `instance_segmentation` (a torchvision variant task, see
# `models/torchvision/presets.variant_task`, whose own `"detect"` spelling is
# normalised to `"detection"` at the adapter boundary).
TASKS = ("detection", "segmentation", "instance_segmentation", "anomaly", "industrial")

# How an exported module wants its input: one batched NCHW tensor, or a list
# of per-image CHW tensors (torchvision's detection/instance-segmentation
# models). `profiling.base.ProfileConfig.input_style` needs the answer to
# feed a profiler a correctly-shaped dummy input, and only the backend knows
# it -- the benchmark UI used to decide with an
# `if backend == "torchvision" and task in (...)` of its own.
EXPORT_INPUT_STYLES = ("batched", "list")


@dataclass
class Artifact:
    """Opaque handle to trained weights, produced by `train()`."""

    path: str
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportedArtifact:
    """Opaque handle to an exported/converted model, produced by `export()`."""

    path: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCapabilities:
    """What a backend can actually do, declared rather than discovered.

    The model-side mirror of `core.dataset_capabilities.DatasetCapabilities`.
    Without this, which `Prediction` fields a backend fills is an implicit
    convention, and an evaluator has no way to know whether a pixel-level
    metric applies to the model it was handed — it can only compute the metric
    and get a meaningless number.

    Every name is validated against the vocabularies above, so a typo
    (`"anomaly_maps"`) fails at import time instead of silently disabling a
    metric.
    """

    tasks: tuple[str, ...]
    prediction_fields: tuple[str, ...]
    required_annotations: tuple[str, ...] = ()
    export_targets: tuple[str, ...] = ()
    supports_amp: bool = False
    # Shape the exported module expects, for whoever has to synthesize an
    # input for it (`profiling.base.ProfileConfig.input_style`). "batched" is
    # the common case, so only the backends that differ say so.
    export_input_style: str = "batched"

    def __post_init__(self) -> None:
        for name, values, vocabulary in (
            ("tasks", self.tasks, TASKS),
            ("prediction_fields", self.prediction_fields, PREDICTION_FIELDS),
            ("required_annotations", self.required_annotations, ANNOTATION_FIELDS),
        ):
            unknown = [v for v in values if v not in vocabulary]
            if unknown:
                raise ValueError(
                    f"ModelCapabilities.{name} has unknown entries {unknown}; "
                    f"expected values from {vocabulary}"
                )
        if not self.tasks:
            raise ValueError("ModelCapabilities.tasks must not be empty")
        if not self.prediction_fields:
            raise ValueError("ModelCapabilities.prediction_fields must not be empty")
        if self.export_input_style not in EXPORT_INPUT_STYLES:
            raise ValueError(
                f"ModelCapabilities.export_input_style={self.export_input_style!r} is unknown; "
                f"expected one of {EXPORT_INPUT_STYLES}"
            )

    def supports_task(self, task: str) -> bool:
        return task in self.tasks

    def fills(self, prediction_field: str) -> bool:
        """Whether `predict()` populates this `Prediction` field."""

        return prediction_field in self.prediction_fields

    def can_export(self, target: str) -> bool:
        return target in self.export_targets


class ModelAdapter(ABC):
    """Base class every concrete model backend (YOLO, Faster R-CNN, PatchCore, ...) implements."""

    name: str
    backend: str

    # Canonical `TrainConfig` field -> this backend's real argument name (see
    # `core.train_config`). Declared per backend because the translation is
    # backend knowledge: `lr` is `lr0` to Ultralytics and `lr` to torchvision.
    # `tests/test_adapter_contract.py` requires every registered backend to
    # fill this in; the empty default only serves throwaway test doubles.
    TRAIN_CONFIG_KEYS: dict[str, str] = {}

    def __init__(self, name: str | None = None, **kwargs):
        if name is not None:
            self.name = name
        self.options = kwargs

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """What this adapter — at its current `self.name` variant — can do.

        An instance method rather than a class attribute because a backend's
        answer can depend on the variant: `TorchvisionAdapter` is a detector
        as `fasterrcnn_resnet50_fpn` and an instance segmenter as
        `maskrcnn_resnet50_fpn`. Constructing an adapter is cheap and loads no
        weights, so callers can still ask before committing to a run.
        """

    @abstractmethod
    def train(self, config: dict[str, Any] | TrainConfig) -> Artifact:
        """Fit the model per `config`, return a handle to the resulting weights.

        A `TrainConfig` carries the portable settings under one validated set
        of names and is translated through `TRAIN_CONFIG_KEYS`; a plain dict
        is passed through untouched, in this backend's own vocabulary.
        """

    @abstractmethod
    def predict(
        self,
        samples: list[Sample],
        artifact: Artifact | None = None,
        output_dir: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Run inference, returning one `Prediction` per input `Sample`.

        `output_dir`, when given, is where a backend that produces per-sample
        image artifacts (anomaly maps) writes them; `config` carries
        inference knobs (confidence/IoU thresholds, batch size). A backend
        ignores whichever of the two its family has no use for — the signature
        is uniform so callers never have to ask which family they hold.
        """

    @abstractmethod
    def export(
        self, artifact: Artifact, target: str, config: dict[str, Any] | None = None
    ) -> ExportedArtifact:
        """Convert `artifact` to a deployment target (e.g. 'onnx', 'tensorrt').

        Backends that cannot export raise `NotImplementedError` and declare an
        empty `ModelCapabilities.export_targets`, so a caller can check first
        rather than discovering it from an exception.
        """

    def raw_module(self) -> Any | None:
        """The live, mutable `torch.nn.Module` behind this adapter's
        currently-loaded model, if any -- for callers that need to
        instrument the model itself rather than an exported copy (e.g.
        FLOPs counting via `profiling.flops.compute_model_flops`, which
        needs to attach hooks a frozen `torch.jit.ScriptModule` export
        can't accept). `None` when nothing is loaded yet.

        The default returns the conventional `self._model` attribute every
        concrete adapter already keeps its loaded model under; override
        this when that attribute isn't a raw module (see
        `UltralyticsAdapter`, whose `_model` is a `YOLO` wrapper around the
        real `torch.nn.Module`).
        """

        return getattr(self, "_model", None)
