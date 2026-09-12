"""Unified data contracts shared by every dataset and model backend.

These mirror the JSON contracts documented in the top-level README
(`Sample`, `Prediction`, `ExperimentResult`) and the schemas under
`schemas/`. Keeping them as plain dataclasses (rather than pydantic or
similar) avoids pulling in a hard dependency before the project has
settled on one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Task = Literal["detection", "segmentation", "anomaly", "industrial"]


@dataclass
class Annotations:
    """Task-native label fields. Only the fields relevant to `task` are set."""

    boxes: list[list[float]] | None = None
    masks: list[Any] | None = None
    labels: list[str] | None = None
    is_anomalous: bool | None = None
    anomaly_mask: str | None = None


@dataclass
class Sample:
    """Unified per-image sample description used across all dataset adapters."""

    id: str
    image_path: str
    task: Task
    annotations: Annotations
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """Unified per-image model output. Backends fill only what they produce."""

    sample_id: str
    boxes: list[list[float]] | None = None
    labels: list[str] | None = None
    scores: list[float] | None = None
    masks: list[Any] | None = None
    anomaly_score: float | None = None
    anomaly_map: str | None = None


@dataclass
class ModelInfo:
    name: str
    backend: str
    task: Task


@dataclass
class DatasetInfo:
    name: str
    split: str


@dataclass
class RuntimeInfo:
    device: str
    engine: str
    precision: str
    input_size: tuple[int, int]


@dataclass
class ExperimentResult:
    """Final, comparable output of a train/predict/evaluate/profile run."""

    experiment_id: str
    model: ModelInfo
    dataset: DatasetInfo
    runtime: RuntimeInfo
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
