"""JSON (de)serialization for the unified contracts in `core/types.py`,
matching `schemas/{sample,prediction,experiment_result}.schema.json`
exactly — this is what closes the README's Phase 1 "emit real predictions
and experiment-result JSON" item: every `ModelAdapter.train`/`predict` call already returns
`Artifact`/`Prediction` objects; this module is what turns those into the
actual on-disk JSON a leaderboard/frontend would read.

Uses `dataclasses.asdict()` rather than hand-writing a field-by-field
mapping — our dataclasses already mirror the schemas field-for-field, so a
generic recursive dict conversion is both correct and immediately obvious
to keep in sync when a field is added; `from_dict` reconstructs the
dataclasses in the one place ordering/nesting actually matters.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fabric_defect_hub.core.types import (
    Annotations,
    DatasetInfo,
    ExperimentResult,
    ModelInfo,
    Prediction,
    RuntimeInfo,
    Sample,
)


# ---------------------------------------------------------------------- #
# Sample
# ---------------------------------------------------------------------- #
def sample_to_dict(sample: Sample) -> dict:
    return asdict(sample)


def sample_from_dict(data: dict) -> Sample:
    annotations = data.get("annotations") or {}
    return Sample(
        id=data["id"],
        image_path=data["image_path"],
        task=data["task"],
        annotations=Annotations(**annotations),
        metadata=data.get("metadata", {}),
    )


def save_samples(samples: list[Sample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([sample_to_dict(s) for s in samples], indent=2, ensure_ascii=False, allow_nan=False)
    )
    return path


def load_samples(path: str | Path) -> list[Sample]:
    data = json.loads(Path(path).read_text())
    return [sample_from_dict(entry) for entry in data]


# ---------------------------------------------------------------------- #
# Prediction
# ---------------------------------------------------------------------- #
def prediction_to_dict(prediction: Prediction) -> dict:
    return asdict(prediction)


def prediction_from_dict(data: dict) -> Prediction:
    return Prediction(**data)


def save_predictions(predictions: list[Prediction], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([prediction_to_dict(p) for p in predictions], indent=2, ensure_ascii=False, allow_nan=False)
    )
    return path


def load_predictions(path: str | Path) -> list[Prediction]:
    data = json.loads(Path(path).read_text())
    return [prediction_from_dict(entry) for entry in data]


# ---------------------------------------------------------------------- #
# ExperimentResult
# ---------------------------------------------------------------------- #
def experiment_result_to_dict(result: ExperimentResult) -> dict:
    return asdict(result)


def experiment_result_from_dict(data: dict) -> ExperimentResult:
    runtime = data["runtime"]
    return ExperimentResult(
        experiment_id=data["experiment_id"],
        model=ModelInfo(**data["model"]),
        dataset=DatasetInfo(**data["dataset"]),
        runtime=RuntimeInfo(
            device=runtime["device"],
            engine=runtime["engine"],
            precision=runtime["precision"],
            input_size=tuple(runtime["input_size"]),
        ),
        metrics=data.get("metrics", {}),
        artifacts=data.get("artifacts", {}),
    )


def save_experiment_result(result: ExperimentResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(experiment_result_to_dict(result), indent=2, ensure_ascii=False, allow_nan=False)
    )
    return path


def load_experiment_result(path: str | Path) -> ExperimentResult:
    return experiment_result_from_dict(json.loads(Path(path).read_text()))


def validate_experiment_result(result: ExperimentResult) -> None:
    """Validate `result` against `schemas/experiment_result.schema.json`.
    Raises `jsonschema.ValidationError` on mismatch. Requires the `dev`
    extra (`pip install -e ".[dev]"`) for `jsonschema`.

    Validates the round-tripped-through-JSON form (`json.loads(json.dumps(...))`)
    rather than the raw `asdict()` output: JSON has no tuple type, so a
    field like `runtime.input_size` (a Python `tuple`) is a `list` once it's
    actually gone through JSON, which is what `jsonschema`'s `"type":
    "array"` check expects — validating the raw dataclass dict would reject
    a perfectly valid tuple for the wrong reason.
    """

    import jsonschema

    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "experiment_result.schema.json"
    schema = json.loads(schema_path.read_text())
    instance = json.loads(json.dumps(experiment_result_to_dict(result)))
    jsonschema.validate(instance=instance, schema=schema)
