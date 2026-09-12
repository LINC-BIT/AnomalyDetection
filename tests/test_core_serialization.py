"""Real round-trip and schema-validation tests for `core.serialization`.

The result schema keeps `model.backend` extensible so future adapters can
participate without changing the shared contract.
"""

import json

import jsonschema
import pytest

from fabric_defect_hub.core.serialization import (
    experiment_result_from_dict,
    load_experiment_result,
    load_predictions,
    load_samples,
    prediction_from_dict,
    prediction_to_dict,
    sample_from_dict,
    sample_to_dict,
    save_experiment_result,
    save_predictions,
    save_samples,
    validate_experiment_result,
)
from fabric_defect_hub.core.types import (
    Annotations,
    DatasetInfo,
    ExperimentResult,
    ModelInfo,
    Prediction,
    RuntimeInfo,
    Sample,
)


def _sample() -> Sample:
    return Sample(
        id="s1", image_path="images/s1.jpg", task="detection",
        annotations=Annotations(boxes=[[1.0, 2.0, 3.0, 4.0]], labels=["défaut"]),
        metadata={"fabric_type": "cotton"},
    )


def _prediction() -> Prediction:
    return Prediction(sample_id="s1", boxes=[[1.0, 2.0, 3.0, 4.0]], labels=["défaut"], scores=[0.9])


def _result() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp-1",
        model=ModelInfo(name="yolov8n", backend="ultralytics", task="detection"),
        dataset=DatasetInfo(name="zju-leaper", split="test"),
        runtime=RuntimeInfo(device="cpu", engine="pytorch", precision="fp32", input_size=(640, 640)),
        metrics={"map50": 0.81},
        artifacts={"predictions": "predictions.json"},
    )


def test_sample_dict_round_trip():
    sample = _sample()
    assert sample_from_dict(sample_to_dict(sample)) == sample


def test_prediction_dict_round_trip():
    prediction = _prediction()
    assert prediction_from_dict(prediction_to_dict(prediction)) == prediction


def test_save_load_samples_preserves_non_ascii_unescaped(tmp_path):
    path = tmp_path / "samples.json"
    save_samples([_sample()], path)

    raw_text = path.read_text(encoding="utf-8")
    assert "défaut" in raw_text  # ensure_ascii=False actually took effect
    assert "\\u" not in raw_text

    reloaded = load_samples(path)
    assert reloaded == [_sample()]


def test_save_load_predictions_round_trip(tmp_path):
    path = tmp_path / "predictions.json"
    save_predictions([_prediction()], path)
    assert load_predictions(path) == [_prediction()]


def test_save_creates_missing_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "samples.json"
    save_samples([_sample()], nested)
    assert nested.exists()


def test_experiment_result_input_size_round_trips_as_tuple(tmp_path):
    path = save_experiment_result(_result(), tmp_path / "result.json")
    reloaded = load_experiment_result(path)

    assert reloaded == _result()
    assert isinstance(reloaded.runtime.input_size, tuple)
    assert reloaded.runtime.input_size == (640, 640)


def test_experiment_result_from_dict_matches_to_dict_json_shape():
    result = _result()
    dumped = json.loads(json.dumps({
        "experiment_id": result.experiment_id,
        "model": {"name": "yolov8n", "backend": "ultralytics", "task": "detection"},
        "dataset": {"name": "zju-leaper", "split": "test"},
        "runtime": {"device": "cpu", "engine": "pytorch", "precision": "fp32", "input_size": [640, 640]},
        "metrics": {"map50": 0.81},
        "artifacts": {"predictions": "predictions.json"},
    }))
    assert experiment_result_from_dict(dumped) == result


def test_validate_experiment_result_accepts_valid_result():
    validate_experiment_result(_result())  # must not raise


def test_validate_experiment_result_accepts_extensible_backend():
    result = ExperimentResult(
        experiment_id="x",
        model=ModelInfo(name="m", backend="future-backend", task="industrial"),
        dataset=DatasetInfo(name="d", split="test"),
        runtime=RuntimeInfo(device="cpu", engine="e", precision="fp32", input_size=(1, 1)),
    )
    validate_experiment_result(result)


def test_save_result_rejects_non_finite_json_metric(tmp_path):
    bad = _result()
    bad.metrics["invalid"] = float("nan")
    with pytest.raises(ValueError):
        save_experiment_result(bad, tmp_path / "bad.json")


def test_validate_experiment_result_rejects_non_numeric_metric():
    bad = ExperimentResult(
        experiment_id="x",
        model=ModelInfo(name="m", backend="ultralytics", task="detection"),
        dataset=DatasetInfo(name="d", split="test"),
        runtime=RuntimeInfo(device="cpu", engine="e", precision="fp32", input_size=(1, 1)),
        metrics={"bad": "not-a-number"},
    )
    with pytest.raises(jsonschema.ValidationError):
        validate_experiment_result(bad)
