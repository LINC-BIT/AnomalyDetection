"""Fast, framework-free tests for the unified `predict` entry point
(`fabric_defect_hub.inference.runner`): input validation and artifact/sample
resolution helpers, without importing any ML framework or running
inference.
"""

import pytest

import fabric_defect_hub.inference.runner as predict_module
from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.models.base import ModelCapabilities
from fabric_defect_hub.inference.runner import (
    PredictInput,
    PredictRunResult,
    _load_samples,
    _resolve_weights_artifact,
    run_evaluate,
)


def test_resolve_weights_artifact_passes_through_bare_path_for_ultralytics():
    assert _resolve_weights_artifact("ultralytics", "artifacts/models/yolov8n_x.pt") == "artifacts/models/yolov8n_x.pt"


def test_resolve_weights_artifact_passes_through_bare_path_for_torchvision():
    assert _resolve_weights_artifact("torchvision", "artifacts/models/fasterrcnn_x.pt") == "artifacts/models/fasterrcnn_x.pt"


def test_resolve_weights_artifact_wraps_anomalib_checkpoint_as_trusted_artifact():
    artifact = _resolve_weights_artifact("anomalib", "artifacts/models/Patchcore.ckpt")
    assert artifact.path == "artifacts/models/Patchcore.ckpt"
    assert artifact.backend == "anomalib"
    assert artifact.metadata["trusted"] is True
    assert artifact.metadata["model_class"] == "Patchcore"


def test_resolve_weights_artifact_infers_model_class_from_stem():
    artifact = _resolve_weights_artifact("anomalib", "/some/dir/Padim.ckpt")
    assert artifact.metadata["model_class"] == "Padim"


# `_load_samples` reads the model's declared capabilities rather than a list
# of backend names, so these fixtures stand in for any detection / anomaly
# backend rather than naming one.
DETECTS = ModelCapabilities(tasks=("detection",), prediction_fields=("boxes", "labels", "scores"))
SCORES_ANOMALIES = ModelCapabilities(
    tasks=("anomaly",), prediction_fields=("anomaly_score", "anomaly_map")
)


def test_load_samples_rejects_both_images_and_dataset():
    source = PredictInput(images=["a.jpg"], dataset="zju-leaper")
    with pytest.raises(ValueError, match="either --image or --dataset"):
        _load_samples(source, DETECTS)


def test_load_samples_rejects_neither_images_nor_dataset():
    source = PredictInput()
    with pytest.raises(ValueError, match="--image.*--dataset"):
        _load_samples(source, DETECTS)


def test_load_samples_from_images_builds_one_sample_per_path():
    source = PredictInput(images=["/data/a.jpg", "/data/b.png"])
    samples = _load_samples(source, DETECTS)
    assert [s.id for s in samples] == ["a", "b"]
    assert [s.image_path for s in samples] == ["/data/a.jpg", "/data/b.png"]
    assert all(s.task == "detection" for s in samples)


def test_load_samples_takes_the_task_from_declared_capabilities():
    # A bare image path carries no task; the model's own declaration supplies
    # one. Previously this was "is the backend in this hard-coded set of four
    # names", which no new backend could satisfy without editing predict.py.
    source = PredictInput(images=["/data/a.jpg"])
    assert _load_samples(source, SCORES_ANOMALIES)[0].task == "anomaly"
    assert _load_samples(source, DETECTS)[0].task == "detection"

    segments = ModelCapabilities(tasks=("segmentation",), prediction_fields=("masks",))
    assert _load_samples(source, segments)[0].task == "segmentation"


def test_load_samples_from_dataset_raises_without_root_or_default(monkeypatch):
    source = PredictInput(dataset="some-unregistered-dataset")
    with pytest.raises(ValueError, match="no dataset_root"):
        _load_samples(source, SCORES_ANOMALIES)


def _fake_run_predict_result() -> PredictRunResult:
    samples = [
        Sample(id="a", image_path="a.jpg", task="anomaly", annotations=Annotations(is_anomalous=False)),
        Sample(id="b", image_path="b.jpg", task="anomaly", annotations=Annotations(is_anomalous=True)),
    ]
    predictions = [
        Prediction(sample_id="a", anomaly_score=0.1),
        Prediction(sample_id="b", anomaly_score=0.9),
    ]
    return PredictRunResult(backend="anomalib", variant="PatchCore", predictions=predictions, samples=samples)


def test_run_evaluate_scores_predictions_via_task_evaluator(monkeypatch):
    monkeypatch.setattr(predict_module, "run_predict", lambda *args, **kwargs: _fake_run_predict_result())

    result = run_evaluate("some_config", weights="x.ckpt", source=PredictInput(dataset="zju-leaper"))

    assert result.backend == "anomalib"
    assert result.variant == "PatchCore"
    assert result.sample_count == 2
    assert "image_auroc" in result.metrics


def test_run_evaluate_honors_explicit_task_override(monkeypatch):
    captured = {}

    def fake_evaluator_for_task(task):
        captured["task"] = task
        from fabric_defect_hub.evaluation import AnomalyEvaluator

        return AnomalyEvaluator()

    monkeypatch.setattr(predict_module, "run_predict", lambda *args, **kwargs: _fake_run_predict_result())
    monkeypatch.setattr("fabric_defect_hub.evaluation.evaluator_for_task", fake_evaluator_for_task)

    run_evaluate("some_config", weights="x.ckpt", source=PredictInput(dataset="zju-leaper"), task="anomaly")

    assert captured["task"] == "anomaly"


def test_run_evaluate_rejects_image_source_without_dataset():
    with pytest.raises(ValueError, match="--dataset"):
        run_evaluate("some_config", weights="x.ckpt", source=PredictInput(images=["a.jpg"]))


def test_run_evaluate_rejects_empty_sample_resolution(monkeypatch):
    empty_result = PredictRunResult(backend="anomalib", variant="PatchCore", predictions=[], samples=[])
    monkeypatch.setattr(predict_module, "run_predict", lambda *args, **kwargs: empty_result)

    with pytest.raises(ValueError, match="no samples resolved"):
        run_evaluate("some_config", weights="x.ckpt", source=PredictInput(dataset="zju-leaper"))


def test_predict_and_evaluate_report_the_config_they_resolved(monkeypatch, tmp_path):
    """Keyword resolution can pick a config the caller never named (`fdh
    predict padim` -> anomalib_example.yaml), so every entry point that
    accepts a bare model name has to report what it landed on — the same
    rule `fdh train` follows.
    """

    from fabric_defect_hub.training import resolve_model_config_and_variant

    for keyword in ("padim", "patchcore", "fastflow"):
        path, _ = resolve_model_config_and_variant(keyword)
        assert path.is_file()

    # The dataclasses carry the field the CLI reports from.
    from fabric_defect_hub.inference.runner import EvaluateRunResult, PredictRunResult

    assert "config_path" in PredictRunResult.__dataclass_fields__
    assert "config_path" in EvaluateRunResult.__dataclass_fields__


def test_evaluate_forwards_output_dir_so_pixel_metrics_are_reachable(monkeypatch):
    """Without this, `fdh evaluate` could never produce pixel_auroc/aupro/iap
    for an anomaly model, however capable of pixel output the model is: the
    adapters only fill `Prediction.anomaly_map` when handed somewhere to
    write it, and `AnomalyEvaluator` scores pixels from that field alone.
    Confirmed against a real FastFlow checkpoint before this was threaded
    through — image-level metrics only.
    """

    from fabric_defect_hub.inference import runner as predict_module

    captured = {}

    def fake_run_predict(model, **kwargs):
        captured.update(kwargs)
        raise _StopEvaluate

    class _StopEvaluate(Exception):
        pass

    monkeypatch.setattr(predict_module, "run_predict", fake_run_predict)
    with pytest.raises(Exception):
        predict_module.run_evaluate(
            "patchcore",
            weights="w.ckpt",
            source=predict_module.PredictInput(dataset="zju-leaper"),
            output_dir="/tmp/maps",
        )

    assert captured["output_dir"] == "/tmp/maps"


def test_cli_evaluate_exposes_output_dir():
    from fabric_defect_hub.cli import build_parser

    args = build_parser().parse_args(
        ["evaluate", "patchcore", "--weights", "w.ckpt", "--dataset", "tilda-400",
         "--output-dir", "/tmp/maps"]
    )
    assert args.output_dir == "/tmp/maps"
    assert build_parser().parse_args(
        ["evaluate", "patchcore", "--weights", "w.ckpt", "--dataset", "tilda-400"]
    ).output_dir is None
