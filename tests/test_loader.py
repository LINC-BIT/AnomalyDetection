"""Minimal closed-loop smoke test: fake dataset + fake model through
`run_experiment`, proving the core wiring (registry -> loader -> types)
works end-to-end before any real framework backend exists.
"""

from fabric_defect_hub.core.registry import register_dataset, register_model
from fabric_defect_hub.core.types import Annotations, ModelInfo, RuntimeInfo, Sample
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.evaluation.base import Evaluator
from fabric_defect_hub.loader import (
    list_model_backends,
    load_dataset,
    load_model,
    run_experiment,
)
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter, ModelCapabilities
from fabric_defect_hub.profiling.base import BackendProfiler, ProfileConfig
from fabric_defect_hub.profiling.power import PowerCapability, PowerReport
from fabric_defect_hub.core.types import Prediction


@register_dataset("fake-fabric")
class FakeDataset(DatasetAdapter):
    name = "fake-fabric"

    def load_samples(self) -> list[Sample]:
        return [
            Sample(
                id="sample-0001",
                image_path=f"{self.root}/0001.jpg",
                task="detection",
                annotations=Annotations(boxes=[[120, 64, 238, 180]], labels=["broken_end"]),
            )
        ]


@register_model("fake-backend")
class FakeModel(ModelAdapter):
    name = "fake-model"
    backend = "fake-backend"

    def capabilities(self):
        return ModelCapabilities(
            tasks=("detection",),
            prediction_fields=("boxes", "labels", "scores"),
            export_targets=("onnx",),
        )

    def train(self, config):
        return Artifact(path="fake.pt", backend=self.backend)

    def predict(self, samples, artifact=None, output_dir=None, config=None):
        return [Prediction(sample_id=s.id, boxes=[[121, 66, 236, 178]], labels=["broken_end"], scores=[0.9]) for s in samples]

    def export(self, artifact, target, config=None):
        return ExportedArtifact(path=f"fake.{target}", target=target)


class FakeEvaluator(Evaluator):
    task = "detection"

    def evaluate(self, samples, predictions):
        return {"map50": 1.0}


def test_end_to_end_loop():
    dataset = load_dataset("fake-fabric", root="data/fake")
    model = load_model("fake-backend", name="fake-model")

    result = run_experiment(
        experiment_id="exp-test-001",
        dataset=dataset,
        model=model,
        model_info=ModelInfo(name="fake-model", backend="fake-backend", task="detection"),
        runtime=RuntimeInfo(device="cpu", engine="python", precision="fp32", input_size=(640, 640)),
        train_config={},
        evaluator=FakeEvaluator(),
    )

    assert result.experiment_id == "exp-test-001"
    assert result.metrics == {"map50": 1.0}
    assert result.dataset.name == "fake-fabric"


class _CountingModel(FakeModel):
    def __init__(self):
        self.train_calls = 0

    def train(self, config):
        self.train_calls += 1
        return super().train(config)


def test_train_config_none_skips_training():
    dataset = FakeDataset(root="data/fake")
    model = _CountingModel()

    result = run_experiment(
        experiment_id="exp-no-train",
        dataset=dataset,
        model=model,
        model_info=ModelInfo(name="fake-model", backend="fake-backend", task="detection"),
        runtime=RuntimeInfo(device="cpu", engine="python", precision="fp32", input_size=(640, 640)),
        train_config=None,
        evaluator=FakeEvaluator(),
    )

    assert model.train_calls == 0
    assert "model" not in result.artifacts


def test_evaluator_none_gives_empty_metrics():
    dataset = FakeDataset(root="data/fake")
    model = FakeModel()

    result = run_experiment(
        experiment_id="exp-no-evaluator",
        dataset=dataset,
        model=model,
        model_info=ModelInfo(name="fake-model", backend="fake-backend", task="detection"),
        runtime=RuntimeInfo(device="cpu", engine="python", precision="fp32", input_size=(640, 640)),
        train_config={},
        evaluator=None,
    )

    assert result.metrics == {}


class _NonFiniteEvaluator(Evaluator):
    task = "anomaly"

    def evaluate(self, samples, predictions):
        return {"valid": 1.0, "nan": float("nan"), "infinite": float("inf")}


def test_experiment_drops_non_finite_metrics():
    result = run_experiment(
        experiment_id="exp-finite",
        dataset=FakeDataset(root="data/fake"),
        model=FakeModel(),
        model_info=ModelInfo(name="fake-model", backend="fake-backend", task="detection"),
        runtime=RuntimeInfo(device="cpu", engine="python", precision="fp32", input_size=(640, 640)),
        train_config={},
        evaluator=_NonFiniteEvaluator(),
    )
    assert result.metrics == {"valid": 1.0}


def test_output_dir_persists_predictions_and_result(tmp_path):
    from fabric_defect_hub.core.serialization import load_experiment_result, load_predictions

    dataset = FakeDataset(root="data/fake")
    model = FakeModel()

    result = run_experiment(
        experiment_id="exp-persisted",
        dataset=dataset,
        model=model,
        model_info=ModelInfo(name="fake-model", backend="fake-backend", task="detection"),
        runtime=RuntimeInfo(device="cpu", engine="python", precision="fp32", input_size=(640, 640)),
        train_config={},
        evaluator=FakeEvaluator(),
        output_dir=str(tmp_path),
    )

    predictions_path = result.artifacts["predictions"]
    result_path = result.artifacts["result"]
    assert load_predictions(predictions_path) == model.predict(dataset.load_samples(), None)

    # The saved result.json is written before `artifacts["result"]` (its own
    # path) is added to the in-memory object — it can't self-reference a
    # path that doesn't exist yet — so it legitimately has one fewer
    # `artifacts` entry than `result` itself. Compare everything else.
    reloaded = load_experiment_result(result_path)
    assert reloaded.artifacts == {k: v for k, v in result.artifacts.items() if k != "result"}
    assert reloaded.experiment_id == result.experiment_id
    assert reloaded.metrics == result.metrics
    assert reloaded.model == result.model
    assert reloaded.dataset == result.dataset
    assert reloaded.runtime == result.runtime


def test_list_model_backends_is_the_cli_choices_source_of_truth():
    backends = list_model_backends()
    assert backends == sorted(backends)
    assert {"ultralytics", "torchvision", "anomalib", "dinomaly", "moeclip", "mambaad"} <= set(backends)


def test_run_log_path_accumulates_across_separate_runs(tmp_path):
    import json

    dataset = FakeDataset(root="data/fake")
    log_path = tmp_path / "runs_log.jsonl"

    for experiment_id in ("exp-a", "exp-b"):
        run_experiment(
            experiment_id=experiment_id,
            dataset=dataset,
            model=FakeModel(),
            model_info=ModelInfo(name="fake-model", backend="fake-backend", task="detection"),
            runtime=RuntimeInfo(device="cpu", engine="python", precision="fp32", input_size=(640, 640)),
            train_config={},
            evaluator=FakeEvaluator(),
            run_log_path=str(log_path),
        )

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["experiment_id"] for row in rows] == ["exp-a", "exp-b"]
    assert all("provenance" in row for row in rows)


class _ExportingModel(FakeModel):
    def __init__(self, export_path):
        self.export_path = export_path

    def export(self, artifact, target, config=None):
        self.export_path.write_bytes(b"model-bytes")
        return ExportedArtifact(path=str(self.export_path), target=target)


class _FakeProfiler(BackendProfiler):
    engine = "fake-runtime"

    def profile(self, artifact, config):
        self.last_power_report = PowerReport(
            capability=PowerCapability("test", "fake", "board", False, "sensor unavailable"),
            status="unavailable",
            sample_count=0,
            duration_s=0.0,
            reason="sensor unavailable",
        )
        return {"latency_ms_p50": 2.5, "fps": 400.0}


def test_profile_metrics_merge_into_experiment_result(tmp_path):
    model = _ExportingModel(tmp_path / "model.onnx")
    profile_config = ProfileConfig(device="cpu", engine="fake-runtime", measured_runs=1)

    result = run_experiment(
        experiment_id="exp-profiled",
        dataset=FakeDataset(root="data/fake"),
        model=model,
        model_info=ModelInfo(name="fake-model", backend="fake-backend", task="detection"),
        runtime=RuntimeInfo(device="cpu", engine="python", precision="fp32", input_size=(640, 640)),
        artifact=Artifact(path="existing.pt", backend="fake-backend"),
        evaluator=FakeEvaluator(),
        profiler=_FakeProfiler(),
        profile_config=profile_config,
        export_target="onnx",
        output_dir=str(tmp_path),
    )

    assert result.metrics["map50"] == 1.0
    assert result.metrics["latency_ms_p50"] == 2.5
    assert result.metrics["model_size_mb"] > 0
    assert result.runtime.engine == "fake-runtime"
    assert result.artifacts["model_onnx"].endswith("model.onnx")
    assert (tmp_path / "exp-profiled" / "power.json").is_file()
