"""Tests for `benchmark.py`'s cross-backend orchestration + leaderboard.

Uses its own uniquely-named fake dataset/model registrations
("*-bench" suffix) rather than reusing `test_loader.py`'s "fake-fabric"/
"fake-backend" — `core.registry`'s `register_dataset`/`register_model`
raise `ValueError` on a duplicate name and the registry is a module-level
dict with no reset between tests, so pytest collecting both files into one
process would collide on identical names.
"""

from fabric_defect_hub.benchmark import BenchmarkConfig, BenchmarkRun, leaderboard, run_benchmark
from fabric_defect_hub.core.registry import register_dataset, register_model
from fabric_defect_hub.core.types import Annotations, ModelInfo, Prediction, RuntimeInfo, Sample
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.evaluation.base import Evaluator
from fabric_defect_hub.loader import load_dataset
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter, ModelCapabilities


@register_dataset("fake-fabric-bench")
class FakeBenchDataset(DatasetAdapter):
    name = "fake-fabric-bench"

    def load_samples(self) -> list[Sample]:
        return [
            Sample(
                id="sample-0001", image_path=f"{self.root}/0001.jpg", task="detection",
                annotations=Annotations(boxes=[[10, 10, 50, 50]], labels=["defect"]),
            )
        ]


def _make_fake_model_cls(backend_name: str, score: float):
    @register_model(backend_name)
    class _FakeBenchModel(ModelAdapter):
        name = f"{backend_name}-model"
        backend = backend_name

        def capabilities(self):
            return ModelCapabilities(
                tasks=("detection",),
                prediction_fields=("boxes", "labels", "scores"),
                export_targets=("onnx",),
            )

        def train(self, config):
            return Artifact(path=f"{backend_name}.pt", backend=self.backend)

        def predict(self, samples, artifact=None, output_dir=None, config=None):
            return [
                Prediction(sample_id=s.id, boxes=[[10, 10, 50, 50]], labels=["defect"], scores=[score])
                for s in samples
            ]

        def export(self, artifact, target, config=None):
            return ExportedArtifact(path=f"{backend_name}.{target}", target=target)

    return _FakeBenchModel


_make_fake_model_cls("fake-backend-bench-a", score=0.9)
_make_fake_model_cls("fake-backend-bench-b", score=0.3)


class _ScoreEvaluator(Evaluator):
    task = "detection"

    def evaluate(self, samples, predictions):
        return {"quality": sum(p.scores[0] for p in predictions) / len(predictions)}


def _runtime() -> RuntimeInfo:
    return RuntimeInfo(device="cpu", engine="python", precision="fp32", input_size=(640, 640))


def _two_runs() -> list[BenchmarkRun]:
    return [
        BenchmarkRun(
            experiment_id="exp-a", model_backend="fake-backend-bench-a", model_name="model-a",
            model_info=ModelInfo(name="model-a", backend="fake-backend-bench-a", task="detection"),
            runtime=_runtime(), dataset=FakeBenchDataset(root="data/fake"),
            train_config={}, evaluator=_ScoreEvaluator(),
        ),
        BenchmarkRun(
            experiment_id="exp-b", model_backend="fake-backend-bench-b", model_name="model-b",
            model_info=ModelInfo(name="model-b", backend="fake-backend-bench-b", task="detection"),
            runtime=_runtime(), dataset=FakeBenchDataset(root="data/fake"),
            train_config={}, evaluator=_ScoreEvaluator(),
        ),
    ]


def test_run_benchmark_executes_all_runs_in_order():
    results = run_benchmark(_two_runs())

    assert [r.experiment_id for r in results] == ["exp-a", "exp-b"]
    assert results[0].metrics["quality"] == 0.9
    assert results[1].metrics["quality"] == 0.3


def test_leaderboard_sorts_descending_by_default():
    results = run_benchmark(_two_runs())
    board = leaderboard(results, metric="quality")
    assert [r.experiment_id for r in board] == ["exp-a", "exp-b"]


def test_leaderboard_ascending():
    results = run_benchmark(_two_runs())
    board = leaderboard(results, metric="quality", descending=False)
    assert [r.experiment_id for r in board] == ["exp-b", "exp-a"]


def test_leaderboard_drops_results_missing_the_metric():
    results = run_benchmark(_two_runs())
    assert leaderboard(results, metric="nonexistent_metric") == []


def test_leaderboard_drops_only_the_missing_one():
    results = run_benchmark(_two_runs())
    results[1].metrics.pop("quality")
    board = leaderboard(results, metric="quality")
    assert [r.experiment_id for r in board] == ["exp-a"]


def test_leaderboard_drops_nan_and_infinite_metrics():
    results = run_benchmark(_two_runs())
    results[0].metrics["quality"] = float("nan")
    results[1].metrics["quality"] = float("inf")
    assert leaderboard(results, metric="quality") == []


def test_resolve_dataset_returns_prebuilt_dataset_as_is():
    dataset = FakeBenchDataset(root="data/fake")
    run = BenchmarkRun(
        experiment_id="x", model_backend="fake-backend-bench-a", model_name="m",
        model_info=ModelInfo(name="m", backend="fake-backend-bench-a", task="detection"),
        runtime=_runtime(), dataset=dataset,
    )
    assert run.resolve_dataset() is dataset


def test_resolve_dataset_loads_by_name_and_root():
    run = BenchmarkRun(
        experiment_id="x", model_backend="fake-backend-bench-a", model_name="m",
        model_info=ModelInfo(name="m", backend="fake-backend-bench-a", task="detection"),
        runtime=_runtime(), dataset_name="fake-fabric-bench", dataset_root="data/fake",
    )
    resolved = run.resolve_dataset()
    assert isinstance(resolved, FakeBenchDataset)
    assert resolved.root == "data/fake"


def test_resolve_dataset_raises_without_dataset_or_name_and_root():
    run = BenchmarkRun(
        experiment_id="x", model_backend="fake-backend-bench-a", model_name="m",
        model_info=ModelInfo(name="m", backend="fake-backend-bench-a", task="detection"),
        runtime=_runtime(),
    )
    try:
        run.resolve_dataset()
        assert False, "expected ValueError"
    except ValueError as e:
        assert "'x'" in str(e)  # the run's own experiment_id, for easy identification


def test_run_benchmark_persists_output(tmp_path):
    results = run_benchmark(_two_runs(), output_dir=str(tmp_path))
    for result in results:
        assert (tmp_path / result.experiment_id / "predictions.json").exists()
        assert (tmp_path / result.experiment_id / "result.json").exists()


def test_run_benchmark_appends_every_run_to_the_same_log(tmp_path):
    import json

    log_path = tmp_path / "runs_log.jsonl"
    results = run_benchmark(_two_runs(), run_log_path=str(log_path))

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["experiment_id"] for row in rows] == [result.experiment_id for result in results]


def test_benchmark_config_builds_and_runs_framework_free(tmp_path):
    config = BenchmarkConfig.from_dict(
        {
            "output_dir": str(tmp_path / "results"),
            "report_path": str(tmp_path / "leaderboard.md"),
            "runs": [
                {
                    "experiment_id": "configured",
                    "dataset": {"name": "fake-fabric-bench", "root": "data/fake"},
                    "model": {
                        "backend": "fake-backend-bench-a",
                        "name": "model-a",
                        "task": "detection",
                    },
                    "runtime": {"device": "cpu", "engine": "python"},
                    "train": {},
                    "evaluator": False,
                }
            ],
        }
    )

    results = config.run()

    assert [result.experiment_id for result in results] == ["configured"]
    assert (tmp_path / "results" / "configured" / "result.json").exists()
    assert (tmp_path / "leaderboard.md").read_text().startswith("| experiment_id |")


def test_benchmark_config_wires_run_log_path(tmp_path):
    import json

    config = BenchmarkConfig.from_dict(
        {
            "output_dir": str(tmp_path / "results"),
            "run_log_path": str(tmp_path / "runs_log.jsonl"),
            "runs": [
                {
                    "experiment_id": "configured",
                    "dataset": {"name": "fake-fabric-bench", "root": "data/fake"},
                    "model": {
                        "backend": "fake-backend-bench-a",
                        "name": "model-a",
                        "task": "detection",
                    },
                    "runtime": {"device": "cpu", "engine": "python"},
                    "train": {},
                    "evaluator": False,
                }
            ],
        }
    )

    config.run()

    rows = [json.loads(line) for line in (tmp_path / "runs_log.jsonl").read_text().splitlines()]
    assert [row["experiment_id"] for row in rows] == ["configured"]
