"""The lifecycle contract: one run order, shared by every backend's pipeline.

`BasePipeline.run()` is the only place the eight steps of a config-driven run
are written down. These tests pin the order and the branching with a fake
config and a fake adapter — no framework, no dataset, no weights — and then
check that all six real backend pipelines are built on it rather than carrying
their own copy again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from fabric_defect_hub.core.pipeline import AnomalyPipeline, BasePipeline, RunResult
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter, ModelCapabilities


# --------------------------------------------------------------------------- #
# Minimal stand-ins mirroring the real config/adapter shapes
# --------------------------------------------------------------------------- #
@dataclass
class _Stage:
    enabled: bool = False


@dataclass
class _ExportSpec:
    enabled: bool = False
    formats: list[str] = field(default_factory=list)


@dataclass
class _Checkpoint:
    registry_dir: str = "/tmp/registry"


@dataclass
class _FakeConfig:
    train: _Stage = field(default_factory=_Stage)
    val: _Stage = field(default_factory=_Stage)
    export: _ExportSpec = field(default_factory=_ExportSpec)
    checkpoint: _Checkpoint = field(default_factory=_Checkpoint)
    validated: bool = False

    def validate(self) -> None:
        self.validated = True


class _FakeAdapter(ModelAdapter):
    name = "pipeline-fake"
    backend = "pipeline-fake"

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def capabilities(self):
        return ModelCapabilities(tasks=("detection",), prediction_fields=("boxes",))

    def train(self, config):
        self.calls.append("train")
        return Artifact(path="trained.pt", backend=self.backend)

    def register_trained_model(self, artifact, registry_dir):
        self.calls.append("register")
        return Artifact(path="registered.pt", backend=self.backend)

    def load_trained_model(self, path):
        self.calls.append("load")
        return Artifact(path=path, backend=self.backend)

    def predict(self, samples, artifact=None, output_dir=None, config=None):
        self.calls.append("predict")
        return []

    def export(self, artifact, target, config=None):
        self.calls.append(f"export:{target}")
        return ExportedArtifact(path=f"m.{target}", target=target, metadata={"config": config})


class _FakePipeline(BasePipeline):
    def __init__(self, config, existing: Artifact | None = None):
        super().__init__(config)
        self.adapter = _FakeAdapter()
        self.existing = existing
        self.prepared = False

    def build_adapter(self):
        return self.adapter

    def prepare(self):
        self.prepared = True

    def build_train_config(self):
        assert self.prepared, "prepare() must run before build_train_config()"
        return {"epochs": 1}

    def load_existing_artifact(self, adapter):
        return adapter.load_trained_model(self.existing.path) if self.existing else None

    def evaluate(self, adapter, artifact):
        adapter.calls.append("evaluate")
        return {"map50": 1.0}


# --------------------------------------------------------------------------- #
# The lifecycle
# --------------------------------------------------------------------------- #
def test_config_is_validated_before_anything_runs():
    config = _FakeConfig()
    _FakePipeline(config).run()
    assert config.validated


def test_disabled_stages_do_nothing():
    pipeline = _FakePipeline(_FakeConfig())

    result = pipeline.run()

    assert pipeline.adapter.calls == []
    assert result.trained_artifact is None
    assert result.metrics == {}
    assert result.exports == []


def test_full_run_executes_the_stages_in_order():
    config = _FakeConfig(
        train=_Stage(enabled=True),
        val=_Stage(enabled=True),
        export=_ExportSpec(enabled=True, formats=["onnx", "torchscript"]),
    )
    pipeline = _FakePipeline(config)

    result = pipeline.run()

    assert pipeline.adapter.calls == [
        "train",
        "register",
        "evaluate",
        "export:onnx",
        "export:torchscript",
    ]
    assert result.trained_artifact.path == "trained.pt"
    assert result.registered_artifact.path == "registered.pt"
    assert result.metrics == {"map50": 1.0}
    assert [e.target for e in result.exports] == ["onnx", "torchscript"]


def test_registered_artifact_is_what_later_stages_act_on():
    config = _FakeConfig(train=_Stage(enabled=True), export=_ExportSpec(enabled=True, formats=["onnx"]))

    result = _FakePipeline(config).run()

    # The stable registered copy wins over the transient training output.
    assert result.active_artifact is result.registered_artifact


def test_without_training_an_existing_checkpoint_can_still_be_evaluated():
    config = _FakeConfig(val=_Stage(enabled=True))
    pipeline = _FakePipeline(config, existing=Artifact(path="prev.pt", backend="pipeline-fake"))

    result = pipeline.run()

    assert pipeline.adapter.calls == ["load", "evaluate"]
    assert result.active_artifact.path == "prev.pt"


@pytest.mark.parametrize(
    "config",
    [
        _FakeConfig(val=_Stage(enabled=True)),
        _FakeConfig(export=_ExportSpec(enabled=True, formats=["onnx"])),
    ],
    ids=["validate", "export"],
)
def test_a_run_with_nothing_to_act_on_fails_loudly(config):
    """Training off, no checkpoint configured, but the config asks to validate
    or export. This used to return an empty result in silence — for the four
    anomaly backends it was unreachable in *any* config, because none of them
    even had a `model.weights` field to set.
    """

    pipeline = _FakePipeline(config)

    with pytest.raises(ValueError, match="Nothing to validate or export"):
        pipeline.run()

    assert pipeline.adapter.calls == []


def test_a_run_with_every_stage_disabled_is_still_a_valid_no_op():
    # Distinct from the case above: nothing was *asked* for, so nothing is
    # missing. Used by "just load and inspect the config" style runs.
    pipeline = _FakePipeline(_FakeConfig())

    result = pipeline.run()

    assert pipeline.adapter.calls == []
    assert result.metrics == {}


def test_export_config_hook_reaches_the_adapter():
    class _WithExportConfig(_FakePipeline):
        def export_config(self, fmt):
            return {"opset": 17, "format": fmt}

    config = _FakeConfig(train=_Stage(enabled=True), export=_ExportSpec(enabled=True, formats=["onnx"]))
    result = _WithExportConfig(config).run()

    assert result.exports[0].metadata["config"] == {"opset": 17, "format": "onnx"}


# --------------------------------------------------------------------------- #
# The shared anomaly evaluation
# --------------------------------------------------------------------------- #
class _FakeAnomalyPipeline(AnomalyPipeline):
    def __init__(self, config, samples):
        super().__init__(config)
        self.adapter = _FakeAdapter()
        self._samples = samples

    def build_adapter(self):
        return self.adapter

    def prepare(self):
        self.test_samples = self._samples

    def build_train_config(self):
        return {}


def test_anomaly_pipeline_scores_nothing_when_there_are_no_samples():
    # The `data_root` / `datamodule_kwargs` modes: an on-disk folder with no
    # `Sample` objects for the evaluator to score against.
    config = _FakeConfig(train=_Stage(enabled=True), val=_Stage(enabled=True))
    pipeline = _FakeAnomalyPipeline(config, samples=None)

    result = pipeline.run()

    assert result.metrics == {}
    assert "predict" not in pipeline.adapter.calls


# --------------------------------------------------------------------------- #
# Every real backend is built on the shared lifecycle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "module_path, pipeline_name, result_name",
    [
        ("fabric_defect_hub.models.ultralytics.pipeline", "UltralyticsPipeline", "UltralyticsRunResult"),
        ("fabric_defect_hub.models.torchvision.pipeline", "TorchvisionPipeline", "TorchvisionRunResult"),
        ("fabric_defect_hub.models.anomalib.pipeline", "AnomalibPipeline", "AnomalibRunResult"),
        ("fabric_defect_hub.models.dinomaly.pipeline", "DinomalyPipeline", "DinomalyRunResult"),
        ("fabric_defect_hub.models.mambaad.pipeline", "MambaADPipeline", "MambaADRunResult"),
        ("fabric_defect_hub.models.moeclip.pipeline", "MoECLIPPipeline", "MoECLIPRunResult"),
    ],
)
def test_backend_pipeline_uses_the_shared_lifecycle(module_path, pipeline_name, result_name):
    module = pytest.importorskip(module_path)
    pipeline_cls = getattr(module, pipeline_name)

    assert issubclass(pipeline_cls, BasePipeline)
    # It must not re-implement `run()`: the whole point is one lifecycle order.
    assert "run" not in pipeline_cls.__dict__
    # The historical per-backend result name is now the shared type.
    assert getattr(module, result_name) is RunResult
    # And the module-level entry points callers use are still there.
    assert callable(module.run_from_config) and callable(module.run_from_yaml)


ANOMALY_BACKENDS = [
    ("fabric_defect_hub.models.anomalib", "AnomalibConfig"),
    ("fabric_defect_hub.models.dinomaly", "DinomalyConfig"),
    ("fabric_defect_hub.models.mambaad", "MambaADConfig"),
    ("fabric_defect_hub.models.moeclip", "MoECLIPConfig"),
]


@pytest.mark.parametrize("package, config_name", ANOMALY_BACKENDS)
def test_every_anomaly_backend_can_name_an_existing_checkpoint(package, config_name):
    """All four adapters have always had `load_trained_model`, but none of
    their configs had a `model.weights` field to reach it — so "evaluate a
    checkpoint I trained yesterday" was unexpressible, and a config that tried
    produced an empty result in silence.
    """

    config_module = pytest.importorskip(f"{package}.config")
    spec = getattr(config_module, "ModelSpec")

    assert "weights" in {f.name for f in __import__("dataclasses").fields(spec)}
    assert spec().weights is None  # opt-in, never a default


@pytest.mark.parametrize("package, config_name", ANOMALY_BACKENDS)
def test_pipeline_loads_the_configured_checkpoint_when_training_is_off(package, config_name):
    pipeline_module = pytest.importorskip(f"{package}.pipeline")
    pipeline_cls = next(
        obj for obj in vars(pipeline_module).values()
        if isinstance(obj, type) and issubclass(obj, AnomalyPipeline) and obj is not AnomalyPipeline
    )

    # Every one of the four must implement the hook — inheriting the base's
    # `None` default is what the silent no-op looked like.
    assert "load_existing_artifact" in pipeline_cls.__dict__


def test_anomalib_refuses_a_bare_checkpoint_without_an_explicit_opt_in():
    """Anomalib checkpoints are Lightning pickles. `load_trained_model` has
    always required `allow_unsafe_checkpoint=True` for a bare path; the config
    surfaces that opt-in rather than quietly passing it.
    """

    config_module = pytest.importorskip("fabric_defect_hub.models.anomalib.config")

    with pytest.raises(ValueError, match="allow_unsafe_checkpoint"):
        config_module.ModelSpec(name="PatchCore", weights="/tmp/model.ckpt").validate()

    # With the opt-in, it validates.
    config_module.ModelSpec(
        name="PatchCore", weights="/tmp/model.ckpt", allow_unsafe_checkpoint=True
    ).validate()


@pytest.mark.parametrize(
    "module_path",
    [
        "fabric_defect_hub.models.anomalib.pipeline",
        "fabric_defect_hub.models.dinomaly.pipeline",
        "fabric_defect_hub.models.mambaad.pipeline",
        "fabric_defect_hub.models.moeclip.pipeline",
    ],
)
def test_anomaly_backends_share_one_evaluate(module_path):
    module = pytest.importorskip(module_path)
    pipeline_cls = next(
        obj for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, AnomalyPipeline) and obj is not AnomalyPipeline
    )

    # None of the four may carry its own copy of predict-then-score.
    assert "evaluate" not in pipeline_cls.__dict__
    assert pipeline_cls.evaluate is AnomalyPipeline.evaluate
