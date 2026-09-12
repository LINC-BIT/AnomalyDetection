"""Framework-free tests for torchvision config and pipeline orchestration."""

import pytest

from fabric_defect_hub.core.types import Annotations, Sample
from fabric_defect_hub.models.base import Artifact, ExportedArtifact
from fabric_defect_hub.models.torchvision.config import TorchvisionConfig
from fabric_defect_hub.models.torchvision.pipeline import run_from_config


def _config():
    return TorchvisionConfig.from_dict(
        {
            "model": {"variant": "fasterrcnn_resnet50_fpn", "pretrained": False, "offline": True},
            "data": {
                "dataset": "zju-leaper",
                "dataset_root": "/dataset",
                "class_names": ["defect"],
                "train_selection": {"split": "train"},
                "val_selection": {"split": "test"},
            },
            "train": {"epochs": 1, "num_workers": 0},
            "export": {"enabled": True, "formats": ["exported_program"]},
        }
    )


class _FakeAdapter:
    def __init__(self, name):
        self.name = name

    def train(self, config):
        assert config["offline"] is True
        assert config["train_samples"]
        return Artifact("trained.pt", "torchvision", {"variant": self.name})

    def register_trained_model(self, artifact, registry_dir):
        return Artifact("registered.pt", "torchvision", artifact.metadata)

    def validate(self, samples, artifact, config):
        return {"map": 0.5}

    def export(self, artifact, target, config=None):
        return ExportedArtifact(f"model.{target}", target)


def test_config_resolves_safe_worker_and_offline_settings():
    config = _config()
    resolved = config.resolved_train_kwargs()
    assert resolved["num_workers"] == 0
    assert config.model.offline is True


def test_pipeline_orchestrates_without_importing_torch(monkeypatch):
    sample = Sample("sample", "image.jpg", "detection", Annotations())
    monkeypatch.setattr(
        "fabric_defect_hub.models.torchvision.pipeline._load_split_samples",
        lambda config, selection: [sample],
    )
    result = run_from_config(_config(), adapter_factory=_FakeAdapter)
    assert result.metrics == {"map": 0.5}
    assert result.registered_artifact.path == "registered.pt"
    assert result.exports[0].target == "exported_program"


def test_pipeline_passes_device_seed_and_resume_through_to_train(monkeypatch):
    """`device`/`seed` are excluded from `resolved_train_kwargs()` (pipeline-
    level, not native torchvision train() kwargs) and must be re-added
    explicitly in pipeline.py, the same way `weights`/`min_size`/... are —
    previously they were silently dropped and never reached `adapter.train()`.
    """

    captured: dict = {}

    class _CapturingAdapter(_FakeAdapter):
        def train(self, config):
            captured.update(config)
            return super().train(config)

    sample = Sample("sample", "image.jpg", "detection", Annotations())
    monkeypatch.setattr(
        "fabric_defect_hub.models.torchvision.pipeline._load_split_samples",
        lambda config, selection: [sample],
    )
    config_dict = {
        "model": {"variant": "fasterrcnn_resnet50_fpn", "pretrained": False, "offline": True},
        "data": {
            "dataset": "zju-leaper",
            "dataset_root": "/dataset",
            "class_names": ["defect"],
            "train_selection": {"split": "train"},
            "val_selection": {"split": "test"},
        },
        "train": {"epochs": 1, "num_workers": 0, "device": "cpu", "seed": 7, "resume": True},
    }
    run_from_config(TorchvisionConfig.from_dict(config_dict), adapter_factory=_CapturingAdapter)
    assert captured["device"] == "cpu"
    assert captured["seed"] == 7
    assert captured["resume"] is True


def test_pipeline_omits_resume_key_when_not_requested(monkeypatch):
    captured: dict = {}

    class _CapturingAdapter(_FakeAdapter):
        def train(self, config):
            captured.update(config)
            return super().train(config)

    sample = Sample("sample", "image.jpg", "detection", Annotations())
    monkeypatch.setattr(
        "fabric_defect_hub.models.torchvision.pipeline._load_split_samples",
        lambda config, selection: [sample],
    )
    run_from_config(_config(), adapter_factory=_CapturingAdapter)
    assert "resume" not in captured


def test_train_spec_accepts_amp_and_resume():
    config = TorchvisionConfig.from_dict(
        {
            "model": {"variant": "fasterrcnn_resnet50_fpn"},
            "data": {"dataset_root": "/dataset"},
            "train": {"amp": True, "resume": True},
        }
    )
    assert config.train.amp is True
    assert config.train.resume is True
    assert config.resolved_train_kwargs()["amp"] is True


def test_val_spec_rejects_removed_score_threshold_key():
    with pytest.raises(ValueError, match="unknown keys"):
        TorchvisionConfig.from_dict(
            {
                "model": {"variant": "fasterrcnn_resnet50_fpn"},
                "data": {"dataset_root": "/dataset"},
                "val": {"score_threshold": 0.3},
            }
        )
