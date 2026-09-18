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


def test_patched_methods_carry_no_import_statements():
    """`torch.jit.script` — the export `_resolution_sweep` drives for these
    variants — compiles every method's own code object and refuses an `import`
    inside it:

        UnsupportedNodeError: import statements aren't supported

    which cost DETR, Cascade R-CNN and DeepLabV3+ their resolution slope. Any
    name a patched method touches has to be a module global in `presets.py`;
    the module-level factory functions are never scripted and keep their own
    local imports.

    Checked from the source with `ast`, so this holds without building a
    model, downloading weights or running TorchScript.
    """

    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src" / "fabric_defect_hub" / "models" / "torchvision" / "presets.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    offenders = []
    for cls in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
        for fn in [node for node in ast.walk(cls) if isinstance(node, ast.FunctionDef)]:
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    offenders.append(f"{fn.name} (line {fn.lineno}): {ast.unparse(node)}")

    assert not offenders, (
        "an import inside a class method breaks the TorchScript export the "
        "resolution sweep and profiling use: " + "; ".join(offenders)
    )


def test_detr_matcher_survives_a_degenerate_box_instead_of_killing_the_epoch():
    """A zero-width/height box makes `box_iou`/`generalized_box_iou` divide by a
    zero area and return NaN, and scipy rejects the *whole* cost matrix
    ("matrix contains invalid numeric entries") — which killed a 300-epoch
    DETR retrain at epoch 57. ZJU-Leaper's 1-pixel-tall sliver annotations
    reach that state routinely, so the matcher has to cost those pairs out
    rather than raise."""

    torch = pytest.importorskip("torch")
    from fabric_defect_hub.models.torchvision.presets import HungarianMatcher

    # Two queries: one with a real box, one collapsed to zero height.
    outputs = {
        "pred_logits": torch.tensor([[[2.0, -1.0], [-1.0, 2.0]]]),          # (B, Q, C)
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.0]]]),
    }
    targets = [{"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.0]])}]

    src_ind, tgt_ind = HungarianMatcher()(outputs, targets)[0]

    assert len(src_ind) == len(tgt_ind) == 1
    assert int(tgt_ind[0]) == 0
    assert 0 <= int(src_ind[0]) < 2


def test_detr_matcher_still_matches_the_obvious_query():
    """The sanitizing must not disturb a normal assignment."""

    torch = pytest.importorskip("torch")
    from fabric_defect_hub.models.torchvision.presets import HungarianMatcher

    outputs = {
        "pred_logits": torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]]),
        "pred_boxes": torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]]),
    }
    targets = [{"labels": torch.tensor([1]), "boxes": torch.tensor([[0.9, 0.9, 0.1, 0.1]])}]

    src_ind, tgt_ind = HungarianMatcher()(outputs, targets)[0]

    assert int(src_ind[0]) == 1  # the query whose box overlaps the target


def test_a_sample_with_no_ground_truth_gets_an_empty_assignment():
    torch = pytest.importorskip("torch")
    from fabric_defect_hub.models.torchvision.presets import HungarianMatcher

    outputs = {
        "pred_logits": torch.tensor([[[1.0, 0.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
    }
    targets = [{"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty((0, 4))}]

    src_ind, tgt_ind = HungarianMatcher()(outputs, targets)[0]

    assert len(src_ind) == 0 and len(tgt_ind) == 0


def test_backbone_lr_splits_the_optimizer_into_two_groups():
    """DETR's recipe is 1e-4 on the transformer/heads and 1e-5 on the
    pretrained ResNet backbone; one shared 1e-4 is the usual way a fine-tune
    of a pretrained detector stalls near zero mAP. `param_groups[0]` stays the
    head so the epoch log keeps reporting the headline learning rate."""

    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from fabric_defect_hub.models.torchvision.engine import build_optimizer

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

    model = Tiny()
    optimizer = build_optimizer(model, "adamw", lr=1e-4, momentum=0.9, weight_decay=0.0, backbone_lr=1e-5)

    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 1e-5]
    head_params = set(optimizer.param_groups[0]["params"])
    backbone_params = set(optimizer.param_groups[1]["params"])
    assert head_params and backbone_params
    assert not (head_params & backbone_params)
    assert set(model.backbone.parameters()) == backbone_params
    assert set(model.head.parameters()) == head_params


def test_without_backbone_lr_every_parameter_shares_one_group():
    """Resuming an in-flight run depends on this: PyTorch refuses to load a
    one-group optimizer state into a two-group optimizer, so the default has
    to stay exactly one group."""

    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from fabric_defect_hub.models.torchvision.engine import build_optimizer

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

    model = Tiny()
    assert len(build_optimizer(model, "adamw", 1e-4, 0.9, 0.0).param_groups) == 1
    # An explicit equal rate is the same optimizer, not a pointless split.
    assert len(build_optimizer(model, "adamw", 1e-4, 0.9, 0.0, backbone_lr=1e-4).param_groups) == 1
