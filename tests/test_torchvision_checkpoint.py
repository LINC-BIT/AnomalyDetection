from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from fabric_defect_hub.models.torchvision.adapter import TorchvisionAdapter, _validate_checkpoint


def _checkpoint():
    return {
        "variant": "fasterrcnn_resnet50_fpn",
        "class_map": {"defect": 1},
        "state_dict": {"backbone.body.conv1.weight": object()},
    }


def test_checkpoint_validation_accepts_adapter_shape():
    class_map, variant, state_dict = _validate_checkpoint(_checkpoint(), Path("model.pt"), "fallback")
    assert class_map == {"defect": 1}
    assert variant == "fasterrcnn_resnet50_fpn"
    assert state_dict


@pytest.mark.parametrize(
    "checkpoint, message",
    [
        ({}, "missing required keys"),
        ({"state_dict": {}, "class_map": {}}, "invalid non-empty"),
        ({"state_dict": {"x": 1}, "class_map": {"defect": 0}}, "positive integer"),
        ({"state_dict": {"x": 1}, "class_map": {"defect": 1}, "variant": "not-real"}, "unsupported"),
    ],
)
def test_checkpoint_validation_has_actionable_errors(checkpoint, message):
    with pytest.raises(ValueError, match=message):
        _validate_checkpoint(checkpoint, Path("broken.pt"), "fallback")


def test_resume_checkpoint_rebuilds_with_backbone_weights_for_optimizer_shape_match(monkeypatch, tmp_path):
    """Regression test for a resume bug: rebuilding the model with
    `backbone_weights=False` makes torchvision's own `_validate_trainable_layers`
    force every backbone stage trainable (see `presets.build_model`'s "no
    effect without backbone weights" warning), producing a differently-shaped
    flat optimizer parameter group than the original `load_pretrained`-built
    model that saved the checkpoint -- `optimizer.load_state_dict` then always
    rejects it with "doesn't match the size of optimizer's group". Confirms
    `_load_resume_checkpoint` asks for `backbone_weights=True` and threads
    `trainable_backbone_layers` through, instead of silently changing which
    parameters are trainable.
    """

    captured = {}

    def fake_build_model(variant, num_classes, pretrained, backbone_weights, trainable_backbone_layers=None, offline=False):
        captured["backbone_weights"] = backbone_weights
        captured["trainable_backbone_layers"] = trainable_backbone_layers
        captured["pretrained"] = pretrained
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr("fabric_defect_hub.models.torchvision.adapter.build_model", fake_build_model)

    checkpoint_path = tmp_path / "last.pt"
    torch.save(
        {
            "variant": "fasterrcnn_resnet50_fpn",
            "class_map": {"defect": 1},
            "state_dict": {"weight": torch.zeros(1, 1), "bias": torch.zeros(1)},
            "optimizer_state": {"state": {}, "param_groups": [{"lr": 0.01, "params": [0]}]},
            "epoch": 3,
            "best_map": 0.5,
        },
        checkpoint_path,
    )

    adapter = TorchvisionAdapter(name="fasterrcnn_resnet50_fpn")
    resume_state = adapter._load_resume_checkpoint(
        checkpoint_path, device="cpu", trainable_backbone_layers=3, offline=False,
    )

    assert captured["backbone_weights"] is True
    assert captured["trainable_backbone_layers"] == 3
    assert resume_state["epoch"] == 3
    assert resume_state["best_map"] == 0.5
