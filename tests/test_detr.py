import pytest
import torch
from fabric_defect_hub.models.torchvision.presets import build_model


def test_detr_resnet50_compilation():
    model = build_model(
        name="detr_resnet50",
        num_classes=3,
        pretrained=False,
        backbone_weights=False,
    )
    assert model is not None

    # Trace/forward in eval mode
    model.eval()
    x = [torch.rand(3, 256, 256)]
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, list)
    assert len(out) == 1
    assert "boxes" in out[0]
    assert "labels" in out[0]
    assert "scores" in out[0]

    # Forward in training mode
    model.train()
    targets = [{
        "boxes": torch.tensor([[10.0, 15.0, 100.0, 120.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }]
    losses = model(x, targets)
    assert isinstance(losses, dict)
    assert "loss_ce" in losses
    assert "loss_bbox" in losses
    assert "loss_giou" in losses


def test_detr_vgg16_compilation():
    model = build_model(
        name="detr_vgg16",
        num_classes=2,
        pretrained=False,
        backbone_weights=False,
    )
    assert model is not None

    # Forward in eval mode
    model.eval()
    x = [torch.rand(3, 256, 256)]
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, list)
    assert len(out) == 1


def test_detr_shufflenet_v2_compilation():
    model = build_model(
        name="detr_shufflenet_v2_x1_0",
        num_classes=2,
        pretrained=False,
        backbone_weights=False,
    )
    assert model is not None

    # Forward in eval mode
    model.eval()
    x = [torch.rand(3, 256, 256)]
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, list)
    assert len(out) == 1
