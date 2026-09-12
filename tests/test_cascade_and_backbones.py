import pytest
import torch
from fabric_defect_hub.models.torchvision.presets import build_model


def test_custom_backbones():
    # Build VGG-16 Faster R-CNN model
    model = build_model(
        name="fasterrcnn_vgg16_fpn",
        num_classes=3,
        pretrained=False,
        backbone_weights=False,
    )
    assert model is not None

    # Trace/forward
    model.eval()
    x = [torch.rand(3, 256, 256)]
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, list)
    assert len(out) == 1

    # Build ShuffleNet V2 Faster R-CNN model
    model_sn = build_model(
        name="fasterrcnn_shufflenet_v2_x1_0_fpn",
        num_classes=2,
        pretrained=False,
        backbone_weights=False,
    )
    assert model_sn is not None

    model_sn.eval()
    with torch.no_grad():
        out_sn = model_sn(x)
    assert isinstance(out_sn, list)
    assert len(out_sn) == 1


def test_cascade_rcnn_forward():
    # Build ResNet50 Cascade R-CNN model
    model = build_model(
        name="cascadercnn_resnet50_fpn",
        num_classes=3,
        pretrained=False,
        backbone_weights=False,
    )
    assert model is not None
    assert hasattr(model.roi_heads, "box_heads")
    assert len(model.roi_heads.box_heads) == 3

    # Forward in eval mode
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

    # Check that it contains cascade stage losses
    assert "loss_classifier_stage1" in losses
    assert "loss_box_reg_stage1" in losses
    assert "loss_classifier_stage2" in losses
    assert "loss_box_reg_stage2" in losses
    assert "loss_classifier_stage3" in losses
    assert "loss_box_reg_stage3" in losses
