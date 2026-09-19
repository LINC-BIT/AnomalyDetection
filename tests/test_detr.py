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


def test_detr_losses_use_weight_dict():
    """`SetCriterion` must return `weight_dict`-scaled losses.

    `engine.train_one_epoch` only sums the dict the criterion returns, so
    returning the raw components silently trained `loss_bbox`/`loss_giou` at
    weight 1 instead of DETR's 5/2. With 100 queries and ~1.7 objects per
    ZJU-Leaper image that let the classification term dominate and the decoder
    collapse onto one constant box for every query and image — the failure
    behind the published `detr_resnet50.pt` (mAP 0.005, constant prediction).
    """

    from fabric_defect_hub.models.torchvision.presets import HungarianMatcher, SetCriterion

    torch.manual_seed(0)
    outputs = {
        "pred_logits": torch.randn(2, 5, 3),
        "pred_boxes": torch.rand(2, 5, 4),
    }
    targets = [
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])},
        {
            "labels": torch.tensor([1, 2]),
            "boxes": torch.tensor([[0.2, 0.3, 0.1, 0.1], [0.7, 0.7, 0.3, 0.3]]),
        },
    ]

    def criterion(weight_dict):
        return SetCriterion(
            num_classes=3,
            matcher=HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0),
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=["labels", "boxes"],
        )

    reference = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    weighted = criterion(reference)(outputs, targets)
    raw = criterion({key: 1.0 for key in reference})(outputs, targets)

    for key, weight in reference.items():
        assert key in weighted
        assert float(weighted[key]) == pytest.approx(weight * float(raw[key]), rel=1e-5)


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
