import pytest
import torch
import numpy as np
from PIL import Image
from fabric_defect_hub.models.torchvision.presets import build_model
from fabric_defect_hub.models.torchvision.adapter import TorchvisionAdapter
from fabric_defect_hub.core.types import Sample, Annotations
from fabric_defect_hub.models.base import Artifact


def test_unetplusplus_compilation():
    model = build_model(
        name="unetplusplus_resnet34",
        num_classes=1,
        pretrained=False,
    )
    assert model is not None

    model.eval()
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_deeplabv3plus_compilation():
    model = build_model(
        name="deeplabv3plus_resnet50",
        num_classes=1,
        pretrained=False,
    )
    assert model is not None

    model.eval()
    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_segmentation_adapter_full_flow(tmp_path):
    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"

    Image.new("RGB", (64, 64), color="white").save(img_path)
    Image.new("L", (64, 64), color=255).save(mask_path)

    sample = Sample(
        id="sample1",
        image_path=str(img_path),
        task="segmentation",
        annotations=Annotations(anomaly_mask=str(mask_path))
    )
    samples = [sample]

    adapter = TorchvisionAdapter(name="unetplusplus_resnet34")
    adapter.load_scratch(class_names=["defect"])

    # 1. Test validate
    metrics = adapter.validate(samples)
    assert "miou" in metrics
    assert "dice" in metrics
    assert "pixel_f1" in metrics

    # 2. Test train (runs 1 epoch)
    config = {
        "train_samples": samples,
        "val_samples": samples,
        "epochs": 1,
        "batch_size": 1,
        "num_workers": 0,
        "run_dir": str(tmp_path),
        "pretrained": False,
    }
    artifact = adapter.train(config)
    assert isinstance(artifact, Artifact)

    # 3. Test predict
    predictions = adapter.predict(samples)
    assert len(predictions) == 1
    assert predictions[0].masks is not None
    assert len(predictions[0].masks) == 1

    # 4. Test export
    exported = adapter.export(artifact, target="exported_program", config={"input_size": (64, 64)})
    assert exported.path is not None
