import importlib.util
from pathlib import Path

import pytest


def _load_tool(name: str):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_coco_converter_emits_detection_samples():
    tool = _load_tool("convert_annotations")
    samples = tool.coco_detection_samples(
        {
            "images": [{"id": 7, "file_name": "fabric.jpg", "width": 100, "height": 80}],
            "categories": [{"id": 3, "name": "hole"}],
            "annotations": [{"image_id": 7, "category_id": 3, "bbox": [10, 20, 30, 40]}],
        },
        Path("images"),
    )
    assert samples[0].image_path == "images/fabric.jpg"
    assert samples[0].annotations.boxes == [[10.0, 20.0, 40.0, 60.0]]
    assert samples[0].annotations.labels == ["hole"]


def test_export_model_parses_input_size():
    tool = _load_tool("export_model")
    assert tool._parse_input_size("640x640") == (640, 640)
    assert tool._parse_input_size("512X384") == (512, 384)
    with pytest.raises(ValueError, match="HxW"):
        tool._parse_input_size("not-a-size")


def test_export_model_samples_from_dir_builds_one_sample_per_image(tmp_path):
    tool = _load_tool("export_model")
    (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "notes.txt").write_text("not an image")

    samples = tool._samples_from_dir(str(tmp_path))

    assert sorted(s.id for s in samples) == ["a", "b"]
    assert all(s.task == "anomaly" for s in samples)


def test_export_model_samples_from_dir_rejects_empty_directory(tmp_path):
    tool = _load_tool("export_model")
    with pytest.raises(ValueError, match="no calibration images"):
        tool._samples_from_dir(str(tmp_path))


def test_export_model_requires_quantize_output(tmp_path, capsys):
    tool = _load_tool("export_model")
    artifact_path = tmp_path / "weights.pt"
    artifact_path.write_bytes(b"fake")

    class _FakeModel:
        def export(self, artifact, target, config=None):
            from fabric_defect_hub.models.base import ExportedArtifact

            return ExportedArtifact(path=str(tmp_path / "model.onnx"), target="onnx")

    tool.load_model = lambda backend, model: _FakeModel()
    (tmp_path / "model.onnx").write_bytes(b"not a real onnx file")

    with pytest.raises(SystemExit):
        tool.main(
            [
                "--backend", "torchvision",
                "--model", "fasterrcnn_resnet50_fpn",
                "--artifact", str(artifact_path),
                "--target", "onnx",
                "--quantize-level", "fp16",
            ]
        )
    assert "--quantize-output" in capsys.readouterr().err


def test_export_model_quantize_requires_onnx_target(tmp_path, capsys):
    tool = _load_tool("export_model")
    artifact_path = tmp_path / "weights.pt"
    artifact_path.write_bytes(b"fake")

    class _FakeModel:
        def export(self, artifact, target, config=None):
            from fabric_defect_hub.models.base import ExportedArtifact

            return ExportedArtifact(path=str(tmp_path / "model.ts"), target="torchscript")

    tool.load_model = lambda backend, model: _FakeModel()
    (tmp_path / "model.ts").write_bytes(b"fake")

    with pytest.raises(SystemExit):
        tool.main(
            [
                "--backend", "torchvision",
                "--model", "fasterrcnn_resnet50_fpn",
                "--artifact", str(artifact_path),
                "--target", "torchscript",
                "--quantize-level", "fp16",
                "--quantize-output", str(tmp_path / "out.onnx"),
            ]
        )
    assert "--target onnx" in capsys.readouterr().err
