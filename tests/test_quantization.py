"""Tests for `quantization/onnx_quant.py`: real end-to-end runs of every
quantization level against a tiny real ONNX model — built directly via
`onnx.helper` (no torch/model-training needed, mirroring `test_profiling.py`'s
"a tiny real export, not a mock" philosophy) so these stay fast while still
exercising the actual onnxruntime/onnxconverter-common quantization APIs,
not a fake of them.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from fabric_defect_hub.core.types import Annotations, Sample
from fabric_defect_hub.models.base import ExportedArtifact
from fabric_defect_hub.quantization import SampleCalibrationDataReader, quantize_onnx
from fabric_defect_hub.quantization.onnx_quant import load_and_preprocess_image


def _tiny_onnx_model(path):
    """A two-Conv/Relu graph — enough real weight/activation tensors for
    dynamic *and* static INT8 quantization to have something to quantize.
    """

    input_tensor = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 16, 16])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 16, 16])

    rng = np.random.default_rng(0)
    w1 = numpy_helper.from_array(rng.standard_normal((4, 3, 3, 3)).astype(np.float32), name="w1")
    b1 = numpy_helper.from_array(rng.standard_normal((4,)).astype(np.float32), name="b1")
    w2 = numpy_helper.from_array(rng.standard_normal((4, 4, 3, 3)).astype(np.float32), name="w2")
    b2 = numpy_helper.from_array(rng.standard_normal((4,)).astype(np.float32), name="b2")

    conv1 = helper.make_node("Conv", ["images", "w1", "b1"], ["c1"], pads=[1, 1, 1, 1])
    relu1 = helper.make_node("Relu", ["c1"], ["r1"])
    conv2 = helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], pads=[1, 1, 1, 1])
    relu2 = helper.make_node("Relu", ["c2"], ["output"])

    graph = helper.make_graph(
        [conv1, relu1, conv2, relu2], "tiny_cnn", [input_tensor], [output_tensor],
        initializer=[w1, b1, w2, b2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _calibration_samples(tmp_path, count=4):
    from PIL import Image

    root = tmp_path / "calib"
    root.mkdir()
    samples = []
    for i in range(count):
        image_path = root / f"c{i}.png"
        Image.new("RGB", (16, 16), (i * 20, i * 10, i * 5)).save(image_path)
        samples.append(Sample(id=f"c{i}", image_path=str(image_path), task="anomaly", annotations=Annotations()))
    return samples


def _run(path, dummy):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(None, {"images": dummy})[0]


@pytest.fixture
def onnx_artifact(tmp_path):
    path = tmp_path / "tiny.onnx"
    _tiny_onnx_model(path)
    return ExportedArtifact(path=str(path), target="onnx", metadata={})


def test_rejects_non_onnx_target(tmp_path):
    with pytest.raises(ValueError, match="onnx"):
        quantize_onnx(
            ExportedArtifact(path="model.pt", target="torchscript"), "fp16", tmp_path / "out.onnx"
        )


def test_rejects_unknown_level(onnx_artifact, tmp_path):
    with pytest.raises(ValueError, match="unknown quantization level"):
        quantize_onnx(onnx_artifact, "int4", tmp_path / "out.onnx")


def test_rejects_missing_source_file(tmp_path):
    artifact = ExportedArtifact(path=str(tmp_path / "missing.onnx"), target="onnx")
    with pytest.raises(FileNotFoundError):
        quantize_onnx(artifact, "fp16", tmp_path / "out.onnx")


def test_int8_static_requires_calibration_samples(onnx_artifact, tmp_path):
    with pytest.raises(ValueError, match="calibration_samples"):
        quantize_onnx(onnx_artifact, "int8-static", tmp_path / "out.onnx")


def test_fp16_quantization_runs_and_keeps_io_dtype(onnx_artifact, tmp_path):
    dummy = np.random.rand(1, 3, 16, 16).astype(np.float32)
    baseline = _run(onnx_artifact.path, dummy)

    result = quantize_onnx(onnx_artifact, "fp16", tmp_path / "fp16.onnx", input_size=(16, 16))

    assert result.target == "onnx"
    assert result.metadata["quantization_level"] == "fp16"
    assert result.metadata["size_mb"] > 0

    quantized_out = _run(result.path, dummy)
    assert quantized_out.dtype == baseline.dtype == np.float32
    assert quantized_out.shape == baseline.shape


def test_int8_dynamic_quantization_runs(onnx_artifact, tmp_path):
    dummy = np.random.rand(1, 3, 16, 16).astype(np.float32)
    result = quantize_onnx(onnx_artifact, "int8-dynamic", tmp_path / "dyn.onnx", input_size=(16, 16))

    assert result.metadata["quantization_level"] == "int8-dynamic"
    out = _run(result.path, dummy)
    assert out.shape == (1, 4, 16, 16)


def test_int8_static_quantization_runs_with_real_calibration_images(onnx_artifact, tmp_path):
    samples = _calibration_samples(tmp_path)
    dummy = np.random.rand(1, 3, 16, 16).astype(np.float32)

    result = quantize_onnx(
        onnx_artifact, "int8-static", tmp_path / "static.onnx",
        calibration_samples=samples, input_size=(16, 16),
    )

    assert result.metadata["quantization_level"] == "int8-static"
    out = _run(result.path, dummy)
    assert out.shape == (1, 4, 16, 16)


def test_sample_calibration_data_reader_yields_preprocessed_batches_then_stops(tmp_path):
    samples = _calibration_samples(tmp_path, count=2)
    reader = SampleCalibrationDataReader(samples, input_name="images", input_size=(16, 16))

    first = reader.get_next()
    assert set(first) == {"images"}
    assert first["images"].shape == (1, 3, 16, 16)
    assert first["images"].dtype == np.float32

    second = reader.get_next()
    assert second is not None
    assert reader.get_next() is None  # exhausted


def test_sample_calibration_data_reader_rejects_empty_sample_list():
    with pytest.raises(ValueError, match="at least one Sample"):
        SampleCalibrationDataReader([], input_name="images")


def test_load_and_preprocess_image_shape_and_range(tmp_path):
    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(image_path)

    array = load_and_preprocess_image(str(image_path), (16, 16))
    assert array.shape == (1, 3, 16, 16)
    assert array.dtype == np.float32
    assert array.min() >= 0.0 and array.max() <= 1.0
