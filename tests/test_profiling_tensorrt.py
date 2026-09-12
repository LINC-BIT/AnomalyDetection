"""Tests for `profiling/tensorrt.py`.

TensorRT itself cannot be installed on this dev machine (Apple Silicon, no
CUDA GPU — see the module's docstring for the full disclosure). The only
things testable without either a real TensorRT install or mocking its
internals (which this project deliberately avoids — see
`profiling/tensorrt.py`'s docstring) are the validation branches that run
*before* `import tensorrt`/`import pycuda` — and that those imports are
correctly deferred until after validation, so a missing-target error never
gets masked by a missing-dependency error.
"""

import pytest

from fabric_defect_hub.models.base import ExportedArtifact
from fabric_defect_hub.profiling.base import ProfileConfig
from fabric_defect_hub.profiling.tensorrt import TensorRTBuildConfig, TensorRTProfiler, build_tensorrt_engine


def test_rejects_wrong_target_without_needing_tensorrt_installed():
    profiler = TensorRTProfiler()
    config = ProfileConfig(device="cuda:0", engine="tensorrt")
    with pytest.raises(ValueError, match="tensorrt"):
        profiler.profile(ExportedArtifact(path="x", target="onnx"), config)


def test_rejects_non_cuda_device_without_needing_tensorrt_installed():
    profiler = TensorRTProfiler()
    config = ProfileConfig(device="cpu", engine="tensorrt")
    with pytest.raises(ValueError, match="CUDA"):
        profiler.profile(ExportedArtifact(path="x", target="tensorrt"), config)


def test_correct_target_and_device_reaches_the_tensorrt_import():
    # Proves validation happens before the import (not after) — a correctly
    # formed request fails on the *missing dependency*, not a validation
    # error, which is the one meaningful check possible without a real
    # TensorRT/pycuda install.
    profiler = TensorRTProfiler()
    config = ProfileConfig(device="cuda:0", engine="tensorrt")
    with pytest.raises(ModuleNotFoundError):
        profiler.profile(ExportedArtifact(path="x", target="tensorrt"), config)


def test_runtime_info_maps_config_fields():
    config = ProfileConfig(device="cuda:0", engine="tensorrt", precision="fp16", input_size=(512, 512))
    info = TensorRTProfiler().runtime_info(config)
    assert info.device == "cuda:0"
    assert info.engine == "tensorrt"
    assert info.precision == "fp16"
    assert info.input_size == (512, 512)


def test_build_config_rejects_implicit_int8_and_invalid_shapes():
    with pytest.raises(ValueError, match="precision"):
        TensorRTBuildConfig(precision="bf16")
    with pytest.raises(ValueError, match="four positive"):
        TensorRTBuildConfig(min_shape=(1, 3, 0, 640))


def test_engine_builder_rejects_non_onnx_before_needing_tensorrt(tmp_path):
    with pytest.raises(ValueError, match="target='onnx'"):
        build_tensorrt_engine(
            ExportedArtifact(path=str(tmp_path / "model.pt"), target="torchscript"),
            tmp_path / "model.engine",
        )


def test_engine_builder_rejects_int8_without_a_calibrator(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"not a real onnx file")
    with pytest.raises(ValueError, match="calibrator"):
        build_tensorrt_engine(
            ExportedArtifact(path=str(onnx_path), target="onnx"),
            tmp_path / "model.engine",
            config=TensorRTBuildConfig(precision="int8"),
        )


def test_engine_builder_with_int8_and_a_calibrator_reaches_the_tensorrt_import(tmp_path):
    # Proves the calibrator-supplied path clears validation and only fails
    # on the missing tensorrt dependency, same pattern as the profiler tests
    # above — validation happens before the import, not after.
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"not a real onnx file")
    with pytest.raises(ModuleNotFoundError):
        build_tensorrt_engine(
            ExportedArtifact(path=str(onnx_path), target="onnx"),
            tmp_path / "model.engine",
            config=TensorRTBuildConfig(precision="int8"),
            calibrator=object(),
        )
