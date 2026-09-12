"""Tests for `profiling/base.py`'s pure helpers (exact hand-computed values)
plus real end-to-end runs of `PyTorchProfiler`/`ONNXRuntimeProfiler` against
a tiny real TorchScript/ONNX export — CPU-only, since this dev machine has
no CUDA/MPS to exercise those device branches (see the module docstrings
in `profiling/pytorch.py` for why that matters, and `test_profiling_tensorrt.py`
for the TensorRT case, which can't be run at all here).

The toy `nn.Module` below is defined at file scope, not inside a function
or heredoc: `torch.jit.script` needs real source access to compile a
module, and raises `OSError: could not get source code` against anything
defined in a place Python can't retrieve source lines for.
"""

import statistics

import pytest
import torch
import torch.nn as nn

from fabric_defect_hub.models.base import ExportedArtifact
from fabric_defect_hub.profiling.base import BackendProfiler, ProfileConfig, summarize_latencies
from fabric_defect_hub.profiling.onnxruntime import ONNXRuntimeProfiler
from fabric_defect_hub.profiling.pytorch import PyTorchProfiler


class _TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------- #
# profiling/base.py — pure functions, exact hand-computed assertions
# ---------------------------------------------------------------------- #
def test_summarize_latencies_exact_values():
    latencies = [float(x) for x in range(10, 101, 10)]  # 10, 20, ..., 100
    result = summarize_latencies(latencies, batch_size=2, peak_memory_bytes=2 * 1024 * 1024)

    assert result["latency_ms_mean"] == 55.0
    assert result["latency_ms_p50"] == 50.0
    assert result["latency_ms_p95"] == 100.0
    assert result["latency_ms_p99"] == 100.0
    assert result["fps"] == pytest.approx(2000.0 / 55.0)
    assert result["peak_memory_mb"] == 2.0
    # no memory_samples_bytes given -> avg falls back to the single peak value
    assert result["avg_memory_mb"] == 2.0
    # instantaneous fps = 2000/latency for each of the 10 latencies above;
    # hand-computed population stddev/mean of that series.
    instantaneous_fps = [2000.0 / latency for latency in latencies]
    assert result["instantaneous_fps_mean"] == pytest.approx(statistics.fmean(instantaneous_fps))
    assert result["instantaneous_fps_std"] == pytest.approx(statistics.pstdev(instantaneous_fps))
    assert result["instantaneous_fps_cv"] == pytest.approx(statistics.pstdev(instantaneous_fps) / statistics.fmean(instantaneous_fps))


def test_summarize_latencies_averages_memory_samples():
    result = summarize_latencies(
        [10.0, 10.0], batch_size=1, peak_memory_bytes=3 * 1024 * 1024,
        memory_samples_bytes=[1 * 1024 * 1024, 3 * 1024 * 1024],
    )
    assert result["peak_memory_mb"] == 3.0
    assert result["avg_memory_mb"] == 2.0


def test_summarize_latencies_empty_zero_fps():
    result = summarize_latencies([], batch_size=1, peak_memory_bytes=0)
    assert result["fps"] == 0.0
    assert result["latency_ms_p50"] == 0.0
    assert result["instantaneous_fps_std"] == 0.0
    assert result["instantaneous_fps_cv"] == 0.0


class _StubProfiler(BackendProfiler):
    engine = "stub"

    def profile(self, artifact, config):
        return {}


def test_runtime_info_maps_config_fields():
    config = ProfileConfig(device="cuda:0", engine="stub", precision="fp16", input_size=(320, 320))
    info = _StubProfiler().runtime_info(config)
    assert info.device == "cuda:0"
    assert info.engine == "stub"
    assert info.precision == "fp16"
    assert info.input_size == (320, 320)


# ---------------------------------------------------------------------- #
# PyTorchProfiler — real TorchScript export + real timed inference
# ---------------------------------------------------------------------- #
def test_pytorch_profiler_batched_input(tmp_path):
    scripted = torch.jit.script(_TinyConv())
    path = tmp_path / "tiny.torchscript.pt"
    scripted.save(str(path))

    config = ProfileConfig(
        device="cpu", engine="pytorch", input_size=(8, 8), batch_size=2,
        warmup_runs=1, measured_runs=3, input_style="batched", power_mode="disabled",
    )
    metrics = PyTorchProfiler().profile(ExportedArtifact(path=str(path), target="torchscript"), config)

    assert set(metrics) == {
        "latency_ms_mean", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "fps",
        "latency_ms_std", "latency_ms_cv", "instantaneous_fps_mean",
        "instantaneous_fps_std", "instantaneous_fps_cv", "peak_memory_mb", "avg_memory_mb",
    }
    assert metrics["latency_ms_p50"] <= metrics["latency_ms_p95"] <= metrics["latency_ms_p99"]
    assert metrics["fps"] == pytest.approx((1000.0 / metrics["latency_ms_mean"]) * config.batch_size)
    assert metrics["peak_memory_mb"] >= 0
    assert metrics["avg_memory_mb"] >= 0
    assert metrics["instantaneous_fps_std"] >= 0
    assert metrics["instantaneous_fps_cv"] >= 0


def test_pytorch_profiler_list_style_input(tmp_path):
    class _TinyListModule(nn.Module):
        def forward(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
            return [img.mean().reshape(1) for img in images]

    scripted = torch.jit.script(_TinyListModule())
    path = tmp_path / "tiny_list.torchscript.pt"
    scripted.save(str(path))

    config = ProfileConfig(
        device="cpu", engine="pytorch", input_size=(4, 4), batch_size=3,
        warmup_runs=1, measured_runs=2, input_style="list", power_mode="disabled",
    )
    metrics = PyTorchProfiler().profile(ExportedArtifact(path=str(path), target="torchscript"), config)
    assert metrics["fps"] >= 0


def test_pytorch_profiler_rejects_wrong_target():
    config = ProfileConfig(device="cpu", engine="pytorch")
    with pytest.raises(ValueError, match="torchscript"):
        PyTorchProfiler().profile(ExportedArtifact(path="x.onnx", target="onnx"), config)


def test_pytorch_profiler_exported_program(tmp_path):
    model = _TinyConv()
    exported = torch.export.export(model, (torch.rand(1, 3, 8, 8),))
    path = tmp_path / "tiny.pt2"
    torch.export.save(exported, path)
    config = ProfileConfig(
        device="cpu", engine="pytorch", input_size=(8, 8), batch_size=1,
        warmup_runs=1, measured_runs=2, power_mode="disabled",
    )
    metrics = PyTorchProfiler().profile(
        ExportedArtifact(path=str(path), target="exported_program"), config
    )
    assert metrics["fps"] >= 0


def test_pytorch_profiler_rejects_unknown_input_style(tmp_path):
    scripted = torch.jit.script(_TinyConv())
    path = tmp_path / "tiny.torchscript.pt"
    scripted.save(str(path))

    with pytest.raises(ValueError, match="input_style"):
        ProfileConfig(device="cpu", engine="pytorch", input_style="not-a-real-style")


# ---------------------------------------------------------------------- #
# ONNXRuntimeProfiler — real ONNX export + real InferenceSession
# ---------------------------------------------------------------------- #
def test_onnxruntime_profiler_dynamic_batch(tmp_path):
    model = _TinyConv()
    dummy = torch.rand(1, 3, 8, 8)
    onnx_path = tmp_path / "tiny.onnx"
    torch.onnx.export(
        model, (dummy,), str(onnx_path),
        input_names=["images"], output_names=["out"],
        dynamic_axes={"images": {0: "batch"}}, opset_version=18,
    )

    config = ProfileConfig(
        device="cpu", engine="onnxruntime", input_size=(8, 8), batch_size=4,
        warmup_runs=1, measured_runs=3, power_mode="disabled",
    )
    metrics = ONNXRuntimeProfiler().profile(ExportedArtifact(path=str(onnx_path), target="onnx"), config)

    assert set(metrics) == {
        "latency_ms_mean", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "fps",
        "latency_ms_std", "latency_ms_cv", "instantaneous_fps_mean",
        "instantaneous_fps_std", "instantaneous_fps_cv", "peak_memory_mb", "avg_memory_mb",
    }
    assert metrics["latency_ms_p50"] <= metrics["latency_ms_p95"] <= metrics["latency_ms_p99"]
    assert metrics["peak_memory_mb"] > 0
    assert metrics["avg_memory_mb"] > 0


def test_onnxruntime_profiler_rejects_wrong_target():
    config = ProfileConfig(device="cpu", engine="onnxruntime")
    with pytest.raises(ValueError, match="onnx"):
        ONNXRuntimeProfiler().profile(ExportedArtifact(path="x.pt", target="torchscript"), config)
