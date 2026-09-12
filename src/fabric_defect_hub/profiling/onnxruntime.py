"""ONNX Runtime `BackendProfiler`. Loads an ONNX export (`ExportedArtifact.
target == 'onnx'`, produced by `UltralyticsAdapter.export`/
`TorchvisionAdapter.export`/`AnomalibAdapter.export`) into a real
`onnxruntime.InferenceSession` and times repeated `session.run()` calls.

Unlike TorchScript profiling (`profiling/pytorch.py`), there's no `input_style`
ambiguity here: an ONNX graph always has a fixed tensor input signature —
even torchvision detection's Python-side `List[Tensor]` calling convention
gets traced down to a single batched tensor input during export (see the
'best-effort' caveat in `models/torchvision/adapter.py::export`). This
profiler reads the actual input name/shape off the loaded model rather than
assuming one, so it works across all three backends' ONNX exports.
"""

from __future__ import annotations

import time

from fabric_defect_hub.core.registry import register_profiler
from fabric_defect_hub.models.base import ExportedArtifact
from fabric_defect_hub.profiling.base import BackendProfiler, ProfileConfig, summarize_latencies


@register_profiler
class ONNXRuntimeProfiler(BackendProfiler):
    """Profiles an ONNX-exported model via a real `onnxruntime.InferenceSession`."""

    engine = "onnxruntime"

    def memory_context(self, config: ProfileConfig) -> dict[str, object]:
        return {
            "kind": "process_rss", "scope": "whole_process_resident_set",
            "cross_engine_comparable": False,
        }

    def profile(self, artifact: ExportedArtifact, config: ProfileConfig) -> dict[str, float]:
        if artifact.target != "onnx":
            raise ValueError(
                f"ONNXRuntimeProfiler expects an 'onnx' export, got target={artifact.target!r}."
            )

        import numpy as np
        import onnxruntime as ort
        import psutil

        providers = _providers_for_device(config.device)
        session = ort.InferenceSession(artifact.path, providers=providers)

        input_meta = session.get_inputs()[0]
        shape = _resolve_shape(input_meta.shape, config)
        dummy = np.random.rand(*shape).astype(_numpy_dtype(input_meta.type))
        feed = {input_meta.name: dummy}
        output_names = [o.name for o in session.get_outputs()]

        process = psutil.Process()
        for _ in range(config.warmup_runs):
            session.run(output_names, feed)

        latencies_ms: list[float] = []
        memory_samples_bytes: list[int] = []
        peak_rss_bytes = 0
        monitor = self.start_power_monitor(config)
        try:
            for _ in range(config.measured_runs):
                start = time.perf_counter()
                session.run(output_names, feed)
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
                rss_bytes = process.memory_info().rss
                memory_samples_bytes.append(rss_bytes)
                peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
                monitor.sample()
        finally:
            metrics = summarize_latencies(
                latencies_ms, config.batch_size, peak_rss_bytes, memory_samples_bytes
            )
            self.finish_power_monitor(monitor, metrics)

        self.last_instrumentation = self.instrumentation_context(config)
        return metrics


def _providers_for_device(device: str) -> list[str]:
    if device.startswith("cuda") or device == "gpu":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _resolve_shape(declared_shape: list, config: ProfileConfig) -> tuple[int, ...]:
    """ONNX dynamic axes show up as strings (e.g. 'batch') or `None` in
    `declared_shape`; substitute `config.batch_size`/`input_size` for those,
    keep fixed axes as-is.
    """

    h, w = config.input_size
    resolved: list[int] = []
    dynamic_fill = [config.batch_size, 3, h, w]
    for i, dim in enumerate(declared_shape):
        if isinstance(dim, int):
            resolved.append(dim)
        else:
            resolved.append(dynamic_fill[i] if i < len(dynamic_fill) else 1)
    return tuple(resolved)


def _numpy_dtype(onnx_type: str) -> str:
    mapping = {
        "tensor(float)": "float32",
        "tensor(float16)": "float16",
        "tensor(double)": "float64",
        "tensor(uint8)": "uint8",
        "tensor(int64)": "int64",
    }
    return mapping.get(onnx_type, "float32")
