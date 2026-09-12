"""PyTorch (eager or TorchScript) `BackendProfiler` — the PC baseline every
other runtime (ONNX Runtime, TensorRT) gets compared against.

Loads a `torch.export` ExportedProgram or legacy TorchScript artifact and
times repeated forward passes with proper device synchronization — a naive
`time.perf_counter()` around a CUDA/MPS call measures kernel-launch time,
not execution time, unless you synchronize first; this is a genuinely easy
mistake, not a hypothetical one, so it's called out explicitly below.
"""

from __future__ import annotations

import time

from fabric_defect_hub.core.registry import register_profiler
from fabric_defect_hub.models.base import ExportedArtifact
from fabric_defect_hub.profiling.base import BackendProfiler, ProfileConfig, summarize_latencies


@register_profiler
class PyTorchProfiler(BackendProfiler):
    """Profiles a TorchScript-exported model under real (or CPU-simulated)
    latency conditions.
    """

    engine = "pytorch"

    def memory_context(self, config: ProfileConfig) -> dict[str, object]:
        device_type = config.device.split(":", 1)[0]
        if device_type == "cuda":
            return {
                "kind": "device_allocator", "scope": "torch_cuda_allocated_tensors",
                "cross_engine_comparable": False,
            }
        if device_type == "mps":
            return {
                "kind": "device_allocator", "scope": "torch_mps_current_allocated_memory",
                "cross_engine_comparable": False,
            }
        return {
            "kind": "process_rss", "scope": "whole_process_resident_set",
            "cross_engine_comparable": False,
        }

    def profile(self, artifact: ExportedArtifact, config: ProfileConfig) -> dict[str, float]:
        if artifact.target not in {"torchscript", "exported_program"}:
            raise ValueError(
                "PyTorchProfiler expects an 'exported_program' or 'torchscript' artifact, "
                f"got target={artifact.target!r}."
            )

        import torch

        device = torch.device(config.device)
        if artifact.target == "exported_program":
            model = torch.export.load(artifact.path).module().to(device)
        else:
            model = torch.jit.load(artifact.path, map_location=device)
        if hasattr(model, "eval"):
            try:
                model.eval()
            except NotImplementedError:
                pass

        dummy_input = _build_dummy_input(config, device)

        with torch.no_grad():
            for _ in range(config.warmup_runs):
                model(dummy_input)

            latencies_ms: list[float] = []
            memory_samples_bytes: list[int] = []
            peak_memory_bytes = 0
            monitor = self.start_power_monitor(config)
            try:
                for _ in range(config.measured_runs):
                    _reset_peak_memory(device)
                    start = time.perf_counter()
                    model(dummy_input)
                    _synchronize(device)
                    latencies_ms.append((time.perf_counter() - start) * 1000.0)
                    sample_bytes = _peak_memory_bytes(device)
                    memory_samples_bytes.append(sample_bytes)
                    peak_memory_bytes = max(peak_memory_bytes, sample_bytes)
                    monitor.sample()
            finally:
                metrics = summarize_latencies(
                    latencies_ms, config.batch_size, peak_memory_bytes, memory_samples_bytes
                )
                self.finish_power_monitor(monitor, metrics)

        self.last_instrumentation = self.instrumentation_context(config)
        return metrics


def _build_dummy_input(config: ProfileConfig, device):
    import torch

    h, w = config.input_size
    if config.input_style == "list":
        return [torch.rand(3, h, w, device=device) for _ in range(config.batch_size)]
    if config.input_style == "batched":
        return torch.rand(config.batch_size, 3, h, w, device=device)
    raise ValueError(f"unknown input_style {config.input_style!r}; expected 'batched' or 'list'.")


def _synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    # cpu: nothing to synchronize, execution is already blocking


def _reset_peak_memory(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        reset_peak = getattr(torch.mps, "reset_peak_memory_stats", None)
        if callable(reset_peak):
            reset_peak()


def _peak_memory_bytes(device) -> int:
    import torch

    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    import psutil

    return int(psutil.Process().memory_info().rss)
