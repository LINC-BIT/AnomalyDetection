"""BackendProfiler: measures inference performance (latency/FPS/memory/power)
for a given runtime (PyTorch, ONNX Runtime, TensorRT, ...) under a fixed,
recorded set of run conditions so results across models stay comparable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from fabric_defect_hub.core.types import RuntimeInfo
from fabric_defect_hub.models.base import ExportedArtifact


@dataclass
class ProfileConfig:
    device: str
    engine: str
    precision: str = "fp32"
    input_size: tuple[int, int] = (640, 640)
    batch_size: int = 1
    warmup_runs: int = 10
    measured_runs: int = 100
    # 'batched': a single [batch, 3, H, W] tensor input — CNN-style models
    #   (Ultralytics YOLO, Anomalib backbones) exported to TorchScript/ONNX.
    # 'list': a List[Tensor] of `batch_size` [3, H, W] tensors — torchvision
    #   detection models' native TorchScript calling convention (see
    #   `models/torchvision/adapter.py::export`). Only relevant to
    #   `profiling/pytorch.py`; ONNX graphs are always a fixed tensor shape
    #   regardless of the pre-export Python calling convention.
    input_style: str = "batched"
    # auto: collect when a supported sensor is available; required: fail if
    # power cannot be measured; disabled: do not attempt a power reading.
    power_mode: str = "auto"
    power_sample_interval_ms: int = 100
    # Optional explicit Linux sysfs power sensor path, chiefly for a
    # Raspberry Pi connected to an INA219/INA226 power monitor.
    power_sensor_path: str | None = None

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("device must be a non-empty string.")
        if not self.engine:
            raise ValueError("engine must be a non-empty string.")
        if self.precision not in {"fp32", "fp16", "int8", "bf16", "amp-mixed"}:
            raise ValueError("precision must be one of fp32, fp16, int8, bf16, or amp-mixed.")
        if len(self.input_size) != 2 or any(int(value) < 1 for value in self.input_size):
            raise ValueError("input_size must contain two positive dimensions.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative.")
        if self.measured_runs < 1:
            raise ValueError("measured_runs must be at least 1.")
        if self.input_style not in {"batched", "list"}:
            raise ValueError("input_style must be 'batched' or 'list'.")
        if self.power_mode not in {"auto", "required", "disabled"}:
            raise ValueError("power_mode must be 'auto', 'required', or 'disabled'.")
        if self.power_sample_interval_ms < 1:
            raise ValueError("power_sample_interval_ms must be at least 1.")


class BackendProfiler(ABC):
    """Base class for a runtime-specific performance benchmark."""

    engine: str

    @abstractmethod
    def profile(self, artifact: ExportedArtifact, config: ProfileConfig) -> dict[str, float]:
        """Run `measured_runs` inferences after `warmup_runs`, return latency/FPS/memory/etc."""

    def runtime_info(self, config: ProfileConfig) -> RuntimeInfo:
        return RuntimeInfo(
            device=config.device,
            engine=config.engine,
            precision=config.precision,
            input_size=config.input_size,
        )

    def start_power_monitor(self, config: ProfileConfig):
        from fabric_defect_hub.profiling.power import PowerMonitor

        monitor = PowerMonitor.from_profile_config(config)
        monitor.start()
        return monitor

    def finish_power_monitor(self, monitor, metrics: dict[str, float]) -> None:
        report = monitor.stop()
        self.last_power_report = report
        metrics.update(report.metrics())

    def instrumentation_context(self, config: ProfileConfig) -> dict[str, object]:
        power = getattr(self, "last_power_report", None)
        return {
            "timing_scope": "runtime_forward_only_synthetic_input",
            "timing_excludes": ["data_loading", "preprocessing", "postprocessing"],
            "throughput_definition": "total_frames_divided_by_total_measured_time",
            "memory": self.memory_context(config),
            "power": {
                "mode": config.power_mode,
                "status": getattr(power, "status", "not_run"),
                "source": getattr(getattr(power, "capability", None), "source", "none"),
                "scope": getattr(getattr(power, "capability", None), "scope", "unknown"),
            },
        }

    def memory_context(self, config: ProfileConfig) -> dict[str, object]:
        return {"kind": "unknown", "scope": "unknown", "cross_engine_comparable": False}


def summarize_latencies(
    latencies_ms: list[float],
    batch_size: int,
    peak_memory_bytes: int,
    memory_samples_bytes: list[int] | None = None,
) -> dict[str, float]:
    """Shared latency/FPS/memory summary every concrete `BackendProfiler`
    (`pytorch.py`, `onnxruntime.py`, `tensorrt.py`) reduces its measured
    runs down to, so the resulting metric names/definitions are identical
    across runtimes — required for the cross-engine comparison this whole
    module exists for.

    `fps` is aggregate throughput: total frames divided by total measured
    time. Per-iteration reciprocal latency has a different arithmetic mean,
    so its distribution is explicitly named `instantaneous_fps_*` rather
    than being presented as the variance of aggregate `fps`.
    `memory_samples_bytes` is optional per-iteration memory (distinct from
    `peak_memory_bytes`, the max) used for `avg_memory_mb`; callers that
    only tracked a peak (or an engine, like TensorRT's device-buffer sum,
    whose measurement doesn't vary per iteration) can omit it and get
    `avg_memory_mb == peak_memory_mb` instead.
    """

    import statistics

    from fabric_defect_hub.stats import coefficient_of_variation

    sorted_latencies = sorted(latencies_ms)
    mean_latency = statistics.fmean(sorted_latencies) if sorted_latencies else 0.0
    instantaneous_fps = [(1000.0 / latency) * batch_size for latency in latencies_ms if latency > 0]
    instantaneous_fps_mean = statistics.fmean(instantaneous_fps) if instantaneous_fps else 0.0
    memory_samples = memory_samples_bytes if memory_samples_bytes else [peak_memory_bytes]

    return {
        "latency_ms_mean": mean_latency,
        "latency_ms_p50": _percentile(sorted_latencies, 50),
        "latency_ms_p95": _percentile(sorted_latencies, 95),
        "latency_ms_p99": _percentile(sorted_latencies, 99),
        "latency_ms_std": statistics.pstdev(latencies_ms) if len(latencies_ms) >= 2 else 0.0,
        "latency_ms_cv": coefficient_of_variation(latencies_ms),
        "fps": (1000.0 / mean_latency) * batch_size if mean_latency > 0 else 0.0,
        "instantaneous_fps_mean": instantaneous_fps_mean,
        "instantaneous_fps_std": statistics.pstdev(instantaneous_fps) if len(instantaneous_fps) >= 2 else 0.0,
        "instantaneous_fps_cv": coefficient_of_variation(instantaneous_fps),
        "peak_memory_mb": peak_memory_bytes / (1024 * 1024),
        "avg_memory_mb": statistics.fmean(memory_samples) / (1024 * 1024),
    }


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round((pct / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[idx]
