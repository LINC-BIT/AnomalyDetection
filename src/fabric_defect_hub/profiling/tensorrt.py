"""TensorRT `BackendProfiler` — Jetson/edge deployment profiling.

**Not verified on this project's dev machine.** Unlike `pytorch.py` and
`onnxruntime.py`, which were built by loading a real exported model and
timing real inference runs, TensorRT is NVIDIA-proprietary and has no
CUDA-capable install target on Apple Silicon (this repo's dev environment)
— `pip install tensorrt` simply has nothing to install here. This class is
written directly against TensorRT's documented Python API method names and
the CUDA-event benchmarking pattern TensorRT's own docs recommend, not
guessed, but it has never actually been executed. Treat it the same way
the project treated the mmdetection→torchvision swap: don't trust the
numbers until it's run for real on a Jetson/CUDA box.

Concretely unverified/undisclaimed here:

* **API generation**: targets the *named-tensor* execution API current as
  of TensorRT 8.6/10.x (`ICudaEngine.get_tensor_name`,
  `IExecutionContext.set_input_shape`/`set_tensor_address`,
  `execute_async_v3`). TensorRT <= 8.4's integer-`binding_index` API
  (`get_binding_index`, `execute_v2(bindings=[...])`) was deprecated in 8.5
  and removed in 10.x — this class will not work against an older
  TensorRT install or an engine built for one without rewriting the I/O
  binding calls.
* **Memory metric is an approximation, not a measurement**: there is no
  NVML/`nvidia-smi` dependency here, so `peak_memory_mb` is the sum of the
  allocated I/O device buffer sizes — a lower bound on the engine's actual
  working set, not the validated `torch.cuda.max_memory_allocated()`-style
  peak `pytorch.py` reports. Don't treat the two as comparable numbers.
* **Power metric has a separately reported scope**: the common power monitor
  uses Jetson's `tegrastats` board-input reading here (rather than this
  profiler's device-buffer lower bound). Inspect the generated `power.json`
  for source, scope, sample count, and any permission/sensor failure.
* **Engine building** (`build_tensorrt_engine`, below) parses an ONNX export
  via TensorRT's own `OnnxParser` and supports fp32/fp16/INT8 precision.
  INT8 requires a real `IInt8Calibrator` (see
  `quantization.tensorrt_calibration.build_int8_calibrator`) — it is
  rejected outright otherwise, same reasoning as the module docstring
  there. `ModelAdapter.export()` itself still never produces
  `target == "tensorrt"` directly; this function is the ONNX -> TensorRT
  step callers run afterwards (see `tools/export_model.py`).
* Not handled: multiple model outputs, dynamic-batch optimization-profile
  selection (`set_optimization_profile_async`), DLA cores, custom plugins.

Requires the `profiling-tensorrt` extra (`pip install -e ".[profiling-tensorrt]"`)
— `tensorrt` + `pycuda` — on a CUDA-capable machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fabric_defect_hub.core.registry import register_profiler
from fabric_defect_hub.models.base import ExportedArtifact
from fabric_defect_hub.profiling.base import BackendProfiler, ProfileConfig, summarize_latencies


@register_profiler
class TensorRTProfiler(BackendProfiler):
    """Profiles a pre-built TensorRT engine via `execute_async_v3` + CUDA-event timing."""

    engine = "tensorrt"

    def memory_context(self, config: ProfileConfig) -> dict[str, object]:
        return {
            "kind": "io_buffer_lower_bound", "scope": "allocated_input_output_device_buffers_only",
            "cross_engine_comparable": False,
        }

    def profile(self, artifact: ExportedArtifact, config: ProfileConfig) -> dict[str, float]:
        # Validate before importing tensorrt/pycuda: this branch stays
        # testable (and correctly ordered — see the module docstring) even
        # on a machine with neither package installed.
        if artifact.target != "tensorrt":
            raise ValueError(
                f"TensorRTProfiler expects a 'tensorrt' export, got target={artifact.target!r}."
            )
        if not config.device.startswith("cuda"):
            raise ValueError(
                f"TensorRTProfiler requires a CUDA device, got config.device={config.device!r}."
            )

        import pycuda.autoinit  # noqa: F401  (initializes the CUDA context as a side effect)
        import pycuda.driver as cuda
        import tensorrt as trt

        engine = _load_engine(artifact.path)
        context = engine.create_execution_context()
        io = _allocate_io_tensors(engine, context, config)
        stream = cuda.Stream()

        for _ in range(config.warmup_runs):
            _execute(context, stream)

        start_event = cuda.Event()
        end_event = cuda.Event()
        latencies_ms: list[float] = []
        monitor = self.start_power_monitor(config)
        try:
            for _ in range(config.measured_runs):
                start_event.record(stream)
                _execute(context, stream)
                end_event.record(stream)
                end_event.synchronize()
                latencies_ms.append(start_event.time_till(end_event))
                monitor.sample()
        finally:
            peak_memory_bytes = sum(buf.nbytes for buf in io.device_buffers)
            metrics = summarize_latencies(latencies_ms, config.batch_size, peak_memory_bytes)
            self.finish_power_monitor(monitor, metrics)

        self.last_instrumentation = self.instrumentation_context(config)
        return metrics


@dataclass
class TensorRTBuildConfig:
    """ONNX-to-TensorRT engine build options for one deployment target."""

    precision: str = "fp32"
    workspace_size_mb: int = 1024
    min_shape: tuple[int, int, int, int] = (1, 3, 640, 640)
    opt_shape: tuple[int, int, int, int] = (1, 3, 640, 640)
    max_shape: tuple[int, int, int, int] = (1, 3, 640, 640)

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "fp16", "int8"}:
            raise ValueError("TensorRT precision must be 'fp32', 'fp16', or 'int8'.")
        if self.workspace_size_mb < 1:
            raise ValueError("TensorRT workspace_size_mb must be at least 1.")
        for name in ("min_shape", "opt_shape", "max_shape"):
            shape = getattr(self, name)
            if len(shape) != 4 or any(value < 1 for value in shape):
                raise ValueError(f"TensorRT {name} must contain four positive dimensions.")


def build_tensorrt_engine(
    artifact: ExportedArtifact,
    output_path: str | Path,
    config: TensorRTBuildConfig | None = None,
    calibrator=None,
) -> ExportedArtifact:
    """Build a deployable TensorRT engine from an ONNX artifact.

    Dynamic ONNX inputs receive an optimization profile from ``config``.

    ``precision="int8"`` requires ``calibrator`` (an ``IInt8Calibrator``,
    e.g. from ``quantization.tensorrt_calibration.build_int8_calibrator``) —
    building an INT8 engine without a calibrated, dataset-specific
    calibrator would produce an untrustworthy deployment artifact, so this
    is rejected rather than silently falling back to meaningless scales.
    """

    if artifact.target != "onnx":
        raise ValueError(f"TensorRT engine build requires target='onnx', got {artifact.target!r}.")
    onnx_path = Path(artifact.path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX artifact does not exist: {onnx_path}")
    config = config or TensorRTBuildConfig()
    if config.precision == "int8" and calibrator is None:
        raise ValueError(
            "INT8 TensorRT builds require a dataset-specific calibrator and are not implicit. "
            "Pass calibrator= (see quantization.tensorrt_calibration.build_int8_calibrator), "
            "or use fp16/fp32 if no calibration data is available."
        )

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [parser.get_error(index).desc() for index in range(parser.num_errors)]
        raise ValueError(f"TensorRT could not parse ONNX artifact {onnx_path}: {'; '.join(errors)}")

    build_config = builder.create_builder_config()
    _set_workspace_limit(build_config, config.workspace_size_mb, trt)
    if config.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This TensorRT platform does not support fast FP16 engine builds.")
        build_config.set_flag(trt.BuilderFlag.FP16)
    if config.precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("This TensorRT platform does not support fast INT8 engine builds.")
        build_config.set_flag(trt.BuilderFlag.INT8)
        build_config.int8_calibrator = calibrator

    profile = builder.create_optimization_profile()
    has_dynamic_input = False
    for index in range(network.num_inputs):
        input_tensor = network.get_input(index)
        shape = tuple(input_tensor.shape)
        if any(dimension < 0 for dimension in shape):
            has_dynamic_input = True
            profile.set_shape(input_tensor.name, config.min_shape, config.opt_shape, config.max_shape)
    if has_dynamic_input:
        build_config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, build_config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build a serialized engine.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(serialized_engine))
    return ExportedArtifact(
        path=str(output), target="tensorrt",
        metadata={
            "source_onnx": str(onnx_path),
            "precision": config.precision,
            "workspace_size_mb": config.workspace_size_mb,
            "min_shape": list(config.min_shape),
            "opt_shape": list(config.opt_shape),
            "max_shape": list(config.max_shape),
        },
    )


def _set_workspace_limit(build_config, workspace_size_mb: int, trt) -> None:
    bytes_limit = workspace_size_mb * 1024 * 1024
    if hasattr(build_config, "set_memory_pool_limit"):
        build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_limit)
    else:
        build_config.max_workspace_size = bytes_limit


class _IOTensors:
    def __init__(self, device_buffers: list):
        self.device_buffers = device_buffers


def _load_engine(path: str):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def _allocate_io_tensors(engine, context, config: ProfileConfig) -> "_IOTensors":
    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt

    h, w = config.input_size
    device_buffers = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = (config.batch_size, 3, h, w)
            context.set_input_shape(name, shape)
        else:
            shape = tuple(context.get_tensor_shape(name))

        dtype = _trt_dtype_to_numpy(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        device_buffer = cuda.mem_alloc(nbytes)
        device_buffer.nbytes = nbytes  # for the peak-memory approximation in `profile()`
        context.set_tensor_address(name, int(device_buffer))
        device_buffers.append(device_buffer)

    return _IOTensors(device_buffers=device_buffers)


def _execute(context, stream) -> None:
    context.execute_async_v3(stream.handle)


def _trt_dtype_to_numpy(dtype) -> str:
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: "float32",
        trt.DataType.HALF: "float16",
        trt.DataType.INT8: "int8",
        trt.DataType.INT32: "int32",
        trt.DataType.BOOL: "bool",
    }
    return mapping.get(dtype, "float32")
