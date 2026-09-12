"""Post-training ONNX quantization for edge deployment.

Operates purely on an already-exported ONNX `ExportedArtifact`
(`target == "onnx"`, produced by `UltralyticsAdapter.export`/
`TorchvisionAdapter.export`/`AnomalibAdapter.export`), so it is entirely
backend-agnostic — the same code path quantizes a YOLO, Faster R-CNN, or
PatchCore ONNX export identically. Verified against a real onnx/onnxruntime/
onnxconverter-common install (a small hand-built two-Conv ONNX graph, not a
real trained model, but exercising the real quantization APIs end to end —
see `tests/test_quantization.py`).

Four levels, in increasing order of size/latency reduction on typical CPU/
edge hardware (and, for the two INT8 tiers, real risk of accuracy loss that
should be checked against `evaluation.*.Evaluator` output before trusting a
deployment — this module does not itself measure accuracy):

    fp32          the source export, unchanged (not a level this module
                  produces — it's the baseline everything else is compared
                  against).
    fp16          half-precision weights + activations via
                  `onnxconverter_common.float16`; ~2x smaller, no
                  calibration data needed, best on GPUs/accelerators with
                  fast fp16 kernels (Jetson, most desktop/cloud GPUs).
                  Negligible accuracy impact in practice.
    int8-dynamic  INT8 weights, with activations quantized per-inference at
                  runtime (`onnxruntime.quantization.quantize_dynamic`); no
                  calibration data needed, ~4x smaller, CPU-friendly. Usually
                  a smaller latency win than static INT8 since activation
                  quantization happens on every run.
    int8-static   INT8 weights *and* activations, both fixed ahead of time
                  from calibration data (`quantize_static`, QDQ format);
                  best latency on integer-only accelerators (Jetson DLA,
                  many microcontroller/NPU targets), but needs a
                  representative calibration sample set — an unrepresentative
                  one silently produces a *worse* model than dynamic
                  quantization, not just a less-optimal one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.models.base import ExportedArtifact

QuantizationLevel = Literal["fp16", "int8-dynamic", "int8-static"]

_LEVELS: tuple[QuantizationLevel, ...] = ("fp16", "int8-dynamic", "int8-static")


class SampleCalibrationDataReader:
    """Feeds real calibration images (drawn from a `Sample` list) to
    `onnxruntime.quantization.quantize_static`.

    Preprocessing (resize to `input_size`, RGB, CHW, [0, 1] float32) mirrors
    `profiling.onnxruntime.ONNXRuntimeProfiler`'s dummy-input shape handling
    — calibration data must match what the model actually sees at inference
    time, or the resulting static-INT8 model is miscalibrated.

    Implements `onnxruntime.quantization.CalibrationDataReader`'s duck-typed
    interface (`get_next()`) directly rather than importing/subclassing it,
    so this module only needs `onnxruntime` inside the functions that
    actually quantize, not at import time.
    """

    def __init__(self, samples: list[Sample], input_name: str, input_size: tuple[int, int] = (640, 640)):
        if not samples:
            raise ValueError("calibration requires at least one Sample")
        self._input_name = input_name
        self._input_size = input_size
        self._iterator = iter(samples)

    def get_next(self) -> dict[str, "object"] | None:
        sample = next(self._iterator, None)
        if sample is None:
            return None
        return {self._input_name: load_and_preprocess_image(sample.image_path, self._input_size)}

    def rewind(self) -> None:
        pass


def quantize_onnx(
    artifact: ExportedArtifact,
    level: QuantizationLevel,
    output_path: str | Path,
    calibration_samples: list[Sample] | None = None,
    input_size: tuple[int, int] = (640, 640),
) -> ExportedArtifact:
    """Quantize `artifact` (an ONNX export) to `level`, writing the result to
    `output_path`. Returns a new `ExportedArtifact` (`target="onnx"`) with
    `metadata["quantization_level"]` and `metadata["size_mb"]` set, so a
    profiler/evaluator run afterwards can attribute its numbers to the right
    variant.

    `calibration_samples` is required for `level="int8-static"` (see the
    module docstring for why) and ignored otherwise.
    """

    if artifact.target != "onnx":
        raise ValueError(f"quantize_onnx requires an 'onnx' export, got target={artifact.target!r}.")
    if level not in _LEVELS:
        raise ValueError(f"unknown quantization level {level!r}; expected one of {_LEVELS}.")

    source_path = Path(artifact.path)
    if not source_path.is_file():
        raise FileNotFoundError(f"ONNX artifact does not exist: {source_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if level == "fp16":
        _quantize_fp16(source_path, output)
    elif level == "int8-dynamic":
        _quantize_int8_dynamic(source_path, output)
    else:
        if not calibration_samples:
            raise ValueError(
                "int8-static quantization requires calibration_samples (a representative "
                "Sample list); use level='int8-dynamic' if no calibration data is available."
            )
        _quantize_int8_static(source_path, output, calibration_samples, input_size)

    return ExportedArtifact(
        path=str(output),
        target="onnx",
        metadata={
            "source_onnx": str(source_path),
            "quantization_level": level,
            "size_mb": output.stat().st_size / (1024 * 1024),
        },
    )


def quantize_and_register(
    artifact: ExportedArtifact,
    level: QuantizationLevel,
    *,
    backend: str,
    variant: str | None,
    project_root: str | Path | None = None,
    calibration_samples: list[Sample] | None = None,
    input_size: tuple[int, int] = (640, 640),
) -> ExportedArtifact:
    """`quantize_onnx` into the project's quantized-model root, and record the
    result in `weight_manifest.jsonl`.

    This is the entry point anything automated should use. `quantize_onnx`
    stays the primitive — it takes an explicit `output_path` and writes
    nothing else, which is what makes it testable — but a quantized file
    written to an arbitrary path is exactly the artifact that later cannot be
    attributed to a run. Here the destination comes from
    `catalog.quantized_path` and the provenance row is written alongside it,
    so `ΔRecall_small` computed against this file can name the fp32
    checkpoint it is being compared to.
    """

    from fabric_defect_hub.catalog import PROJECT_ROOT, quantized_path
    from fabric_defect_hub.weight_registry import record_quantized_weight

    root = Path(project_root) if project_root else PROJECT_ROOT
    destination = quantized_path(backend, variant or backend, level)
    if project_root is not None:
        destination = (
            Path(project_root) / destination.relative_to(PROJECT_ROOT)
            if destination.is_relative_to(PROJECT_ROOT)
            else destination
        )

    quantized = quantize_onnx(
        artifact, level, destination,
        calibration_samples=calibration_samples, input_size=input_size,
    )
    record_quantized_weight(
        project_root=root,
        backend=backend,
        variant=variant,
        level=level,
        source_path=artifact.path,
        quantized_artifact=quantized,
        calibration_sample_count=len(calibration_samples) if calibration_samples else None,
    )
    return quantized


def _quantize_fp16(source_path: Path, output: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(source_path))
    # keep_io_types=True: the graph's external input/output stay fp32 (so
    # callers built for the fp32 export's tensor dtype keep working
    # unchanged), only internal weights/activations become fp16.
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, str(output))


def _quantize_int8_dynamic(source_path: Path, output: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(source_path), str(output), weight_type=QuantType.QInt8)


def _quantize_int8_static(
    source_path: Path, output: Path, calibration_samples: list[Sample], input_size: tuple[int, int]
) -> None:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    input_name = ort.InferenceSession(
        str(source_path), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name
    reader = SampleCalibrationDataReader(calibration_samples, input_name, input_size)
    quantize_static(
        str(source_path), str(output), reader,
        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
    )


def load_and_preprocess_image(image_path: str, input_size: tuple[int, int]):
    """Load one image as an NCHW, [0, 1]-scaled float32 array of shape
    `(1, 3, *input_size)`. Shared preprocessing between ONNX static-INT8
    calibration here and `tensorrt_calibration.py`'s TensorRT calibrator —
    both need calibration data preprocessed the same way the exported model
    actually expects it at inference time.
    """

    import numpy as np
    from PIL import Image

    height, width = input_size
    with Image.open(image_path) as img:
        resized = img.convert("RGB").resize((width, height))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)[None, ...]  # HWC -> NCHW
    return np.ascontiguousarray(array)
