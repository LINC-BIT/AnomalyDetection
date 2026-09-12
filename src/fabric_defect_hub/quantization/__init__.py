"""Post-training quantization for edge deployment.

`onnx_quant` (ONNX Runtime + onnxconverter-common) is the primary, backend-
agnostic path: it operates on an already-exported `target == "onnx"`
`ExportedArtifact`, so it works identically regardless of which
`ModelAdapter` (Ultralytics/torchvision/Anomalib) produced it. See
`onnx_quant.py`'s module docstring for the four quantization levels.

`tensorrt_calibration` supplies a real image-backed `IInt8Calibrator` for
`profiling.tensorrt.build_tensorrt_engine`'s previously-rejected
`precision="int8"` path — see that module's docstring for why it stayed
unverified/opt-in.
"""

from __future__ import annotations

from fabric_defect_hub.quantization.onnx_quant import (
    QuantizationLevel,
    SampleCalibrationDataReader,
    load_and_preprocess_image,
    quantize_and_register,
    quantize_onnx,
)
from fabric_defect_hub.quantization.tensorrt_calibration import build_int8_calibrator

__all__ = [
    "QuantizationLevel",
    "SampleCalibrationDataReader",
    "load_and_preprocess_image",
    "quantize_and_register",
    "quantize_onnx",
    "build_int8_calibrator",
]
