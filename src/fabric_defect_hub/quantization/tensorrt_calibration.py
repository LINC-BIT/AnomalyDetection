"""INT8 calibrator for `profiling.tensorrt.build_tensorrt_engine`.

**Not verified on this project's dev machine** — same caveat as
`profiling/tensorrt.py` (no CUDA-capable install target on Apple Silicon;
`tensorrt`/`pycuda` have nothing to install here). Written directly against
TensorRT's documented `IInt8EntropyCalibrator2` interface
(`get_batch_size`/`get_batch`/`read_calibration_cache`/
`write_calibration_cache`) and pycuda's documented `mem_alloc`/`memcpy_htod`
pattern — the same one `profiling/tensorrt.py` already uses for I/O buffers
— not guessed, but never actually executed against a real TensorRT/pycuda
install. Treat it the same way the rest of that module is treated: don't
trust it deployed until it's run for real on a Jetson/CUDA box.

`build_tensorrt_engine` previously rejected `precision="int8"` outright,
because building an INT8 engine without a calibrated, dataset-specific
`IInt8Calibrator` produces an untrustworthy deployment artifact (TensorRT
would otherwise calibrate against meaningless data, or refuse to build at
all). `build_int8_calibrator` closes that gap for real: it calibrates
against actual `Sample` images from the same preprocessing path ONNX static
quantization uses (see `onnx_quant.load_and_preprocess_image`), so the two
INT8 paths (ONNX Runtime static quantization, TensorRT INT8 engines) are
calibrated consistently.
"""

from __future__ import annotations

from pathlib import Path

from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.quantization.onnx_quant import load_and_preprocess_image


def build_int8_calibrator(
    samples: list[Sample],
    input_size: tuple[int, int] = (640, 640),
    batch_size: int = 1,
    cache_path: str | Path | None = None,
):
    """Build a TensorRT `IInt8EntropyCalibrator2` over real calibration
    images.

    `cache_path`, if given, persists the computed calibration scales to disk
    (TensorRT's documented calibration-cache mechanism) so a later rebuild
    of the same model can reuse them instead of re-running calibration
    inference over every sample again.

    Requires the `profiling-tensorrt` extra (`tensorrt` + `pycuda`) on a
    CUDA-capable machine — imported lazily so this function stays importable
    (just not callable) without them installed.
    """

    if not samples:
        raise ValueError("INT8 calibration requires at least one Sample")

    import numpy as np
    import pycuda.driver as cuda
    import tensorrt as trt

    height, width = input_size
    nbytes_per_image = 3 * height * width * 4  # float32

    class _SampleInt8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            trt.IInt8EntropyCalibrator2.__init__(self)
            self._samples = list(samples)
            self._index = 0
            self._batch_size = batch_size
            self._cache_path = Path(cache_path) if cache_path else None
            self._device_input = cuda.mem_alloc(batch_size * nbytes_per_image)

        def get_batch_size(self) -> int:
            return self._batch_size

        def get_batch(self, names):  # noqa: ARG002  (TensorRT's documented signature)
            if self._index >= len(self._samples):
                return None
            batch_samples = self._samples[self._index : self._index + self._batch_size]
            self._index += self._batch_size

            arrays = [load_and_preprocess_image(sample.image_path, input_size)[0] for sample in batch_samples]
            while len(arrays) < self._batch_size:
                # Pad a final partial batch by repeating the last image —
                # simpler and just as statistically valid for entropy
                # calibration as dropping the remainder, and avoids a
                # variable-batch-size call into a fixed-size device buffer.
                arrays.append(arrays[-1])

            batch = np.ascontiguousarray(np.stack(arrays, axis=0).astype(np.float32))
            cuda.memcpy_htod(self._device_input, batch)
            return [int(self._device_input)]

        def read_calibration_cache(self):
            if self._cache_path is not None and self._cache_path.is_file():
                return self._cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            if self._cache_path is not None:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._cache_path.write_bytes(cache)

    return _SampleInt8Calibrator()
