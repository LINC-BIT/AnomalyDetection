"""AnomalyDetection loader and optimization strategies."""

from fabric_defect_hub.strategies.loader_strategies import (
    BatchNormCalibrator,
    SlidingWindowTiler,
    SparseSubsampler,
    TTAInferenceWrapper,
)

__all__ = [
    "SparseSubsampler",
    "SlidingWindowTiler",
    "TTAInferenceWrapper",
    "BatchNormCalibrator",
]
