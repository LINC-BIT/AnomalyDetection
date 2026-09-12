"""Unit tests for Strategy-Driven Loading & Protocol Engine (SDLP)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import fabric_defect_hub as fdh
from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.strategies.loader_strategies import (
    SlidingWindowTiler,
    SparseSubsampler,
    TTAInferenceWrapper,
)


def test_sparse_subsampler():
    """Test proportionate sparse ratio subsampling."""
    samples = []
    for i in range(100):
        s = Sample(
            id=f"sample_{i}",
            image_path="",
            task="anomaly",
            annotations=Annotations(is_anomalous=False),
            metadata={"pattern": f"p_{i % 3}"},
        )
        setattr(s, "image", Image.new("RGB", (64, 64)))
        samples.append(s)

    # Test 10% sparse ratio
    sparse_10 = SparseSubsampler.apply_sparse_ratio(samples, sparse_ratio=0.1)
    assert len(sparse_10) == 10

    # Test stratified pattern sampling
    stratified = SparseSubsampler.apply_stratified_pattern(samples, sparse_ratio=0.2)
    assert len(stratified) > 0


def test_sliding_window_tiler():
    """Test 4K high-res sliding window tiling and prediction stitching."""
    tiler = SlidingWindowTiler(tile_size=(256, 256), overlap=0.25)
    large_img = Image.new("RGB", (512, 512))
    sample = Sample(
        id="large_0",
        image_path="",
        task="anomaly",
        annotations=Annotations(is_anomalous=True),
    )
    setattr(sample, "image", large_img)

    tiles, meta_info = tiler.split_sample(sample)
    assert meta_info["tiled"] is True
    assert len(tiles) > 1

    # Simulate tile predictions
    tile_preds = [
        Prediction(sample_id=t.id, anomaly_score=0.8, anomaly_map=np.ones((256, 256), dtype=np.float32))
        for t in tiles
    ]
    for p, t in zip(tile_preds, tiles):
        setattr(p, "metadata", t.metadata)

    stitched = tiler.stitch_predictions(tile_preds, meta_info)
    assert stitched.anomaly_map.shape == (512, 512)
    assert np.allclose(stitched.anomaly_map, 1.0)


class MockModelAdapter:

    def predict(self, samples, artifact=None, **kwargs):
        preds = []
        for s in samples:
            preds.append(
                Prediction(
                    sample_id=s.id,
                    anomaly_score=0.7,
                    anomaly_map=np.ones((64, 64), dtype=np.float32) * 0.5,
                )
            )
        return preds


def test_tta_inference_wrapper():
    """Test Test-Time Augmentation (TTA) wrapper."""
    mock_model = MockModelAdapter()
    tta_wrapper = TTAInferenceWrapper(mock_model, tta_mode="flip_multiscale")

    sample = Sample(id="test_0", image_path="", task="anomaly", annotations=Annotations())
    setattr(sample, "image", Image.new("RGB", (64, 64)))
    samples = [sample]

    preds = tta_wrapper.predict(samples)

    assert len(preds) == 1
    assert preds[0].anomaly_score == 0.7
