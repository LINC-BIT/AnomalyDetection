"""Real-value tests for `evaluation.anomaly.AnomalyEvaluator`. Numbers
below were computed by actually running the evaluator, not estimated.
"""

import math

import numpy as np

from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.evaluation.anomaly import (
    AnomalyEvaluator,
    _best_f1_threshold,
    _integrate_trapezoid,
)


def _image_level_dataset():
    samples = [
        Sample(id="a", image_path="a.jpg", task="anomaly", annotations=Annotations(is_anomalous=True)),
        Sample(id="b", image_path="b.jpg", task="anomaly", annotations=Annotations(is_anomalous=True)),
        Sample(id="c", image_path="c.jpg", task="anomaly", annotations=Annotations(is_anomalous=False)),
        Sample(id="d", image_path="d.jpg", task="anomaly", annotations=Annotations(is_anomalous=False)),
    ]
    predictions = [
        Prediction(sample_id="a", anomaly_score=0.9),
        Prediction(sample_id="b", anomaly_score=0.8),
        Prediction(sample_id="c", anomaly_score=0.2),
        Prediction(sample_id="d", anomaly_score=0.1),
    ]
    return samples, predictions


def test_perfect_image_level_separation():
    samples, predictions = _image_level_dataset()
    metrics = AnomalyEvaluator(allow_oracle_threshold=True).evaluate(samples, predictions)

    assert metrics["image_auroc"] == 1.0
    assert metrics["image_f1"] == 1.0
    assert metrics["image_precision"] == 1.0
    assert metrics["image_recall"] == 1.0
    assert metrics["image_threshold"] == 0.8


def test_single_class_ground_truth_gives_nan_auroc():
    samples = [
        Sample(id="a", image_path="a.jpg", task="anomaly", annotations=Annotations(is_anomalous=True)),
        Sample(id="b", image_path="b.jpg", task="anomaly", annotations=Annotations(is_anomalous=True)),
    ]
    predictions = [
        Prediction(sample_id="a", anomaly_score=0.9),
        Prediction(sample_id="b", anomaly_score=0.1),
    ]
    metrics = AnomalyEvaluator(allow_oracle_threshold=True).evaluate(samples, predictions)
    assert math.isnan(metrics["image_auroc"])


def test_no_predictions_returns_empty_dict():
    samples, _ = _image_level_dataset()
    assert AnomalyEvaluator().evaluate(samples, []) == {}


def test_best_f1_threshold_single_class_shortcircuits():
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.1, 0.5, 0.9])
    assert _best_f1_threshold(y_true, y_score) == 0.5


def test_trapezoid_compatibility_falls_back_to_legacy_name():
    class LegacyNumpy:
        @staticmethod
        def trapz(values, coordinates):
            return 0.75

    assert _integrate_trapezoid(LegacyNumpy, [1, 2], [0, 1]) == 0.75


def test_best_f1_threshold_monotonic_scores():
    # perfectly separable at 0.5: negatives below, positives at/above
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.3, 0.6, 0.9])
    assert _best_f1_threshold(y_true, y_score) == 0.6


def _write_mask_and_map(tmp_path, name: str, shape=(10, 10), defect_box=(2, 5, 2, 5)):
    from PIL import Image

    y0, y1, x0, x1 = defect_box
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    mask_path = tmp_path / f"{name}_mask.png"
    Image.fromarray(mask).save(mask_path)

    score_map = np.zeros(shape, dtype=np.float32)
    score_map[y0:y1, x0:x1] = 0.9
    score_map[y1:, x1:] = 0.05
    map_path = tmp_path / f"{name}_map.npy"
    np.save(map_path, score_map)
    return str(mask_path), str(map_path)


def test_pixel_level_perfect_separation(tmp_path):
    mask_path, map_path = _write_mask_and_map(tmp_path, "a")
    sample = Sample(
        id="a", image_path="a.jpg", task="anomaly",
        annotations=Annotations(is_anomalous=True, anomaly_mask=mask_path),
    )
    prediction = Prediction(sample_id="a", anomaly_score=0.9, anomaly_map=map_path)

    metrics = AnomalyEvaluator(allow_oracle_threshold=True).evaluate([sample], [prediction])
    assert metrics["pixel_auroc"] == 1.0
    assert metrics["pixel_f1"] == 1.0
    assert 0.0 <= metrics["pixel_aupro"] <= 1.0
    assert metrics["pixel_aupro"] > 0.9
    assert 0.0 <= metrics["iap"] <= 1.0
    assert metrics["iap"] > 0.9


def _mixed_pixel_dataset(tmp_path):
    rng = np.random.default_rng(42)
    samples, predictions = [], []
    for i in range(6):
        is_anom = i % 2 == 0
        mask = np.zeros((20, 20), dtype=np.uint8)
        if is_anom:
            mask[5:10, 5:10] = 255
        mask_path = tmp_path / f"mask_{i}.png"
        from PIL import Image

        Image.fromarray(mask).save(mask_path)

        score_map = rng.random((20, 20)).astype(np.float32)
        if is_anom:
            score_map[5:10, 5:10] += 1.0
        map_path = tmp_path / f"map_{i}.npy"
        np.save(map_path, score_map)

        samples.append(
            Sample(
                id=str(i), image_path=f"{i}.jpg", task="anomaly",
                annotations=Annotations(is_anomalous=is_anom, anomaly_mask=str(mask_path)),
            )
        )
        predictions.append(
            Prediction(sample_id=str(i), anomaly_score=float(score_map.max()), anomaly_map=str(map_path))
        )
    return samples, predictions


def test_iap_equal_weights_small_missed_region_against_large_found_one(tmp_path):
    """A large region perfectly found + a tiny region completely missed:
    plain pixel-level metrics barely notice the miss (dominated by the
    large region's pixel count), but IAP equal-weights both regions, so it
    should sit well below the near-perfect pixel_auroc/pixel_f1 those two
    give here.
    """

    from PIL import Image

    shape = (20, 20)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[0:10, 0:10] = 255  # large region: 100px, found
    mask[15:16, 15:16] = 255  # tiny region: 1px, missed
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)

    score_map = np.zeros(shape, dtype=np.float32)
    score_map[0:10, 0:10] = 1.0  # large region scored at the dataset max
    map_path = tmp_path / "map.npy"
    np.save(map_path, score_map)

    sample = Sample(
        id="a", image_path="a.jpg", task="anomaly",
        annotations=Annotations(is_anomalous=True, anomaly_mask=str(mask_path)),
    )
    prediction = Prediction(sample_id="a", anomaly_score=1.0, anomaly_map=str(map_path))

    metrics = AnomalyEvaluator().evaluate([sample], [prediction])
    assert metrics["pixel_auroc"] > 0.99  # barely dented by missing 1 of 101 positive px
    assert 0.0 <= metrics["iap"] <= 1.0
    assert metrics["iap"] < 0.95


def test_pixel_level_subsampling_is_deterministic_for_same_seed(tmp_path):
    samples, predictions = _mixed_pixel_dataset(tmp_path)

    # max_pixels=100 < 6*400=2400 total pixels, max_aupro_images=2 < 6 images:
    # both subsampling paths are actually exercised, not just theoretically reachable.
    metrics_1 = AnomalyEvaluator(max_pixels=100, max_aupro_images=2, seed=7).evaluate(samples, predictions)
    metrics_2 = AnomalyEvaluator(max_pixels=100, max_aupro_images=2, seed=7).evaluate(samples, predictions)
    assert metrics_1 == metrics_2
