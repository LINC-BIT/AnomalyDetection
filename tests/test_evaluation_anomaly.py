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


def test_nonfinite_scores_are_reported_without_failing_the_remaining_evaluation():
    samples, predictions = _image_level_dataset()
    predictions[0].anomaly_score = float("nan")

    metrics = AnomalyEvaluator().evaluate(samples, predictions)

    assert metrics["invalid_anomaly_score_count"] == 1.0
    assert metrics["image_auroc"] == 1.0


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


def test_calibration_fits_a_threshold_the_test_split_never_saw():
    """The threshold comes from a *calibration* split and is then handed to
    the evaluator that scores a different one -- the only way
    `image_f1`/`precision`/`recall` can be reported without the evaluator
    falling back to the same-set optimum it refuses to pick itself."""

    from fabric_defect_hub.evaluation.anomaly import calibrate_thresholds

    samples, predictions = _image_level_dataset()
    threshold, pixel_threshold, scored = calibrate_thresholds(samples, predictions)

    assert scored == 4
    # Every defective score sits above every normal one, so an F1-optimal
    # threshold lands inside the gap.
    assert 0.2 < threshold <= 0.9
    # These predictions carry no anomaly map, so there are no pixels to fit.
    assert pixel_threshold is None

    scored_metrics = AnomalyEvaluator(image_threshold=threshold).evaluate(samples, predictions)
    assert scored_metrics["image_f1"] == 1.0
    assert scored_metrics["image_precision"] == 1.0
    assert scored_metrics["image_recall"] == 1.0
    assert scored_metrics["image_threshold"] == threshold


def test_calibration_fits_a_pixel_threshold_from_the_anomaly_maps(tmp_path):
    """`pixel_f1` has the same problem `image_f1` had: it needs a threshold,
    and the evaluator will not fit one on the split it reports on. The
    calibration pass flattens the same pixels the evaluator will score."""

    from fabric_defect_hub.evaluation.anomaly import calibrate_thresholds

    samples, predictions = _mixed_pixel_dataset(tmp_path)
    image_threshold, pixel_threshold, scored = calibrate_thresholds(samples, predictions)

    assert scored == 6
    assert image_threshold is not None
    # The defect regions are scored above the rest, so a separating pixel
    # threshold exists.
    assert pixel_threshold is not None

    metrics = AnomalyEvaluator(pixel_threshold=pixel_threshold).evaluate(samples, predictions)
    assert metrics["pixel_f1"] > 0.9
    assert 0.0 <= metrics["pixel_auroc"] <= 1.0


def test_calibration_reports_no_pixel_threshold_for_a_single_class_calibration_set(tmp_path):
    """A normal-only calibration split has no defective pixels, so the pixel
    threshold cannot be fitted even when the maps are there."""

    from fabric_defect_hub.evaluation.anomaly import calibrate_thresholds

    samples, predictions = _mixed_pixel_dataset(tmp_path)
    normal_only = [s for s in samples if not s.annotations.is_anomalous]
    normal_ids = {s.id for s in normal_only}
    normal_predictions = [p for p in predictions if p.sample_id in normal_ids]

    image_threshold, pixel_threshold, scored = calibrate_thresholds(normal_only, normal_predictions)
    assert image_threshold is None
    assert pixel_threshold is None
    assert scored == len(normal_only)


def test_calibration_refuses_a_single_class_split():
    """MVTec AD and the flat-folder datasets hand out normal-only training
    images: nothing separates, so the caller is told `None` and has to leave
    the thresholded metrics unmeasured rather than publish a made-up 0.5."""

    from fabric_defect_hub.evaluation.anomaly import calibrate_thresholds

    samples = [
        Sample(id="a", image_path="a.jpg", task="anomaly", annotations=Annotations(is_anomalous=False)),
        Sample(id="b", image_path="b.jpg", task="anomaly", annotations=Annotations(is_anomalous=False)),
    ]
    predictions = [
        Prediction(sample_id="a", anomaly_score=0.9),
        Prediction(sample_id="b", anomaly_score=0.1),
    ]

    threshold, pixel_threshold, scored = calibrate_thresholds(samples, predictions)
    assert threshold is None
    # No anomaly maps were supplied either, so the pixel threshold is
    # unmeasured for the same reason.
    assert pixel_threshold is None
    assert scored == 2


def test_calibration_ignores_predictions_with_no_usable_score():
    """A missing or non-finite score cannot vote on a threshold; counting it
    as a normal image (score 0) would drag the fitted threshold down."""

    from fabric_defect_hub.evaluation.anomaly import calibrate_thresholds

    samples = [
        Sample(id="a", image_path="a.jpg", task="anomaly", annotations=Annotations(is_anomalous=True)),
        Sample(id="b", image_path="b.jpg", task="anomaly", annotations=Annotations(is_anomalous=False)),
        Sample(id="c", image_path="c.jpg", task="anomaly", annotations=Annotations(is_anomalous=True)),
        Sample(id="d", image_path="d.jpg", task="anomaly", annotations=Annotations(is_anomalous=False)),
    ]
    predictions = [
        Prediction(sample_id="a", anomaly_score=0.9),
        Prediction(sample_id="b", anomaly_score=0.1),
        Prediction(sample_id="c", anomaly_score=float("nan")),
        Prediction(sample_id="d", anomaly_score=None),
    ]

    threshold, _pixel_threshold, scored = calibrate_thresholds(samples, predictions)
    assert scored == 2  # the two usable scores, not four
    assert 0.1 < threshold <= 0.9


def test_a_saturated_prediction_set_is_reported_not_scored():
    """One score for every image is a model whose output saturated, not a
    model that scores 0.5 AUROC. The evaluator cannot fix that, but it can
    refuse to let it pass for a measurement."""

    from fabric_defect_hub.evaluation.anomaly import DEGENERATE_SCORE_MIN_SAMPLES

    count = DEGENERATE_SCORE_MIN_SAMPLES + 2
    samples = [
        Sample(
            id=f"s{i}", image_path=f"{i}.jpg", task="anomaly",
            annotations=Annotations(is_anomalous=bool(i % 2)),
        )
        for i in range(count)
    ]
    saturated = [Prediction(sample_id=s.id, anomaly_score=1.0) for s in samples]
    varying = [
        Prediction(sample_id=s.id, anomaly_score=0.9 if s.annotations.is_anomalous else 0.1)
        for s in samples
    ]

    saturated_metrics = AnomalyEvaluator().evaluate(samples, saturated)
    varying_metrics = AnomalyEvaluator().evaluate(samples, varying)

    assert saturated_metrics["constant_anomaly_scores"] == float(count)
    # AUROC is still 0.5 and AUPRO/IAP are absent here -- the point of the
    # flag is that those numbers must not be read as a measurement.
    assert saturated_metrics["image_auroc"] == 0.5
    assert "constant_anomaly_scores" not in varying_metrics
    assert varying_metrics["image_auroc"] == 1.0


def test_a_small_tied_prediction_set_is_not_called_saturated():
    """Two images can legitimately share a score; that is not a diagnosis."""

    from fabric_defect_hub.evaluation.anomaly import DEGENERATE_SCORE_MIN_SAMPLES

    count = DEGENERATE_SCORE_MIN_SAMPLES - 1
    samples = [
        Sample(
            id=f"s{i}", image_path=f"{i}.jpg", task="anomaly",
            annotations=Annotations(is_anomalous=bool(i % 2)),
        )
        for i in range(count)
    ]
    tied = [Prediction(sample_id=s.id, anomaly_score=0.5) for s in samples]

    assert "constant_anomaly_scores" not in AnomalyEvaluator().evaluate(samples, tied)
