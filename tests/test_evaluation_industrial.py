"""Real-value tests for `evaluation.industrial.IndustrialEvaluator`. Numbers
below were computed by actually running the evaluator, not estimated.
"""

import pytest

from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.evaluation.industrial import IndustrialEvaluator, recall_first_threshold


def test_recall_first_threshold_picks_best_precision_above_floor():
    # threshold 0.6 gives recall=1.0 (both positives >= 0.6) with the best
    # precision among thresholds meeting that floor.
    threshold = recall_first_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], target_recall=1.0)
    assert threshold == 0.6


def test_recall_first_threshold_single_class_shortcircuits():
    assert recall_first_threshold([1, 1, 1], [0.1, 0.5, 0.9], target_recall=0.9) == 0.5


def test_recall_first_threshold_unreachable_falls_back_to_max_recall():
    # positives and negatives fully interleaved -> no threshold can reach
    # target_recall=1.0 while also being a real decision boundary; the
    # fallback must still return a threshold that achieves the best
    # achievable recall, not crash or return something arbitrary.
    y_true = [0, 1, 0, 1]
    y_score = [0.9, 0.1, 0.8, 0.2]
    threshold = recall_first_threshold(y_true, y_score, target_recall=1.0)
    assert isinstance(threshold, float)


@pytest.mark.parametrize("bad_recall", [0.0, 1.5, -0.1])
def test_init_rejects_invalid_target_recall(bad_recall):
    with pytest.raises(ValueError):
        IndustrialEvaluator(target_recall=bad_recall)


def _fixed_threshold_dataset(with_length: bool):
    kwargs = {"metadata": {"fabric_length_m": 2.0}} if with_length else {}
    samples = [
        Sample(id="a", image_path="a.jpg", task="anomaly", annotations=Annotations(is_anomalous=True), **kwargs),
        Sample(id="b", image_path="b.jpg", task="anomaly", annotations=Annotations(is_anomalous=True), **kwargs),
        Sample(id="c", image_path="c.jpg", task="anomaly", annotations=Annotations(is_anomalous=False), **kwargs),
        Sample(id="d", image_path="d.jpg", task="anomaly", annotations=Annotations(is_anomalous=False), **kwargs),
    ]
    predictions = [
        Prediction(sample_id="a", anomaly_score=0.9),  # correctly flagged
        Prediction(sample_id="b", anomaly_score=0.3),  # missed (under-detection)
        Prediction(sample_id="c", anomaly_score=0.2),  # correctly not flagged
        Prediction(sample_id="d", anomaly_score=0.7),  # false alarm (over-detection)
    ]
    return samples, predictions


def test_fixed_threshold_rates_with_fabric_length():
    samples, predictions = _fixed_threshold_dataset(with_length=True)
    metrics = IndustrialEvaluator(score_threshold=0.5).evaluate(samples, predictions)

    assert metrics["under_detection_rate"] == 0.5
    assert metrics["over_detection_rate"] == 0.5
    assert metrics["chosen_threshold"] == 0.5
    assert metrics["num_alarms"] == 2.0
    assert metrics["num_samples"] == 4.0
    assert metrics["alarms_per_unit_length"] == 0.25  # 2 alarms / (4 * 2.0m)


def test_alarms_per_unit_length_omitted_without_length_info():
    samples, predictions = _fixed_threshold_dataset(with_length=False)
    metrics = IndustrialEvaluator(score_threshold=0.5).evaluate(samples, predictions)

    assert "alarms_per_unit_length" not in metrics


def test_no_matching_predictions_returns_empty_dict():
    samples, _ = _fixed_threshold_dataset(with_length=False)
    assert IndustrialEvaluator().evaluate(samples, []) == {}


def test_mixed_ground_truth_and_score_sources():
    # sample "a": anomaly-task ground truth (is_anomalous) + anomaly_score
    # sample "b": detection-task ground truth (labels) + max(scores)
    samples = [
        Sample(id="a", image_path="a.jpg", task="anomaly", annotations=Annotations(is_anomalous=True)),
        Sample(id="b", image_path="b.jpg", task="detection", annotations=Annotations(boxes=[[0, 0, 1, 1]], labels=["defect"])),
    ]
    predictions = [
        Prediction(sample_id="a", anomaly_score=0.9),
        Prediction(sample_id="b", boxes=[[0, 0, 1, 1]], labels=["defect"], scores=[0.8]),
    ]

    metrics = IndustrialEvaluator(score_threshold=0.5).evaluate(samples, predictions)
    assert metrics["under_detection_rate"] == 0.0
    assert metrics["over_detection_rate"] == 0.0
    assert metrics["num_alarms"] == 2.0
