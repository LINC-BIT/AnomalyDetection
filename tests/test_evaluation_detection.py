"""Real-value tests for `evaluation.detection.DetectionEvaluator`. Numbers
below were computed by actually running the evaluator, not estimated.
"""

import pytest

from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.evaluation.detection import (
    DetectionEvaluator,
    _box_iou,
    _precision_recall_f1,
    quantization_recall_decay,
    recall_by_size,
)


def test_box_iou_partial_overlap():
    # intersection = 5x5 = 25, union = 100+100-25 = 175
    assert _box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175


def test_box_iou_no_overlap():
    assert _box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_box_iou_identical():
    assert _box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def _perfect_match_pair():
    sample = Sample(
        id="a", image_path="a.jpg", task="detection",
        annotations=Annotations(boxes=[[10, 10, 50, 50]], labels=["defect"]),
    )
    prediction = Prediction(sample_id="a", boxes=[[10, 10, 50, 50]], labels=["defect"], scores=[0.9])
    return sample, prediction


def test_perfect_match_gives_perfect_scores():
    sample, prediction = _perfect_match_pair()
    metrics = DetectionEvaluator().evaluate([sample], [prediction])

    assert metrics["map"] == 1.0
    assert metrics["map_50"] == 1.0
    assert metrics["precision_at_threshold"] == 1.0
    assert metrics["recall_at_threshold"] == 1.0
    assert metrics["f1_at_threshold"] == 1.0
    assert metrics["true_positives"] == 1.0
    assert metrics["false_positives"] == 0.0
    assert metrics["false_negatives"] == 0.0


def test_disjoint_prediction_scores_zero():
    sample = Sample(
        id="a", image_path="a.jpg", task="detection",
        annotations=Annotations(boxes=[[10, 10, 50, 50]], labels=["defect"]),
    )
    prediction = Prediction(sample_id="a", boxes=[[200, 200, 240, 240]], labels=["defect"], scores=[0.9])

    metrics = DetectionEvaluator().evaluate([sample], [prediction])

    assert metrics["map"] == 0.0
    assert metrics["map_50"] == 0.0
    assert metrics["false_positives"] == 1.0
    assert metrics["false_negatives"] == 1.0


def test_empty_prediction_boxes_only_contributes_false_negative():
    sample = Sample(
        id="a", image_path="a.jpg", task="detection",
        annotations=Annotations(boxes=[[10, 10, 50, 50]], labels=["defect"]),
    )
    prediction = Prediction(sample_id="a", boxes=[], labels=[], scores=[])

    metrics = DetectionEvaluator().evaluate([sample], [prediction])

    assert metrics["true_positives"] == 0.0
    assert metrics["false_positives"] == 0.0
    assert metrics["false_negatives"] == 1.0


def test_no_matching_predictions_returns_empty_dict():
    sample, _ = _perfect_match_pair()
    assert DetectionEvaluator().evaluate([sample], []) == {}


def test_class_names_fixes_ordering():
    ev = DetectionEvaluator(class_names=["background_like", "defect"])
    class_map = ev._build_class_map([])
    assert class_map == {"background_like": 0, "defect": 1}


def test_class_names_inferred_when_omitted():
    sample, prediction = _perfect_match_pair()
    ev = DetectionEvaluator()
    class_map = ev._build_class_map([(sample, prediction)])
    assert class_map == {"defect": 0}


def test_precision_recall_f1_hand_computed_scenario():
    # sample "a": one gt box matched by one correct prediction (tp=1)
    # sample "b": one gt box, no prediction at all (fn=1)
    # sample "c": no gt box, one spurious prediction (fp=1)
    a = Sample(id="a", image_path="a.jpg", task="detection", annotations=Annotations(boxes=[[0, 0, 10, 10]], labels=["defect"]))
    a_pred = Prediction(sample_id="a", boxes=[[0, 0, 10, 10]], labels=["defect"], scores=[0.9])
    b = Sample(id="b", image_path="b.jpg", task="detection", annotations=Annotations(boxes=[[0, 0, 10, 10]], labels=["defect"]))
    b_pred = Prediction(sample_id="b", boxes=[], labels=[], scores=[])
    c = Sample(id="c", image_path="c.jpg", task="detection", annotations=Annotations(boxes=[], labels=[]))
    c_pred = Prediction(sample_id="c", boxes=[[5, 5, 15, 15]], labels=["defect"], scores=[0.9])

    pairs = [(a, a_pred), (b, b_pred), (c, c_pred)]
    result = _precision_recall_f1(pairs, class_map={"defect": 0}, score_threshold=0.5)

    assert result["true_positives"] == 1.0
    assert result["false_positives"] == 1.0
    assert result["false_negatives"] == 1.0
    assert result["precision_at_threshold"] == 0.5
    assert result["recall_at_threshold"] == 0.5
    assert result["f1_at_threshold"] == 0.5


def test_recall_by_size_buckets_small_and_normal_defects_separately():
    # "small": an 8x8 gt box (shorter side 8 < 10px), matched -> tp
    small_hit = Sample(
        id="s1", image_path="s1.jpg", task="detection",
        annotations=Annotations(boxes=[[0, 0, 8, 8]], labels=["defect"]),
    )
    small_hit_pred = Prediction(sample_id="s1", boxes=[[0, 0, 8, 8]], labels=["defect"], scores=[0.9])

    # "small": another 8x8 gt box, missed entirely -> fn
    small_miss = Sample(
        id="s2", image_path="s2.jpg", task="detection",
        annotations=Annotations(boxes=[[100, 100, 108, 108]], labels=["defect"]),
    )
    small_miss_pred = Prediction(sample_id="s2", boxes=[], labels=[], scores=[])

    # "normal": a 50x50 gt box (shorter side 50 >= 10px), matched -> tp
    normal_hit = Sample(
        id="n1", image_path="n1.jpg", task="detection",
        annotations=Annotations(boxes=[[0, 0, 50, 50]], labels=["defect"]),
    )
    normal_hit_pred = Prediction(sample_id="n1", boxes=[[0, 0, 50, 50]], labels=["defect"], scores=[0.9])

    samples = [small_hit, small_miss, normal_hit]
    predictions = [small_hit_pred, small_miss_pred, normal_hit_pred]

    result = recall_by_size(samples, predictions, small_max_px=10.0)
    assert result["recall_small"] == pytest.approx(0.5)  # 1 tp, 1 fn
    assert result["recall_normal"] == pytest.approx(1.0)  # 1 tp, 0 fn


def test_recall_by_size_empty_bucket_is_zero_not_error():
    sample = Sample(
        id="n1", image_path="n1.jpg", task="detection",
        annotations=Annotations(boxes=[[0, 0, 50, 50]], labels=["defect"]),
    )
    prediction = Prediction(sample_id="n1", boxes=[[0, 0, 50, 50]], labels=["defect"], scores=[0.9])
    result = recall_by_size([sample], [prediction], small_max_px=10.0)
    assert result["recall_small"] == 0.0
    assert result["recall_normal"] == pytest.approx(1.0)


def test_quantization_recall_decay_hand_computed():
    result = quantization_recall_decay(
        recall_fp32_small=0.80, recall_quant_small=0.55,
        recall_fp32_normal=0.90, recall_quant_normal=0.85,
    )
    assert result["delta_recall_small"] == pytest.approx(0.25)
    assert result["delta_recall_normal"] == pytest.approx(0.05)
