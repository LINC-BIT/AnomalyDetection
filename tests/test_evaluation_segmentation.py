"""Real-value tests for `evaluation.segmentation.SegmentationEvaluator`."""

import numpy as np

from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.evaluation.segmentation import (
    SegmentationEvaluator,
    _dice,
    _iou,
    _load_binary_mask,
    _pixel_f1,
    _resize_like,
    _union,
)


def test_resize_like_nearest_checkerboard():
    mask = np.array([[True, False], [False, True]])
    resized = _resize_like(mask, (4, 4))
    expected = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )
    assert np.array_equal(resized, expected)


def test_resize_like_noop_when_already_matching():
    mask = np.array([[True, False], [False, True]])
    assert _resize_like(mask, (2, 2)) is mask


def test_iou_dice_pixel_f1_hand_computed():
    gt = np.array([[True, True, False], [False, False, False]])
    pred = np.array([[True, False, False], [False, False, True]])
    # intersection=1, union=3 (gt has 2, pred has 2, overlap 1)
    assert _iou(gt, pred) == 1 / 3
    assert _dice(gt, pred) == 2 * 1 / (2 + 2)
    # tp=1, fp=1, fn=1 -> precision=recall=0.5 -> f1=0.5
    assert _pixel_f1(gt, pred) == 0.5


def test_iou_dice_both_empty_is_perfect_agreement():
    empty = np.zeros((3, 3), dtype=bool)
    assert _iou(empty, empty) == 1.0
    assert _dice(empty, empty) == 1.0
    assert _pixel_f1(empty, empty) == 1.0


def test_load_binary_mask_none():
    assert _load_binary_mask(None) is None


def test_load_binary_mask_nested_list():
    mask = _load_binary_mask([[True, False], [False, True]])
    assert mask.tolist() == [[True, False], [False, True]]


def test_load_binary_mask_stack_unions_instances():
    stack = [[[True, False], [False, False]], [[False, False], [False, True]]]
    mask = _load_binary_mask(stack)
    assert mask.tolist() == [[True, False], [False, True]]


def test_load_binary_mask_file_paths(tmp_path):
    from PIL import Image

    mask_a = np.zeros((4, 4), dtype=np.uint8)
    mask_a[0, 0] = 255
    mask_b = np.zeros((4, 4), dtype=np.uint8)
    mask_b[3, 3] = 255
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    Image.fromarray(mask_a).save(path_a)
    Image.fromarray(mask_b).save(path_b)

    mask = _load_binary_mask([str(path_a), str(path_b)])
    assert mask[0, 0] and mask[3, 3]
    assert not mask[1, 1]


def test_union_logical_or():
    masks = [np.array([True, False]), np.array([False, True])]
    assert _union(masks).tolist() == [True, True]


def test_evaluator_identical_masks_perfect_score():
    mask = [[True, True], [False, False]]
    sample = Sample(id="a", image_path="a.jpg", task="segmentation", annotations=Annotations(masks=[mask]))
    prediction = Prediction(sample_id="a", masks=mask)

    metrics = SegmentationEvaluator().evaluate([sample], [prediction])
    assert metrics["miou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["pixel_f1"] == 1.0
    assert metrics["num_evaluated"] == 1.0


def test_evaluator_disjoint_masks_zero_iou():
    gt_mask = [[True, False], [False, False]]
    pred_mask = [[False, True], [False, False]]
    sample = Sample(id="a", image_path="a.jpg", task="segmentation", annotations=Annotations(masks=[gt_mask]))
    prediction = Prediction(sample_id="a", masks=pred_mask)

    metrics = SegmentationEvaluator().evaluate([sample], [prediction])
    assert metrics["miou"] == 0.0


def test_evaluator_averages_across_samples():
    perfect_mask = [[True, False], [False, False]]
    disjoint_gt = [[True, False], [False, False]]
    disjoint_pred = [[False, True], [False, False]]

    samples = [
        Sample(id="a", image_path="a.jpg", task="segmentation", annotations=Annotations(masks=[perfect_mask])),
        Sample(id="b", image_path="b.jpg", task="segmentation", annotations=Annotations(masks=[disjoint_gt])),
    ]
    predictions = [
        Prediction(sample_id="a", masks=perfect_mask),
        Prediction(sample_id="b", masks=disjoint_pred),
    ]

    metrics = SegmentationEvaluator().evaluate(samples, predictions)
    assert metrics["miou"] == 0.5
    assert metrics["dice"] == 0.5
    assert metrics["pixel_f1"] == 0.5
    assert metrics["num_evaluated"] == 2.0


def test_evaluator_cross_representation_file_path_vs_nested_list(tmp_path):
    from PIL import Image

    gt_arr = np.zeros((4, 4), dtype=np.uint8)
    gt_arr[0:2, 0:2] = 255
    gt_path = tmp_path / "gt.png"
    Image.fromarray(gt_arr).save(gt_path)

    pred_mask = np.zeros((4, 4), dtype=bool)
    pred_mask[0:2, 0:2] = True

    sample = Sample(id="a", image_path="a.jpg", task="segmentation", annotations=Annotations(masks=[str(gt_path)]))
    prediction = Prediction(sample_id="a", masks=pred_mask.tolist())

    metrics = SegmentationEvaluator().evaluate([sample], [prediction])
    assert metrics["miou"] == 1.0


def test_evaluator_skips_sample_missing_from_predictions():
    sample = Sample(id="a", image_path="a.jpg", task="segmentation", annotations=Annotations(masks=None))
    assert SegmentationEvaluator().evaluate([sample], []) == {}


def test_evaluator_excludes_both_empty_masks_from_average():
    empty = [[False, False], [False, False]]
    sample = Sample(id="normal", image_path="normal.jpg", task="segmentation", annotations=Annotations(masks=[empty]))
    prediction = Prediction(sample_id="normal", masks=empty)

    assert SegmentationEvaluator().evaluate([sample], [prediction]) == {}
