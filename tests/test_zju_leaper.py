import json

import pytest

from fabric_defect_hub.datasets.zju_leaper import ZJULeaperDataset


def _dataset_root(tmp_path, index):
    image_sets = tmp_path / "ImageSets"
    image_sets.mkdir()
    (image_sets / "total.json").write_text(json.dumps(index))
    (tmp_path / "Images").mkdir()
    (tmp_path / "Annotations" / "xmls").mkdir(parents=True)
    return tmp_path


def test_invalid_imageset_structure_has_contextual_error(tmp_path):
    root = _dataset_root(tmp_path, {"normal": {"test": "not-a-list"}, "defect": {"test": []}})

    with pytest.raises(ValueError, match=r"normal\.test.*list of image-id strings"):
        ZJULeaperDataset(root=str(root)).load_samples()


def _pattern_root(tmp_path, patterns: dict[int, dict]):
    patterns_dir = tmp_path / "ImageSets" / "Patterns"
    patterns_dir.mkdir(parents=True)
    for pattern_id, index in patterns.items():
        (patterns_dir / f"pattern{pattern_id}.json").write_text(json.dumps(index))
    (tmp_path / "Images").mkdir()
    (tmp_path / "Annotations" / "xmls").mkdir(parents=True)
    return tmp_path


def _index(normal_ids, defect_ids=()):
    return {
        "normal": {"train": list(normal_ids), "test": list(normal_ids)},
        "defect": {"train": list(defect_ids), "test": list(defect_ids)},
    }


def test_multi_pattern_list_pools_and_splits_evenly(tmp_path):
    # pattern1 has far more images than pattern2/pattern3 — a naive merge-
    # and-slice would let it dominate the low-shot budget.
    root = _pattern_root(
        tmp_path,
        {
            1: _index([f"p1-{i}" for i in range(1000)]),
            2: _index([f"p2-{i}" for i in range(20)]),
            3: _index([f"p3-{i}" for i in range(20)]),
        },
    )

    samples = ZJULeaperDataset(
        root=str(root), pattern=[1, 2, 3], num_samples=30, use_defect=False, seed=0
    ).load_samples()

    prefixes = {s.id.split("-")[0] for s in samples}
    counts = {prefix: sum(1 for s in samples if s.id.startswith(prefix)) for prefix in prefixes}
    assert len(samples) == 30
    assert prefixes == {"p1", "p2", "p3"}
    assert counts == {"p1": 10, "p2": 10, "p3": 10}


def test_multi_pattern_full_data_pools_everything(tmp_path):
    root = _pattern_root(
        tmp_path,
        {1: _index(["p1-a", "p1-b"]), 2: _index(["p2-a"])},
    )

    samples = ZJULeaperDataset(root=str(root), pattern=[1, 2], use_defect=False).load_samples()

    assert {s.id for s in samples} == {"p1-a", "p1-b", "p2-a"}


def test_malformed_bbox_is_skipped_with_warning(tmp_path):
    root = _dataset_root(tmp_path, {"normal": {"test": []}, "defect": {"test": ["bad-box"]}})
    xml = root / "Annotations" / "xmls" / "bad-box.xml"
    xml.write_text(
        "<annotation><bbox><xmin>1</xmin><ymin>2</ymin>"
        "<xmax>bad</xmax></bbox></annotation>"
    )

    with pytest.warns(RuntimeWarning, match="Skipping malformed bbox"):
        samples = ZJULeaperDataset(root=str(root), task="detection").load_samples()

    assert len(samples) == 1
    assert samples[0].annotations.boxes is None
    assert samples[0].annotations.labels is None
