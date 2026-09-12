"""Tests for the VisA adapter: per-category Normal/Anomaly layout, pixel
masks on the Anomaly split, synthesized normal train/test partition,
category filtering.
"""

import pytest

from fabric_defect_hub.datasets.visa import VisADataset


def _category(root, name, normals=(), anomalies=(), masks=()):
    img = root / name / "Data" / "Images"
    (img / "Normal").mkdir(parents=True)
    (img / "Anomaly").mkdir(parents=True)
    (root / name / "Data" / "Masks" / "Anomaly").mkdir(parents=True)
    for stem in normals:
        (img / "Normal" / f"{stem}.JPG").write_bytes(b"n")
    for stem in anomalies:
        (img / "Anomaly" / f"{stem}.JPG").write_bytes(b"a")
    for stem in masks:
        (root / name / "Data" / "Masks" / "Anomaly" / f"{stem}.png").write_bytes(b"m")


def test_test_split_attaches_masks_when_present(tmp_path):
    _category(tmp_path, "candle", normals=["n0"], anomalies=["a0", "a1"], masks=["a0"])

    samples = VisADataset(root=str(tmp_path), split="test", category="candle", train_ratio=0.0).load_samples()
    by_id = {s.id: s for s in samples}

    assert by_id["candle/Anomaly/a0"].annotations.anomaly_mask.endswith("a0.png")
    # a1 has no mask file -> mask stays None, sample still anomalous.
    assert by_id["candle/Anomaly/a1"].annotations.anomaly_mask is None
    assert by_id["candle/Anomaly/a1"].annotations.is_anomalous is True


def test_segmentation_task_populates_masks_list(tmp_path):
    _category(tmp_path, "candle", anomalies=["a0"], masks=["a0"])

    samples = VisADataset(
        root=str(tmp_path), split="test", category="candle", task="segmentation", train_ratio=0.0
    ).load_samples()
    anomaly = next(s for s in samples if s.annotations.is_anomalous)
    assert anomaly.annotations.masks == [anomaly.annotations.anomaly_mask]


def test_category_none_aggregates(tmp_path):
    _category(tmp_path, "candle", normals=["n0"], anomalies=["a0"])
    _category(tmp_path, "pcb1", normals=["n0"], anomalies=["a0"])

    samples = VisADataset(root=str(tmp_path), split="test", category=None, train_ratio=0.0).load_samples()
    assert {s.metadata["category"] for s in samples} == {"candle", "pcb1"}


def test_non_category_dirs_are_ignored(tmp_path):
    _category(tmp_path, "candle", normals=["n0"])
    (tmp_path / "split_csv").mkdir()
    (tmp_path / "LICENSE-DATASET").write_bytes(b"x")

    ds = VisADataset(root=str(tmp_path), split="test", train_ratio=0.0)
    assert ds._available_categories() == ["candle"]


def test_unknown_category_raises(tmp_path):
    _category(tmp_path, "candle", normals=["n0"])
    with pytest.raises(ValueError, match="unknown VisA category"):
        VisADataset(root=str(tmp_path), split="test", category="nope").load_samples()
