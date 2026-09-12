"""Tests for the flat-folder, image-level anomaly adapters (TILDA-400 and
the Fabric Defects Dataset) and their shared `FlatFolderAnomalyDataset` base:
synthesized leak-free train/test split, defect handling, no masks.
"""

import pytest

from fabric_defect_hub.datasets.fabric_defects import FabricDefectsDataset
from fabric_defect_hub.datasets.tilda import TILDA400Dataset


def _make_tilda(root, normals=(), defects=None):
    defects = defects or {}
    (root / "good").mkdir(parents=True)
    for stem in normals:
        (root / "good" / f"{stem}.png").write_bytes(b"n")
    for defect_type, stems in defects.items():
        (root / defect_type).mkdir(parents=True)
        for stem in stems:
            (root / defect_type / f"{stem}.png").write_bytes(b"d")


def test_train_split_is_normal_only_and_deterministic(tmp_path):
    _make_tilda(tmp_path, normals=[f"n{i}" for i in range(10)], defects={"hole": ["d0", "d1"]})

    train = TILDA400Dataset(root=str(tmp_path), split="train", train_ratio=0.6).load_samples()

    assert all(s.annotations.is_anomalous is False for s in train)
    # 60% of 10 normals -> first 6 by sorted position.
    assert len(train) == 6


def test_train_and_test_normal_pools_are_disjoint(tmp_path):
    _make_tilda(tmp_path, normals=[f"n{i}" for i in range(10)], defects={"hole": ["d0"]})

    train = TILDA400Dataset(root=str(tmp_path), split="train", train_ratio=0.5).load_samples()
    test = TILDA400Dataset(root=str(tmp_path), split="test", train_ratio=0.5).load_samples()

    train_imgs = {s.image_path for s in train}
    test_normal_imgs = {s.image_path for s in test if not s.annotations.is_anomalous}
    assert train_imgs.isdisjoint(test_normal_imgs)
    # Every normal image appears in exactly one of the two splits.
    assert len(train_imgs) + len(test_normal_imgs) == 10


def test_test_split_includes_defects_without_masks(tmp_path):
    _make_tilda(tmp_path, normals=["n0", "n1"], defects={"hole": ["d0"], "oil spot": ["d1"]})

    test = TILDA400Dataset(root=str(tmp_path), split="test", train_ratio=0.0).load_samples()
    defect = [s for s in test if s.annotations.is_anomalous]

    assert {s.metadata["defect_type"] for s in defect} == {"hole", "oil spot"}
    assert all(s.annotations.anomaly_mask is None for s in defect)  # no pixel masks


def test_use_defect_false_excludes_defects(tmp_path):
    _make_tilda(tmp_path, normals=["n0", "n1"], defects={"hole": ["d0"]})

    test = TILDA400Dataset(root=str(tmp_path), split="test", use_defect=False, train_ratio=0.0).load_samples()
    assert all(s.annotations.is_anomalous is False for s in test)


def test_num_samples_honours_defect_ratio(tmp_path):
    _make_tilda(
        tmp_path,
        normals=[f"n{i}" for i in range(20)],
        defects={"hole": [f"d{i}" for i in range(20)]},
    )

    test = TILDA400Dataset(
        root=str(tmp_path), split="test", num_samples=10, defect_ratio=0.5, train_ratio=0.0
    ).load_samples()

    n_defect = sum(s.annotations.is_anomalous for s in test)
    assert len(test) == 10
    assert n_defect == 5


def test_fabric_defects_reads_nested_subdir(tmp_path):
    base = tmp_path / "Fabric Defect Dataset"
    (base / "defect free").mkdir(parents=True)
    (base / "stain").mkdir(parents=True)
    (base / "defect free" / "1.jpg").write_bytes(b"n")
    (base / "stain" / "2.jpg").write_bytes(b"d")

    test = FabricDefectsDataset(root=str(tmp_path), split="test", train_ratio=0.0).load_samples()

    by_anom = {s.annotations.is_anomalous for s in test}
    assert by_anom == {False, True}
    assert any(s.metadata["defect_type"] == "stain" for s in test)


def test_fabric_defects_processed_files_are_masks_not_photos(tmp_path):
    # hole/Vertical/horizontal ship "<stem>_processed (<n>)" binary masks
    # alongside the base photo -- these must never be counted as their own
    # Sample, and must be attached as anomaly_mask on the base image instead.
    base = tmp_path / "Fabric Defect Dataset"
    (base / "defect free").mkdir(parents=True)
    (base / "hole").mkdir(parents=True)
    (base / "hole" / "10.jpg").write_bytes(b"photo")
    (base / "hole" / "10_processed (1).jpg").write_bytes(b"mask1")
    (base / "hole" / "10_processed (2).jpg").write_bytes(b"mask2")

    test = FabricDefectsDataset(root=str(tmp_path), split="test", train_ratio=0.0).load_samples()

    assert len(test) == 1  # only the base photo, masks are not separate samples
    sample = test[0]
    assert sample.id == "hole/10"
    assert sample.annotations.anomaly_mask.endswith("10_processed (1).jpg")


def test_fabric_defects_segmentation_task_attaches_all_masks(tmp_path):
    base = tmp_path / "Fabric Defect Dataset"
    (base / "defect free").mkdir(parents=True)
    (base / "hole").mkdir(parents=True)
    (base / "hole" / "10.jpg").write_bytes(b"photo")
    (base / "hole" / "10_processed (1).jpg").write_bytes(b"mask1")
    (base / "hole" / "10_processed (2).jpg").write_bytes(b"mask2")

    test = FabricDefectsDataset(
        root=str(tmp_path), split="test", task="segmentation", train_ratio=0.0
    ).load_samples()

    assert len(test[0].annotations.masks) == 2


def test_fabric_defects_handles_processed_naming_without_space(tmp_path):
    # The raw dataset mixes "_processed (1)" and "_processed(1)" (no space)
    # for the same convention -- both must be recognised as masks.
    base = tmp_path / "Fabric Defect Dataset"
    (base / "defect free").mkdir(parents=True)
    (base / "horizontal").mkdir(parents=True)
    (base / "horizontal" / "15.jpg").write_bytes(b"photo")
    (base / "horizontal" / "15_processed(2).jpg").write_bytes(b"mask")

    test = FabricDefectsDataset(root=str(tmp_path), split="test", train_ratio=0.0).load_samples()

    assert len(test) == 1
    assert test[0].annotations.anomaly_mask.endswith("15_processed(2).jpg")


def test_fabric_defects_photo_without_mask_leaves_anomaly_mask_none(tmp_path):
    base = tmp_path / "Fabric Defect Dataset"
    (base / "defect free").mkdir(parents=True)
    (base / "lines").mkdir(parents=True)  # lines has no masks in the real dataset
    (base / "lines" / "1.jpg").write_bytes(b"photo")

    test = FabricDefectsDataset(root=str(tmp_path), split="test", train_ratio=0.0).load_samples()

    assert test[0].annotations.anomaly_mask is None


def test_invalid_split_raises(tmp_path):
    _make_tilda(tmp_path, normals=["n0"])
    with pytest.raises(ValueError, match="split must be"):
        TILDA400Dataset(root=str(tmp_path), split="val")
