"""Tests for the MVTec LOCO adapter: MVTec-style train/test split with
logical + structural anomaly folders and per-image ground-truth mask
directories (`ground_truth/<type>/<id>/<k>.png`).
"""

import pytest

from fabric_defect_hub.datasets.mvtec_loco import MVTecLOCODataset


def _category(root, name, train_good=(), test_good=(), defects=None):
    """defects: dict[type -> list[stem]], each anomalous stem also getting a
    ground_truth/<type>/<stem>/000.png mask directory."""

    defects = defects or {}
    cat = root / name
    (cat / "train" / "good").mkdir(parents=True)
    (cat / "test" / "good").mkdir(parents=True)
    for stem in train_good:
        (cat / "train" / "good" / f"{stem}.png").write_bytes(b"n")
    for stem in test_good:
        (cat / "test" / "good" / f"{stem}.png").write_bytes(b"n")
    for defect_type, stems in defects.items():
        (cat / "test" / defect_type).mkdir(parents=True)
        for stem in stems:
            (cat / "test" / defect_type / f"{stem}.png").write_bytes(b"d")
            gt = cat / "ground_truth" / defect_type / stem
            gt.mkdir(parents=True)
            (gt / "000.png").write_bytes(b"m")


def test_train_split_is_normal_only(tmp_path):
    _category(tmp_path, "breakfast_box", train_good=["a", "b"],
              defects={"logical_anomalies": ["c"]})

    samples = MVTecLOCODataset(root=str(tmp_path), split="train", category="breakfast_box").load_samples()
    assert {s.id for s in samples} == {"breakfast_box/good/a", "breakfast_box/good/b"}


def test_logical_and_structural_both_loaded_with_masks(tmp_path):
    _category(
        tmp_path, "juice_bottle", test_good=["n0"],
        defects={"logical_anomalies": ["l0"], "structural_anomalies": ["s0"]},
    )

    samples = MVTecLOCODataset(root=str(tmp_path), split="test", category="juice_bottle").load_samples()
    by_id = {s.id: s for s in samples}

    assert by_id["juice_bottle/logical_anomalies/l0"].annotations.anomaly_mask.endswith("000.png")
    assert by_id["juice_bottle/structural_anomalies/s0"].annotations.is_anomalous is True
    assert by_id["juice_bottle/good/n0"].annotations.is_anomalous is False


def test_missing_mask_dir_leaves_mask_none(tmp_path):
    cat = tmp_path / "pushpins"
    (cat / "train" / "good").mkdir(parents=True)
    (cat / "test" / "structural_anomalies").mkdir(parents=True)
    (cat / "test" / "structural_anomalies" / "s0.png").write_bytes(b"d")  # no ground_truth dir

    samples = MVTecLOCODataset(root=str(tmp_path), split="test", category="pushpins").load_samples()
    anomaly = next(s for s in samples if s.annotations.is_anomalous)
    assert anomaly.annotations.anomaly_mask is None
    assert anomaly.annotations.is_anomalous is True


def test_unknown_category_raises(tmp_path):
    _category(tmp_path, "pushpins", train_good=["a"])
    with pytest.raises(ValueError, match="unknown MVTec LOCO category"):
        MVTecLOCODataset(root=str(tmp_path), split="test", category="nope").load_samples()
