import pytest

from fabric_defect_hub.datasets.mvtec_ad import MVTecADDataset


def _category(root, name, train_good=(), test_good=(), defects=None):
    """defects: dict[str, list[str]] mapping defect_type -> stems, each also
    getting a matching ground-truth mask."""

    defects = defects or {}
    cat = root / name
    (cat / "train" / "good").mkdir(parents=True)
    (cat / "test" / "good").mkdir(parents=True)
    for stem in train_good:
        (cat / "train" / "good" / f"{stem}.png").write_bytes(b"normal")
    for stem in test_good:
        (cat / "test" / "good" / f"{stem}.png").write_bytes(b"normal")
    for defect_type, stems in defects.items():
        (cat / "test" / defect_type).mkdir(parents=True)
        (cat / "ground_truth" / defect_type).mkdir(parents=True)
        for stem in stems:
            (cat / "test" / defect_type / f"{stem}.png").write_bytes(b"defect")
            (cat / "ground_truth" / defect_type / f"{stem}_mask.png").write_bytes(b"mask")
    return cat


def test_single_category_train_split_is_normal_only(tmp_path):
    _category(tmp_path, "bottle", train_good=["a", "b"], defects={"broken": ["c"]})

    samples = MVTecADDataset(root=str(tmp_path), split="train", category="bottle").load_samples()

    assert {sample.id for sample in samples} == {"bottle/good/a", "bottle/good/b"}
    assert all(sample.annotations.is_anomalous is False for sample in samples)


def test_multiple_defect_types_are_all_loaded_with_masks(tmp_path):
    _category(
        tmp_path,
        "bottle",
        test_good=["n1"],
        defects={"broken_large": ["d1"], "contamination": ["d2"]},
    )

    samples = MVTecADDataset(root=str(tmp_path), split="test", category="bottle").load_samples()
    by_id = {sample.id: sample for sample in samples}

    assert by_id["bottle/good/n1"].annotations.is_anomalous is False
    assert by_id["bottle/broken_large/d1"].annotations.is_anomalous is True
    assert by_id["bottle/broken_large/d1"].annotations.anomaly_mask.endswith("d1_mask.png")
    assert by_id["bottle/contamination/d2"].annotations.is_anomalous is True
    assert by_id["bottle/contamination/d2"].metadata["defect_type"] == "contamination"


def test_category_none_aggregates_every_category(tmp_path):
    _category(tmp_path, "bottle", test_good=["b1"], defects={"broken": ["bd1"]})
    _category(tmp_path, "cable", test_good=["c1"], defects={"cut": ["cd1"]})

    samples = MVTecADDataset(root=str(tmp_path), split="test", category=None).load_samples()

    categories = {sample.metadata["category"] for sample in samples}
    assert categories == {"bottle", "cable"}
    assert len(samples) == 4


def test_unknown_category_raises_a_helpful_error(tmp_path):
    _category(tmp_path, "bottle", test_good=["b1"])

    with pytest.raises(ValueError, match="unknown MVTec AD category 'nope'"):
        MVTecADDataset(root=str(tmp_path), split="test", category="nope").load_samples()


def test_low_shot_honours_count_and_defect_ratio(tmp_path):
    _category(
        tmp_path,
        "bottle",
        test_good=[f"n{i}" for i in range(20)],
        defects={"broken": [f"d{i}" for i in range(20)]},
    )

    samples = MVTecADDataset(
        root=str(tmp_path), split="test", category="bottle", num_samples=10, defect_ratio=0.3, seed=0
    ).load_samples()

    defect_count = sum(1 for sample in samples if sample.annotations.is_anomalous)
    assert len(samples) == 10
    assert defect_count == 3


def test_use_defect_false_returns_only_normal_samples(tmp_path):
    _category(tmp_path, "bottle", test_good=["n1"], defects={"broken": ["d1"]})

    samples = MVTecADDataset(root=str(tmp_path), split="test", category="bottle", use_defect=False).load_samples()

    assert len(samples) == 1
    assert samples[0].annotations.is_anomalous is False
