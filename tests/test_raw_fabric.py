from fabric_defect_hub.datasets.raw_fabric import RawFabricDataset


def _dataset_root(tmp_path, train_good=(), test_good=(), test_defect=(), masks=()):
    mvtec = tmp_path / "MVTec"
    (mvtec / "train" / "good").mkdir(parents=True)
    (mvtec / "test" / "good").mkdir(parents=True)
    (mvtec / "test" / "defect").mkdir(parents=True)
    (mvtec / "ground_truth" / "defect").mkdir(parents=True)
    for name in train_good:
        (mvtec / "train" / "good" / f"{name}.png").write_bytes(b"normal")
    for name in test_good:
        (mvtec / "test" / "good" / f"{name}.png").write_bytes(b"normal")
    for name in test_defect:
        (mvtec / "test" / "defect" / f"{name}.png").write_bytes(b"defect")
    for name in masks:
        (mvtec / "ground_truth" / "defect" / f"{name}.png").write_bytes(b"mask")
    return tmp_path


def test_train_split_is_normal_only_even_when_use_defect_is_true(tmp_path):
    root = _dataset_root(tmp_path, train_good=["a", "b"], test_defect=["c"])

    samples = RawFabricDataset(root=str(root), split="train", use_defect=True).load_samples()

    assert {sample.id for sample in samples} == {"a", "b"}
    assert all(sample.annotations.is_anomalous is False for sample in samples)


def test_test_split_loads_normal_and_defect_with_masks(tmp_path):
    root = _dataset_root(
        tmp_path, test_good=["n1"], test_defect=["d1"], masks=["d1"]
    )

    samples = RawFabricDataset(root=str(root), split="test").load_samples()
    by_id = {sample.id: sample for sample in samples}

    assert by_id["n1"].annotations.is_anomalous is False
    assert by_id["d1"].annotations.is_anomalous is True
    assert by_id["d1"].annotations.anomaly_mask is not None
    assert by_id["d1"].annotations.anomaly_mask.endswith("d1.png")


def test_num_samples_none_uses_everything(tmp_path):
    root = _dataset_root(
        tmp_path,
        test_good=[f"n{i}" for i in range(5)],
        test_defect=[f"d{i}" for i in range(3)],
        masks=[f"d{i}" for i in range(3)],
    )

    samples = RawFabricDataset(root=str(root), split="test", num_samples=None).load_samples()

    assert len(samples) == 8


def test_low_shot_honours_count_and_defect_ratio(tmp_path):
    root = _dataset_root(
        tmp_path,
        test_good=[f"n{i}" for i in range(20)],
        test_defect=[f"d{i}" for i in range(20)],
        masks=[f"d{i}" for i in range(20)],
    )

    samples = RawFabricDataset(
        root=str(root), split="test", num_samples=10, defect_ratio=0.3, seed=0
    ).load_samples()

    defect_count = sum(1 for sample in samples if sample.annotations.is_anomalous)
    assert len(samples) == 10
    assert defect_count == 3


def test_use_defect_false_returns_only_normal_samples(tmp_path):
    root = _dataset_root(tmp_path, test_good=["n1"], test_defect=["d1"], masks=["d1"])

    samples = RawFabricDataset(root=str(root), split="test", use_defect=False).load_samples()

    assert len(samples) == 1
    assert samples[0].annotations.is_anomalous is False
