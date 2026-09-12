import json

import pytest

from fabric_defect_hub.datasets.tianchi import TianchiDataset


def _make_part(root, subdir, normals=(), defects=()):
    """`defects` is a list of (stem, [(defect_name, bbox), ...])."""

    part = root / subdir
    (part / "normal_Images").mkdir(parents=True)
    (part / "defect_Images").mkdir(parents=True)
    (part / "Annotations").mkdir(parents=True)

    for stem in normals:
        (part / "normal_Images" / f"{stem}.jpg").write_bytes(b"n")

    entries = []
    for stem, boxes in defects:
        (part / "defect_Images" / f"{stem}.jpg").write_bytes(b"d")
        for defect_name, bbox in boxes:
            entries.append({"name": f"{stem}.jpg", "defect_name": defect_name, "bbox": bbox})
    (part / "Annotations" / "anno_train.json").write_text(json.dumps(entries, ensure_ascii=False))


def _train2_root(tmp_path, normals=(), defects=()):
    _make_part(tmp_path, "guangdong1_round1_train2_20190828", normals=normals, defects=defects)
    return tmp_path


def test_normal_only_selection_excludes_defects(tmp_path):
    root = _train2_root(
        tmp_path,
        normals=[f"n{i}" for i in range(4)],
        defects=[("d0", [("破洞", [1.0, 2.0, 3.0, 4.0])])],
    )

    samples = TianchiDataset(root=str(root), split="train", parts="train2", use_defect=False).load_samples()

    assert len(samples) > 0
    assert all(sample.annotations.is_anomalous is False for sample in samples)


def test_defect_sample_carries_boxes_and_labels(tmp_path):
    root = _train2_root(
        tmp_path,
        normals=["n0"],
        defects=[("d0", [("破洞", [1.0, 2.0, 3.0, 4.0]), ("水渍", [5.0, 6.0, 7.0, 8.0])])],
    )

    samples = TianchiDataset(
        root=str(root), split="train", parts="train2", task="detection", train_ratio=1.0
    ).load_samples()
    defect_sample = next(s for s in samples if s.annotations.is_anomalous)

    assert defect_sample.annotations.boxes == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    assert defect_sample.annotations.labels == ["破洞", "水渍"]
    assert defect_sample.task == "detection"


def test_train_test_split_is_disjoint_and_deterministic(tmp_path):
    root = _train2_root(
        tmp_path,
        normals=[f"n{i}" for i in range(10)],
        defects=[(f"d{i}", [("破洞", [0.0, 0.0, 1.0, 1.0])]) for i in range(10)],
    )

    train = TianchiDataset(root=str(root), split="train", parts="train2", train_ratio=0.7).load_samples()
    test = TianchiDataset(root=str(root), split="test", parts="train2", train_ratio=0.7).load_samples()

    train_ids = {s.id for s in train}
    test_ids = {s.id for s in test}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(test_ids) == 20


def test_unknown_part_raises():
    with pytest.raises(ValueError, match="unknown tianchi part"):
        TianchiDataset(root="/nonexistent", parts="bogus").load_samples()


def test_unlabelled_test_directories_are_never_read(tmp_path):
    # testA/testB have no Annotations/normal_Images/defect_Images split at
    # all -- if the adapter ever tried to read them as a part it would
    # error immediately, since they don't have that shape.
    (tmp_path / "guangdong1_round1_testA_20190818").mkdir(parents=True)
    (tmp_path / "guangdong1_round1_testA_20190818" / "some_image.jpg").write_bytes(b"x")
    root = _train2_root(tmp_path, normals=["n0"])

    samples = TianchiDataset(root=str(root), split="train", parts="train2").load_samples()

    assert all("testA" not in sample.image_path for sample in samples)


def test_parts_all_pools_every_labelled_collection(tmp_path):
    _make_part(tmp_path, "guangdong1_round1_train1_20190828/partA", normals=["a0"])
    _make_part(tmp_path, "guangdong1_round1_train1_20190828/partB", normals=["b0"])
    _make_part(tmp_path, "guangdong1_round1_train2_20190828", normals=["t0"])

    samples = TianchiDataset(root=str(tmp_path), split="train", parts="all", train_ratio=1.0).load_samples()

    parts = {sample.metadata["part"] for sample in samples}
    assert parts == {"train1-partA", "train1-partB", "train2"}


def test_num_samples_and_defect_ratio_cap_the_selection(tmp_path):
    root = _train2_root(
        tmp_path,
        normals=[f"n{i}" for i in range(20)],
        defects=[(f"d{i}", [("破洞", [0.0, 0.0, 1.0, 1.0])]) for i in range(20)],
    )

    samples = TianchiDataset(
        root=str(root), split="train", parts="train2", num_samples=10, use_defect=True, defect_ratio=0.3, seed=0
    ).load_samples()

    assert len(samples) == 10
    n_defect = sum(1 for s in samples if s.annotations.is_anomalous)
    assert n_defect == 3
