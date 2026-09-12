"""Tests for the `fabric-train` composite dataset: unions the in-domain
fabric members, prefixes ids per source, applies the global sample budget,
and skips members whose directory is absent.
"""

import fabric_defect_hub.datasets  # noqa: F401  (registers member adapters)
from fabric_defect_hub.datasets.fabric_train import FabricTrainDataset


def _make_flat(base, dirname, normals):
    (base / dirname / "good").mkdir(parents=True)
    for stem in normals:
        (base / dirname / "good" / f"{stem}.png").write_bytes(b"n")


def _make_fabric_defects(base, normals):
    d = base / "Fabric Defects Dataset" / "Fabric Defect Dataset" / "defect free"
    d.mkdir(parents=True)
    for stem in normals:
        (d / f"{stem}.png").write_bytes(b"n")


def test_unions_present_members_and_skips_absent(tmp_path):
    # Only TILDA-400 and Fabric Defects present; ZJU-Leaper / RAW_FABRID
    # absent -> silently skipped, not an error.
    _make_flat(tmp_path, "TILDA_400", ["t0", "t1"])
    _make_fabric_defects(tmp_path, ["f0", "f1", "f2"])

    samples = FabricTrainDataset(root=str(tmp_path), split="train").load_samples()

    sources = {s.metadata["source_dataset"] for s in samples}
    assert sources == {"tilda-400", "fabric-defects"}
    assert all(s.annotations.is_anomalous is False for s in samples)  # train = normal only


def test_ids_are_prefixed_by_source(tmp_path):
    _make_flat(tmp_path, "TILDA_400", ["t0"])

    samples = FabricTrainDataset(root=str(tmp_path), split="train").load_samples()
    assert all(s.id.startswith("tilda-400/") for s in samples)


def test_global_num_samples_caps_the_union(tmp_path):
    _make_flat(tmp_path, "TILDA_400", [f"t{i}" for i in range(10)])
    _make_fabric_defects(tmp_path, [f"f{i}" for i in range(10)])

    samples = FabricTrainDataset(root=str(tmp_path), split="train", num_samples=5).load_samples()
    assert len(samples) == 5


def test_empty_base_dir_yields_nothing(tmp_path):
    assert FabricTrainDataset(root=str(tmp_path), split="train").load_samples() == []
