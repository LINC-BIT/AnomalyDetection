from fabric_defect_hub.core.dataset_capabilities import (
    capabilities_for,
    default_dataset_roots,
    names_with_role,
)


def test_tianchi_is_both_a_detection_and_anomaly_training_source():
    caps = capabilities_for("tianchi")
    assert caps.supports("detection_train")
    assert caps.supports("anomaly_train")
    assert caps.supports("fabric_train_member")
    assert caps.default_root == "datasets/textile/tianchi"


def test_cross_domain_benchmarks_are_not_anomaly_trainable():
    for name in ("mvtec-ad", "mvtec-loco", "visa"):
        assert not capabilities_for(name).supports("anomaly_train")
        assert capabilities_for(name).supports("zero_shot_train")


def test_fabric_train_composite_is_not_its_own_member():
    assert "fabric-train" not in names_with_role("fabric_train_member")


def test_unknown_dataset_has_no_roles_or_root():
    caps = capabilities_for("some-future-dataset")
    assert caps.roles == frozenset()
    assert caps.default_root is None


def test_default_dataset_roots_include_every_declared_dataset():
    roots = default_dataset_roots()
    assert roots["tianchi"] == "datasets/textile/tianchi"
    assert roots["zju-leaper"] == "datasets/textile/ZJU-Leaper"
