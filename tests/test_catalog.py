"""Fast, framework-free tests for `fabric_defect_hub.catalog`: the
train->frontend bridge (see catalog.py's module docstring).
"""

from pathlib import Path

import pytest

from fabric_defect_hub.catalog import (
    CANONICAL_MODELS,
    find_canonical_model,
    metadata_for,
    publish_artifact,
    published_path,
)


def test_canonical_models_has_twenty_entries():
    assert len(CANONICAL_MODELS) == 20


def test_canonical_model_keys_are_unique():
    keys = [model.key for model in CANONICAL_MODELS]
    assert len(keys) == len(set(keys))


def test_canonical_model_labels_are_unique():
    labels = [model.label for model in CANONICAL_MODELS]
    assert len(labels) == len(set(labels))


@pytest.mark.parametrize("model", CANONICAL_MODELS, ids=lambda entry: entry.key)
def test_every_published_model_is_reachable_by_its_own_name(model):
    """`fdh train <name>` must work for every model this project publishes,
    and must land on the config that entry itself declares.

    This is the invariant that was broken before keyword resolution grew its
    catalog step: half of these (PaDiM, RD4AD, WinCLIP, DETR, UNet++, ...)
    were unreachable by name because no config declared them, and PatchCore
    was ambiguous across three configs. Parameterized over the catalog so
    adding an entry without a resolvable config fails here rather than in
    somebody's training run.
    """

    from fabric_defect_hub.training import resolve_model_config_and_variant

    path, variant = resolve_model_config_and_variant(model.variant)

    assert path.name == model.config
    assert variant is not None
    assert variant.strip().lower() == model.variant.strip().lower()


def test_every_catalogued_config_exists_on_disk():
    """`CanonicalModel.config` is what keyword resolution reads to answer
    "which config trains this model", so a typo there is a broken
    `fdh train <name>`, not just stale documentation.
    """

    from fabric_defect_hub.training import DEFAULT_MODEL_CONFIG_DIR

    missing = sorted(
        {model.config for model in CANONICAL_MODELS
         if not (Path(DEFAULT_MODEL_CONFIG_DIR) / model.config).is_file()}
    )
    assert not missing, f"catalog names configs that do not exist: {missing}"


@pytest.mark.parametrize(
    ("backend", "variant", "expected_key"),
    [
        ("ultralytics", "yolov8n", "yolov8n"),
        ("ultralytics", "YOLOV8N", "yolov8n"),  # case-insensitive
        ("torchvision", "cascadercnn_resnet50_fpn", "cascadercnn_resnet50_fpn"),
        ("torchvision", "detr_resnet50", "detr_resnet50"),
        ("torchvision", "unetplusplus_resnet34", "unetplusplus_resnet34"),
        ("torchvision", "deeplabv3plus_resnet50", "deeplabv3plus_resnet50"),
        ("anomalib", "PatchCore", "PatchCore"),
        ("anomalib", "patchcore", "PatchCore"),  # case-insensitive
        ("anomalib", "WinClip", "WinCLIP"),
        ("anomalib", "winclip", "WinCLIP"),  # case-insensitive
        ("anomalib", "STFPM", "STFPM"),
        ("anomalib", "Stfpm", "STFPM"),  # case-insensitive
        ("anomalib", "GANomaly", "GANomaly"),
        ("anomalib", "ganomaly", "GANomaly"),  # case-insensitive
        ("dinomaly", "dinov2reg_vit_base_14", "Dinomaly"),
        ("moeclip", "ViT-L-14-336", "MoECLIP"),
        ("moeclip", "vit-l-14-336", "MoECLIP"),  # case-insensitive
        ("mambaad", "resnet34", "MambaAD"),
    ],
)
def test_find_canonical_model_matches(backend, variant, expected_key):
    model = find_canonical_model(backend, variant)
    assert model is not None
    assert model.key == expected_key


def test_find_canonical_model_returns_none_for_unknown_variant():
    assert find_canonical_model("ultralytics", "yolov5x") is None


def test_find_canonical_model_returns_none_for_backend_mismatch():
    # "yolov8n" exists, but not under torchvision
    assert find_canonical_model("torchvision", "yolov8n") is None


def test_published_path_uses_pt_for_ultralytics_and_torchvision():
    yolo = find_canonical_model("ultralytics", "yolov8n")
    tv = find_canonical_model("torchvision", "fasterrcnn_resnet50_fpn")
    assert published_path(yolo).suffix == ".pt"
    assert published_path(tv).suffix == ".pt"


def test_published_path_uses_ckpt_for_anomalib():
    model = find_canonical_model("anomalib", "PatchCore")
    assert published_path(model).suffix == ".ckpt"


def test_published_path_uses_pth_for_dinomaly():
    model = find_canonical_model("dinomaly", "dinov2reg_vit_base_14")
    assert published_path(model).suffix == ".pth"


def test_published_path_uses_pth_for_moeclip():
    model = find_canonical_model("moeclip", "ViT-L-14-336")
    assert published_path(model).suffix == ".pth"


def test_published_path_uses_pth_for_mambaad():
    model = find_canonical_model("mambaad", "resnet34")
    assert published_path(model).suffix == ".pth"


def test_published_path_is_keyed_by_model_key():
    model = find_canonical_model("ultralytics", "yolov8s")
    assert published_path(model).name == "yolov8s.pt"


def test_metadata_for_non_anomalib_has_no_model_class():
    model = find_canonical_model("ultralytics", "yolov8n")
    metadata = metadata_for(model)
    assert metadata["trusted"] is True
    assert "model_class" not in metadata


def test_metadata_for_anomalib_resolves_model_class():
    model = find_canonical_model("anomalib", "RD4AD")
    metadata = metadata_for(model)
    assert metadata["trusted"] is True
    assert metadata["model_class"] == "ReverseDistillation"


def test_metadata_for_winclip_resolves_model_class():
    model = find_canonical_model("anomalib", "WinClip")
    metadata = metadata_for(model)
    assert metadata["trusted"] is True
    assert metadata["model_class"] == "WinClip"
    assert metadata["zero_shot"] is True
    assert metadata["model_kwargs"]["k_shot"] == 0
    assert metadata["model_kwargs"]["class_name"] == "fabric"


def test_metadata_for_dinomaly_resolves_architecture_fields():
    model = find_canonical_model("dinomaly", "dinov2reg_vit_base_14")
    metadata = metadata_for(model)
    assert metadata["trusted"] is True
    assert metadata["encoder_name"] == "dinov2reg_vit_base_14"
    assert metadata["target_layers"] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert metadata["image_size"] == 448
    assert metadata["crop_size"] == 392


def test_metadata_for_moeclip_resolves_architecture_fields():
    model = find_canonical_model("moeclip", "ViT-L-14-336")
    metadata = metadata_for(model)
    assert metadata["trusted"] is True
    assert metadata["model_class"] == "MoECLIP"
    assert metadata["model_name"] == "ViT-L-14-336"
    # The architecture knobs predict() needs to rebuild the same model.
    assert metadata["img_size"] == 518
    assert metadata["moe_layers"] == [5, 11, 17, 23]
    assert metadata["moe_num_experts"] == 4


def test_metadata_for_mambaad_resolves_architecture_fields():
    model = find_canonical_model("mambaad", "resnet34")
    metadata = metadata_for(model)
    assert metadata["trusted"] is True
    assert metadata["model_class"] == "MambaADNet"
    assert metadata["encoder_name"] == "resnet34"
    assert metadata["dims_decoder"] == [512, 256, 128, 64]
    assert metadata["scan_type"] == "hilbert"
    assert metadata["num_direction"] == 8


def test_publish_artifact_returns_none_for_non_canonical_variant(tmp_path):
    src = tmp_path / "weights.pt"
    src.write_bytes(b"fake")
    assert publish_artifact("ultralytics", "yolov5x", str(src)) is None


def test_publish_artifact_copies_to_published_path(tmp_path, monkeypatch):
    import fabric_defect_hub.catalog as catalog_module

    dest_root = tmp_path / "published"
    monkeypatch.setattr(catalog_module, "PUBLISHED_MODEL_ROOT", dest_root)

    src = tmp_path / "run" / "best.pt"
    src.parent.mkdir()
    src.write_bytes(b"trained weights")

    result = publish_artifact("ultralytics", "yolov8n", str(src))

    assert result == dest_root / "yolov8n.pt"
    assert result.read_bytes() == b"trained weights"


def test_publish_artifact_overwrites_previous_publish(tmp_path, monkeypatch):
    import fabric_defect_hub.catalog as catalog_module

    dest_root = tmp_path / "published"
    monkeypatch.setattr(catalog_module, "PUBLISHED_MODEL_ROOT", dest_root)

    first = tmp_path / "first.pt"
    first.write_bytes(b"run 1")
    second = tmp_path / "second.pt"
    second.write_bytes(b"run 2")

    publish_artifact("ultralytics", "yolov8n", str(first))
    result = publish_artifact("ultralytics", "yolov8n", str(second))

    assert result.read_bytes() == b"run 2"
