"""Fast, framework-free tests for the Ultralytics config/preset layer.

These exercise the config-driven logic (variant resolution, hyperparameter
layering, YAML validation) without importing `ultralytics` or running any
training, so they stay in the default test suite.
"""

import pytest

from fabric_defect_hub.models.ultralytics.adapter import UltralyticsAdapter
from fabric_defect_hub.models.ultralytics.config import UltralyticsConfig, resolve_variant_profile
from fabric_defect_hub.models.ultralytics.presets import (
    default_train_kwargs,
    list_supported_variants,
    resolve_variant,
    variant_weights,
)


def test_variant_resolution_and_aliases():
    assert set(list_supported_variants()) == {"yolov8n", "yolov8s", "yolo11n", "fabricmamba"}
    assert resolve_variant("YOLOv8s") == "yolov8s"
    assert resolve_variant("v11n") == "yolo11n"
    assert resolve_variant("yolo11n.pt") == "yolo11n"
    with pytest.raises(KeyError):
        resolve_variant("yolov99x")


def test_variant_weights_pretrained_vs_scratch():
    assert variant_weights("yolov8n", pretrained=True) == "yolov8n.pt"
    assert variant_weights("yolov8n", pretrained=False) == "yolov8n.yaml"


def test_fabric_defaults_are_single_class():
    kwargs = default_train_kwargs("yolov8n")
    assert kwargs["single_cls"] is True
    assert kwargs["epochs"] > 0
    # per-variant override applies for the 's' model
    assert default_train_kwargs("yolov8s")["batch"] == 12


def test_config_from_dict_layers_and_resolves():
    cfg = UltralyticsConfig.from_dict(
        {
            "model": {"variant": "yolo11n", "pretrained": True},
            "data": {"data_yaml": "data.yaml"},
            "train": {"epochs": 5, "imgsz": 512, "extra": {"cache": True}},
            "checkpoint": {"project": "runs/x", "name": "exp1"},
        }
    )
    assert cfg.model.initial_weights() == "yolo11n.pt"
    resolved = cfg.resolved_train_kwargs()
    # config named field overrides the fabric preset
    assert resolved["epochs"] == 5
    assert resolved["imgsz"] == 512
    # fabric preset still fills unspecified knobs
    assert resolved["single_cls"] is True
    # extra is merged verbatim
    assert resolved["cache"] is True
    # checkpoint overrides included
    assert resolved["project"] == "runs/x"
    assert resolved["name"] == "exp1"


def test_config_typed_augmentation_and_inference_tiling_are_resolved():
    cfg = UltralyticsConfig.from_dict(
        {
            "data": {"data_yaml": "data.yaml"},
            "train": {"augmentation": {"mosaic": 0.2, "mixup": 0.0}},
            "predict": {"tta_mode": "flip_multiscale", "tiling": True, "tile_size": [512, 512]},
        }
    )
    assert cfg.resolved_train_kwargs()["mosaic"] == 0.2
    assert cfg.predict.as_overrides()["tiling"] is True
    assert cfg.predict.as_overrides()["tile_size"] == (512, 512)


def test_variant_profile_overrides_shared_settings_without_mutating_base_config():
    raw = {
        "model": {"variant": "yolov8n"},
        "data": {"data_yaml": "data.yaml"},
        "train": {"batch": 32, "epochs": 100},
        "variants": {
            "yolov8n": {"train": {"batch": 24}, "checkpoint": {"name": "v8n"}},
            "yolov8s": {"train": {"batch": 16}, "checkpoint": {"name": "v8s"}},
        },
    }
    resolved = resolve_variant_profile(raw)
    assert resolved["train"]["batch"] == 24
    assert resolved["train"]["epochs"] == 100
    assert resolved["checkpoint"]["name"] == "v8n"
    assert raw["train"]["batch"] == 32


def test_variant_profile_requires_a_matching_model_entry():
    with pytest.raises(ValueError, match="no profile"):
        resolve_variant_profile(
            {"model": {"variant": "yolo11n"}, "variants": {"yolov8n": {}}}
        )


def test_config_rejects_unknown_and_conflicting_keys():
    with pytest.raises(ValueError):
        UltralyticsConfig.from_dict({"trian": {}})  # typo in top-level key
    with pytest.raises(ValueError):
        UltralyticsConfig.from_dict({"model": {"variant": "yolov8n", "bogus": 1}, "data": {"data_yaml": "d"}})
    with pytest.raises(ValueError):
        UltralyticsConfig.from_dict(
            {"data": {"data_yaml": "d", "dataset": "zju-leaper", "dataset_root": "/r"}}
        )
    with pytest.raises(ValueError):
        UltralyticsConfig.from_dict({"data": {"dataset": "zju-leaper"}})  # missing dataset_root
    with pytest.raises(ValueError):
        UltralyticsConfig.from_dict(
            {"data": {"data_yaml": "d"}, "model": {"loss_fn": "AFDLoss"}}
        )
    with pytest.raises(ValueError, match="tile_overlap"):
        UltralyticsConfig.from_dict({"data": {"data_yaml": "d"}, "predict": {"tile_overlap": 1.0}})


def test_scratch_init_uses_architecture_yaml():
    cfg = UltralyticsConfig.from_dict(
        {"model": {"variant": "yolov8n", "pretrained": False}, "data": {"data_yaml": "d.yaml"}}
    )
    assert cfg.model.initial_weights() == "yolov8n.yaml"


def test_validation_metric_normalisation_rejects_empty_results():
    class EmptyMetrics:
        results_dict = {}
        maps = None

    with pytest.raises(RuntimeError, match="no recognized metrics"):
        UltralyticsAdapter._normalise_metrics(EmptyMetrics())


def test_raw_module_returns_none_before_anything_is_loaded():
    adapter = UltralyticsAdapter(name="yolov8n")
    assert adapter.raw_module() is None


def test_raw_module_unwraps_the_yolo_wrapper_to_the_torch_module():
    class _FakeTorchModule:
        pass

    class _FakeYOLO:
        model = _FakeTorchModule()

    adapter = UltralyticsAdapter(name="yolov8n")
    adapter._model = _FakeYOLO()
    assert adapter.raw_module() is adapter._model.model
    assert isinstance(adapter.raw_module(), _FakeTorchModule)
