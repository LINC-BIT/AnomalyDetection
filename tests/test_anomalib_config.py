"""Fast, framework-free tests for the Anomalib config/preset layer.

These exercise the config-driven logic (model-name resolution,
hyperparameter layering, YAML validation) without importing `anomalib` or
running any training — `models/anomalib/presets.py` only imports
`anomalib` lazily inside functions that actually need it (`resolve_model_
class`), so this stays in the default test suite. Mirrors
`tests/test_ultralytics_config.py`'s structure.
"""

import importlib.util
import inspect

import pytest

from fabric_defect_hub.models.anomalib.adapter import AnomalibAdapter
from fabric_defect_hub.models.anomalib.config import AnomalibConfig
from fabric_defect_hub.models.anomalib.presets import (
    IMAGE_LEVEL_ONLY,
    MODEL_PRESETS,
    default_model_kwargs,
    list_supported_variants,
    resolve_model_class_name,
)

requires_anomalib = pytest.mark.skipif(
    importlib.util.find_spec("anomalib") is None,
    reason="anomalib is an optional extra; the rest of this module stays framework-free",
)


def test_model_name_resolution_and_aliases():
    assert set(list_supported_variants()) == {
        # memory bank / statistics
        "Patchcore", "Padim",
        # teacher-student
        "ReverseDistillation", "Stfpm", "UniNet",
        # reconstruction
        "Draem", "Dsr",
        # adversarial
        "Ganomaly",
        # normalizing flow
        "Fastflow",
        # synthetic anomaly
        "Supersimplenet", "Glass",
        # zero-shot VLM
        "WinClip", "AnomalyDINO",
        # regularized distillation
        "EfficientAd",
    }
    assert resolve_model_class_name("PatchCore") == "Patchcore"
    assert resolve_model_class_name("rd4ad") == "ReverseDistillation"
    assert resolve_model_class_name("EfficientAD") == "EfficientAd"
    assert resolve_model_class_name("WinCLIP") == "WinClip"
    assert resolve_model_class_name("Patchcore") == "Patchcore"  # literal class name passes through
    # The families added for the reconstruction/adversarial/teacher-student
    # coverage this benchmark claims, reachable by their paper names.
    assert resolve_model_class_name("STFPM") == "Stfpm"
    assert resolve_model_class_name("GANomaly") == "Ganomaly"
    assert resolve_model_class_name("DRAEM") == "Draem"
    assert resolve_model_class_name("FastFlow") == "Fastflow"
    assert resolve_model_class_name("AnomalyDINO") == "AnomalyDINO"
    with pytest.raises(KeyError):
        resolve_model_class_name("NotAModel")


@requires_anomalib
def test_every_preset_key_is_a_real_constructor_kwarg():
    """The preset layer's core risk: anomalib renames or drops a constructor
    argument on upgrade, and a preset silently keeps passing the old name —
    which surfaces as a TypeError deep inside a training run, on whichever
    machine happens to run that model next.

    Checking against the *installed* anomalib rather than a hand-maintained
    list is the point: this fails on the upgrade that breaks it, not months
    later. It is also what lets `presets.py`'s module docstring claim the
    kwargs were introspected rather than guessed.
    """

    from fabric_defect_hub.models.anomalib.presets import resolve_model_class

    for class_name, preset in MODEL_PRESETS.items():
        params = inspect.signature(resolve_model_class(class_name).__init__).parameters
        unknown = sorted(set(preset) - set(params))
        assert not unknown, f"{class_name} preset passes non-constructor kwargs: {unknown}"


@requires_anomalib
def test_image_level_only_models_really_produce_no_anomaly_map():
    """`IMAGE_LEVEL_ONLY` drives what `capabilities()` advertises, so it has
    to track anomalib's actual inference output rather than a note someone
    wrote once. Every model *not* listed there must return an `anomaly_map`
    from its torch model's forward; the listed ones must not.
    """

    from fabric_defect_hub.models.anomalib.presets import resolve_model_class

    for class_name in MODEL_PRESETS:
        lightning_cls = resolve_model_class(class_name)
        torch_model_module = inspect.getmodule(lightning_cls).__name__.replace(
            "lightning_model", "torch_model"
        )
        try:
            module = importlib.import_module(torch_model_module)
        except ImportError:  # pragma: no cover - model laid out differently
            continue
        forwards = [
            inspect.getsource(obj.forward)
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if obj.__module__ == torch_model_module and hasattr(obj, "forward")
        ]
        if not forwards:  # pragma: no cover
            continue
        emits_map = any("anomaly_map" in src for src in forwards)
        if class_name in IMAGE_LEVEL_ONLY:
            assert not emits_map, f"{class_name} is listed IMAGE_LEVEL_ONLY but emits an anomaly_map"
        else:
            assert emits_map, f"{class_name} emits no anomaly_map but is not in IMAGE_LEVEL_ONLY"


def test_ganomaly_capabilities_omit_pixel_level_map():
    # GANomaly scores the L2 distance between two latent vectors; there is no
    # spatial map, so pixel AUROC/AUPRO must be declared uncomputable rather
    # than silently evaluated over absent maps.
    caps = AnomalibAdapter(name="GANomaly").capabilities()
    assert "anomaly_map" not in caps.prediction_fields
    assert "anomaly_score" in caps.prediction_fields

    assert "anomaly_map" in AnomalibAdapter(name="PatchCore").capabilities().prediction_fields


def test_draem_refuses_to_start_without_a_staged_texture_source(tmp_path):
    # anomalib's Draem downloads ~600MB of DTD mid-training when the
    # directory is missing; the adapter turns that into an up-front error.
    adapter = AnomalibAdapter(name="DRAEM")
    with pytest.raises(ValueError, match="DTD"):
        adapter._validate_model_kwargs({"dtd_dir": str(tmp_path / "nope")})

    # ...unless the caller opts into the download explicitly, in which case
    # the flag is stripped before the kwargs reach anomalib's constructor.
    kwargs = {"dtd_dir": str(tmp_path / "nope"), "allow_dtd_download": True}
    adapter._validate_model_kwargs(kwargs)
    assert "allow_dtd_download" not in kwargs

    staged = tmp_path / "DTD"
    staged.mkdir()
    adapter._validate_model_kwargs({"dtd_dir": str(staged)})


def test_ganomaly_preset_pins_batch_size_because_trainconfig_cannot_reach_it():
    # `batch_size` is a GANomaly *constructor* argument, so the canonical
    # TrainConfig.batch_size never reaches it (TRAIN_CONFIG_KEYS maps only
    # genuinely flat keys). Pinning it in the preset is what keeps the value
    # explicit and reviewable instead of implicit.
    assert default_model_kwargs("GANomaly")["batch_size"] == 32
    assert "batch_size" not in AnomalibAdapter.TRAIN_CONFIG_KEYS


def test_anomalydino_disables_object_masking_for_full_frame_fabric():
    # Upstream's masking segments a foreground object from its background,
    # which is meaningless on full-frame fabric texture.
    assert default_model_kwargs("AnomalyDINO")["masking"] is False


def test_efficientad_requires_imagenet_dir_with_no_fabric_default():
    kwargs = default_model_kwargs("EfficientAD")
    assert kwargs["imagenet_dir"] is None


def test_supersimplenet_defaults_to_supervised():
    # ZJU-Leaper ships real defect masks, so the fabric preset should use
    # them (supervised=True) rather than anomalib's synthetic-anomaly default.
    assert default_model_kwargs("SuperSimpleNet")["supervised"] is True


def test_winclip_defaults_to_zero_shot():
    # WinCLIP is CLIP-based; k_shot=0 keeps it a pure zero-shot baseline that
    # consumes no fabric training images, and class_name gives its prompt
    # ensemble a domain-appropriate noun.
    kwargs = default_model_kwargs("WinCLIP")
    assert kwargs["k_shot"] == 0
    assert kwargs["class_name"] == "fabric"


def test_winclip_zero_shot_skips_data_dependent_fit():
    adapter = AnomalibAdapter(name="WinCLIP")
    assert adapter._is_zero_shot_winclip({"k_shot": 0}) is True
    assert adapter._is_zero_shot_winclip({"k_shot": 5}) is False


def test_config_from_dict_layers_and_resolves():
    cfg = AnomalibConfig.from_dict(
        {
            "model": {"name": "PatchCore"},
            "data": {
                "dataset": "zju-leaper",
                "dataset_root": "/data/zju-leaper",
                "train_selection": {"use_defect": False},
                "test_selection": {"use_defect": True},
            },
            "train": {
                "model_kwargs": {"coreset_sampling_ratio": 0.25},
                "engine_kwargs": {"max_epochs": 3},
            },
            "checkpoint": {"default_root_dir": "results/x", "registry_dir": "artifacts/x"},
        }
    )

    resolved_model = cfg.resolved_model_kwargs()
    # config override wins
    assert resolved_model["coreset_sampling_ratio"] == 0.25
    # fabric preset still fills unspecified knobs
    assert resolved_model["backbone"] == "wide_resnet50_2"
    assert resolved_model["num_neighbors"] == 9

    resolved_engine = cfg.resolved_engine_kwargs()
    assert resolved_engine["max_epochs"] == 3
    # checkpoint.default_root_dir is merged in automatically
    assert resolved_engine["default_root_dir"] == "results/x"


def test_config_rejects_unknown_and_conflicting_keys():
    with pytest.raises(ValueError):
        AnomalibConfig.from_dict({"modle": {}})  # typo in top-level key
    with pytest.raises(ValueError):
        AnomalibConfig.from_dict(
            {"model": {"name": "PatchCore", "bogus": 1}, "data": {"datamodule_kwargs": {"root": "/x"}}}
        )
    with pytest.raises(ValueError):
        AnomalibConfig.from_dict(
            {"data": {"datamodule_kwargs": {"root": "/x"}, "dataset": "zju-leaper", "dataset_root": "/r"}}
        )
    with pytest.raises(ValueError):
        AnomalibConfig.from_dict({"data": {"dataset": "zju-leaper"}})  # missing dataset_root
    with pytest.raises(ValueError):
        AnomalibConfig.from_dict({"data": {}})  # neither datamodule_kwargs nor dataset


def test_config_rejects_unknown_model_name():
    with pytest.raises(KeyError):
        AnomalibConfig.from_dict(
            {"model": {"name": "NotAModel"}, "data": {"datamodule_kwargs": {"root": "/x"}}}
        )


def test_datamodule_kwargs_mode_does_not_use_adapter():
    cfg = AnomalibConfig.from_dict({"data": {"datamodule_kwargs": {"root": "/x"}}})
    assert cfg.data.uses_adapter() is False


def test_dataset_mode_uses_adapter():
    cfg = AnomalibConfig.from_dict(
        {"data": {"dataset": "zju-leaper", "dataset_root": "/r"}}
    )
    assert cfg.data.uses_adapter() is True


def test_default_train_spec_has_zero_workers():
    # See config.py's TrainSpec docstring: staged dirs are transient
    # symlinks, so num_workers=0 avoids a shutdown race, unlike the
    # torchvision backend's default of 2.
    cfg = AnomalibConfig.from_dict({"data": {"datamodule_kwargs": {"root": "/x"}}})
    assert cfg.train.num_workers == 0
