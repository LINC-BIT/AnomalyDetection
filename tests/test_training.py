"""Fast, framework-free tests for the unified `train` entry point
(`fabric_defect_hub.training`): backend keyword detection and dataset/shot
overlay logic, without importing any ML framework or running training.
"""

import pytest

from fabric_defect_hub import training
from fabric_defect_hub.training import (
    ANOMALY_TRAINABLE_DATASETS,
    DEFAULT_DATASET_ROOTS,
    ZERO_SHOT_TRAINABLE_DATASETS,
    TEST_SHOT_NUM_SAMPLES,
    DatasetOverrides,
    _apply_test_speed_overrides,
    _enforce_trainable_dataset,
    apply_available_dataset,
    apply_dataset_overrides,
    apply_default_dataset_root,
    apply_model_overrides,
    apply_raw_overrides,
    find_model_configs,
    infer_backend,
    resolve_model_config,
)


@pytest.fixture
def config_dir(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "ultralytics_example.yaml").write_text("model:\n  variant: yolov8n\n")
    (directory / "anomalib_example.yaml").write_text("model:\n  name: PatchCore\n")
    return directory


def test_resolve_model_config_by_existing_path(config_dir):
    path = config_dir / "ultralytics_example.yaml"
    assert resolve_model_config(str(path), config_dir=config_dir) == path


def test_resolve_model_config_by_filename_stem(config_dir):
    assert resolve_model_config("ultralytics_example", config_dir=config_dir) == (
        config_dir / "ultralytics_example.yaml"
    )
    # with the .yaml suffix too
    assert resolve_model_config("ultralytics_example.yaml", config_dir=config_dir) == (
        config_dir / "ultralytics_example.yaml"
    )


def test_resolve_model_config_by_model_keyword(config_dir):
    assert resolve_model_config("yolov8n", config_dir=config_dir) == (
        config_dir / "ultralytics_example.yaml"
    )
    # case-insensitive, matches model.name too
    assert resolve_model_config("patchcore", config_dir=config_dir) == (
        config_dir / "anomalib_example.yaml"
    )


def test_resolve_model_config_missing_raises_with_available_list(config_dir):
    with pytest.raises(FileNotFoundError, match="ultralytics_example"):
        resolve_model_config("nonexistent", config_dir=config_dir)


# --------------------------------------------------------------------------- #
# Keyword resolution steps 3-5 (see resolve_model_config_and_variant)
# --------------------------------------------------------------------------- #
def test_a_name_several_configs_declare_is_settled_by_the_catalog(config_dir):
    """Two configs declaring `yolov8n` is no longer an error: step 3 defers,
    and the catalog says which config trains the published yolov8n. This is
    the mechanism that makes a bare `fdh train patchcore` work with three
    PatchCore configs on disk.
    """

    (config_dir / "ultralytics_example2.yaml").write_text("model:\n  variant: yolov8n\n")

    assert resolve_model_config("yolov8n", config_dir=config_dir) == (
        config_dir / "ultralytics_example.yaml"
    )


def test_an_ambiguous_name_the_catalog_does_not_know_still_raises(config_dir):
    """...but ambiguity the catalog *cannot* settle must still be reported,
    naming the culprits — otherwise the deferral in step 3 would turn a
    clear error into a confusing "could not resolve".
    """

    (config_dir / "custom_a.yaml").write_text("model:\n  variant: mymodel\n")
    (config_dir / "custom_b.yaml").write_text("model:\n  variant: mymodel\n")

    with pytest.raises(ValueError, match="declared by multiple configs"):
        resolve_model_config("mymodel", config_dir=config_dir)


def test_backend_supported_variant_falls_back_to_that_backends_example(config_dir):
    """Step 5: FastFlow is runnable but unpublished, so no config declares it
    and the catalog has no entry — it should still resolve, to the anomalib
    general-purpose config, with the variant carried through.
    """

    path, variant = training.resolve_model_config_and_variant("fastflow", config_dir=config_dir)

    assert path == config_dir / "anomalib_example.yaml"
    assert variant == "fastflow"


def test_a_variant_two_backends_both_claim_is_rejected(config_dir, monkeypatch):
    """Step 5 is only unambiguous because no two backends currently share a
    variant name. If that ever changes, resolution must say so rather than
    silently picking whichever backend enumerates first.
    """

    monkeypatch.setattr(
        "fabric_defect_hub.api.list_models",
        lambda backend=None: {"anomalib": ["Collide"], "dinomaly": ["collide"]},
    )
    with pytest.raises(ValueError, match="more than one backend"):
        resolve_model_config("collide", config_dir=config_dir)


def test_find_model_configs_lists_yaml_files_sorted(config_dir):
    found = find_model_configs(config_dir)
    assert found == sorted(config_dir.glob("*.yaml"))


def test_find_model_configs_empty_dir_returns_empty(tmp_path):
    assert find_model_configs(tmp_path / "does-not-exist") == []


@pytest.mark.parametrize(
    ("raw", "backend"),
    [
        ({"model": {"variant": "yolov8n"}}, "ultralytics"),
        # Registered non-yolo-prefixed variant: must resolve through the
        # ultralytics variant registry, not the name-prefix fallback.
        ({"model": {"variant": "fabricmamba"}}, "ultralytics"),
        ({"model": {"variant": "fasterrcnn_resnet50_fpn"}}, "torchvision"),
        ({"model": {"name": "PatchCore"}}, "anomalib"),
        ({"model": {"name": "dinov2reg_vit_base_14"}}, "dinomaly"),
        ({"model": {"name": "ViT-L-14-336"}}, "moeclip"),
        ({"model": {"name": "resnet34"}}, "mambaad"),
        ({"backend": "torchvision", "model": {"variant": "yolov8n"}}, "torchvision"),
    ],
)
def test_infer_backend(raw, backend):
    assert infer_backend(raw) == backend


@pytest.mark.parametrize("dataset", sorted(ANOMALY_TRAINABLE_DATASETS))
def test_enforce_allows_trainable_fabric_datasets(dataset):
    # Should not raise for any one-class backend.
    _enforce_trainable_dataset({"data": {"dataset": dataset}}, "anomalib")
    _enforce_trainable_dataset({"data": {"dataset": dataset}}, "dinomaly")
    _enforce_trainable_dataset({"data": {"dataset": dataset}}, "mambaad")


@pytest.mark.parametrize("dataset", ["visa", "mvtec-ad", "mvtec-loco"])
def test_enforce_allows_general_datasets_for_anomaly_backends(dataset):
    _enforce_trainable_dataset({"data": {"dataset": dataset}}, "anomalib")
    _enforce_trainable_dataset({"data": {"dataset": dataset}}, "mambaad")


@pytest.mark.parametrize("dataset", sorted(ZERO_SHOT_TRAINABLE_DATASETS))
def test_enforce_allows_cross_domain_corpora_for_zero_shot_backend(dataset):
    # The mirror image of the one-class rule: MoECLIP is trained on an
    # auxiliary cross-domain corpus and applied to unseen fabric.
    _enforce_trainable_dataset({"data": {"dataset": dataset}}, "moeclip")


@pytest.mark.parametrize("dataset", sorted(ANOMALY_TRAINABLE_DATASETS - ZERO_SHOT_TRAINABLE_DATASETS))
def test_enforce_rejects_fabric_training_for_zero_shot_backend(dataset):
    # Training MoECLIP on fabric would make its fabric scores in-domain and
    # void the zero-shot claim the benchmark is measuring.
    with pytest.raises(ValueError, match="zero-shot"):
        _enforce_trainable_dataset({"data": {"dataset": dataset}}, "moeclip")


def test_test_dataset_override_sets_zero_shot_eval_target():
    raw = {"data": {"dataset": "visa", "test_dataset": "raw-fabric"}}
    result = apply_dataset_overrides(
        raw, "moeclip", DatasetOverrides(test_dataset="tilda-400")
    )
    assert result["data"]["test_dataset"] == "tilda-400"
    assert result["data"]["dataset"] == "visa"  # training corpus untouched


def test_test_dataset_override_is_rejected_for_other_backends():
    with pytest.raises(ValueError, match="only apply to the zero-shot backends"):
        apply_dataset_overrides(
            {"data": {"dataset": "raw-fabric"}}, "dinomaly",
            DatasetOverrides(test_dataset="tilda-400"),
        )


def test_default_dataset_root_resolves_zero_shot_eval_target():
    raw = apply_default_dataset_root({"data": {"dataset": "visa", "test_dataset": "raw-fabric"}})
    assert raw["data"]["dataset_root"] == DEFAULT_DATASET_ROOTS["visa"]
    assert raw["data"]["test_dataset_root"] == DEFAULT_DATASET_ROOTS["raw-fabric"]


def test_enforce_is_noop_for_detection_backends():
    # Detection backends legitimately train on detection datasets; the
    # anomaly allowlist must not apply to them.
    _enforce_trainable_dataset({"data": {"dataset": "sdust-fdd"}}, "ultralytics")
    _enforce_trainable_dataset({"data": {"dataset": "mvtec-ad"}}, "torchvision")


def test_enforce_is_noop_without_registered_dataset():
    # data_root / datamodule_kwargs mode: no registered dataset name.
    _enforce_trainable_dataset({"data": {"data_root": "/some/mvtec/bottle"}}, "anomalib")


def test_infer_backend_requires_model_or_backend_key():
    with pytest.raises(ValueError):
        infer_backend({})


def test_infer_backend_rejects_unknown_explicit_backend():
    with pytest.raises(ValueError):
        infer_backend({"backend": "tensorflow"})


def test_empty_overrides_are_a_no_op():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {"num_samples": 300}}}
    assert apply_dataset_overrides(raw, "ultralytics", DatasetOverrides()) is raw


def test_overrides_require_dataset_adapter_config():
    raw = {"data": {"data_yaml": "data.yaml"}}
    with pytest.raises(ValueError):
        apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="test"))


def test_dataset_and_root_override_clear_conflicting_keys():
    raw = {"data": {"data_yaml": "data.yaml"}}
    out = apply_dataset_overrides(
        raw, "ultralytics", DatasetOverrides(dataset="zju-leaper", dataset_root="/data/zju")
    )
    assert out["data"]["dataset"] == "zju-leaper"
    assert out["data"]["dataset_root"] == "/data/zju"
    assert "data_yaml" not in out["data"]


def test_test_mode_forces_eight_samples_on_both_splits():
    raw = {
        "data": {
            "dataset": "zju-leaper",
            "train_selection": {"num_samples": 300},
            "val_selection": {"num_samples": 100},
        }
    }
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="test"))
    assert out["data"]["train_selection"]["num_samples"] == TEST_SHOT_NUM_SAMPLES
    assert out["data"]["val_selection"]["num_samples"] == TEST_SHOT_NUM_SAMPLES


def test_full_mode_clears_num_samples():
    raw = {
        "data": {
            "dataset": "zju-leaper",
            "train_selection": {"num_samples": 300},
            "val_selection": {"num_samples": 100},
        }
    }
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="full"))
    assert out["data"]["train_selection"]["num_samples"] is None
    assert out["data"]["val_selection"]["num_samples"] is None


def test_few_mode_leaves_configured_count_untouched():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {"num_samples": 300}}}
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="few"))
    assert out["data"]["train_selection"]["num_samples"] == 300


def test_few_mode_leaves_configured_pattern_untouched():
    raw = {
        "data": {
            "dataset": "zju-leaper",
            "train_selection": {"pattern": [1, 2, 3, 4], "num_samples": 300},
        }
    }
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="few"))
    assert out["data"]["train_selection"]["pattern"] == [1, 2, 3, 4]


def test_medium_mode_caps_samples_per_explicit_pattern():
    raw = {
        "data": {
            "dataset": "zju-leaper",
            "train_selection": {"pattern": [1, 2, 3, 4], "num_samples": 300},
            "val_selection": {"pattern": [1, 2, 3, 4], "num_samples": 100},
        }
    }
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="medium"))
    assert out["data"]["train_selection"]["pattern"] == [1, 2, 3, 4]
    assert out["data"]["val_selection"]["pattern"] == [1, 2, 3, 4]
    assert out["data"]["train_selection"]["num_samples"] == 150 * 4
    assert out["data"]["val_selection"]["num_samples"] == 50 * 4


def test_full_mode_keeps_explicit_pattern_subset():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {"pattern": [1, 2, 3, 4]}}}
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="full"))
    assert out["data"]["train_selection"]["pattern"] == [1, 2, 3, 4]


def test_cli_pattern_wins_over_configured_pattern():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {"pattern": [1, 2, 3, 4]}}}
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="full", pattern="pattern7"))
    assert out["data"]["train_selection"]["pattern"] == "pattern7"


def test_cli_pattern_number_selects_one_pattern():
    """`--pattern 5` reaches the adapter as pattern 5, not as the string "5".

    The adapter rejects a bare digit string (it falls through to the texture
    name lookup), so the CLI normalizes it first.
    """

    from fabric_defect_hub.datasets.zju_leaper import coerce_pattern

    raw = {"data": {"dataset": "zju-leaper", "train_selection": {"pattern": [1, 2, 3, 4]}}}
    out = apply_dataset_overrides(
        raw, "ultralytics", DatasetOverrides(mode="full", pattern=coerce_pattern("5"))
    )
    assert out["data"]["train_selection"]["pattern"] == 5


def test_unrestricted_zju_medium_mode_uses_all_patterns():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {"num_samples": 300}}}
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(mode="medium"))
    assert out["data"]["train_selection"]["pattern"] == list(range(1, 20))
    assert out["data"]["train_selection"]["num_samples"] == 150 * 19


def test_mode_pattern_handling_is_zju_leaper_only():
    raw = {"data": {"dataset": "mvtec-ad", "train_selection": {"category": "bottle"}}}
    out = apply_dataset_overrides(raw, "torchvision", DatasetOverrides(mode="full"))
    assert "pattern" not in out["data"]["train_selection"]


def test_explicit_num_samples_wins_over_mode():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {}, "val_selection": {}}}
    out = apply_dataset_overrides(
        raw, "ultralytics", DatasetOverrides(mode="full", num_samples=42)
    )
    assert out["data"]["train_selection"]["num_samples"] == 42
    assert out["data"]["val_selection"]["num_samples"] == 42


def test_val_num_samples_overrides_only_the_second_split():
    raw = {"data": {"dataset": "zju-leaper", "train_selection": {}, "val_selection": {}}}
    out = apply_dataset_overrides(
        raw, "ultralytics", DatasetOverrides(num_samples=8, val_num_samples=20)
    )
    assert out["data"]["train_selection"]["num_samples"] == 8
    assert out["data"]["val_selection"]["num_samples"] == 20


def test_anomalib_train_split_is_forced_normal_only():
    raw = {
        "data": {
            "dataset": "zju-leaper",
            "train_selection": {"num_samples": 300},
            "test_selection": {"num_samples": 100, "use_defect": True, "defect_ratio": 0.3},
        }
    }
    out = apply_dataset_overrides(
        raw, "anomalib", DatasetOverrides(mode="test", use_defect=True, defect_ratio=0.7)
    )
    train_selection = out["data"]["train_selection"]
    assert train_selection["use_defect"] is False
    assert "defect_ratio" not in train_selection
    assert train_selection["num_samples"] == TEST_SHOT_NUM_SAMPLES
    # The test split still receives the requested defect mix.
    test_selection = out["data"]["test_selection"]
    assert test_selection["use_defect"] is True
    assert test_selection["defect_ratio"] == 0.7
    assert test_selection["num_samples"] == TEST_SHOT_NUM_SAMPLES


def test_apply_model_overrides_is_a_no_op_when_variant_is_none():
    raw = {"model": {"variant": "yolov8n"}}
    assert apply_model_overrides(raw, "ultralytics", None) is raw


def test_apply_model_overrides_sets_variant_for_ultralytics():
    raw = {"model": {"variant": "yolov8n"}}
    out = apply_model_overrides(raw, "ultralytics", "yolov8s")
    assert out["model"]["variant"] == "yolov8s"


def test_apply_model_overrides_sets_variant_for_torchvision():
    raw = {"model": {"variant": "fasterrcnn_resnet50_fpn"}}
    out = apply_model_overrides(raw, "torchvision", "maskrcnn_resnet50_fpn")
    assert out["model"]["variant"] == "maskrcnn_resnet50_fpn"


def test_apply_model_overrides_sets_name_for_anomalib():
    raw = {"model": {"name": "PatchCore"}}
    out = apply_model_overrides(raw, "anomalib", "PaDiM")
    assert out["model"]["name"] == "PaDiM"


def test_apply_model_overrides_does_not_mutate_input():
    raw = {"model": {"variant": "yolov8n"}}
    apply_model_overrides(raw, "ultralytics", "yolov8s")
    assert raw["model"]["variant"] == "yolov8n"


def test_apply_model_overrides_namespaces_checkpoint_name():
    raw = {"model": {"variant": "yolov8n"}, "checkpoint": {"name": "pattern1", "project": "runs"}}
    out = apply_model_overrides(raw, "ultralytics", "yolov8s")
    assert out["checkpoint"]["name"] == "yolov8s_pattern1"
    assert out["checkpoint"]["project"] == "runs"  # untouched


def test_apply_model_overrides_does_not_double_prefix_checkpoint_name():
    raw = {"model": {"variant": "yolov8n"}, "checkpoint": {"name": "yolov8s_pattern1"}}
    out = apply_model_overrides(raw, "ultralytics", "yolov8s")
    assert out["checkpoint"]["name"] == "yolov8s_pattern1"


def test_apply_model_overrides_without_checkpoint_section_is_fine():
    raw = {"model": {"variant": "yolov8n"}}
    out = apply_model_overrides(raw, "ultralytics", "yolov8s")
    assert "checkpoint" not in out


def test_pattern_category_and_seed_apply_to_both_splits():
    raw = {"data": {"dataset": "mvtec-ad", "train_selection": {}, "val_selection": {}}}
    out = apply_dataset_overrides(
        raw, "torchvision", DatasetOverrides(category="bottle", seed=7)
    )
    for key in ("train_selection", "val_selection"):
        assert out["data"][key]["category"] == "bottle"
        assert out["data"][key]["seed"] == 7


# -- apply_default_dataset_root ------------------------------------------- #


@pytest.mark.parametrize("dataset", sorted(DEFAULT_DATASET_ROOTS))
def test_default_dataset_root_fills_missing_root_for_every_registered_dataset(dataset):
    raw = {"data": {"dataset": dataset}}
    out = apply_default_dataset_root(raw)
    assert out["data"]["dataset_root"] == DEFAULT_DATASET_ROOTS[dataset]


def test_default_dataset_root_fills_in_when_root_key_absent():
    raw = {"data": {"dataset": "zju-leaper"}}
    out = apply_default_dataset_root(raw)
    assert out["data"]["dataset_root"] == "datasets/textile/ZJU-Leaper"


def test_default_dataset_root_fills_in_when_root_is_empty_string():
    raw = {"data": {"dataset": "zju-leaper", "dataset_root": ""}}
    out = apply_default_dataset_root(raw)
    assert out["data"]["dataset_root"] == "datasets/textile/ZJU-Leaper"


def test_default_dataset_root_fills_in_when_env_var_placeholder_unexpanded():
    # os.path.expandvars leaves ${VAR} untouched (not blank, not an error)
    # when the environment variable isn't set, so this is what a real
    # unexpanded config looks like by the time it reaches here.
    raw = {"data": {"dataset": "zju-leaper", "dataset_root": "${ZJU_LEAPER_ROOT}"}}
    out = apply_default_dataset_root(raw)
    assert out["data"]["dataset_root"] == "datasets/textile/ZJU-Leaper"


def test_default_dataset_root_does_not_override_an_explicit_path():
    raw = {"data": {"dataset": "zju-leaper", "dataset_root": "/custom/path"}}
    out = apply_default_dataset_root(raw)
    assert out["data"]["dataset_root"] == "/custom/path"


def test_default_dataset_root_does_not_override_a_successfully_expanded_env_var():
    raw = {"data": {"dataset": "zju-leaper", "dataset_root": "/Volumes/SSD/datasets/ZJU-Leaper"}}
    out = apply_default_dataset_root(raw)
    assert out["data"]["dataset_root"] == "/Volumes/SSD/datasets/ZJU-Leaper"


def test_default_dataset_root_is_a_no_op_without_a_dataset():
    raw = {"data": {"data_yaml": "/path/to/data.yaml"}}
    assert apply_default_dataset_root(raw) is raw


def test_default_dataset_root_is_a_no_op_for_unregistered_dataset_name():
    raw = {"data": {"dataset": "some-future-dataset"}}
    assert apply_default_dataset_root(raw) is raw


def test_default_dataset_root_is_a_no_op_without_data_section():
    raw = {"model": {"variant": "yolov8n"}}
    assert apply_default_dataset_root(raw) is raw


def test_default_dataset_root_does_not_mutate_input():
    raw = {"data": {"dataset": "zju-leaper"}}
    apply_default_dataset_root(raw)
    assert "dataset_root" not in raw["data"]


# -- switching --dataset drops a stale root for the old dataset ----------- #


def test_dataset_override_without_root_drops_stale_root_for_old_dataset():
    raw = {
        "data": {
            "dataset": "zju-leaper",
            "dataset_root": "/Volumes/SSD/datasets/ZJU-Leaper",
            "train_selection": {},
            "val_selection": {},
        }
    }
    out = apply_dataset_overrides(raw, "ultralytics", DatasetOverrides(dataset="raw-fabric"))
    assert out["data"]["dataset"] == "raw-fabric"
    assert "dataset_root" not in out["data"]
    # ... and apply_default_dataset_root then resolves it correctly for the new dataset.
    resolved = apply_default_dataset_root(out)
    assert resolved["data"]["dataset_root"] == "datasets/textile/RAW_FABRID"


# -- --mode test must actually cap epochs, not just sample count --------- #


def test_test_speed_overrides_force_epochs_even_when_config_sets_its_own():
    # Regression: a plain `setdefault` here was a no-op against configs that
    # already declare `epochs`/`patience` (every example config does), so
    # `--mode test` silently ran the config's full schedule instead of a
    # fast 1-epoch smoke run.
    raw = {"train": {"epochs": 100, "patience": 30, "lr0": 0.01}}
    out = _apply_test_speed_overrides(raw, "ultralytics")
    assert out["train"]["epochs"] == 1
    assert out["train"]["patience"] == 1
    assert out["train"]["lr0"] == 0.01  # unrelated keys untouched


def test_test_speed_overrides_force_max_epochs_for_anomalib():
    raw = {"train": {"engine_kwargs": {"max_epochs": 50, "accelerator": "gpu"}}}
    out = _apply_test_speed_overrides(raw, "anomalib")
    assert out["train"]["engine_kwargs"]["max_epochs"] == 1
    assert out["train"]["engine_kwargs"]["accelerator"] == "gpu"


def test_dataset_override_with_explicit_root_keeps_it():
    raw = {
        "data": {
            "dataset": "zju-leaper",
            "dataset_root": "/Volumes/SSD/datasets/ZJU-Leaper",
            "train_selection": {},
            "val_selection": {},
        }
    }
    out = apply_dataset_overrides(
        raw, "ultralytics", DatasetOverrides(dataset="raw-fabric", dataset_root="/custom/raw-fabric")
    )
    assert out["data"]["dataset_root"] == "/custom/raw-fabric"


# --------------------------------------------------------------------------- #
# apply_available_dataset — substitute a *staged* alternative rather than
# failing deep inside a backend when a config names a dataset this machine
# doesn't have (a cloud training box especially won't have every dataset the
# project knows about). Fully isolates `_BACKEND_TRAINABLE_DATASETS` /
# `DEFAULT_DATASET_ROOTS` per test so this never touches this machine's own
# real `data/<Dataset>` symlinks.
# --------------------------------------------------------------------------- #
def test_apply_available_dataset_is_a_true_noop_when_requested_is_staged(tmp_path, monkeypatch):
    (tmp_path / "f.jpg").write_text("x")
    monkeypatch.setattr(training, "DEFAULT_DATASET_ROOTS", {"zju-leaper": str(tmp_path), "raw-fabric": "/nope"})
    monkeypatch.setattr(
        training, "_BACKEND_TRAINABLE_DATASETS", {"anomalib": ({"zju-leaper", "raw-fabric"}, "one-class")}
    )
    raw = {"data": {"dataset": "zju-leaper", "dataset_root": str(tmp_path)}}

    assert apply_available_dataset(raw, "anomalib") is raw


def test_apply_available_dataset_substitutes_a_staged_alternative_and_warns(tmp_path, monkeypatch):
    (tmp_path / "f.jpg").write_text("x")
    monkeypatch.setattr(training, "DEFAULT_DATASET_ROOTS", {"zju-leaper": str(tmp_path), "raw-fabric": "/nope"})
    monkeypatch.setattr(
        training, "_BACKEND_TRAINABLE_DATASETS", {"anomalib": ({"zju-leaper", "raw-fabric"}, "one-class")}
    )
    raw = {"data": {"dataset": "raw-fabric", "dataset_root": "/nope"}}

    with pytest.warns(UserWarning, match="substituted"):
        out = apply_available_dataset(raw, "anomalib")

    assert out["data"]["dataset"] == "zju-leaper"
    assert out["data"]["dataset_root"] == str(tmp_path)


def test_apply_available_dataset_raises_when_nothing_is_staged(monkeypatch):
    monkeypatch.setattr(training, "DEFAULT_DATASET_ROOTS", {"zju-leaper": "/nope1", "raw-fabric": "/nope2"})
    monkeypatch.setattr(
        training, "_BACKEND_TRAINABLE_DATASETS", {"anomalib": ({"zju-leaper", "raw-fabric"}, "one-class")}
    )
    raw = {"data": {"dataset": "zju-leaper"}}

    with pytest.raises(FileNotFoundError, match="stageable"):
        apply_available_dataset(raw, "anomalib")


def test_apply_available_dataset_is_noop_for_detection_backends():
    raw = {"data": {"dataset": "sdust-fdd"}}
    assert apply_available_dataset(raw, "ultralytics") is raw


def test_apply_available_dataset_is_noop_without_a_named_dataset():
    # data_root / datamodule_kwargs mode -- mirrors `_enforce_trainable_dataset`'s
    # own no-op there, since no registered dataset name is involved.
    raw = {"data": {"data_root": "/some/mvtec/bottle"}}
    assert apply_available_dataset(raw, "anomalib") is raw


# --------------------------------------------------------------------------- #
# apply_raw_overrides — the tuning window: dotted-path config overrides
# --------------------------------------------------------------------------- #
def test_apply_raw_overrides_sets_a_nested_existing_path():
    raw = {"train": {"epochs": 10, "model_kwargs": {"lr": 0.01}}}

    out = apply_raw_overrides(raw, {"train.model_kwargs.lr": 0.0005, "train.epochs": 50})

    assert out["train"]["epochs"] == 50
    assert out["train"]["model_kwargs"]["lr"] == 0.0005
    # Original is untouched -- every other apply_* function in this module
    # returns a new dict rather than mutating the caller's.
    assert raw["train"]["epochs"] == 10
    assert raw["train"]["model_kwargs"]["lr"] == 0.01


def test_apply_raw_overrides_creates_missing_intermediate_sections():
    out = apply_raw_overrides({}, {"a.b.c": 1})
    assert out == {"a": {"b": {"c": 1}}}


def test_apply_raw_overrides_is_a_noop_for_none_or_empty():
    raw = {"train": {"epochs": 10}}
    assert apply_raw_overrides(raw, None) is raw
    assert apply_raw_overrides(raw, {}) is raw
