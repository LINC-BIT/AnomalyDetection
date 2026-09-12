"""Drift guard: every recipe's hyperparameters name *real* arguments of the
backend it targets, with values anchored to that backend's upstream-verified
defaults. This is what stops a recipe from sliding back into invented knobs
(`d_conv`, `routing_temperature`, `n_neighbors`) or values that contradict the
paper (MambaAD `lr=0.0001` vs upstream `0.005`).

For the multi-config backends (anomalib, ultralytics) the recipe additionally
*overrides* backend defaults; for the single-architecture clean-room backends
(mambaad, moeclip) the backend presets already ARE the published recipe, so
the recipe object mirrors them — these tests pin that mirror in place.
"""

from fabric_defect_hub.recipes.apply import resolve_recipe


def test_patchcore_recipe_names_are_real_anomalib_args():
    from fabric_defect_hub.models.anomalib.presets import MODEL_PRESETS

    hparams = resolve_recipe("patchcore").get_default_hyperparameters()
    assert set(hparams) <= set(MODEL_PRESETS["Patchcore"])
    assert hparams["backbone"] == "wide_resnet50_2"
    assert hparams["num_neighbors"] == 9


def test_rd4ad_recipe_names_are_real_anomalib_args():
    from fabric_defect_hub.models.anomalib.presets import MODEL_PRESETS

    hparams = resolve_recipe("rd4ad").get_default_hyperparameters()
    assert set(hparams) <= set(MODEL_PRESETS["ReverseDistillation"])
    assert hparams["backbone"] == "wide_resnet50_2"
    assert hparams["anomaly_map_mode"] == "add"
    # The invented distillation knobs are gone.
    assert "distillation_temp" not in hparams
    assert "cosine_loss_weight" not in hparams


def test_yolov8_recipe_uses_ultralytics_loss_gain_names():
    hparams = resolve_recipe("yolov8").get_default_hyperparameters()
    # Reconciled to YOLO's real names; the `*_loss_weight` aliases are gone.
    assert {"box", "cls", "dfl"} <= set(hparams)
    assert "box_loss_weight" not in hparams
    assert "cls_loss_weight" not in hparams


def test_moeclip_recipe_names_are_real_backend_arch_knobs():
    from fabric_defect_hub.models.moeclip.presets import DEFAULT_ARCH_KWARGS

    hparams = resolve_recipe("moeclip").get_default_hyperparameters()
    assert set(hparams) <= set(DEFAULT_ARCH_KWARGS)
    # Values mirror upstream's argparse defaults.
    for key in hparams:
        assert hparams[key] == DEFAULT_ARCH_KWARGS[key], key
    # Invented / misnamed knobs are gone.
    assert "routing_temperature" not in hparams
    assert "lora_rank" not in hparams


def test_dinomaly_recipe_matches_upstream_verified_defaults():
    from fabric_defect_hub.models.dinomaly.presets import DEFAULT_TRAIN_KWARGS, ENCODER_PRESETS

    hparams = resolve_recipe("dinomaly").get_default_hyperparameters()
    assert hparams["encoder_name"] in ENCODER_PRESETS
    assert hparams["encoder_name"] == "dinov2reg_vit_base_14"
    # Training schedule matches upstream's `dinomaly_mvtec_sep.py` defaults.
    assert hparams["lr"] == DEFAULT_TRAIN_KWARGS["lr"] == 2e-3
    assert hparams["final_lr"] == DEFAULT_TRAIN_KWARGS["final_lr"]
    assert hparams["image_size"] == DEFAULT_TRAIN_KWARGS["image_size"]
    assert hparams["crop_size"] == DEFAULT_TRAIN_KWARGS["crop_size"]


def test_mambaad_recipe_matches_upstream_verified_defaults():
    from fabric_defect_hub.models.mambaad.presets import DEFAULT_TRAIN_KWARGS, ENCODER_PRESETS

    hparams = resolve_recipe("mambaad").get_default_hyperparameters()
    # Encoder is a real preset; the flagship teacher.
    assert hparams["encoder_name"] in ENCODER_PRESETS
    assert hparams["encoder_name"] == "resnet34"
    # Training schedule matches upstream's published recipe (and the earlier
    # lr=0.0001 that contradicted upstream's 0.005 is corrected).
    assert hparams["lr"] == DEFAULT_TRAIN_KWARGS["lr"] == 0.005
    assert hparams["weight_decay"] == DEFAULT_TRAIN_KWARGS["weight_decay"]
    assert hparams["loss_lambda"] == DEFAULT_TRAIN_KWARGS["loss_lambda"]
    # Fixed-in-construction Mamba internals are not exposed as tunable knobs.
    assert "d_state" not in hparams
    assert "scan_directions" not in hparams
