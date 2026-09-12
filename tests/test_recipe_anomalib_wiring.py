"""Stage 3: the `patchcore` recipe's hyperparameters are reconciled to
anomalib's real `Patchcore` constructor vocabulary, and `model.recipe` is
actually consumed by `AnomalibConfig.resolved_model_kwargs()`.

Guards the exact failure that motivated this work: the recipe used to say
`backbone="wideresnet50"` / `n_neighbors=9` (both wrong names/values), which
either silently no-op'd or would have raised inside anomalib's `__init__`.
"""

from fabric_defect_hub.models.anomalib.config import AnomalibConfig
from fabric_defect_hub.models.anomalib.presets import MODEL_PRESETS
from fabric_defect_hub.recipes.apply import recipe_model_kwargs, resolve_recipe


def test_recipe_model_kwargs_drops_unaccepted_keys():
    hparams = {"backbone": "wide_resnet50_2", "num_neighbors": 9, "made_up_flag": True}
    accepted = {"backbone", "num_neighbors", "coreset_sampling_ratio"}

    assert recipe_model_kwargs(hparams, accepted) == {
        "backbone": "wide_resnet50_2",
        "num_neighbors": 9,
    }


def test_patchcore_recipe_uses_only_real_anomalib_constructor_args():
    # Every key the recipe emits must be a real Patchcore constructor arg,
    # i.e. present in the introspection-verified preset. This is what makes it
    # safe to feed straight into anomalib.
    recipe = resolve_recipe("patchcore")
    hparams = recipe.get_default_hyperparameters()
    accepted = set(MODEL_PRESETS["Patchcore"])

    assert set(hparams) <= accepted, f"recipe emits non-constructor keys: {set(hparams) - accepted}"
    # And the values are the paper's reproduction settings, correctly named.
    assert hparams["backbone"] == "wide_resnet50_2"
    assert hparams["num_neighbors"] == 9
    assert "n_neighbors" not in hparams  # the old, wrong name is gone


def test_resolved_model_kwargs_layers_preset_recipe_and_explicit():
    # preset < recipe < explicit train.model_kwargs
    config = AnomalibConfig.from_dict(
        {
            "model": {"name": "PatchCore", "recipe": "patchcore"},
            "data": {"dataset": "mvtec-ad", "dataset_root": "/tmp/mvtec"},
            "train": {"model_kwargs": {"coreset_sampling_ratio": 0.25}},
        }
    )

    resolved = config.resolved_model_kwargs()

    # From the recipe layer (overriding preset where they'd differ):
    assert resolved["backbone"] == "wide_resnet50_2"
    assert resolved["num_neighbors"] == 9
    # Explicit train.model_kwargs wins over the recipe:
    assert resolved["coreset_sampling_ratio"] == 0.25
    # No unknown kwargs leaked in that would break anomalib's __init__:
    assert set(resolved) <= set(MODEL_PRESETS["Patchcore"])


def test_resolved_model_kwargs_without_recipe_is_just_the_preset():
    config = AnomalibConfig.from_dict(
        {
            "model": {"name": "PatchCore"},
            "data": {"dataset": "mvtec-ad", "dataset_root": "/tmp/mvtec"},
        }
    )

    assert config.resolved_model_kwargs() == MODEL_PRESETS["Patchcore"]
