"""Wiring that makes a config profile *actually take effect* during a run,
rather than only being printed by `fdh list-recipes`.

Before this module the `recipes/*.py` classes were resolvable
(`get_recipe`) and self-describing (`get_recipe_summary`) but nothing ever
called `get_default_hyperparameters` on the train/predict path — the recipe
was documentation, not behaviour.

A recipe influences a run in exactly one concrete, inspectable way:
hyperparameters are resolved once and attached as ``_recipe_hparams`` so a
backend can fold the trainer-relevant subset into its own trainer.

`load_model(..., recipe=...)` calls `attach_recipe`; `run_experiment` calls
`apply_recipe_to_training` immediately before `model.train`.

A profile may *only* supply settings — never a loss, an architecture change,
or an augmentation pipeline (see `core.base_recipe`): a benchmark row labelled
"YOLOv8" has to be stock YOLOv8 for the comparison to mean anything.

Deliberately *not* done here: flattening every recipe hyperparameter into the
backend's ``train(**kwargs)``. Recipe hyperparameter dicts use names that don't
always match a backend's trainer (YOLO wants ``box``, some upstream papers say
``box_loss_weight``), and carry model-constructor knobs that are not trainer
args at all. Blindly forwarding them would crash a real trainer, so each
backend opts in to the keys it understands via `recipe_trainer_overrides`
(see `UltralyticsAdapter.train`).
"""

from __future__ import annotations

from typing import Any

# Recipe hyperparameter keys that name a real, framework-agnostic *training*
# knob (as opposed to a model-constructor argument such as `backbone`, which
# reaches the backend via `recipe_model_kwargs` instead). A backend adapter
# intersects this with its own trainer's accepted arguments before use.
_TRAINER_HPARAM_KEYS = frozenset(
    {
        "lr0",
        "lrf",
        "lr",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_iters",
        "epochs",
        "total_iters",
        "batch",
        "batch_size",
        "optimizer",
        # Ultralytics loss-gain names (its `box`/`cls`/`dfl`, not `*_loss_weight`).
        "box",
        "cls",
        "dfl",
    }
)


def resolve_recipe(recipe: Any) -> Any:
    """Accept a recipe id, a target model name, or an already-instantiated
    recipe object, and return the recipe instance (or ``None``).

    The `fabric_defect_hub.recipes` import is lazy so that model loading with
    no recipe never pays for it and never risks an import cycle.
    """

    if recipe is None:
        return None
    if isinstance(recipe, str):
        import fabric_defect_hub.recipes  # noqa: F401  (triggers @register_recipe)
        from fabric_defect_hub.core.registry import get_recipe

        return get_recipe(recipe)
    return recipe


def attach_recipe(model_adapter: Any, recipe: Any) -> Any:
    """Resolve `recipe` and stash the instance on `model_adapter._recipe`.

    Returns the adapter unchanged (aside from the attribute) so callers can
    chain it. A ``None`` recipe is a no-op.
    """

    recipe_obj = resolve_recipe(recipe)
    if recipe_obj is not None:
        model_adapter._recipe = recipe_obj
    return model_adapter


def recipe_trainer_overrides(hparams: dict[str, Any], accepted: set[str] | frozenset[str]) -> dict[str, Any]:
    """The subset of a recipe's hyperparameters that is safe to hand to a
    specific backend trainer: keys that are both recognised trainer knobs
    (`_TRAINER_HPARAM_KEYS`) *and* accepted by that backend (`accepted`).

    Pure and dependency-free so backends and tests can call it directly.
    """

    safe = _TRAINER_HPARAM_KEYS & set(accepted)
    return {k: v for k, v in hparams.items() if k in safe}


def recipe_model_kwargs(hparams: dict[str, Any], accepted: set[str] | frozenset[str]) -> dict[str, Any]:
    """The subset of a recipe's hyperparameters that name a real *model
    constructor* argument for a given backend (`accepted`).

    Backends whose model is built from a free-form kwargs dict (e.g. the
    anomalib `Folder` models, constructed from
    `AnomalibConfig.resolved_model_kwargs()`) pass their introspection-verified
    accepted-argument set here so a recipe can only override values of real
    constructor parameters — never inject an unknown kwarg that would raise
    deep inside the model's ``__init__``. Pure and dependency-free.
    """

    return {k: v for k, v in hparams.items() if k in accepted}


def apply_recipe_to_training(model_adapter: Any, train_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Resolve the attached recipe's settings just before training and attach
    them to `model_adapter` for the backend to consume.

    Idempotent side effect: ``_recipe_hparams`` — the resolved default
    hyperparameters. The model itself is never touched; a profile supplies
    settings only.

    Returns `train_config` unchanged; the backend is responsible for reading
    the attached attribute (via `recipe_trainer_overrides` for trainer args).
    """

    recipe_obj = getattr(model_adapter, "_recipe", None)
    if recipe_obj is None:
        return train_config

    model_adapter._recipe_hparams = dict(recipe_obj.get_default_hyperparameters())
    return train_config
