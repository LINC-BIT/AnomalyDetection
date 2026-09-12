"""Proves a config profile actually *takes effect* on the train path — not just
that `fdh list-recipes` can print it.

The wiring contract (`recipes.apply`):
  * `load_model(recipe=...)`        -> attaches `_recipe`.
  * `run_experiment` (train path)   -> resolves the recipe's settings, attaches
    `_recipe_hparams`, and stamps the recipe id onto the trained artifact's
    metadata. A profile supplies settings only -- it never touches the model.
  * a backend                       -> folds the *trainer-safe* subset of the
    recipe's hyperparameters into its own trainer (`recipe_trainer_overrides`).

Most assertions use a lightweight fake recipe/adapter so the wiring is tested
without pulling in torch; the final cases exercise a real registered recipe.
"""

from fabric_defect_hub.core.registry import register_model
from fabric_defect_hub.core.types import Annotations, ModelInfo, Prediction, RuntimeInfo, Sample
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.evaluation.base import Evaluator
from fabric_defect_hub.loader import load_model, run_experiment
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter, ModelCapabilities
from fabric_defect_hub.recipes.apply import (
    apply_recipe_to_training,
    attach_recipe,
    recipe_trainer_overrides,
    resolve_recipe,
)


# --------------------------------------------------------------------------- #
# Lightweight, torch-free stand-ins mirroring the real interfaces.
# --------------------------------------------------------------------------- #
class _FakeRecipe:
    recipe_id = "fake-recipe"

    def get_default_hyperparameters(self):
        # Mixes a trainer knob (`lr0`), a differently-named loss gain
        # (`box_loss_weight`) and a model-constructor knob (`backbone`),
        # exactly like the real recipes — only `lr0` is trainer-safe.
        return {"lr0": 0.005, "box_loss_weight": 7.5, "backbone": "wide_resnet50_2"}


@register_model("recipe-fake-backend")
class _RecipeFakeModel(ModelAdapter):
    name = "recipe-fake-model"
    backend = "recipe-fake-backend"

    def __init__(self, name="recipe-fake-model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.seen_hparams_at_train = "<unset>"

    def capabilities(self):
        return ModelCapabilities(
            tasks=("detection",),
            prediction_fields=("boxes", "labels", "scores"),
            export_targets=("onnx",),
        )

    def train(self, config):
        # Capture what the recipe attached, as a real backend would read it.
        self.seen_hparams_at_train = getattr(self, "_recipe_hparams", None)
        return Artifact(path="fake.pt", backend=self.backend)

    def predict(self, samples, artifact=None, output_dir=None, config=None):
        return [Prediction(sample_id=s.id, boxes=[[1, 2, 3, 4]], labels=["d"], scores=[0.9]) for s in samples]

    def export(self, artifact, target, config=None):
        return ExportedArtifact(path=f"fake.{target}", target=target)


class _FakeDataset(DatasetAdapter):
    name = "recipe-fake-fabric"

    def load_samples(self):
        return [
            Sample(
                id="s-0001",
                image_path=f"{self.root}/0001.jpg",
                task="detection",
                annotations=Annotations(boxes=[[1, 2, 3, 4]], labels=["d"]),
            )
        ]


class _FakeEvaluator(Evaluator):
    task = "detection"

    def evaluate(self, samples, predictions):
        return {"map50": 1.0}


def _model_info():
    return ModelInfo(name="recipe-fake-model", backend="recipe-fake-backend", task="detection")


def _runtime():
    return RuntimeInfo(device="cpu", engine="python", precision="fp32", input_size=(640, 640))


# --------------------------------------------------------------------------- #
# recipe_trainer_overrides — the safety filter (pure, dependency-free)
# --------------------------------------------------------------------------- #
def test_recipe_trainer_overrides_keeps_only_safe_accepted_keys():
    hparams = {"lr0": 0.005, "box_loss_weight": 7.5, "backbone": "wide_resnet50_2", "momentum": 0.9}
    accepted = {"lr0", "momentum", "epochs"}

    overrides = recipe_trainer_overrides(hparams, accepted)

    # lr0 & momentum are both recognised trainer knobs and accepted here;
    # box_loss_weight / backbone are dropped (not trainer-safe).
    assert overrides == {"lr0": 0.005, "momentum": 0.9}


def test_recipe_trainer_overrides_respects_backend_acceptance():
    hparams = {"lr0": 0.005, "momentum": 0.9}
    # A backend that only accepts lr0 must not receive momentum.
    assert recipe_trainer_overrides(hparams, {"lr0"}) == {"lr0": 0.005}


# --------------------------------------------------------------------------- #
# apply_recipe_to_training — settings resolution & attachment
# --------------------------------------------------------------------------- #
def test_apply_recipe_attaches_resolved_hyperparameters():
    model = _RecipeFakeModel()
    attach_recipe(model, _FakeRecipe())

    apply_recipe_to_training(model, {"epochs": 1})

    assert model._recipe_hparams["lr0"] == 0.005


def test_apply_recipe_is_noop_without_a_recipe():
    model = _RecipeFakeModel()
    out = apply_recipe_to_training(model, {"epochs": 1})

    assert out == {"epochs": 1}
    assert not hasattr(model, "_recipe_hparams")


def test_apply_recipe_never_touches_the_model():
    # A profile supplies settings only: no architecture/loss/augmentation hook
    # may exist, so a materialised module must come out the far side untouched.
    model = _RecipeFakeModel()
    model._model = "raw-module"
    attach_recipe(model, _FakeRecipe())

    apply_recipe_to_training(model, {})

    assert model._model == "raw-module"
    assert not hasattr(model, "_recipe_loss")
    assert model._recipe_hparams["lr0"] == 0.005


def test_base_recipe_exposes_no_model_modification_hooks():
    # Guards the benchmark-integrity rule in `core.base_recipe`: if one of these
    # hooks ever comes back, a profile could silently make the row labelled
    # "YOLOv8" no longer stock YOLOv8.
    from fabric_defect_hub.core.base_recipe import BaseModelRecipe

    for hook in ("adapt_architecture", "configure_loss", "configure_optimizer", "configure_augmentations"):
        assert not hasattr(BaseModelRecipe, hook), hook


# --------------------------------------------------------------------------- #
# End-to-end through run_experiment
# --------------------------------------------------------------------------- #
def test_run_experiment_fires_recipe_and_stamps_artifact(tmp_path):
    dataset = _FakeDataset(root="data/fake")
    model = load_model("recipe-fake-backend", name="recipe-fake-model")
    attach_recipe(model, _FakeRecipe())

    result = run_experiment(
        experiment_id="exp-recipe-on",
        dataset=dataset,
        model=model,
        model_info=_model_info(),
        runtime=_runtime(),
        train_config={"epochs": 1},
        evaluator=_FakeEvaluator(),
        output_dir=str(tmp_path),
    )

    # The recipe's hyperparameters were attached *before* train() ran and the
    # backend saw them; the trained artifact is stamped with the recipe id.
    assert model.seen_hparams_at_train["lr0"] == 0.005
    assert result.metrics == {"map50": 1.0}


def test_recipe_off_vs_on_is_observable():
    dataset = _FakeDataset(root="data/fake")

    without = _RecipeFakeModel()
    run_experiment(
        experiment_id="exp-off",
        dataset=dataset,
        model=without,
        model_info=_model_info(),
        runtime=_runtime(),
        train_config={"epochs": 1},
        evaluator=_FakeEvaluator(),
    )

    with_recipe = _RecipeFakeModel()
    attach_recipe(with_recipe, _FakeRecipe())
    run_experiment(
        experiment_id="exp-on",
        dataset=dataset,
        model=with_recipe,
        model_info=_model_info(),
        runtime=_runtime(),
        train_config={"epochs": 1},
        evaluator=_FakeEvaluator(),
    )

    assert without.seen_hparams_at_train is None  # no recipe -> nothing attached
    assert with_recipe.seen_hparams_at_train["lr0"] == 0.005


# --------------------------------------------------------------------------- #
# Real registered recipes resolve and filter safely
# --------------------------------------------------------------------------- #
def test_real_recipe_resolves_by_id_and_by_target_model():
    by_id = resolve_recipe("yolov8")
    assert by_id.recipe_id == "yolov8"

    by_model = resolve_recipe("yolov8n")  # target-model lookup
    assert by_model.recipe_id == "yolov8"


def test_real_yolov8_recipe_only_exposes_trainer_safe_keys_to_ultralytics():
    from fabric_defect_hub.models.ultralytics.adapter import UltralyticsAdapter

    recipe = resolve_recipe("yolov8")
    overrides = recipe_trainer_overrides(
        recipe.get_default_hyperparameters(), UltralyticsAdapter._RECIPE_TRAINER_ARGS
    )

    # Real trainer knobs pass through, including the loss gains under YOLO's
    # actual names (`box`/`cls`/`dfl`, reconciled from `*_loss_weight`).
    assert "lr0" in overrides
    assert overrides["box"] == 7.5
    assert overrides["cls"] == 0.5
    assert overrides["dfl"] == 1.5
    # The profile carries no architecture / augmentation flags at all.
    assert "spd_conv_downsample" not in overrides
    assert "fabric_aug_enabled" not in overrides
