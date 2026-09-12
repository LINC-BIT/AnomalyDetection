"""The single `predict` entry point, mirroring `training.run_train`: pick a
model config the same way `fdh train` does (a path, a filename stem under
`configs/models/`, or a model keyword), point it at a previously trained
artifact instead of training a new one, and run inference over either
explicit image paths or a slice of a registered dataset.

Before this module, every backend's `predict()`/`load_trained_model()` was
only reachable through the interactive web UI's `InferenceSessionManager`
(see `inference/session.py`) or by hand-writing a Python script — training
a model was config/CLI-driven via `fdh train`, but running it afterwards
was not. This closes that gap: every model this project can train, it can
also run inference for, from the command line.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.models.base import Artifact, ModelCapabilities
from fabric_defect_hub.training import (
    DEFAULT_DATASET_ROOTS,
    DEFAULT_MODEL_CONFIG_DIR,
    BACKEND_MODEL_KEY,
    apply_default_dataset_root,
    apply_model_overrides,
    infer_backend,
    load_raw_config,
    resolve_model_config_and_variant,
)

# Per backend: (module, class name) of its `ModelAdapter` — kept separate
# from `training._BACKEND_PIPELINE_MODULES` (which points at the
# train/val/export *pipeline* module) because predict only ever needs the
# adapter itself.
_ADAPTER_MODULES: dict[str, tuple[str, str]] = {
    "ultralytics": ("fabric_defect_hub.models.ultralytics.adapter", "UltralyticsAdapter"),
    "torchvision": ("fabric_defect_hub.models.torchvision.adapter", "TorchvisionAdapter"),
    "anomalib": ("fabric_defect_hub.models.anomalib.adapter", "AnomalibAdapter"),
    "dinomaly": ("fabric_defect_hub.models.dinomaly.adapter", "DinomalyAdapter"),
    "moeclip": ("fabric_defect_hub.models.moeclip.adapter", "MoECLIPAdapter"),
    "mambaad": ("fabric_defect_hub.models.mambaad.adapter", "MambaADAdapter"),
}

# NOTE: there used to be an `_ANOMALY_MAP_BACKENDS = {"anomalib", "dinomaly",
# "moeclip", "mambaad"}` set here, consulted for three unrelated questions:
# what task a bare image path implies, which config key names the model, and
# whether `predict()` should be handed an `output_dir`. The three answers
# coincided for those four backends by accident, not by definition, and each
# already has an owner:
#
#   task            -> ModelCapabilities.supports_task / .tasks
#   config key      -> training.BACKEND_MODEL_KEY
#   output_dir      -> ModelCapabilities.fills("anomaly_map")
#
# Asking the owner keeps a seventh backend from having to be added to a list
# it doesn't know exists — and stopped `predict` from advertising a pixel map
# for GANomaly, an anomalib model that has none.


@dataclass
class PredictInput:
    """Where to source `Sample`s for prediction — either raw image paths or
    a slice of a registered dataset (mutually exclusive; see `_load_samples`).
    """

    images: list[str] = field(default_factory=list)
    dataset: str | None = None
    dataset_root: str | None = None
    split: str = "test"
    num_samples: int | None = None
    pattern: str | int | None = None
    category: str | None = None
    seed: int = 0
    # Which ground truth to attach when loading the dataset — distinct from
    # `run_evaluate`'s `task`, which picks the *evaluator*. A dataset that
    # serves several tasks (ZJU-Leaper: anomaly, segmentation, detection)
    # attaches only one of them, and its default is not necessarily the one
    # the model needs: loading ZJU-Leaper's default `anomaly` samples and
    # then scoring a YOLO run against them yields boxes with no box ground
    # truth to match, i.e. no metrics at all. Ignored when the dataset does
    # not declare the task in `core.dataset_capabilities`.
    task: str | None = None


@dataclass
class PredictRunResult:
    backend: str
    variant: str
    predictions: list[Prediction]
    samples: list[Sample] = field(default_factory=list)
    # Which config the `model` argument resolved to — same reason
    # `TrainRunResult` carries it: keyword resolution can pick a config the
    # caller never named, so the choice has to be inspectable.
    config_path: str | None = None


@dataclass
class EvaluateRunResult:
    backend: str
    variant: str
    metrics: dict[str, float]
    sample_count: int
    config_path: str | None = None


def build_adapter(backend: str, variant: str):
    """Construct a backend's `ModelAdapter` by name.

    Public because `_ADAPTER_MODULES` is the project's one backend->module
    table and a second caller (`metric_sweep`, which needs `capabilities()`
    and `export()`) must not carry its own copy. Note this deliberately does
    *not* go through `core.registry.get_model_cls`: that registry is
    populated by `@register_model` at import time, so it is empty until
    something has already imported the backend — which is exactly the
    ordering bug that made an unattended sweep silently skip every model.
    """

    module_name, cls_name = _ADAPTER_MODULES[backend]
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(name=variant)


_build_adapter = build_adapter


def _resolve_weights_artifact(backend: str, weights: str) -> Artifact | str:
    """Anomalib's `load_trained_model` refuses a bare path — Lightning
    checkpoints can deserialize arbitrary Python objects, so it requires a
    *trusted* `Artifact` instead (see `AnomalibAdapter.load_trained_model`).
    `register_trained_model` names registry files `<ModelClass>.ckpt`
    (e.g. `Patchcore.ckpt`), so the model class is recovered from the
    filename stem when the caller just passes a path. Ultralytics/
    torchvision checkpoints embed their own architecture/variant info and
    accept a bare path directly.
    """

    if backend != "anomalib":
        return weights
    model_class = Path(weights).stem
    return Artifact(path=weights, backend="anomalib", metadata={"model_class": model_class, "trusted": True})


def _load_samples(source: PredictInput, capabilities: ModelCapabilities) -> list[Sample]:
    """Resolve `source` to `Sample`s.

    A raw image path carries no task of its own, so the model's declared
    capabilities supply one: "anomaly" when the model serves it, otherwise
    whatever it does serve. Reading it from `capabilities` rather than from a
    list of backend names means a backend that grows a second task, or a new
    backend entirely, needs no edit here.
    """

    if source.images and source.dataset:
        raise ValueError("pass either --image or --dataset, not both")
    if source.images:
        task = "anomaly" if capabilities.supports_task("anomaly") else capabilities.tasks[0]
        return [
            Sample(id=Path(image_path).stem, image_path=image_path, task=task, annotations=Annotations())
            for image_path in source.images
        ]
    if not source.dataset:
        raise ValueError("pass --image (one or more) or --dataset to select what to run inference on")

    from fabric_defect_hub.loader import load_dataset

    raw = apply_default_dataset_root({"data": {"dataset": source.dataset, "dataset_root": source.dataset_root}})
    root = raw["data"]["dataset_root"]
    if not root:
        raise ValueError(
            f"no dataset_root for {source.dataset!r}; pass --dataset-root explicitly "
            f"(only {', '.join(sorted(DEFAULT_DATASET_ROOTS))} have a project default)"
        )

    kwargs: dict[str, Any] = {"seed": source.seed}
    if source.num_samples is not None:
        kwargs["num_samples"] = source.num_samples
    if source.pattern is not None:
        kwargs["pattern"] = source.pattern
    if source.category is not None:
        kwargs["category"] = source.category
    if source.task is not None and _dataset_serves(source.dataset, source.task):
        kwargs["task"] = source.task

    dataset = load_dataset(source.dataset, root=root, split=source.split, **kwargs)
    return dataset.load_samples()


def _dataset_serves(dataset: str, task: str) -> bool:
    """Whether `dataset` declares ground truth for `task`.

    Asking `core.dataset_capabilities` rather than passing `task` blindly:
    not every `DatasetAdapter` takes a `task` kwarg, and a dataset that has
    no boxes cannot be made to produce them by being asked.
    """

    from fabric_defect_hub.core.dataset_capabilities import capabilities_for

    try:
        return task in capabilities_for(dataset).tasks
    except KeyError:
        return False


def run_predict(
    model: str | Path,
    weights: str,
    source: PredictInput,
    backend: str | None = None,
    variant: str | None = None,
    config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR,
    output_dir: str | None = None,
    enable_tiling: bool = False,
    enable_tta: bool = False,
    tile_size: int | None = None,
    tile_overlap: float | None = None,
) -> PredictRunResult:
    """The unified inference entry point, mirroring `training.run_train`.

    1. Resolve `model` to a config path exactly like `run_train` does (a
       full path, a filename stem under `config_dir`, or a model keyword
       like "yolov8n"/"patchcore"), and resolve its backend.
    2. Layer `variant` onto the model section (same mechanism as
       `training.apply_model_overrides`), so inference can target any
       variant that backend supports, not just whatever the config declares.
    3. Instantiate that backend's adapter and load `weights` — a path to a
       previously trained/registered artifact (see `fdh train`'s
       `registered_artifact.path` output).
    4. Load `source` (explicit image paths or a dataset selection) into
       `Sample`s and run `adapter.predict(...)`.

    `output_dir` is only meaningful for anomalib: it additionally persists
    each sample's pixel-level anomaly map (see `AnomalibAdapter.predict`).
    """

    model_config, implied_variant = resolve_model_config_and_variant(str(model), config_dir=config_dir)
    variant = variant if variant is not None else implied_variant
    raw = load_raw_config(model_config)
    resolved_backend = backend or infer_backend(raw)
    if resolved_backend not in _ADAPTER_MODULES:
        raise ValueError(f"unknown backend '{resolved_backend}'; expected one of {sorted(_ADAPTER_MODULES)}")

    raw = apply_model_overrides(raw, resolved_backend, variant)
    if resolved_backend == "ultralytics":
        from fabric_defect_hub.models.ultralytics.config import resolve_variant_profile

        raw = resolve_variant_profile(raw)
        predict = dict(raw.get("predict") or {})
        if enable_tiling:
            predict["tiling"] = True
        if tile_size is not None:
            predict["tile_size"] = [tile_size, tile_size]
        if tile_overlap is not None:
            predict["tile_overlap"] = tile_overlap
        if enable_tta:
            predict["tta_mode"] = "flip_multiscale"
        raw["predict"] = predict
    # Which key under `model:` names the model is a fact about each backend's
    # config shape, owned by `training.BACKEND_MODEL_KEY` — the same table
    # `apply_model_overrides` writes through, so the two cannot disagree.
    model_key = BACKEND_MODEL_KEY[resolved_backend]
    resolved_variant = raw.get("model", {}).get(model_key)
    if not resolved_variant:
        raise ValueError(f"config has no model.{model_key}; pass --variant explicitly")

    adapter = _build_adapter(resolved_backend, resolved_variant)
    capabilities = adapter.capabilities()
    artifact = adapter.load_trained_model(_resolve_weights_artifact(resolved_backend, weights))

    samples = _load_samples(source, capabilities)
    if not samples:
        raise ValueError("no samples resolved to run inference on")

    if resolved_backend == "ultralytics":
        from fabric_defect_hub.models.ultralytics.config import UltralyticsConfig

        config = UltralyticsConfig.from_dict(raw)
        predictions = adapter.predict(samples, artifact, config=config.predict.as_overrides() | {
            "tta_mode": config.predict.tta_mode,
        })
    elif capabilities.fills("anomaly_map"):
        # Only a model that actually produces a pixel map gets somewhere to
        # write one. GANomaly is an anomalib model that does not (see
        # `anomalib.presets.IMAGE_LEVEL_ONLY`), and under the old
        # backend-name check it was handed an `output_dir` regardless.
        predictions = adapter.predict(samples, artifact, output_dir=output_dir)
    else:
        predictions = adapter.predict(samples, artifact)
    return PredictRunResult(
        backend=resolved_backend,
        variant=resolved_variant,
        predictions=predictions,
        samples=samples,
        config_path=str(model_config),
    )


def run_evaluate(
    model: str | Path,
    weights: str,
    source: PredictInput,
    backend: str | None = None,
    variant: str | None = None,
    config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR,
    task: str | None = None,
    output_dir: str | None = None,
    enable_tiling: bool = False,
    enable_tta: bool = False,
    tile_size: int | None = None,
    tile_overlap: float | None = None,
) -> EvaluateRunResult:
    """Score a trained checkpoint against `source`'s ground truth — the
    CLI-scriptable equivalent of the web Benchmark tab's leaderboard, for
    validating a model against any registered dataset (e.g. `tilda-400`,
    `fabric-defects`) without a browser. Runs inference exactly like
    `run_predict` (same tiling/TTA support, since it simply calls it), then
    scores the result via the task-appropriate `Evaluator` (see
    `evaluation.evaluator_for_task`).

    `task`, if omitted, is taken from the resolved samples' own `.task`
    (every `Sample` already carries the task it was built for) — pass it
    explicitly only to force a different evaluator than the dataset's
    default (e.g. scoring a segmentation-capable dataset's masks as
    image-level anomaly detection instead).

    `output_dir` is what makes **pixel-level** metrics possible for the
    anomaly backends. `AnomalyEvaluator` reads each prediction's
    `anomaly_map`, and the adapters only persist that map (and fill the
    field) when given somewhere to write it — so without this argument an
    anomaly model returns image AUROC/F1 and nothing else, however capable
    of pixel-level output it is. Pass a directory to also get
    `pixel_auroc` / `pixel_aupro` / `iap`.

    Not the default: the maps are one `.npy` per sample, so writing them is
    the caller's decision, exactly as in `run_predict`. It has no effect for
    a model whose `capabilities()` does not fill `anomaly_map` (GANomaly), or
    for the detection/segmentation backends, which score from boxes/masks.

    `source` must resolve via `--dataset` (a `PredictInput.images` source
    carries no ground truth to score against).
    """

    from fabric_defect_hub.evaluation import evaluator_for_task

    if source.images and not source.dataset:
        raise ValueError("evaluate requires --dataset (raw --image sources have no ground truth to score)")

    run = run_predict(
        model, weights=weights, source=source, backend=backend, variant=variant, config_dir=config_dir,
        output_dir=output_dir,
        enable_tiling=enable_tiling, enable_tta=enable_tta, tile_size=tile_size, tile_overlap=tile_overlap,
    )
    if not run.samples:
        raise ValueError("no samples resolved to run evaluation on")
    resolved_task = task or run.samples[0].task
    metrics = evaluator_for_task(resolved_task).evaluate(run.samples, run.predictions)
    return EvaluateRunResult(
        backend=run.backend,
        variant=run.variant,
        metrics={key: float(value) for key, value in metrics.items()},
        sample_count=len(run.samples),
        config_path=run.config_path,
    )
