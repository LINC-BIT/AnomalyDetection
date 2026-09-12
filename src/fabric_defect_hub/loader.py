"""Unified dataset and model resolution loader."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.registry import get_dataset_cls, get_model_cls
from fabric_defect_hub.core.serialization import save_experiment_result, save_predictions
from fabric_defect_hub.core.types import DatasetInfo, ExperimentResult, ModelInfo, RuntimeInfo
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.models.base import Artifact, ModelAdapter
from fabric_defect_hub.reporting import append_run_log

# Backend package import paths, used to lazily register model/dataset
# implementations without requiring every optional framework to be
# installed just to import this module.
_MODEL_BACKEND_MODULES = {
    "ultralytics": "fabric_defect_hub.models.ultralytics",
    "torchvision": "fabric_defect_hub.models.torchvision.adapter",
    "anomalib": "fabric_defect_hub.models.anomalib",
    "dinomaly": "fabric_defect_hub.models.dinomaly",
    "moeclip": "fabric_defect_hub.models.moeclip",
    "mambaad": "fabric_defect_hub.models.mambaad",
}


def list_model_backends() -> list[str]:
    """Every model backend keyword `load_model`/`fdh` accepts. The single
    source of truth for `cli.py`'s several `--backend` argument `choices`
    tuples, which used to each hardcode their own copy of this list —
    a backend added here without updating every copy would silently accept
    it in `fdh run` but reject it in `fdh train`/`fdh predict`.
    """

    return sorted(_MODEL_BACKEND_MODULES)


def import_all_model_backends() -> None:
    """Best-effort import every registered backend's module, so its
    `@register_model` decorator runs even on a machine that only has some
    of the optional ML frameworks installed — one missing framework just
    means that one backend's import is skipped, not that the rest fail.
    After this call, `core.registry.list_models()` reflects what's actually
    usable *here*, not just what the project knows how to talk to (that
    static list is `list_model_backends()`). Used by `fdh list`.
    """

    for module_path in _MODEL_BACKEND_MODULES.values():
        try:
            importlib.import_module(module_path)
        except ImportError:
            continue


def load_dataset(
    name: str,
    root: str,
    split: str = "test",
    sparse_ratio: float | None = None,
    stratified_by: str | None = None,
    tiling: bool = False,
    tile_size: tuple[int, int] = (256, 256),
    overlap: float = 0.25,
    **kwargs,
) -> DatasetAdapter:
    """Resolve and instantiate a registered `DatasetAdapter` by name,

    with opt-in support for sparse ratio subsampling and sliding-window tiling.
    """
    importlib.import_module("fabric_defect_hub.datasets")  # triggers @register_dataset
    cls = get_dataset_cls(name)
    adapter = cls(root=root, split=split, **kwargs)

    # Attach SDLP loading strategies onto the adapter instance
    if sparse_ratio is not None or stratified_by is not None:
        orig_load = adapter.load_samples

        def load_samples_with_sparse(*args, **kw):
            samples = orig_load(*args, **kw)
            from fabric_defect_hub.strategies.loader_strategies import SparseSubsampler

            if stratified_by is not None:
                return SparseSubsampler.apply_stratified_pattern(samples, sparse_ratio=sparse_ratio or 0.1)
            return SparseSubsampler.apply_sparse_ratio(samples, sparse_ratio=sparse_ratio)

        adapter.load_samples = load_samples_with_sparse

    if tiling:
        adapter._tiling_enabled = True
        adapter._tile_size = tile_size
        adapter._tile_overlap = overlap

    return adapter


def load_model(
    backend: str,
    name: str,
    tta_mode: str | None = None,
    calibrate_bn: bool = False,
    precision_mode: str | None = None,
    recipe: str | None = None,
    **kwargs,
) -> ModelAdapter:
    """Resolve and instantiate a registered `ModelAdapter` by backend + name,

    with opt-in Test-Time Augmentation (TTA), BatchNorm calibration, and a
    paper-anchored config `recipe` (see `fabric_defect_hub.recipes`). When
    given, the recipe is resolved and attached here; its hooks fire in
    `run_experiment` just before training (see `recipes.apply`).
    """
    module_path = _MODEL_BACKEND_MODULES.get(backend)
    if module_path is not None:
        importlib.import_module(module_path)  # triggers @register_model
    cls = get_model_cls(backend)
    model_adapter = cls(name=name, **kwargs)

    if tta_mode is not None and tta_mode != "none":
        from fabric_defect_hub.strategies.loader_strategies import TTAInferenceWrapper

        model_adapter = TTAInferenceWrapper(model_adapter, tta_mode=tta_mode)

    if recipe is not None:
        from fabric_defect_hub.recipes.apply import attach_recipe

        attach_recipe(model_adapter, recipe)

    return model_adapter


def run_experiment(
    experiment_id: str,
    dataset: DatasetAdapter,
    model: ModelAdapter,
    model_info: ModelInfo,
    runtime: RuntimeInfo,
    train_config: dict[str, Any] | None = None,
    evaluator=None,
    output_dir: str | None = None,
    artifact: Artifact | None = None,
    profiler=None,
    profile_config=None,
    export_target: str | None = None,
    export_config: dict[str, Any] | None = None,
    run_log_path: str | None = None,
) -> ExperimentResult:

    samples = dataset.load_samples()

    # Fire any attached recipe's hooks (loss / hyperparameters / architecture)
    # right before training, and record which recipe tuned the resulting model.
    if train_config is not None:
        from fabric_defect_hub.recipes.apply import apply_recipe_to_training

        train_config = apply_recipe_to_training(model, train_config)
    active_artifact = model.train(train_config) if train_config is not None else artifact
    _recipe = getattr(model, "_recipe", None)
    if _recipe is not None and active_artifact is not None:
        active_artifact.metadata.setdefault("recipe", _recipe.recipe_id)

    # Check if sliding-window tiling strategy is enabled on dataset
    if getattr(dataset, "_tiling_enabled", False):
        from fabric_defect_hub.strategies.loader_strategies import SlidingWindowTiler

        tiler = SlidingWindowTiler(
            tile_size=getattr(dataset, "_tile_size", (256, 256)),
            overlap=getattr(dataset, "_tile_overlap", 0.25),
        )
        predictions = []
        for s in samples:
            tiles, meta_info = tiler.split_sample(s)
            if meta_info.get("tiled", False):
                tile_preds = model.predict(tiles, active_artifact)
                stitched_pred = tiler.stitch_predictions(tile_preds, meta_info)
                predictions.append(stitched_pred)
            else:
                predictions.extend(model.predict([s], active_artifact))
    else:
        predictions = model.predict(samples, active_artifact)

    evaluated_metrics = evaluator.evaluate(samples, predictions) if evaluator is not None else {}
    metrics = {
        name: float(value)
        for name, value in evaluated_metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }

    artifacts: dict[str, str] = {}
    if active_artifact is not None:
        artifacts["model"] = active_artifact.path

    if profiler is not None:
        if active_artifact is None:
            raise ValueError("profiling requires a trained or explicitly supplied model artifact")
        if profile_config is None or export_target is None:
            raise ValueError("profiling requires both profile_config and export_target")
        # Every backend's `export` takes the same three arguments (see
        # `ModelAdapter.export`), so this is a plain call — it used to have to
        # `inspect.signature` the adapter first to find out whether it would
        # accept `config`.
        exported = model.export(active_artifact, target=export_target, config=export_config or {})
        export_path = Path(exported.path)
        if not export_path.is_file():
            raise FileNotFoundError(f"exported model does not exist: {export_path}")
        profile_metrics = profiler.profile(exported, profile_config)
        profile_metrics["model_size_mb"] = export_path.stat().st_size / (1024 * 1024)
        metrics.update(profile_metrics)
        runtime = profiler.runtime_info(profile_config)
        artifacts[f"model_{export_target}"] = str(export_path)
    if output_dir is not None:
        base = f"{output_dir.rstrip('/')}/{experiment_id}"
        artifacts["predictions"] = str(save_predictions(predictions, f"{base}/predictions.json"))
        power_report = getattr(profiler, "last_power_report", None) if profiler is not None else None
        if power_report is not None:
            from fabric_defect_hub.profiling.power import save_power_report

            artifacts["power_measurement"] = str(save_power_report(power_report, f"{base}/power.json"))

    result = ExperimentResult(
        experiment_id=experiment_id,
        model=model_info,
        dataset=DatasetInfo(name=dataset.name, split=dataset.split),
        runtime=runtime,
        metrics=metrics,
        artifacts=artifacts,
    )

    if output_dir is not None:
        result_path = save_experiment_result(result, f"{output_dir.rstrip('/')}/{experiment_id}/result.json")
        result.artifacts["result"] = str(result_path)

    if run_log_path is not None:
        append_run_log(result, run_log_path)

    return result
