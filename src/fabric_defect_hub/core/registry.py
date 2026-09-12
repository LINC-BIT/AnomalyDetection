"""Name-based registries so the loader can resolve datasets, models,
evaluators, and profilers by string name instead of importing every
framework eagerly, and so `benchmark.py`'s YAML-driven config doesn't have
to hardcode "which class does 'detection' mean" anywhere.

Framework-specific packages (`models/ultralytics`, `models/torchvision`,
`models/anomalib`, individual `datasets/*` modules) register themselves via
`register_dataset`/`register_model` when imported. Optional backends are
only imported on demand inside `loader.py`, so a machine without, say,
Anomalib installed can still use the Ultralytics or torchvision paths.

`register_evaluator`/`register_profiler` are plain class decorators (no
name argument) rather than decorator factories like the two above: every
`Evaluator` already carries a `task` class attribute and every
`BackendProfiler` an `engine` one (read by `evaluation/base.py` and
`profiling/base.py` respectively, and already relied on elsewhere, e.g.
`evaluation/base.py`'s docstring example `{'map50': 0.81}` keyed by task).
Requiring a *second*, separately-typed-out string at the registration site
would just be a second name that can silently drift from the first; reading
`cls.task`/`cls.engine` directly makes that impossible. Both evaluator and
profiler modules are safe to import unconditionally regardless of which
optional ML framework is installed — every concrete class only imports its
actual dependency (`sklearn`, `torchmetrics`, `torch`, `onnxruntime`, ...)
lazily inside its methods, never at module level — so, unlike datasets and
models, there is no lazy-module-path dance needed for these two.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_DATASET_REGISTRY: dict[str, type] = {}
_MODEL_REGISTRY: dict[str, type] = {}
_EVALUATOR_REGISTRY: dict[str, type] = {}
_PROFILER_REGISTRY: dict[str, type] = {}
_RECIPE_REGISTRY: dict[str, Any] = {}


def register_dataset(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if name in _DATASET_REGISTRY:
            raise ValueError(f"dataset '{name}' is already registered")
        _DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def register_model(backend: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if backend in _MODEL_REGISTRY:
            raise ValueError(f"model backend '{backend}' is already registered")
        _MODEL_REGISTRY[backend] = cls
        return cls

    return decorator


def register_evaluator(cls: type[T]) -> type[T]:
    task = cls.task
    if task in _EVALUATOR_REGISTRY:
        raise ValueError(f"evaluator for task '{task}' is already registered")
    _EVALUATOR_REGISTRY[task] = cls
    return cls


def register_profiler(cls: type[T]) -> type[T]:
    engine = cls.engine
    if engine in _PROFILER_REGISTRY:
        raise ValueError(f"profiler for engine '{engine}' is already registered")
    _PROFILER_REGISTRY[engine] = cls
    return cls


def register_recipe(recipe_id: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if recipe_id in _RECIPE_REGISTRY:
            raise ValueError(f"recipe '{recipe_id}' is already registered")
        _RECIPE_REGISTRY[recipe_id] = cls
        return cls

    return decorator


def get_dataset_cls(name: str) -> type:
    try:
        return _DATASET_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_DATASET_REGISTRY)) or "<none registered>"
        raise KeyError(f"unknown dataset '{name}'. Known datasets: {known}") from exc


def get_model_cls(backend: str) -> type:
    try:
        return _MODEL_REGISTRY[backend]
    except KeyError as exc:
        known = ", ".join(sorted(_MODEL_REGISTRY)) or "<none registered>"
        raise KeyError(f"unknown model backend '{backend}'. Known backends: {known}") from exc


def get_evaluator_cls(task: str) -> type:
    try:
        return _EVALUATOR_REGISTRY[task]
    except KeyError as exc:
        known = ", ".join(sorted(_EVALUATOR_REGISTRY)) or "<none registered>"
        raise KeyError(f"unknown evaluator task '{task}'. Known tasks: {known}") from exc


def get_profiler_cls(engine: str) -> type:
    try:
        return _PROFILER_REGISTRY[engine]
    except KeyError as exc:
        known = ", ".join(sorted(_PROFILER_REGISTRY)) or "<none registered>"
        raise KeyError(f"unknown profiler engine '{engine}'. Known engines: {known}") from exc


def get_recipe(recipe_id_or_model: str) -> Any:
    # Check exact match first
    if recipe_id_or_model in _RECIPE_REGISTRY:
        recipe_item = _RECIPE_REGISTRY[recipe_id_or_model]
        return recipe_item() if isinstance(recipe_item, type) else recipe_item

    # Search for recipe matching target model
    for recipe in _RECIPE_REGISTRY.values():
        instance = recipe() if isinstance(recipe, type) else recipe
        if hasattr(instance, "target_models") and any(recipe_id_or_model.startswith(m) for m in instance.target_models):
            return instance

    known = ", ".join(sorted(_RECIPE_REGISTRY)) or "<none registered>"
    raise KeyError(f"No optimization recipe found for '{recipe_id_or_model}'. Known recipes: {known}")


def list_datasets() -> list[str]:
    return sorted(_DATASET_REGISTRY)


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def list_evaluators() -> list[str]:
    return sorted(_EVALUATOR_REGISTRY)


def list_profilers() -> list[str]:
    return sorted(_PROFILER_REGISTRY)


def list_recipes() -> list[str]:
    return sorted(_RECIPE_REGISTRY)
