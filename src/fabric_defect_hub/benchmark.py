"""Cross-backend benchmark orchestration: run the same (or different)
dataset(s) through several models — any mix of Ultralytics/torchvision/
Anomalib backends — and produce a sorted leaderboard, closing the README's
Phase 2 "unified train/predict/evaluate/artifact-management pipeline" item.

`loader.run_experiment` already runs one (dataset, model, evaluator) triple
end to end and can persist its `ExperimentResult`. What was still missing
is the layer above that: describing *several* runs declaratively and
getting back a comparable, sorted list — this is what turns individual
experiments into an actual benchmark. `evaluation/*.py` being backend-
agnostic (see their module docstrings) is what makes a fair leaderboard
possible here: two rows produced by different frameworks were still scored
by the exact same metric code.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.types import ExperimentResult, ModelInfo, RuntimeInfo
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.evaluation.base import Evaluator
from fabric_defect_hub.loader import load_dataset, load_model, run_experiment
from fabric_defect_hub.models.base import Artifact
from fabric_defect_hub.profiling.base import ProfileConfig


@dataclass
class BenchmarkRun:
    """One row of a benchmark: which dataset, which model, how to train/
    evaluate it. Either pass a ready-built `dataset`, or `dataset_name` +
    `dataset_root` (+ `dataset_kwargs`) to have it loaded for you.
    """

    experiment_id: str
    model_backend: str
    model_name: str
    model_info: ModelInfo
    runtime: RuntimeInfo
    dataset: DatasetAdapter | None = None
    dataset_name: str | None = None
    dataset_root: str | None = None
    dataset_split: str = "test"
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    train_config: dict[str, Any] | None = None
    evaluator: Evaluator | None = None
    artifact: Artifact | None = None
    train_dataset: DatasetAdapter | None = None
    validation_dataset: DatasetAdapter | None = None
    profiler: Any | None = None
    profile_config: ProfileConfig | None = None
    export_target: str | None = None
    export_config: dict[str, Any] = field(default_factory=dict)

    def resolve_dataset(self) -> DatasetAdapter:
        if self.dataset is not None:
            return self.dataset
        if self.dataset_name is None or self.dataset_root is None:
            raise ValueError(
                f"BenchmarkRun {self.experiment_id!r}: provide either 'dataset', or "
                "'dataset_name' + 'dataset_root'."
            )
        return load_dataset(
            self.dataset_name, root=self.dataset_root, split=self.dataset_split, **self.dataset_kwargs
        )

    def resolved_train_config(self) -> dict[str, Any] | None:
        if self.train_config is None:
            return None
        config = dict(self.train_config)
        if self.train_dataset is None:
            return config
        train_samples = self.train_dataset.load_samples()
        validation_samples = (
            self.validation_dataset.load_samples()
            if self.validation_dataset is not None
            else self.resolve_dataset().load_samples()
        )
        if self.model_backend == "ultralytics":
            config.setdefault("samples", {"train": train_samples, "val": validation_samples})
        elif self.model_backend == "torchvision":
            config.setdefault("train_samples", train_samples)
            config.setdefault("val_samples", validation_samples)
        elif self.model_backend in ("anomalib", "dinomaly"):
            config.setdefault("train_samples", train_samples)
            config.setdefault("test_samples", validation_samples)
        elif self.model_backend in ("moeclip", "mambaad"):
            # Neither takes a validation split during train(): MoECLIP has
            # no in-loop validation pass, and MambaADAdapter.train() only
            # ever consumes train_samples too (see their adapters).
            config.setdefault("train_samples", train_samples)
        return config


def run_benchmark(
    runs: list[BenchmarkRun], output_dir: str | None = None, run_log_path: str | None = None
) -> list[ExperimentResult]:
    """Execute every `BenchmarkRun` in order and return their `ExperimentResult`s.

    A failure in one run is not swallowed — if you need partial results
    from a long benchmark despite one row failing, wrap individual runs
    yourself; silently continuing past a broken run would make the
    leaderboard's comparability claim (identical metric code across rows)
    untrustworthy.

    `run_log_path`, if given, is passed through to every row's
    `run_experiment(..., run_log_path=...)` call, so every model/backend in
    the benchmark appends to the same shared, never-overwritten run log
    (see `reporting.append_run_log`) — one accumulating source of truth for
    plotting across benchmarks, not just within this one call's `results`.
    """

    results: list[ExperimentResult] = []
    for run in runs:
        dataset = run.resolve_dataset()
        model = load_model(run.model_backend, run.model_name, **run.model_kwargs)
        result = run_experiment(
            experiment_id=run.experiment_id,
            dataset=dataset,
            model=model,
            model_info=run.model_info,
            runtime=run.runtime,
            train_config=run.resolved_train_config(),
            evaluator=run.evaluator,
            output_dir=output_dir,
            artifact=run.artifact,
            profiler=run.profiler,
            profile_config=run.profile_config,
            export_target=run.export_target,
            export_config=run.export_config,
            run_log_path=run_log_path,
        )
        results.append(result)
    return results


def leaderboard(
    results: list[ExperimentResult], metric: str, descending: bool = True
) -> list[ExperimentResult]:
    """Sort `results` by `metric` (e.g. 'map50', 'image_auroc'). Results that
    didn't compute `metric` (different task, evaluator not run, ...) are
    dropped rather than sorted arbitrarily — mixing a missing-metric
    sentinel into a ranked list would silently misrank real runs.
    """

    ranked = [
        result
        for result in results
        if metric in result.metrics and math.isfinite(result.metrics[metric])
    ]
    return sorted(ranked, key=lambda r: r.metrics[metric], reverse=descending)


@dataclass
class BenchmarkConfig:
    """Declarative, YAML-loadable description of a complete benchmark."""

    runs: list[BenchmarkRun]
    output_dir: str = "artifacts/benchmarks"
    leaderboard_metric: str | None = None
    descending: bool = True
    report_path: str | None = None
    run_log_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        import yaml

        with open(path) as file:
            raw = yaml.safe_load(file) or {}
        if not isinstance(raw, dict):
            raise ValueError("Benchmark config top level must be a mapping.")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchmarkConfig":
        allowed = {"runs", "output_dir", "leaderboard", "report_path", "run_log_path"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"BenchmarkConfig: unknown keys {sorted(unknown)}.")
        run_entries = raw.get("runs")
        if not isinstance(run_entries, list) or not run_entries:
            raise ValueError("BenchmarkConfig requires a non-empty 'runs' list.")
        board = raw.get("leaderboard") or {}
        if not isinstance(board, dict):
            raise ValueError("BenchmarkConfig 'leaderboard' must be a mapping.")
        unknown_board = set(board) - {"metric", "descending"}
        if unknown_board:
            raise ValueError(f"BenchmarkConfig leaderboard: unknown keys {sorted(unknown_board)}.")
        return cls(
            runs=[_benchmark_run_from_dict(entry, index) for index, entry in enumerate(run_entries)],
            output_dir=_expand_path(str(raw.get("output_dir", "artifacts/benchmarks"))),
            leaderboard_metric=board.get("metric"),
            descending=bool(board.get("descending", True)),
            report_path=_expand_path(raw["report_path"]) if raw.get("report_path") else None,
            run_log_path=_expand_path(raw["run_log_path"]) if raw.get("run_log_path") else None,
        )

    def run(self) -> list[ExperimentResult]:
        results = run_benchmark(self.runs, output_dir=self.output_dir, run_log_path=self.run_log_path)
        if self.leaderboard_metric:
            results = leaderboard(results, self.leaderboard_metric, self.descending)
        if self.report_path:
            from fabric_defect_hub.reporting import save_leaderboard

            save_leaderboard(results, self.report_path)
        return results


def _benchmark_run_from_dict(raw: object, index: int) -> BenchmarkRun:
    if not isinstance(raw, dict):
        raise ValueError(f"BenchmarkConfig runs[{index}] must be a mapping.")
    allowed = {
        "experiment_id", "dataset", "train_dataset", "validation_dataset",
        "model", "runtime", "train", "evaluator", "artifact", "profile",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"BenchmarkConfig runs[{index}]: unknown keys {sorted(unknown)}.")

    experiment_id = _required_string(raw, "experiment_id", index)
    dataset_spec = _required_mapping(raw, "dataset", index)
    model_spec = _required_mapping(raw, "model", index)
    runtime_spec = _required_mapping(raw, "runtime", index)
    dataset = _dataset_from_spec(dataset_spec, f"runs[{index}].dataset")

    backend = _mapping_string(model_spec, "backend", f"runs[{index}].model")
    model_name = _mapping_string(model_spec, "name", f"runs[{index}].model")
    task = _mapping_string(model_spec, "task", f"runs[{index}].model")
    model_kwargs = _mapping_dict(model_spec, "kwargs", f"runs[{index}].model")
    unknown_model = set(model_spec) - {"backend", "name", "task", "kwargs"}
    if unknown_model:
        raise ValueError(f"runs[{index}].model: unknown keys {sorted(unknown_model)}.")

    runtime = _runtime_from_spec(runtime_spec, f"runs[{index}].runtime")
    evaluator = _evaluator_from_spec(raw.get("evaluator"), task, index)
    artifact = _artifact_from_spec(raw.get("artifact"), backend, index)
    profile = _profile_from_spec(raw.get("profile"), runtime, index)

    train_config = raw.get("train")
    if train_config is not None and not isinstance(train_config, dict):
        raise ValueError(f"runs[{index}].train must be a mapping or null.")

    train_dataset = _optional_dataset(raw.get("train_dataset"), f"runs[{index}].train_dataset")
    validation_dataset = _optional_dataset(
        raw.get("validation_dataset"), f"runs[{index}].validation_dataset"
    )
    return BenchmarkRun(
        experiment_id=experiment_id,
        model_backend=backend,
        model_name=model_name,
        model_info=ModelInfo(name=model_name, backend=backend, task=task),
        runtime=runtime,
        dataset=dataset,
        model_kwargs=model_kwargs,
        train_config=dict(train_config) if train_config is not None else None,
        evaluator=evaluator,
        artifact=artifact,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        profiler=profile[0] if profile else None,
        profile_config=profile[1] if profile else None,
        export_target=profile[2] if profile else None,
        export_config=profile[3] if profile else {},
    )


def _dataset_from_spec(spec: dict[str, Any], context: str) -> DatasetAdapter:
    unknown = set(spec) - {"name", "root", "split", "kwargs"}
    if unknown:
        raise ValueError(f"{context}: unknown keys {sorted(unknown)}.")
    name = _mapping_string(spec, "name", context)
    root = _expand_path(_mapping_string(spec, "root", context))
    split = spec.get("split", "test")
    if not isinstance(split, str):
        raise ValueError(f"{context}.split must be a string.")
    return load_dataset(name, root=root, split=split, **_mapping_dict(spec, "kwargs", context))


def _optional_dataset(raw: object, context: str) -> DatasetAdapter | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping.")
    return _dataset_from_spec(raw, context)


def _runtime_from_spec(spec: dict[str, Any], context: str) -> RuntimeInfo:
    unknown = set(spec) - {"device", "engine", "precision", "input_size"}
    if unknown:
        raise ValueError(f"{context}: unknown keys {sorted(unknown)}.")
    size = spec.get("input_size", [640, 640])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"{context}.input_size must contain [height, width].")
    return RuntimeInfo(
        device=_mapping_string(spec, "device", context),
        engine=str(spec.get("engine", "python")),
        precision=str(spec.get("precision", "fp32")),
        input_size=(int(size[0]), int(size[1])),
    )


def _evaluator_from_spec(raw: object, default_task: str, index: int) -> Evaluator | None:
    if raw is False or raw is None:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"runs[{index}].evaluator must be a mapping, true, or false.")
    unknown = set(raw) - {"type", "kwargs"}
    if unknown:
        raise ValueError(f"runs[{index}].evaluator: unknown keys {sorted(unknown)}.")
    evaluator_type = str(raw.get("type", default_task))
    kwargs = _mapping_dict(raw, "kwargs", f"runs[{index}].evaluator")
    import fabric_defect_hub.evaluation  # noqa: F401 -- triggers @register_evaluator
    from fabric_defect_hub.core.registry import get_evaluator_cls, list_evaluators

    try:
        return get_evaluator_cls(evaluator_type)(**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"runs[{index}].evaluator.type must be one of {list_evaluators()}."
        ) from exc


def _artifact_from_spec(raw: object, backend: str, index: int) -> Artifact | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"runs[{index}].artifact must be a mapping.")
    unknown = set(raw) - {"path", "metadata"}
    if unknown:
        raise ValueError(f"runs[{index}].artifact: unknown keys {sorted(unknown)}.")
    path = _expand_path(_mapping_string(raw, "path", f"runs[{index}].artifact"))
    return Artifact(
        path=path,
        backend=backend,
        metadata=_mapping_dict(raw, "metadata", f"runs[{index}].artifact"),
    )


def _profile_from_spec(raw: object, runtime: RuntimeInfo, index: int):
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"runs[{index}].profile must be a mapping or false.")
    unknown = set(raw) - {"engine", "target", "config", "export_config"}
    if unknown:
        raise ValueError(f"runs[{index}].profile: unknown keys {sorted(unknown)}.")
    engine = str(raw.get("engine", runtime.engine)).lower()
    import fabric_defect_hub.profiling  # noqa: F401 -- triggers @register_profiler
    from fabric_defect_hub.core.registry import get_profiler_cls, list_profilers

    try:
        profiler = get_profiler_cls(engine)()
    except KeyError as exc:
        raise ValueError(f"runs[{index}].profile.engine must be one of {list_profilers()}.") from exc
    config_raw = _mapping_dict(raw, "config", f"runs[{index}].profile")
    size = config_raw.pop("input_size", runtime.input_size)
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"runs[{index}].profile.config.input_size must contain two integers.")
    config = ProfileConfig(
        device=str(config_raw.pop("device", runtime.device)),
        engine=engine,
        precision=str(config_raw.pop("precision", runtime.precision)),
        input_size=tuple(size),
        **config_raw,
    )
    target = raw.get("target") or ("torchscript" if engine == "pytorch" else "onnx")
    return profiler, config, str(target), _mapping_dict(
        raw, "export_config", f"runs[{index}].profile"
    )


def _required_mapping(raw: dict[str, Any], key: str, index: int) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"BenchmarkConfig runs[{index}].{key} must be a mapping.")
    return value


def _required_string(raw: dict[str, Any], key: str, index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"BenchmarkConfig runs[{index}].{key} must be a non-empty string.")
    return value


def _mapping_string(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string.")
    return value


def _mapping_dict(raw: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a mapping.")
    return dict(value)


def _expand_path(path: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(path))
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in path {path!r}")
    return expanded
