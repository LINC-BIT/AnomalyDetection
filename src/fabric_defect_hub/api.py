"""Public top-level API facade: load_config, load_dataset, load_model, train, predict, evaluate."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from fabric_defect_hub.models.base import Artifact

# Preset module + its published-variant lister, per backend. Every backend's
# preset module exposes `list_supported_variants()` under that one name (see
# each `models/<backend>/presets.py`), so this stays a table lookup rather
# than a per-backend branch.
_PRESET_MODULES: dict[str, str] = {
    "ultralytics": "fabric_defect_hub.models.ultralytics.presets",
    "torchvision": "fabric_defect_hub.models.torchvision.presets",
    "anomalib": "fabric_defect_hub.models.anomalib.presets",
    "dinomaly": "fabric_defect_hub.models.dinomaly.presets",
    "moeclip": "fabric_defect_hub.models.moeclip.presets",
    "mambaad": "fabric_defect_hub.models.mambaad.presets",
}


# --------------------------------------------------------------------------- #
# Layer 1 -- configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunConfig:
    """A resolved experiment configuration: which config file, which model
    inside it, and the override layers to apply on top.

    Deliberately a *description* of a run rather than a live object -- it
    holds exactly the arguments `training.run_train` takes, so `train(cfg)`
    is a call, not a translation. `raw` is the parsed config as it stands
    before the override layers are applied, for inspection.
    """

    model: str
    backend: str
    config_path: Path
    variant: str | None = None
    overrides: Any = None  # training.DatasetOverrides
    set_overrides: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def with_set(self, **dotted: Any) -> "RunConfig":
        """Return a copy with extra dotted-path overrides layered on, e.g.
        `cfg.with_set(**{"train.engine_kwargs.max_epochs": 5})`.
        """

        from dataclasses import replace

        return replace(self, set_overrides={**self.set_overrides, **dotted})


def load_config(
    model: str,
    *,
    dataset: str | None = None,
    dataset_root: str | None = None,
    variant: str | None = None,
    epochs: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    config_dir: str | Path | None = None,
    set: dict[str, Any] | None = None,  # noqa: A002 - mirrors `fdh train --set`
) -> RunConfig:
    """Resolve `model` to a config and layer the given overrides onto it.

    `model` accepts everything `fdh train` does: a config path, a config
    filename stem, or a bare model keyword (`"stfpm"`, `"yolov8n"`,
    `"patchcore"`) matched against every config's declared model.

    `dataset`/`dataset_root`/`num_samples`/`seed` become a
    `training.DatasetOverrides` (the same layer `--dataset`/`--num-samples`
    use). `set` is the arbitrary dotted-path escape hatch, highest priority,
    identical to `fdh train --set a.b.c=v`.

    `epochs` is resolved through `training.RUN_LENGTH_KEYS` into whichever
    config key that backend counts its run length in. It raises for the two
    backends that count optimizer iterations rather than epochs (Dinomaly,
    MambaAD) instead of quietly writing an epoch count they would ignore --
    those take `set={"train.total_iters": N}`.
    """

    from fabric_defect_hub.training import (
        DEFAULT_MODEL_CONFIG_DIR,
        RUN_LENGTH_KEYS,
        DatasetOverrides,
        infer_backend,
        load_raw_config,
        resolve_model_config_and_variant,
    )

    config_path, implied_variant = resolve_model_config_and_variant(
        model, config_dir=config_dir or DEFAULT_MODEL_CONFIG_DIR
    )
    raw = load_raw_config(config_path)
    backend = infer_backend(raw)

    set_overrides = dict(set or {})
    if epochs is not None:
        key, unit = RUN_LENGTH_KEYS[backend]
        if unit != "epochs":
            raise ValueError(
                f"the {backend} backend counts its run length in {unit}, not epochs; "
                f"pass set={{{key!r}: N}} instead of epochs={epochs}"
            )
        set_overrides.setdefault(key, epochs)

    return RunConfig(
        model=model,
        backend=backend,
        config_path=config_path,
        variant=variant if variant is not None else implied_variant,
        overrides=DatasetOverrides(
            dataset=dataset,
            dataset_root=dataset_root,
            num_samples=num_samples,
            seed=seed,
        ),
        set_overrides=set_overrides,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Layer 2 -- weights
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PretrainedWeights:
    """A published checkpoint from this project's own model catalog.

    Passes straight into `predict`/`evaluate`'s `weights=` (it implements
    `__fspath__`), while still carrying the provenance the frontend shows --
    which backend and variant produced it, and what `source` string the
    leaderboard labels it with.
    """

    key: str
    path: str
    backend: str
    variant: str
    task: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path

    def as_artifact(self) -> Artifact:
        """The `models.base.Artifact` handle an adapter's `predict`/`export`
        takes directly, for callers going through the adapter layer rather
        than through `api.predict`.
        """

        return Artifact(path=self.path, backend=self.backend, metadata=dict(self.metadata))


def from_pretrained(key: str) -> PretrainedWeights:
    """Look up a published checkpoint by its catalog key (`"PatchCore"`,
    `"yolov8n"`, `"STFPM"`, ... -- see `fdh.list_pretrained()`).

    Raises `FileNotFoundError` when the model is in the catalog but has not
    been trained and published on this machine, which is the common case on
    a fresh checkout: the catalog names what this project *publishes*, and
    the weights themselves are produced by `fdh train`, not shipped in git.
    """

    from fabric_defect_hub.catalog import (
        describe_published, find_canonical_model_by_key, metadata_for,
        published_is_usable, published_path, published_status,
    )

    model = find_canonical_model_by_key(key)
    path = published_path(model)
    if not published_is_usable(path):
        if published_status(path) == "broken_link":
            # A different problem from "never trained", and telling the two
            # apart is the difference between copying one directory and
            # spending GPU-hours retraining something already on disk.
            raise FileNotFoundError(f"published weights for {key!r} are unreachable: {describe_published(path)}")
        raise FileNotFoundError(
            f"no published weights for {key!r} at {path}. Train and publish it first "
            f"(`fdh train {model.config}` with variant {model.variant!r}), or pick another "
            f"model from fdh.list_pretrained()."
        )
    return PretrainedWeights(
        key=model.key,
        path=str(path),
        backend=model.backend,
        variant=model.variant,
        task=model.task,
        source=model.source,
        metadata=metadata_for(model),
    )


def list_pretrained(*, available_only: bool = False) -> list[str]:
    """Catalog keys `from_pretrained` accepts. With `available_only`, only
    those whose weights are actually present on this machine.
    """

    from fabric_defect_hub.catalog import CANONICAL_MODELS, published_is_usable, published_path

    return [
        model.key
        for model in CANONICAL_MODELS
        if not available_only or published_is_usable(published_path(model))
    ]


# --------------------------------------------------------------------------- #
# Layer 3 -- discovery
# --------------------------------------------------------------------------- #
def list_models(backend: str | None = None) -> dict[str, list[str]] | list[str]:
    """Every model this project can run: `{backend: [variant, ...]}`, or
    just one backend's list when `backend` is given.

    Reads each backend's own preset module, so it reflects what is actually
    registered rather than a hand-maintained list -- adding an alias to
    `models/anomalib/presets.py::MODEL_ALIASES` shows up here immediately.
    """

    import importlib

    backends = [backend] if backend is not None else list(_PRESET_MODULES)
    listed = {
        name: importlib.import_module(_PRESET_MODULES[name]).list_supported_variants()
        for name in backends
    }
    return listed[backend] if backend is not None else listed


def list_datasets() -> list[str]:
    """Every registered dataset name `load_dataset`/`predict` accept."""

    import importlib

    importlib.import_module("fabric_defect_hub.datasets")  # triggers @register_dataset
    from fabric_defect_hub.core.registry import list_datasets as _list

    return _list()


# --------------------------------------------------------------------------- #
# Verbs
# --------------------------------------------------------------------------- #
def train(
    config: RunConfig | str,
    *,
    publish: bool = True,
    config_dir: str | Path | None = None,
    **kwargs: Any,
):
    """Run training. `config` is a `RunConfig` from `load_config`, or a bare
    model keyword for a run that needs no overrides (`fdh.train("stfpm")`).

    Returns `training.TrainRunResult`. `kwargs` pass through to
    `training.run_train` (e.g. `profile=`).

    `config_dir` is named explicitly rather than left to `**kwargs` because it
    has to reach *both* calls below: resolving a bare keyword reads the config
    to infer its backend, so letting it through to `run_train` alone would
    infer the backend from one directory's config and then train from
    another's.
    """

    from fabric_defect_hub.training import run_train

    if config_dir is not None:
        kwargs["config_dir"] = config_dir
    cfg = (
        config
        if isinstance(config, RunConfig)
        else load_config(config, config_dir=config_dir)
    )
    return run_train(
        cfg.model,
        backend=cfg.backend,
        variant=cfg.variant,
        overrides=cfg.overrides,
        set_overrides=cfg.set_overrides,
        publish=publish,
        **kwargs,
    )


def predict(
    model: str,
    weights: str | os.PathLike[str] | PretrainedWeights,
    source: str | list[str] | None = None,
    *,
    dataset: str | None = None,
    dataset_root: str | None = None,
    split: str = "test",
    num_samples: int | None = None,
    pattern: str | int | None = None,
    category: str | None = None,
    seed: int = 0,
    output_dir: str | None = None,
    **kwargs: Any,
):
    """Run inference. Give either `source` (one image path or a list of
    them) or `dataset` (a registered dataset name, optionally sliced by
    `num_samples`/`pattern`/`category`).

    Returns `predict.PredictRunResult`. `kwargs` pass through to
    `inference.runner.run_predict` (`enable_tiling=`, `enable_tta=`, `tile_size=`,
    `tile_overlap=`, `backend=`, `variant=`, `config_dir=`).
    """

    from fabric_defect_hub.inference.runner import run_predict

    return run_predict(
        model,
        weights=os.fspath(weights),
        source=_as_predict_input(
            source, dataset, dataset_root, split, num_samples, pattern, category, seed
        ),
        output_dir=output_dir,
        **kwargs,
    )


def evaluate(
    model: str,
    weights: str | os.PathLike[str] | PretrainedWeights,
    *,
    dataset: str,
    dataset_root: str | None = None,
    split: str = "test",
    num_samples: int | None = None,
    pattern: str | int | None = None,
    category: str | None = None,
    seed: int = 0,
    task: str | None = None,
    output_dir: str | None = None,
    **kwargs: Any,
):
    """Score a checkpoint against a dataset's ground truth -- the scriptable
    equivalent of the web Benchmark tab's leaderboard row.

    Returns `predict.EvaluateRunResult`. A dataset is required (raw image
    paths carry no ground truth to score against).

    Pass `output_dir` to get pixel-level metrics from an anomaly model: the
    adapters only fill `Prediction.anomaly_map` when given somewhere to write
    it, so without it the result is image-level only. See `run_evaluate`.
    """

    from fabric_defect_hub.inference.runner import run_evaluate

    return run_evaluate(
        model,
        weights=os.fspath(weights),
        source=_as_predict_input(
            None, dataset, dataset_root, split, num_samples, pattern, category, seed
        ),
        task=task,
        output_dir=output_dir,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Layer 4 -- measurement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BenchmarkResult:
    """A finished measurement run: the raw rows, plus the tables they group into.

    Returned by `benchmark()` and by `load_results()`, so a script that just
    measured and a page reading yesterday's JSONL hold the same object and
    render the same tables. That equivalence is the point — the Gradio app is
    a *reader* of this, never a second implementation of it.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    header: dict[str, Any] = field(default_factory=dict)
    output_path: str | None = None

    def tables(self, *, include_unimplemented: bool = True) -> list[Any]:
        """Every declared table, in taxonomy order (`MetricTable` objects)."""

        from fabric_defect_hub.metrics_taxonomy import build_tables

        return build_tables(self.rows, include_unimplemented=include_unimplemented)

    def by_category(self) -> dict[str, list[Any]]:
        """`{"technical": [...], "overhead": [...]}` — the two-part split."""

        from fabric_defect_hub.metrics_taxonomy import tables_by_category

        return tables_by_category(self.tables())

    def table(self, name: str) -> Any:
        """One table by name, e.g. `"pixel_level"` or `"compute"`."""

        for candidate in self.tables():
            if candidate.name == name:
                return candidate
        from fabric_defect_hub.metrics_taxonomy import TABLES

        raise KeyError(f"unknown table {name!r}; expected one of {list(TABLES)}")

    def unrecognised_metrics(self) -> list[str]:
        """Measured keys with no home in the taxonomy — a metric that is
        computed and then shown nowhere is a silent hole, so it is reported."""

        from fabric_defect_hub.metrics_taxonomy import unrecognised

        return unrecognised(
            key for row in self.rows for key in (row.get("metrics") or {})
        )

    def summary(self) -> dict[str, Any]:
        from fabric_defect_hub.metric_sweep import summarize

        return summarize(self.rows)

    def to_json(self, path: str | Path) -> Path:
        """Write the grouped tables as one JSON document.

        Distinct from the sweep's JSONL, which is the append-only measurement
        log: this is the *presentation* shape (two categories, named tables,
        formatted cells) that a report or a page consumes.
        """

        import json

        payload = {
            "header": self.header,
            "categories": {
                category: [
                    {
                        "name": table.name,
                        "status": table.status,
                        "note": table.note,
                        "header": table.header(),
                        "rows": table.as_matrix(),
                    }
                    for table in tables
                ]
                for category, tables in self.by_category().items()
            },
            "unrecognised_metrics": self.unrecognised_metrics(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        return destination


def measure(
    dataset: str,
    *,
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    groups: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    dataset_root: str | None = None,
    split: str = "test",
    num_samples: int | None = 100,
    pattern: str | int | None = None,
    category: str | None = None,
    seed: int = 0,
    device: str = "cpu",
    anomaly_map_dir: str | Path | None = None,
    cross_domain_patterns: Sequence[str] = (),
    cross_domain_k: int = 3,
    cross_domain_mode: str = "worst",
    frame_budget_ms: float = 33.0,
    max_streams_to_try: int = 8,
    measured_runs: int = 50,
    warmup_runs: int = 5,
    on_row: Any = None,
) -> BenchmarkResult:
    """Measure every implemented metric over every model that has weights.

    The fourth verb beside `train` / `predict` / `evaluate`, and named
    `measure` rather than `benchmark` for a concrete reason: a top-level
    module `fabric_defect_hub.benchmark` already exists (the config-driven
    runner behind `fdh benchmark <config.yaml>`), and Python binds a
    submodule as an attribute of its package on import — so a facade function
    called `benchmark` would be silently replaced by the module the first
    time anything imported it. `test_api_facade` catches exactly that.

    The library entry point the project is meant to be driven by:

        import fabric_defect_hub as fdh

        result = fdh.measure("zju-leaper", num_samples=200)
        result.to_json("report.json")
        for table in result.by_category()["technical"]:
            print(table.name, table.header(), table.as_matrix())

    Nothing raises per model: a backend that is not installed, weights that
    vanished, or an export that fails become rows with a status and a reason,
    so an unattended run after a training batch finishes with results for
    everything that could be measured.

    `on_row` is called with each `SweepRow` as it completes — how a progress
    bar or a live table updates without this layer knowing about any UI.
    """

    from fabric_defect_hub.metric_sweep import METRIC_GROUPS, SweepRequest, read_sweep, run_sweep

    root = Path(project_root) if project_root else Path.cwd()
    destination = Path(output_path) if output_path else root / "artifacts" / "runtime" / "benchmark.jsonl"
    request = SweepRequest(
        project_root=root,
        dataset=dataset,
        output_path=destination,
        groups=tuple(groups) if groups else METRIC_GROUPS,
        models=tuple(models) if models else None,
        dataset_root=dataset_root,
        split=split,
        num_samples=num_samples,
        pattern=str(pattern) if pattern is not None else None,
        category=category,
        seed=seed,
        device=device,
        anomaly_map_dir=Path(anomaly_map_dir) if anomaly_map_dir else None,
        cross_domain_patterns=tuple(str(p) for p in cross_domain_patterns),
        cross_domain_k=cross_domain_k,
        cross_domain_mode=cross_domain_mode,
        frame_budget_ms=frame_budget_ms,
        max_streams_to_try=max_streams_to_try,
        measured_runs=measured_runs,
        warmup_runs=warmup_runs,
    )
    run_sweep(request, on_row=on_row)
    header, rows = read_sweep(destination)
    return BenchmarkResult(rows=rows, header=header, output_path=str(destination))


def load_results(path: str | Path) -> BenchmarkResult:
    """Re-open a sweep JSONL as a `BenchmarkResult`.

    What lets a page render tables without re-measuring, and what makes
    "yesterday's run" and "the run I just did" the same object.
    """

    from fabric_defect_hub.metric_sweep import read_sweep

    header, rows = read_sweep(path)
    return BenchmarkResult(rows=rows, header=header, output_path=str(path))


def list_metric_tables() -> dict[str, list[str]]:
    """The declared table names per category, without measuring anything —
    so a front end can lay out its sections before any run exists."""

    from fabric_defect_hub.metrics_taxonomy import OVERHEAD_TABLES, TECHNICAL_TABLES

    return {"technical": list(TECHNICAL_TABLES), "overhead": list(OVERHEAD_TABLES)}


def _as_predict_input(
    source: str | list[str] | None,
    dataset: str | None,
    dataset_root: str | None,
    split: str,
    num_samples: int | None,
    pattern: str | int | None,
    category: str | None,
    seed: int,
):
    """Pack the flat inference arguments into the `PredictInput` that
    `run_predict`/`run_evaluate` already take. Shared by both verbs so the
    two keep identical source-selection semantics.
    """

    from fabric_defect_hub.inference.runner import PredictInput

    images = [source] if isinstance(source, str) else list(source or [])
    return PredictInput(
        images=images,
        dataset=dataset,
        dataset_root=dataset_root,
        split=split,
        num_samples=num_samples,
        pattern=pattern,
        category=category,
        seed=seed,
    )
