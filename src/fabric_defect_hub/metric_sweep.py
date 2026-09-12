"""One entry point that measures every implemented metric over every model
that actually has weights, and writes the results as JSONL.

The problem this solves is not that the individual metrics were missing —
accuracy lives behind `fdh evaluate`, runtime behind `fdh benchmark`, the
resolution slope behind the Gradio tab, and `max_concurrent_streams` behind
nothing at all. It is that measuring "all of them, for everything trained so
far" meant driving four different front ends by hand and reconciling their
output formats afterwards.

Design rules, all of them consequences of "this runs unattended after a long
training batch":

* **Nothing raises.** A model whose backend is not installed, whose weights
  vanished, or whose export blows up produces a row with `status` of
  `skipped`/`failed` and a reason. One broken model must not cost the other
  seventeen their measurements.
* **Rows are written as they finish**, not at the end, so a sweep killed by
  a disconnected SSH session still leaves everything measured up to that
  point.
* **Discovery is from the weight manifest**, so "models that have been
  trained" is read from provenance rather than guessed from a hard-coded
  list that would drift.
* **One row per (model, metric group).** A flat table would force every
  group's columns onto every model; separate rows let a reader load the
  JSONL and pivot on whichever group they care about.

The output is deliberately JSONL rather than a report: this is the input to
whatever table, plot, or LaTeX export comes next, and `reporting/` already
owns turning rows into a document.
"""

from __future__ import annotations

import json
import hashlib
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from fabric_defect_hub.core.provenance import collect_provenance
from fabric_defect_hub.weight_registry import read_weight_manifest

METRIC_GROUPS: tuple[str, ...] = (
    "accuracy", "cross_domain", "runtime", "scaling", "concurrency", "communication",
)

# Groups that need an exported artifact rather than a raw checkpoint. Kept
# here so `plan_sweep` can report "would export once" instead of each group
# re-exporting the same model.
_EXPORT_GROUPS = frozenset({"runtime", "scaling", "concurrency"})

# Keys that describe the run rather than measure the model. A row carrying
# only these has measured nothing, however successfully it executed.
_BOOKKEEPING_KEYS = frozenset({
    "sample_count", "frame_budget_ms", "resolution_sweep_points",
    "concurrency_probe_points", "metric", "mode", "k", "source_value",
})


@dataclass(frozen=True)
class TrainedModel:
    """A checkpoint the sweep can actually load, found via provenance."""

    backend: str
    variant: str
    weights: Path
    model_key: str | None = None
    config: str | None = None
    discovered_via: str = "manifest"

    @property
    def label(self) -> str:
        return self.model_key or f"{self.backend}/{self.variant}"

    @property
    def config_hint(self) -> str:
        """What to hand `run_evaluate` as its `model` argument.

        The manifest stores the config path as it was on the machine that
        trained the model, which for this project is usually the cloud box
        (`/root/autodl-tmp/FabricDefectHub/configs/models/...`). That path
        does not exist locally, so only the *filename* is portable —
        `resolve_model_config` looks it up under the local `--config-dir`.
        Falling back to the variant lets keyword resolution take over when
        even the filename is gone.
        """

        if self.config:
            stem = Path(self.config).stem
            if stem:
                return stem
        return self.variant


@dataclass
class SweepRequest:
    """Everything one sweep needs. Defaults are the unattended-run defaults."""

    project_root: Path
    dataset: str
    output_path: Path
    groups: tuple[str, ...] = METRIC_GROUPS
    models: tuple[str, ...] | None = None  # None = every discovered model
    dataset_root: str | None = None
    split: str = "test"
    num_samples: int | None = 100
    pattern: str | None = None
    category: str | None = None
    seed: int = 0
    device: str = "cpu"
    anomaly_map_dir: Path | None = None
    cross_domain_patterns: tuple[str, ...] = ()
    cross_domain_k: int = 3
    cross_domain_mode: str = "worst"
    frame_budget_ms: float = 33.0
    max_streams_to_try: int = 8
    measured_runs: int = 50
    warmup_runs: int = 5
    power_mode: str = "disabled"
    overwrite: bool = False

    def __post_init__(self) -> None:
        unknown = sorted(set(self.groups) - set(METRIC_GROUPS))
        if unknown:
            raise ValueError(f"unknown metric group(s) {unknown}; expected {list(METRIC_GROUPS)}")
        self.project_root = Path(self.project_root)
        self.output_path = Path(self.output_path)
        if self.num_samples is not None and self.num_samples < 1:
            raise ValueError("num_samples must be positive or None")
        if self.cross_domain_k < 1:
            raise ValueError("cross_domain_k must be at least 1")
        if self.frame_budget_ms <= 0:
            raise ValueError("frame_budget_ms must be positive")
        if self.max_streams_to_try < 1:
            raise ValueError("max_streams_to_try must be at least 1")
        if self.measured_runs < 1:
            raise ValueError("measured_runs must be at least 1")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if self.power_mode not in {"required", "disabled"}:
            raise ValueError("formal sweeps require power_mode='required' or 'disabled'")


@dataclass(frozen=True)
class SweepRow:
    """One measurement attempt. `status` is always one of ok/skipped/failed."""

    model: str
    backend: str
    variant: str
    group: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    duration_s: float = 0.0
    weights: str | None = None
    instrumentation: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return payload


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_trained_models(project_root: str | Path) -> list[TrainedModel]:
    """Every (backend, variant) with a checkpoint still on disk.

    Reads `weight_manifest.jsonl` newest-first and keeps the first surviving
    checkpoint per (backend, variant): re-running a model should sweep its
    latest weights, and a manifest row whose file has since been pruned must
    not shadow an older row whose file is still there.

    Published symlinks are consulted as a fallback so a tree that was
    populated before the manifest existed still sweeps.
    """

    root = Path(project_root)
    found: dict[tuple[str, str], TrainedModel] = {}

    for record in reversed(read_weight_manifest(root)):
        if record.get("kind") == "quantized":
            continue
        backend, variant = record.get("backend"), record.get("variant")
        # Case-folded key: the manifest records whatever spelling the config
        # used (`ganomaly`) while `catalog` uses the canonical one
        # (`GANomaly`), and a case-sensitive key sweeps the same checkpoint
        # twice under two names.
        if not backend or not variant or _key(backend, variant) in found:
            continue
        artifact = record.get("artifact") or {}
        for key in ("registered_path", "published_path", "primary_path"):
            candidate = artifact.get(key)
            if candidate and Path(candidate).is_file():
                found[_key(backend, variant)] = TrainedModel(
                    backend=backend, variant=variant, weights=Path(candidate),
                    model_key=record.get("model_key"),
                    config=record.get("config_source_path"),
                )
                break

    published = root / "artifacts" / "models" / "published"
    if published.is_dir():
        from fabric_defect_hub.catalog import find_canonical_model_by_key, published_is_usable

        for entry in sorted(published.iterdir()):
            # `published_is_usable` accepts a real file or a resolving
            # symlink alike — a tree copied down from the training box holds
            # real files, a locally published one holds links, and the sweep
            # must not care which. A dangling link is neither, and is
            # surfaced by `broken_published_links()` rather than silently
            # dropping the model.
            if entry.is_dir() or not published_is_usable(entry):
                continue
            try:
                model = find_canonical_model_by_key(entry.stem)
            except KeyError:
                continue
            key = _key(model.backend, model.variant)
            if key not in found:
                found[key] = TrainedModel(
                    backend=model.backend, variant=model.variant,
                    weights=entry.resolve(), model_key=model.key,
                    config=model.config, discovered_via="published",
                )

    return sorted(found.values(), key=lambda m: (m.backend, m.variant.lower()))


def _key(backend: str, variant: str) -> tuple[str, str]:
    return (backend.lower(), variant.lower())


def broken_published_links(project_root: str | Path) -> list[tuple[Path, str]]:
    """Published slots that are symlinks to something no longer there.

    Reported rather than repaired: the fix depends on why the target is
    missing. Copying `published/` down from the training box without
    `artifacts/models/` leaves every link dangling and wants the rest of the
    tree copied; a pruned checkpoint wants a re-publish. Both look identical
    from here, and both are invisible to `discover_trained_models`, which
    correctly refuses to sweep weights it cannot open.
    """

    from fabric_defect_hub.catalog import describe_published, published_status

    published = Path(project_root) / "artifacts" / "models" / "published"
    if not published.is_dir():
        return []
    return [
        (entry, describe_published(entry))
        for entry in sorted(published.iterdir())
        if published_status(entry) == "broken_link"
    ]


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #
def run_sweep(
    request: SweepRequest,
    *,
    on_row: Callable[[SweepRow], None] | None = None,
) -> list[SweepRow]:
    """Measure every requested group for every selected model.

    Returns the rows and writes them to `request.output_path` as they are
    produced. `on_row` is called after each row is written, which is how a
    front end shows progress without this module knowing about any UI.
    """

    models = discover_trained_models(request.project_root)
    if request.models:
        wanted = {name.lower() for name in request.models}
        models = [
            model for model in models
            if model.label.lower() in wanted or model.variant.lower() in wanted
        ]

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[SweepRow] = []
    mode = "w" if request.overwrite else "x"
    with request.output_path.open(mode, encoding="utf-8") as handle:
        header = {
            "kind": "sweep_header",
            "dataset": request.dataset,
            "split": request.split,
            "device": request.device,
            "groups": list(request.groups),
            "model_count": len(models),
            "num_samples": request.num_samples,
            "pattern": request.pattern,
            "category": request.category,
            "seed": request.seed,
            "cross_domain_patterns": list(request.cross_domain_patterns),
            "cross_domain_k": request.cross_domain_k,
            "cross_domain_mode": request.cross_domain_mode,
            "frame_budget_ms": request.frame_budget_ms,
            "max_streams_to_try": request.max_streams_to_try,
            "measured_runs": request.measured_runs,
            "warmup_runs": request.warmup_runs,
            "power_mode": request.power_mode,
            "models": [_model_manifest_entry(model, request.project_root) for model in models],
            "provenance": collect_provenance(request.project_root),
        }
        handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()

        for model in models:
            exported: Any = None
            export_error: str | None = None
            for group in request.groups:
                started = time.perf_counter()
                if group in _EXPORT_GROUPS and exported is None and export_error is None:
                    exported, export_error = _try_export(model, request)
                row = _measure(model, group, request, exported, export_error)
                row = SweepRow(**{**asdict(row), "duration_s": time.perf_counter() - started})
                rows.append(row)
                handle.write(json.dumps(row.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                if on_row:
                    on_row(row)
    return rows


def _measure(
    model: TrainedModel,
    group: str,
    request: SweepRequest,
    exported: Any,
    export_error: str | None,
) -> SweepRow:
    """Run one group for one model, converting every failure into a row.

    The broad `except Exception` is deliberate and is the module's whole
    contract: a sweep exists to survive whatever eighteen heterogeneous
    research backends throw, and a traceback that kills the process loses
    every measurement not yet written.
    """

    base = {
        "model": model.label, "backend": model.backend, "variant": model.variant,
        "group": group, "weights": str(model.weights),
    }
    if group in _EXPORT_GROUPS and exported is None:
        return SweepRow(**base, status="skipped", reason=export_error or "no exported artifact")
    if group == "cross_domain" and not request.cross_domain_patterns:
        return SweepRow(**base, status="skipped", reason="no --cross-domain-patterns given")

    try:
        if group == "accuracy":
            metrics = _accuracy(model, request)
        elif group == "cross_domain":
            metrics = _cross_domain(model, request)
        elif group == "runtime":
            metrics = _runtime(exported, request)
        elif group == "scaling":
            metrics = _scaling(exported, request)
        elif group == "concurrency":
            metrics = _concurrency(exported, request)
        elif group == "communication":
            metrics = _communication(model, exported)
        else:  # unreachable: SweepRequest validates the vocabulary
            return SweepRow(**base, status="skipped", reason=f"unknown group {group}")
    except NotImplementedError as exc:
        return SweepRow(**base, status="skipped", reason=f"backend does not support it: {exc}")
    except ImportError as exc:
        return SweepRow(**base, status="skipped", reason=f"dependency missing: {exc}")
    except Exception as exc:  # noqa: BLE001 -- see docstring
        return SweepRow(
            **base, status="failed",
            reason=f"{type(exc).__name__}: {exc}".strip()[:500],
            metrics={"traceback_tail": traceback.format_exc(limit=3)[-500:]},
        )

    instrumentation = metrics.pop("__instrumentation__", {})
    # `sample_count` and friends say the run happened, not that anything was
    # measured. Counting them as metrics is how an evaluator/model task
    # mismatch first showed up as a green `ok` row carrying nothing at all.
    if not set(metrics) - _BOOKKEEPING_KEYS:
        return SweepRow(
            **base, status="skipped", metrics=metrics, instrumentation=instrumentation,
            reason=(
                "no metrics produced — nothing this group measures applies to this model "
                "(e.g. an evaluator with no matching ground truth, or a fixed-shape export "
                "that cannot be swept across resolutions)"
            ),
        )
    return SweepRow(**base, status="ok", metrics=metrics, instrumentation=instrumentation)


# --------------------------------------------------------------------------- #
# Per-group measurement
# --------------------------------------------------------------------------- #
def _predict_source(request: SweepRequest, pattern: str | None = None, task: str | None = None):
    from fabric_defect_hub.inference.runner import PredictInput

    return PredictInput(
        dataset=request.dataset,
        dataset_root=request.dataset_root,
        split=request.split,
        num_samples=request.num_samples,
        pattern=_coerce_pattern(pattern if pattern is not None else request.pattern),
        category=request.category,
        seed=request.seed,
        task=task,
    )


def _coerce_pattern(pattern: str | int | None) -> str | int | None:
    """`"5"` -> `5`.

    Patterns arrive from a comma-separated command-line list, so they are
    strings; `ZJULeaperDataset` accepts `patternN` or an *int* N but rejects
    the bare string `"5"`. Converting here rather than in the CLI keeps the
    Gradio front end and any automation script from having to know the same
    quirk.
    """

    if isinstance(pattern, str) and pattern.strip().isdigit():
        return int(pattern.strip())
    return pattern


def _accuracy(model: TrainedModel, request: SweepRequest) -> dict[str, Any]:
    from fabric_defect_hub.inference.runner import run_evaluate

    # Accuracy sweeps persist anomaly maps by default. This is required for
    # pixel-level metrics; detection/segmentation adapters ignore the output
    # directory unless they declare an anomaly map capability. A caller can
    # still provide an explicit directory to control where maps are stored.
    map_root = request.anomaly_map_dir or (
        request.project_root / "artifacts" / "runtime" / "anomaly_maps"
    )
    output_dir = str(Path(map_root) / model.backend / model.variant)

    task = _evaluation_task(model)
    run = run_evaluate(
        model.config_hint,
        weights=str(model.weights),
        source=_predict_source(request, task=task),
        backend=model.backend,
        variant=model.variant,
        task=task,
        output_dir=output_dir,
    )
    return {**run.metrics, "sample_count": run.sample_count}


def _evaluation_task(model: TrainedModel) -> str | None:
    """The evaluator to score this model with, taken from the *model*.

    Without this the sweep inherits whatever task the dataset samples declare,
    which on ZJU-Leaper is `anomaly` — so a YOLO run was scored by
    `AnomalyEvaluator`, which has no `anomaly_score` to read from a detection
    prediction and silently returned nothing. Asking `ModelCapabilities`
    keeps the choice with the layer that knows what the model produces;
    `ground_truth_task` folds instance segmentation into the segmentation
    bucket the way `evaluator_for_task` expects.

    Returns `None` (meaning "let the samples decide") if the backend cannot
    be introspected here — a sweep must not fail over a task hint.
    """

    from fabric_defect_hub.evaluation import ground_truth_task
    from fabric_defect_hub.inference.runner import build_adapter

    tasks = build_adapter(model.backend, model.variant).capabilities().tasks
    return ground_truth_task(tasks[0]) if tasks else None


def _cross_domain(model: TrainedModel, request: SweepRequest) -> dict[str, Any]:
    """The held-out-pattern protocol, delegated to `pattern_sweep_degradation`.

    That function already owns the whole reduction — per-pattern degradation,
    top-k mean with `k` clamped to what was actually scored, patterns listed
    as skipped rather than counted as a 0% drop, and a bootstrap CI that
    resamples *patterns* (the right population for "would this hold on a
    different set of fabrics"). Re-deriving any of it here would be a second
    definition of the project's headline robustness number.
    """

    from fabric_defect_hub.evaluation.cross_domain import pattern_sweep_degradation
    from fabric_defect_hub.inference.runner import run_evaluate

    task = _evaluation_task(model)
    source_run = run_evaluate(
        model.config_hint, weights=str(model.weights),
        source=_predict_source(request, task=task),
        backend=model.backend, variant=model.variant, task=task,
    )
    metric_key = _headline_metric(source_run.metrics)
    if metric_key is None:
        return {}

    def score(pattern: str) -> float | None:
        """One held-out pattern, or `None` when it cannot be scored here."""

        try:
            run = run_evaluate(
                model.config_hint, weights=str(model.weights),
                source=_predict_source(request, pattern=pattern, task=task),
                backend=model.backend, variant=model.variant, task=task,
            )
        except (FileNotFoundError, KeyError) as exc:
            # A missing directory/pattern is a data-availability skip. Other
            # exceptions must escape so _measure records a real failure.
            return None
        except ValueError as exc:
            message = str(exc).lower()
            if any(token in message for token in ("no samples", "not found", "unknown pattern", "does not exist")):
                return None
            raise
        value = run.metrics.get(metric_key)
        return float(value) if isinstance(value, (int, float)) else None

    result = pattern_sweep_degradation(
        float(source_run.metrics[metric_key]),
        list(request.cross_domain_patterns),
        score,
        k=request.cross_domain_k,
        mode=request.cross_domain_mode,
    )
    return {"metric": metric_key, **result}


def _model_manifest_entry(model: TrainedModel, project_root: Path) -> dict[str, Any]:
    config_path = Path(model.config) if model.config else None
    if config_path is not None and not config_path.is_file():
        local = project_root / "configs" / "models" / config_path.name
        config_path = local if local.is_file() else None
    config_sha256 = None
    if config_path is not None:
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    stat = model.weights.stat()
    return {
        "label": model.label,
        "backend": model.backend,
        "variant": model.variant,
        "weights": str(model.weights),
        "weights_size_bytes": stat.st_size,
        "weights_mtime_ns": stat.st_mtime_ns,
        "config": str(config_path) if config_path else model.config,
        "config_sha256": config_sha256,
    }


def _profile_config(
    request: SweepRequest,
    input_style: str = "batched",
    engine: str = "pytorch",
    power_mode: str | None = None,
):
    from fabric_defect_hub.profiling.base import ProfileConfig

    return ProfileConfig(
        device=request.device, engine=engine,
        measured_runs=request.measured_runs, warmup_runs=request.warmup_runs,
        input_style=input_style, power_mode=power_mode or request.power_mode,
    )


def _profiler_for(artifact: Any):
    """The profiler that can actually read this artifact's format.

    Each backend exports to whatever it supports — anomalib gives ONNX,
    Ultralytics and torchvision give TorchScript — so hard-wiring
    `PyTorchProfiler` made every anomalib model fail with "expects a
    torchscript artifact, got onnx". The engine follows the artifact.
    """

    if artifact.target == "onnx":
        from fabric_defect_hub.profiling.onnxruntime import ONNXRuntimeProfiler

        return ONNXRuntimeProfiler(), "onnxruntime"
    from fabric_defect_hub.profiling.pytorch import PyTorchProfiler

    return PyTorchProfiler(), "pytorch"


def _runtime(exported: Any, request: SweepRequest) -> dict[str, Any]:
    artifact, input_style, adapter = _export_parts(exported)
    profiler, engine = _profiler_for(artifact)
    timing_config = _profile_config(request, input_style, engine, power_mode="disabled")
    metrics = profiler.profile(artifact, timing_config)
    instrumentation = dict(getattr(profiler, "last_instrumentation", {}))
    instrumentation["performance_pass_instrumented_for_power"] = False
    if request.power_mode == "required":
        power_profiler, _ = _profiler_for(artifact)
        power_config = _profile_config(request, input_style, engine, power_mode="required")
        power_metrics = power_profiler.profile(artifact, power_config)
        metrics.update({
            key: value for key, value in power_metrics.items()
            if key.startswith("power_") or key == "energy_j"
        })
        power_context = getattr(power_profiler, "last_instrumentation", {}).get("power", {})
        instrumentation["power"] = {**power_context, "separate_measurement_pass": True}
    metrics["model_size_mb"] = Path(artifact.path).stat().st_size / (1024 * 1024)
    if adapter is not None:
        metrics.update(_model_complexity(adapter, request))
        memory_kind = instrumentation.get("memory", {}).get("kind")
        if (
            metrics.get("fps", 0) > 0
            and metrics.get("peak_memory_mb", 0) > 0
            and memory_kind == "device_allocator"
        ):
            try:
                from fabric_defect_hub.evaluation.lmei_profiler import calculate_lmei

                if "flops_g" in metrics and "params_m" in metrics:
                    metrics["lmei"] = calculate_lmei(
                        fps=metrics["fps"],
                        vram_mb=metrics["peak_memory_mb"],
                        flops_g=metrics["flops_g"],
                        params_m=metrics["params_m"],
                    )
            except (ImportError, TypeError, ValueError):
                pass
    if "flops_g" in metrics:
        instrumentation["flops"] = {
            "kind": "static_hook_estimate",
            "scope": "eager_model_forward_at_configured_input_size",
            "convention": "two_flops_per_mac",
            "cross_backend_comparable": False,
        }
    instrumentation["derived_metric_policy"] = {
        "lmei_requires_memory_kind": "device_allocator",
        "cross_scope_memory_ranking_allowed": False,
    }
    metrics["__instrumentation__"] = instrumentation
    return metrics


def _scaling(exported: Any, request: SweepRequest) -> dict[str, Any]:
    from fabric_defect_hub.profiling.sweeps import resolution_scaling

    artifact, input_style, _ = _export_parts(exported)
    profiler, engine = _profiler_for(artifact)
    config = _profile_config(request, input_style, engine, power_mode="disabled")
    metrics = resolution_scaling(profiler, artifact, config)
    metrics["__instrumentation__"] = {
        **profiler.instrumentation_context(config),
        "performance_pass_instrumented_for_power": False,
    }
    return metrics


def _concurrency(exported: Any, request: SweepRequest) -> dict[str, Any]:
    from fabric_defect_hub.profiling.sweeps import concurrency_capacity

    artifact, input_style, _ = _export_parts(exported)
    profiler, engine = _profiler_for(artifact)
    config = _profile_config(request, input_style, engine, power_mode="disabled")
    metrics = concurrency_capacity(
        profiler, artifact, config,
        frame_budget_ms=request.frame_budget_ms,
        max_streams_to_try=request.max_streams_to_try,
    )
    metrics["__instrumentation__"] = {
        **profiler.instrumentation_context(config),
        "performance_pass_instrumented_for_power": False,
        "power": {"mode": "disabled", "status": "not_measured_for_concurrency"},
    }
    return metrics


def _communication(model: TrainedModel, exported: Any | None) -> dict[str, float]:
    """Measure transferable artifact sizes as a communication proxy.

    This is deliberately not called network bandwidth: no network endpoint is
    involved. It quantifies the bytes that must cross a deployment boundary
    for model distribution, while real link throughput remains environment-
    specific and must be measured separately at deployment time.
    """

    metrics = {
        "model_transfer_mb": model.weights.stat().st_size / (1024 * 1024),
        "model_transfer_bytes": float(model.weights.stat().st_size),
    }
    if exported is not None:
        artifact, _, _ = _export_parts(exported)
        path = Path(artifact.path)
        if path.is_file():
            metrics["export_transfer_mb"] = path.stat().st_size / (1024 * 1024)
            metrics["export_transfer_bytes"] = float(path.stat().st_size)
    return metrics


def _try_export(model: TrainedModel, request: SweepRequest) -> tuple[Any, str | None]:
    """Export once per model, shared by every profiling group.

    Returns `((artifact, input_style), None)` or `(None, reason)` — the
    input style comes from `ModelCapabilities.export_input_style`, because
    only the backend knows whether its exported module wants one batched
    tensor or a list of per-image tensors.
    """

    try:
        from fabric_defect_hub.inference.runner import build_adapter

        adapter = build_adapter(model.backend, model.variant)
        caps = adapter.capabilities()
        if not caps.export_targets:
            return None, f"{model.backend} declares no export targets"

        # These are weights this project trained and registered itself, so
        # the unsafe-checkpoint guard (which exists to stop a *downloaded*
        # Lightning checkpoint deserializing arbitrary objects) is satisfied
        # by provenance. Asked by signature rather than by backend name so a
        # seventh backend with the same guard needs no edit here.
        load_kwargs = (
            {"allow_unsafe_checkpoint": True}
            if _accepts_arg(adapter.load_trained_model, "allow_unsafe_checkpoint") else {}
        )
        artifact = adapter.load_trained_model(str(model.weights), **load_kwargs)
        target = "torchscript" if "torchscript" in caps.export_targets else caps.export_targets[0]
        # No export config: a backend's `config` dict is forwarded straight
        # into its own exporter (Ultralytics rejects any key YOLO does not
        # define), so the sweep passes nothing and takes each backend's
        # default destination.
        exported = adapter.export(artifact, target)
        return (exported, caps.export_input_style, adapter), None
    except Exception as exc:  # noqa: BLE001 -- export failure skips profiling, not the sweep
        return None, f"export failed ({type(exc).__name__}: {str(exc)[:200]})"


def _accepts_arg(func: Any, parameter: str) -> bool:
    import inspect

    try:
        return parameter in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _export_parts(exported: Any) -> tuple[Any, str, Any | None]:
    """Unpack current and legacy export tuples for compatibility."""

    if len(exported) == 2:
        return exported[0], exported[1], None
    return exported[0], exported[1], exported[2]


def _model_complexity(adapter: Any, request: SweepRequest) -> dict[str, float]:
    """Measure optional FLOPs/parameters/LMEI from the live adapter module."""

    raw_module = adapter.raw_module() if hasattr(adapter, "raw_module") else None
    if raw_module is None:
        return {}
    try:
        from fabric_defect_hub.model_statistics import parameter_counts
        from fabric_defect_hub.profiling.flops import compute_model_flops

        flops_g = compute_model_flops(
            raw_module, input_size=(640, 640),
            input_style=adapter.capabilities().export_input_style,
            device=request.device,
        )
        params_m = parameter_counts(raw_module).get("parameter_count", 0) / 1e6
        return {"flops_g": float(flops_g), "params_m": float(params_m)}
    except (ImportError, NotImplementedError, RuntimeError, TypeError, ValueError):
        # FLOPs is an optional dependency and some backends expose no
        # instrumentable eager module. Runtime metrics remain valid.
        return {}


def _headline_metric(metrics: dict[str, Any]) -> str | None:
    """The metric a degradation is computed over, in task priority order."""

    for key in ("map_50", "map", "image_auroc", "auroc", "miou", "dice", "f1", "iou"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return key
    return None


# --------------------------------------------------------------------------- #
# Reading results back
# --------------------------------------------------------------------------- #
def read_sweep(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """`(header, rows)` from a sweep JSONL — the secondary-analysis entry.

    Tolerates a truncated final line, which is what a sweep killed mid-write
    leaves behind and is precisely when the partial results matter most.
    """

    header: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("kind") == "sweep_header":
            header = payload
        else:
            rows.append(payload)
    return header, rows


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counts by status and by group — what a script prints when it finishes."""

    rows = list(rows)
    by_status: dict[str, int] = {}
    by_group: dict[str, dict[str, int]] = {}
    for row in rows:
        status = row.get("status", "?")
        group = row.get("group", "?")
        by_status[status] = by_status.get(status, 0) + 1
        by_group.setdefault(group, {})
        by_group[group][status] = by_group[group].get(status, 0) + 1
    return {"total": len(rows), "by_status": by_status, "by_group": by_group}
