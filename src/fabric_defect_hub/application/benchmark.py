"""Backend glue for the Benchmark tab: load, evaluate, and profile models."""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from fabric_defect_hub.core.registry import get_profiler_cls
from fabric_defect_hub.core.types import ModelInfo, RuntimeInfo
from fabric_defect_hub.evaluation import evaluator_for_task, ground_truth_task
from fabric_defect_hub.evaluation.cross_domain import cross_domain_degradation
from fabric_defect_hub.i18n import DEFAULT_LANGUAGE, tr
from fabric_defect_hub.inference.session import clear_accelerator_cache
from fabric_defect_hub.application import load_dataset, load_model, run_experiment
from fabric_defect_hub.profiling.base import ProfileConfig
from fabric_defect_hub.scoring import SCORE_PRESETS, score_rows
from fabric_defect_hub.application.workspace import (
    DATASET_CATALOG,
    MODEL_CATALOG,
    available_model_labels,
    shot_text,
    task_text,
    artifact_for_model,
    dataset_tasks,
    default_dataset_root,
    shot_regime_kwargs,
    slice_value,
)

DEFAULT_RUN_LOG_PATH = "runs/leaderboard_log.jsonl"
BENCHMARK_ANOMALY_MAP_ROOT = "artifacts/runtime/anomaly_maps/benchmark"

# The metric each task's `Evaluator` treats as its headline accuracy number
# (see `evaluation/{anomaly,detection,segmentation}.py`) -- what
# `cross_domain_degradation` is computed over.
_PRIMARY_ACCURACY_METRIC = {"anomaly": "image_auroc", "detection": "map", "segmentation": "miou"}

# Input side lengths swept for `_resolution_sweep`. Kept short (4 points,
# 2 warmup/5 measured runs each below) since this already runs on top of
# whatever `include_profiling`'s own pass costs -- enough points for a
# least-squares slope, not a dense curve.
RESOLUTION_SWEEP_SIZES: tuple[int, ...] = (320, 480, 640, 800)


def score_preset_choices(lang: str = DEFAULT_LANGUAGE) -> list[tuple[str, str]]:
    """Gradio `(display_label, value)` tuples for the score-preset dropdown
    — the value halves (`scoring.SCORE_PRESETS` keys, plus `"custom"`) are
    compared elsewhere (`run_benchmark`'s `score_preset == "custom"` branch),
    so only the display half is localized; see `i18n.py`'s module docstring."""

    return [
        (tr(lang, "choice_score_accuracy_first"), "accuracy_first"),
        (tr(lang, "choice_score_balanced"), "balanced"),
        (tr(lang, "choice_score_efficiency_first"), "efficiency_first"),
        (tr(lang, "choice_score_custom"), "custom"),
    ]


def compatible_models(dataset_label: str) -> list[str]:
    """Models this dataset can supply real ground truth for *and* that are
    staged here — i.e. every available catalog model whose task the dataset's
    `tasks` set covers (ZJU-Leaper has boxes *and* masks, so both detection
    and segmentation models are compatible; RAW-FABRID/MVTec AD have anomaly
    labels and masks but no boxes, so only anomaly and segmentation models
    are).

    The staging half comes from `available_model_labels`, so the benchmark
    checklist never offers a general-domain training slot nothing has been
    published into yet — the model would be evaluated against zero weights.
    """

    tasks = dataset_tasks(DATASET_CATALOG[dataset_label]["name"])
    return [
        label for label in available_model_labels()
        if ground_truth_task(MODEL_CATALOG[label]["task"]) in tasks
    ]


def _detect_devices(torch_module: Any | None = None) -> list[str]:
    """Return every usable local accelerator, or one serial fallback device."""

    try:
        torch = torch_module
        if torch is None:
            import torch

        if torch.cuda.is_available():
            return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if torch.backends.mps.is_available():
            return ["mps"]
    except ImportError:
        pass
    return ["cpu"]


def _detect_device() -> str:
    return _detect_devices()[0]


def _activate_device(device: str) -> None:
    """Select this worker's CUDA device for backends that use torch defaults."""

    if not device.startswith("cuda:"):
        return
    import torch

    torch.cuda.set_device(int(device.partition(":")[2]))


def _profile_setup(model: Any, device: str):
    """Build the (profiler, config, export_target) triple `run_experiment`
    needs to also measure FPS/latency/memory for this model, mirroring
    `benchmark.py::_profile_from_spec`'s pytorch-engine defaults but with
    lighter warmup/measured-run counts so an interactive UI click doesn't
    stall for as long as an unattended CLI benchmark would tolerate.

    `input_style` comes from the model's own `capabilities()`. This module
    used to derive it from `backend == "torchvision" and task in (...)` --
    a fact about torchvision's exported forward signature, asserted by the
    benchmark tab, which is not in a position to know it.
    """

    capabilities = model.capabilities()
    # The profiler follows the artifact, rather than the export being forced
    # to suit one profiler. Restricting this to torchscript/exported_program
    # meant every anomalib model (PatchCore, PaDiM, RD4AD, ...) reported
    # "no PyTorchProfiler-compatible export target" and silently lost its
    # whole overhead row — anomalib exports ONNX. Same rule as
    # `metric_sweep._profiler_for`, deliberately spelled the same way.
    engine_for_target = {
        "exported_program": "pytorch", "torchscript": "pytorch", "onnx": "onnxruntime",
    }
    export_target = next(
        (target for target in engine_for_target if target in capabilities.export_targets), None
    )
    if export_target is None:
        return None

    engine = engine_for_target[export_target]
    profiler = get_profiler_cls(engine)()
    config = ProfileConfig(
        device=device, engine=engine, precision="fp32", input_size=(640, 640),
        input_style=capabilities.export_input_style, warmup_runs=5, measured_runs=20,
        power_mode="disabled",
    )
    return profiler, config, export_target


def _native_profile_model(model: Any, artifact: Any, samples: list[Any], device: str) -> dict[str, float]:
    """Measure an adapter directly when it has no exportable runtime.

    This path is intentionally labelled native: it includes the adapter's
    preprocessing and postprocessing, and is therefore not mixed with the
    export-only profiler results.
    """
    if not samples:
        raise ValueError("native profiling requires at least one sample")
    import statistics
    import time as _time

    sample = samples[0]
    warmup, measured = 2, 8
    for _ in range(warmup):
        model.predict([sample], artifact)
    latencies: list[float] = []
    for _ in range(measured):
        started = _time.perf_counter()
        model.predict([sample], artifact)
        latencies.append((_time.perf_counter() - started) * 1000.0)
    mean_ms = statistics.fmean(latencies)
    metrics: dict[str, float] = {
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2],
        "latency_ms_p95": sorted(latencies)[min(len(latencies) - 1, int(round(.95 * (len(latencies) - 1))))],
        "latency_ms_p99": sorted(latencies)[min(len(latencies) - 1, int(round(.99 * (len(latencies) - 1))))],
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "profiling_mode": "native",
        "memory_measurement_kind": "process_rss",
        "memory_measurement_scope": "host_process",
        "memory_cross_engine_comparable": False,
    }
    try:
        import os
        import psutil

        metrics["peak_memory_mb"] = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        metrics["peak_memory_mb"] = 0.0
    return metrics


def _profile_model(model: Any, artifact: Any, device: str, samples: list[Any] | None = None) -> dict[str, float]:
    """Export and profile one model without letting optional overhead data
    invalidate its already-computed accuracy result.

    Some backends intentionally do not provide a PyTorch-loadable export
    (Anomalib's ``torch`` package, for example); others may have model or
    runtime-specific export limitations.  The benchmark's accuracy row is
    still valid in either case, so callers handle failures as a skipped
    optional metric rather than a failed model.
    """

    setup = _profile_setup(model, device)
    if setup is None:
        return _native_profile_model(model, artifact, samples or [], device)
    profiler, config, export_target = setup
    # No export config: the dict is forwarded verbatim into the backend's own
    # exporter, and each backend has its own vocabulary — Ultralytics rejects
    # any key YOLO does not define ("'input_size' is not a valid YOLO
    # argument"), which made profiling fail for every YOLO model while the
    # error surfaced only as a status-line footnote. `config.input_size`
    # shapes the *profiler's* dummy input; the export keeps its defaults,
    # which are what `metric_sweep._try_export` profiles successfully.
    exported = model.export(artifact, target=export_target)
    export_path = Path(exported.path)
    if not export_path.is_file():
        raise FileNotFoundError(f"exported model does not exist: {export_path}")
    metrics = profiler.profile(exported, config)
    memory_context = getattr(profiler, "last_instrumentation", {}).get("memory", {})
    metrics["memory_measurement_kind"] = memory_context.get("kind", "unknown")
    metrics["memory_measurement_scope"] = memory_context.get("scope", "unknown")
    metrics["memory_cross_engine_comparable"] = bool(
        memory_context.get("cross_engine_comparable", False)
    )
    metrics["model_size_mb"] = export_path.stat().st_size / (1024 * 1024)
    return metrics


def _resolution_sweep(model: Any, artifact: Any, device: str) -> dict[str, float]:
    """Export once, profile that same export at `RESOLUTION_SWEEP_SIZES`,
    and fit the throughput decay slope. See the module docstring's
    `include_resolution_sweep` entry for why this doesn't just call
    `run_experiment` once per resolution (that would redundantly re-run
    accuracy evaluation too).
    """

    from fabric_defect_hub.profiling.sweeps import resolution_scaling

    profiler = get_profiler_cls("pytorch")()
    input_style = model.capabilities().export_input_style
    exported = model.export(artifact, target="torchscript")
    config = ProfileConfig(
        device=device, engine="pytorch", precision="fp32",
        input_style=input_style, warmup_runs=2, measured_runs=5,
    )
    # `resolution_scaling` (the same driver `fdh.measure` uses) drops sizes
    # the export cannot run instead of dying on the first one — an
    # Ultralytics TorchScript bakes in its imgsz, so this loop used to raise
    # at the first non-native size and forfeit the whole sweep.
    metrics = resolution_scaling(profiler, exported, config, sides=RESOLUTION_SWEEP_SIZES)
    if not metrics:
        raise RuntimeError(
            "fewer than two resolutions were measurable — the export is fixed-shape, "
            "so a decay slope cannot be fitted for this model"
        )
    return metrics


def _flops_and_lmei(
    model: Any, model_spec: dict[str, Any], device: str, fps: float | None,
    vram_mb: float | None, memory_kind: str | None,
) -> dict[str, float]:
    """FLOPs + parameter count from the adapter's live model (`ModelAdapter
    .raw_module()`), then the LMEI edge-deployment trade-off score those
    combine with `fps`/`vram_mb` into (see `evaluation.lmei_profiler
    .calculate_lmei`). Deliberately *not* computed from the TorchScript
    export `include_profiling` already produced: `thop`'s hook-based
    counter needs to `register_buffer` bookkeeping tensors onto the model,
    which a frozen `torch.jit.ScriptModule` refuses ("Can't add a new
    parameter after ScriptModule construction") -- only the live,
    pre-export module accepts that. Returns `{}` (no columns added) when
    there's no raw module to instrument, or `fps`/`vram_mb` aren't
    available -- `calculate_lmei` itself would just return 0.0 for a
    missing input, which would misleadingly look like a real "worst
    possible" score.
    """

    raw_module = model.raw_module() if hasattr(model, "raw_module") else None
    if raw_module is None or not fps or not vram_mb:
        return {}

    from fabric_defect_hub.evaluation.lmei_profiler import calculate_lmei
    from fabric_defect_hub.model_statistics import parameter_counts
    from fabric_defect_hub.profiling.flops import compute_model_flops

    flops_g = compute_model_flops(
        raw_module, input_size=(640, 640),
        input_style=model.capabilities().export_input_style, device=device,
    )
    params_m = parameter_counts(raw_module).get("parameter_count", 0) / 1e6
    metrics = {
        "flops_g": round(flops_g, 4),
        "params_m": round(params_m, 4),
    }
    if memory_kind == "device_allocator":
        metrics["lmei"] = calculate_lmei(
            fps=fps, vram_mb=vram_mb, flops_g=flops_g, params_m=params_m
        )
    return metrics


def _cross_domain_probe(
    model: Any,
    artifact: Any,
    dataset_task: str,
    target_label: str,
    num_samples: int | None,
    defect_ratio: float,
) -> dict[str, float] | None:
    """Evaluate the already-loaded `model` against a second ("target")
    dataset -- the whole dataset, no texture/pattern slicing -- to measure
    how far its accuracy falls outside its primary ("source") domain.
    Returns `None` (letting the caller skip the degradation column) when
    the target dataset can't supply this task, isn't staged on this
    machine, or has no matching samples, so a mismatched pairing never
    fails the row it's attached to.
    """

    target_spec = DATASET_CATALOG.get(target_label)
    if target_spec is None or dataset_task not in dataset_tasks(target_spec["name"]):
        return None
    root = default_dataset_root(target_label)
    if not root:
        return None
    dataset = load_dataset(
        target_spec["name"], root=root, task=dataset_task, split="test",
        use_defect=True, num_samples=num_samples, defect_ratio=defect_ratio,
    )
    samples = dataset.load_samples()
    if not samples:
        return None
    predictions = model.predict(samples, artifact)
    return evaluator_for_task(dataset_task).evaluate(samples, predictions)


def _release_model(model: Any) -> None:
    """Mirrors `InferenceSessionManager._unload_active` (which the Single
    Image tab uses): call the adapter's own `unload()` if it has one (only
    the Ultralytics and Anomalib adapters do), drop our reference, then
    force a GC pass and clear the CUDA/MPS allocator cache so the next
    model's `load_model` isn't fighting the previous one's still-cached
    memory."""

    unload = getattr(model, "unload", None)
    if callable(unload):
        unload()
    del model
    gc.collect()
    clear_accelerator_cache()


def run_benchmark(
    dataset_label: str,
    texture_label: str,
    shot_mode: str,
    model_labels: list[str],
    lang: str = DEFAULT_LANGUAGE,
    include_profiling: bool = False,
    include_resolution_sweep: bool = False,
    cross_domain_dataset_label: str | None = None,
    score_preset: str = "balanced",
    custom_technical_weight: float | None = None,
    run_log_path: str | None = DEFAULT_RUN_LOG_PATH,
) -> Iterator[tuple[list[str], list[list[Any]], str, list[dict[str, Any]]]]:
    """Evaluate every model in `model_labels` against the same dataset
    sample (test split only — the benchmark tab never trains). A single
    device runs models serially as mount -> test -> unmount -> next model
    (`_release_model`); CUDA hosts run up to one model per detected GPU in
    parallel, with each worker pinned to its own `cuda:N`. Yields `(columns,
    rows, status)` after every completed model so the leaderboard fills in
    live instead of appearing all at once; `columns` is the superset of
    metric names produced by any model evaluated so far, so every row stays
    padded to the same shape.

    `include_profiling` additionally runs a `PyTorchProfiler` pass per model
    (see `_profile_setup`) so overhead metrics (fps, latency_ms_*,
    peak_memory_mb, model_size_mb) land in the same row as the accuracy
    metrics. `include_resolution_sweep` and `cross_domain_dataset_label` are
    two further opt-ins — see the module docstring — that add
    `resolution_slope_beta`/`resolution_slope_alpha` and
    `cross_domain_delta_acc_pct` columns respectively. `score_preset` (one
    of `scoring.SCORE_PRESETS`, or `"custom"` with `custom_technical_weight`
    in [0, 1]) blends whatever technical/overhead metrics are present into a
    `composite_score` column, recomputed across all rows collected so far
    after every model. `run_log_path`, if not `None`, appends every
    completed row to that shared JSONL log via `reporting.append_run_log`.
    """

    if not model_labels:
        yield [], [], tr(lang, "bench_select_model"), []
        return

    root = default_dataset_root(dataset_label)
    if not root:
        yield [], [], tr(lang, "bench_dataset_unavailable", label=dataset_label), []
        return

    if score_preset == "custom":
        weight = 0.5 if custom_technical_weight is None else custom_technical_weight
        technical_weight, overhead_weight = weight, 1.0 - weight
    else:
        technical_weight, overhead_weight = SCORE_PRESETS.get(score_preset, SCORE_PRESETS["balanced"])

    spec = DATASET_CATALOG[dataset_label]
    supported_tasks = dataset_tasks(spec["name"])
    num_samples, defect_ratio = shot_regime_kwargs(shot_mode)
    base_dataset_kwargs: dict[str, Any] = dict(
        root=root,
        split="test",
        use_defect=True,
        num_samples=num_samples,
        defect_ratio=defect_ratio,
    )
    if spec["slice_kwarg"] is not None:
        base_dataset_kwargs[spec["slice_kwarg"]] = slice_value(dataset_label, texture_label)

    devices = _detect_devices()

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    sample_count: int | None = None
    total = len(model_labels)
    yield [], [], tr(lang, "bench_starting", total=total), []

    def evaluate_model(
        index: int, model_label: str, device: str,
    ) -> tuple[int, str, dict[str, Any] | None, list[str], int | None]:
        model_spec = MODEL_CATALOG[model_label]
        dataset_task = ground_truth_task(model_spec["task"])
        if dataset_task not in supported_tasks:
            return index, model_label, None, [tr(
                lang, "bench_task_mismatch",
                model=model_label, dataset=dataset_label, task=task_text(lang, model_spec["task"]),
            )], None

        model = None
        warnings: list[str] = []
        try:
            _activate_device(device)
            dataset = load_dataset(spec["name"], task=dataset_task, **base_dataset_kwargs)
            model = load_model(model_spec["backend"], model_spec["name"])
            evaluator = evaluator_for_task(dataset_task)
            started = time.perf_counter()
            result = run_experiment(
                experiment_id=f"benchmark-{_slug(model_label)}",
                dataset=dataset,
                model=model,
                model_info=ModelInfo(
                    name=model_spec["name"], backend=model_spec["backend"], task=model_spec["task"]
                ),
                runtime=RuntimeInfo(device=device, engine="python", precision="fp32", input_size=(640, 640)),
                evaluator=evaluator,
                artifact=artifact_for_model(model_spec),
                output_dir=str(Path(BENCHMARK_ANOMALY_MAP_ROOT) / _slug(model_label)),
                run_log_path=run_log_path,
            )
            count = len(dataset.load_samples())
            row: dict[str, Any] = {
                "model": model_label,
                "runtime_s": round(time.perf_counter() - started, 1),
                **result.metrics,
            }
            if include_profiling:
                try:
                    profile_started = time.perf_counter()
                    row.update(_profile_model(model, artifact_for_model(model_spec), device, dataset.load_samples()))
                    row["runtime_s"] = round(row["runtime_s"] + time.perf_counter() - profile_started, 1)
                except Exception as exc:
                    warnings.append(f"{model_label}: profiling skipped ({type(exc).__name__}: {exc})")
            # Every opt-in addition below is best-effort: a failure in one
            # (e.g. thop missing for FLOPs, a target dataset erroring mid-
            # probe) only forfeits that addition's columns, never the base
            # accuracy/profiling row already computed above.
            if include_resolution_sweep:
                try:
                    row.update(_resolution_sweep(model, artifact_for_model(model_spec), device))
                except Exception as exc:
                    warnings.append(f"{model_label}: resolution sweep skipped ({type(exc).__name__}: {exc})")
            if include_profiling:
                try:
                    row.update(_flops_and_lmei(
                        model, model_spec, device, fps=row.get("fps"), vram_mb=row.get("peak_memory_mb"),
                        memory_kind=row.get("memory_measurement_kind"),
                    ))
                except Exception as exc:
                    warnings.append(f"{model_label}: FLOPs/LMEI skipped ({type(exc).__name__}: {exc})")
            if cross_domain_dataset_label:
                try:
                    metric_key = _PRIMARY_ACCURACY_METRIC.get(dataset_task)
                    acc_src = result.metrics.get(metric_key) if metric_key else None
                    if acc_src is not None:
                        target_metrics = _cross_domain_probe(
                            model, artifact_for_model(model_spec), dataset_task,
                            cross_domain_dataset_label, num_samples, defect_ratio,
                        )
                        acc_tgt = target_metrics.get(metric_key) if target_metrics else None
                        if acc_tgt is not None and acc_src != 0:
                            row["cross_domain_delta_acc_pct"] = cross_domain_degradation(acc_src, acc_tgt)
                except Exception as exc:
                    warnings.append(f"{model_label}: cross-domain probe skipped ({type(exc).__name__}: {exc})")
            return index, model_label, row, warnings, count
        except Exception as exc:
            return index, model_label, None, [f"{model_label}: {type(exc).__name__}: {exc}"], None
        finally:
            if model is not None:
                _release_model(model)

    scheduled = list(enumerate(model_labels, start=1))
    if len(devices) == 1:
        completed = (
            evaluate_model(index, model_label, devices[0])
            for index, model_label in scheduled
        )
    else:
        def parallel_completed():
            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                futures = [
                    executor.submit(evaluate_model, index, model_label, devices[(index - 1) % len(devices)])
                    for index, model_label in scheduled
                ]
                yield from (future.result() for future in as_completed(futures))

        completed = parallel_completed()

    for index, model_label, row, warnings, count in completed:
        if row is not None:
            rows.append(row)
        errors.extend(warnings)
        if sample_count is None and count is not None:
            sample_count = count
        status = tr(lang, "bench_progress", index=index, total=total, model=model_label)
        yield _render(rows, sample_count, shot_mode, errors, status, lang, technical_weight, overhead_weight)

    yield _render(rows, sample_count, shot_mode, errors, lang=lang, technical_weight=technical_weight, overhead_weight=overhead_weight)


def _render(
    rows: list[dict[str, Any]],
    sample_count: int | None,
    shot_mode: str,
    errors: list[str],
    status: str | None = None,
    lang: str = DEFAULT_LANGUAGE,
    technical_weight: float = 0.5,
    overhead_weight: float = 0.5,
) -> tuple[list[str], list[list[Any]], str, list[dict[str, Any]]]:
    """Returns `(columns, table, status, scored)`. `table` is the
    positional, display-formatted form the `gr.Dataframe` wants; `scored` is
    the same rows as metric-name-keyed dicts, which is what `web/tables.py`
    needs (a chart looks metrics up by name, it can't use column offsets).
    Both come from one `score_rows` call so the charts and the table can
    never show different numbers."""

    if not rows:
        base = tr(lang, "bench_no_results") if not errors else "🔴 " + "; ".join(errors)
        return [], [], base, []

    scored = score_rows(rows, technical_weight, overhead_weight)
    scored.sort(key=lambda row: (row["composite_score"] is None, -(row["composite_score"] or 0)))

    score_columns = ["composite_score", "technical_score", "overhead_score"]
    metric_columns = sorted({
        key for row in scored if row
        for key in row if key not in ("model", "runtime_s", *score_columns)
    })
    columns = ["model", *score_columns, "runtime_s", *metric_columns]
    table = [
        [_display_value(row.get(column, ""), column, score_columns) for column in columns]
        for row in scored
    ]
    if status is None:
        status = tr(
            lang, "bench_done", count=len(rows),
            samples=sample_count if sample_count is not None else "?", shot=shot_text(lang, shot_mode),
        )
    if errors:
        status += " ⚠️ " + "; ".join(errors)
    return columns, table, status, scored


def _display_value(value: Any, column: str, score_columns: list[str]) -> Any:
    if column in score_columns and isinstance(value, (int, float)):
        return round(value, 1)
    return value


def _slug(label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in label).strip("-")
