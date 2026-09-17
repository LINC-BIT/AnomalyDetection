"""Backend glue for the Benchmark tab: load, evaluate, and profile models."""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from fabric_defect_hub.core.execution import model_execution, model_tracing
from fabric_defect_hub.core.registry import get_profiler_cls
from fabric_defect_hub.core.types import ModelInfo, RuntimeInfo
from fabric_defect_hub.evaluation import evaluator_for_task, ground_truth_task
from fabric_defect_hub.evaluation.cross_domain import cross_domain_degradation
from fabric_defect_hub.i18n import DEFAULT_LANGUAGE, tr
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
# Where `_record_benchmark_failure` appends the stacks the status line cannot
# carry. Beside the per-model anomaly maps, so one run's failures live with the
# artifacts they belong to (both are gitignored `artifacts/runtime/` output).
BENCHMARK_FAILURE_LOG = Path(BENCHMARK_ANOMALY_MAP_ROOT) / "failures.log"
_FAILURE_LOG_LOCK = threading.Lock()


def _poisoned_cuda_hint(exc: BaseException | str) -> str:
    """Extra sentence for the one CUDA failure that is not about this model.

    An illegal memory access leaves the process's CUDA context permanently
    broken ("Sticky error detected"). Until each model got its own process that
    meant every model evaluated afterwards failed with the same message no
    matter how healthy it was — which read as ten broken models instead of one
    poisoned context. Say so either way, because the symptom still looks the
    same from the panel.
    """

    text = str(exc)
    if "illegal memory access" not in text and "Sticky error" not in text:
        return ""
    return (
        " — this is not this model's fault: an illegal memory access poisoned a CUDA context, "
        "and the models sharing that process failed with it. Each model now runs in its own "
        "process, so the rest of the batch is unaffected; re-run to retry this one"
    )


def _append_benchmark_failure(model_label: str, detail: str) -> None:
    """Append one failure block to `BENCHMARK_FAILURE_LOG`, and say where.

    Best-effort by construction: this runs for a failure whose whole job is to
    not take the batch down, so a read-only filesystem or a missing directory
    must not turn a reported model failure into a crashed benchmark. The panel
    keeps its one-line summary; this is the part that says *which frame* raised.
    """

    try:
        with _FAILURE_LOG_LOCK:
            BENCHMARK_FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with BENCHMARK_FAILURE_LOG.open("a", encoding="utf-8") as handle:
                handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {model_label} =====\n")
                handle.write(detail if detail.endswith("\n") else detail + "\n")
        last_line = detail.strip().splitlines()[-1] if detail.strip() else "failed"
        print(
            f"[benchmark] {model_label}: {last_line[:200]} "
            f"(full traceback: {BENCHMARK_FAILURE_LOG})",
            file=sys.stderr, flush=True,
        )
    except Exception:
        return


def _record_benchmark_failure(model_label: str, exc: BaseException) -> None:
    """Log an exception raised in *this* process (see `_append_benchmark_failure`)."""

    _append_benchmark_failure(
        model_label, "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )


def _record_worker_failure(model_label: str, error: str, traceback_text: str | None) -> None:
    """Log a failure a worker process reported, with the worker's own stack.

    The worker catches its own exceptions and sends the traceback back, so the
    log still shows the frame that raised even though the failure happened in
    another process.
    """

    _append_benchmark_failure(model_label, traceback_text or f"{error}\n")


# How long one model's worker may run before the parent gives up on it. A
# 350-sample anomalib evaluation is minutes, an export-heavy profiling pass can
# add more, so the default is generous -- but finite, because a wedged worker
# holds a GPU the rest of the batch could be using.
BENCHMARK_WORKER_TIMEOUT_SECONDS = float(os.environ.get("FDH_BENCHMARK_WORKER_TIMEOUT", "3600"))
_BENCHMARK_WORKER_MODULE = "fabric_defect_hub.application.benchmark_worker"

# Every worker this process has running right now. Kept so a quit (Ctrl+C on the
# UI, or a normal exit) can kill them: a stranded worker keeps a GPU and a
# 350-sample evaluation's worth of memory until it finishes on its own.
_LIVE_WORKERS: set[subprocess.Popen] = set()
_LIVE_WORKERS_LOCK = threading.Lock()


def _child_setup() -> None:  # pragma: no cover - runs in the forked child
    """Ask the kernel to signal a worker if this process dies.

    `preexec_fn` runs between fork and exec, so it must not allocate or import
    anything heavy. Linux-only (`prctl`); elsewhere the `atexit` cleanup is the
    only guard, which is why it is registered unconditionally.
    """

    try:
        import ctypes
        import signal as _signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, _signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG = 1
    except Exception:
        return


def _spawn_worker(payload: dict[str, Any], results_path: Path, env: dict[str, str]) -> subprocess.Popen:
    """Start one worker process, and remember it until it exits."""

    process = subprocess.Popen(
        [sys.executable, "-m", _BENCHMARK_WORKER_MODULE],
        stdin=subprocess.PIPE, text=True, env=env,
        preexec_fn=_child_setup if sys.platform.startswith("linux") else None,
    )
    with _LIVE_WORKERS_LOCK:
        _LIVE_WORKERS.add(process)
    assert process.stdin is not None  # noqa: S101 - guaranteed by stdin=PIPE
    process.stdin.write(json.dumps({**payload, "results_path": str(results_path)}))
    process.stdin.close()
    return process


def _terminate_process(process: subprocess.Popen, grace_seconds: float = 3.0) -> None:
    """SIGTERM a worker, SIGKILL it if it will not go.

    SIGTERM first: a worker between samples dies at once. SIGKILL only for one
    that ignores it — stuck inside a CUDA call, say — because waiting for that
    one is exactly the "quitting takes minutes" behaviour this exists to remove.
    """

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=max(0.0, grace_seconds))
    except subprocess.TimeoutExpired:
        process.kill()


def terminate_benchmark_workers(grace_seconds: float = 3.0) -> int:
    """Stop every worker process this one started; return how many it stopped.

    Called from the exit hooks below, and after an interrupted wait — a
    stranded worker keeps a GPU and a whole evaluation's worth of memory until
    it finishes on its own.
    """

    with _LIVE_WORKERS_LOCK:
        processes = list(_LIVE_WORKERS)
    for process in processes:
        _terminate_process(process, grace_seconds)
        with _LIVE_WORKERS_LOCK:
            _LIVE_WORKERS.discard(process)
    return len(processes)


_fast_exit_installed = False


def install_fast_exit() -> bool:
    """Make quitting this process quick, and never leave a worker behind.

    The UI's benchmark waits on its workers from thread-pool threads, and
    uvicorn's graceful shutdown waits for the request that is doing the waiting:
    without this, Ctrl+C on a running benchmark sat there until the current
    model finished — minutes, for a 350-sample evaluation. So the first SIGINT
    (or SIGTERM) kills the workers *before* uvicorn's own handler runs, which
    lets that request unwind immediately; the second signal still gets
    uvicorn's force-exit, so a shutdown stuck on anything else is one more
    Ctrl+C away rather than a `kill -9`.

    uvicorn registers `Server.handle_exit` as its signal handler when it starts,
    so the wrapper has to be in place before `launch()` runs — which is why the
    UI launcher calls this first. Returns whether the handler was wrapped.
    """

    global _fast_exit_installed
    atexit.register(terminate_benchmark_workers)
    if _fast_exit_installed:
        return True
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - the UI extra always brings uvicorn
        return False

    original = uvicorn.Server.handle_exit

    def handle_exit(self, sig, frame):  # type: ignore[no-untyped-def]
        stopped = terminate_benchmark_workers()
        if stopped:
            print(
                f"[benchmark] {signal.Signals(sig).name}: stopped {stopped} worker process(es) "
                f"before shutting down",
                file=sys.stderr, flush=True,
            )
        return original(self, sig, frame)

    uvicorn.Server.handle_exit = handle_exit  # type: ignore[method-assign]
    _fast_exit_installed = True
    return True


def _run_model_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one model in its own process, and return its report.

    stdout/stderr are inherited on purpose: the worker's per-image progress
    lines and the frameworks' warnings have to keep reaching the terminal the
    UI was launched from. The structured report travels through a file instead,
    so a chatty backend cannot corrupt it.
    """

    with tempfile.TemporaryDirectory(prefix="fdh-benchmark-") as tmp:
        results_path = Path(tmp) / "result.json"
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = _spawn_worker(payload, results_path, env)
        try:
            process.wait(timeout=BENCHMARK_WORKER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _terminate_process(process)
            return _worker_report(
                f"worker timed out after {BENCHMARK_WORKER_TIMEOUT_SECONDS:.0f}s "
                f"(raise FDH_BENCHMARK_WORKER_TIMEOUT to allow longer runs)"
            )
        finally:
            with _LIVE_WORKERS_LOCK:
                _LIVE_WORKERS.discard(process)
            if process.poll() is None:
                # This call was interrupted (Ctrl+C in the parent) rather than
                # the worker exiting: do not leave it running with a GPU.
                terminate_benchmark_workers()
        if not results_path.is_file():
            return _worker_report(
                f"worker exited with code {process.returncode} without writing a report — it was "
                f"killed before it could catch anything (an out-of-memory kill looks like this, "
                f"and so does quitting the UI mid-run)"
            )
        try:
            return json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _worker_report(f"worker wrote an unreadable report: {exc}")


def _worker_report(error: str) -> dict[str, Any]:
    return {
        "row": None, "warnings": [], "notes": [], "count": None,
        "error": error, "traceback": None,
    }


class MetricNotApplicable(RuntimeError):
    """An optional metric that this model *cannot* produce, not one that failed.

    The benchmark's opt-in columns (resolution slope, cross-domain delta) are
    best-effort, and a few of the reasons they are missing are properties of the
    model rather than problems: a fixed-shape export cannot be run at the other
    resolutions a decay slope needs, and a backend with no profiler-compatible
    export has nothing to sweep. Raising this instead of a bare `RuntimeError`
    lets the status line say "not applicable" (ℹ️) where a genuine failure still
    says "skipped" (⚠️) — the distinction the panel was missing.
    """


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


def _idle_cuda_device_indices(device_count: int) -> list[int]:
    """Return GPUs without a foreign compute process, when `nvidia-smi` is available."""

    try:
        gpu_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        uuid_to_index = {
            uuid.strip(): int(index.strip())
            for line in gpu_query.stdout.splitlines() if "," in line
            for index, uuid in [line.split(",", maxsplit=1)]
        }
        process_query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        busy = {
            uuid_to_index[uuid.strip()]
            for line in process_query.stdout.splitlines() if "," in line
            for pid, uuid in [line.split(",", maxsplit=1)]
            if int(pid.strip()) != os.getpid() and uuid.strip() in uuid_to_index
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    return [index for index in range(device_count) if index not in busy]


def _detect_devices(torch_module: Any | None = None) -> list[str]:
    """Prefer CUDA GPUs without other jobs; otherwise return a serial fallback."""

    try:
        torch = torch_module
        if torch is None:
            import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            idle = _idle_cuda_device_indices(device_count)
            indices = idle or list(range(device_count))
            return [f"cuda:{index}" for index in indices]
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
    #
    # Order matters, and `torchscript` goes first on purpose:
    #   * `torch.export` rejects the detection heads ("operators not yet
    #     supported"), so preferring it cost Faster/Cascade/Mask R-CNN their
    #     profiling *and* their resolution sweep;
    #   * it also patches module dispatch process-wide while it traces (see
    #     `core.execution`), which is a hazard this module would rather not
    #     depend on at all.
    # `exported_program` stays as the last resort, for a backend that offers
    # nothing else.
    engine_for_target = {
        "torchscript": "pytorch", "onnx": "onnxruntime", "exported_program": "pytorch",
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
    # Same reason as `loader.run_experiment`: every forward is a reader of the
    # tracing guard, so a profiling export in a sibling worker never traces
    # through one of these calls.
    with model_execution():
        for _ in range(warmup):
            model.predict([sample], artifact)
    latencies: list[float] = []
    for _ in range(measured):
        started = _time.perf_counter()
        with model_execution():
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
    # A backend can refuse to export a given model (WinCLIP's ONNX export hits
    # `aten::_native_multi_head_attention`, which the ONNX exporter does not
    # support at any opset this repo can ask for). Measuring the adapter
    # directly is strictly better than dropping the row's overhead columns, and
    # `_native_profile_model` labels itself as `profiling_mode="native"` so the
    # two kinds of number are never silently mixed.
    try:
        # Tracing patches module dispatch process-wide while it runs, so this is
        # a writer: it must not overlap any forward pass or another export (see
        # `core.execution`). The profiler pass below is not a trace and stays
        # parallel.
        with model_tracing():
            exported = model.export(artifact, target=export_target)
        export_path = Path(exported.path)
        if not export_path.is_file():
            raise FileNotFoundError(f"exported model does not exist: {export_path}")
    except Exception:
        return _native_profile_model(model, artifact, samples or [], device)
    exported_input_size = exported.metadata.get("input_size")
    if exported_input_size is not None:
        config = replace(config, input_size=tuple(exported_input_size))
    # The profiler pass runs the exported model on the device, so it is GPU work
    # like any forward: it must not overlap a sibling worker's trace, which
    # patches module dispatch process-wide (see `core.execution`).
    with model_execution():
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

    setup = _profile_setup(model, device)
    if setup is None:
        raise MetricNotApplicable(
            "this backend has no export format a benchmark profiler can drive"
        )
    profiler, config, export_target = setup
    with model_tracing():
        exported = model.export(artifact, target=export_target)
    exported_input_size = exported.metadata.get("input_size")
    if exported_input_size is not None:
        config = replace(config, input_size=tuple(exported_input_size))
    config = replace(config, warmup_runs=2, measured_runs=5)
    # `resolution_scaling` (the same driver `fdh.measure` uses) drops sizes
    # the export cannot run instead of dying on the first one — an
    # Ultralytics TorchScript bakes in its imgsz, so this loop used to raise
    # at the first non-native size and forfeit the whole sweep.
    # Runs the export at several sizes on the device -- GPU work, so it is a
    # reader of the tracing guard like every other forward pass.
    with model_execution():
        metrics = resolution_scaling(profiler, exported, config, sides=RESOLUTION_SWEEP_SIZES)
    if not metrics:
        raise MetricNotApplicable(
            "its export is fixed-shape, so fewer than two resolutions were measurable and a "
            "decay slope cannot be fitted"
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

    # Ask the model what it can be probed at instead of assuming 640x640: the
    # patch-based backbones assert on a side that is not a whole number of
    # patches (Dinomaly's ViTill) or mis-shape their positional embeddings
    # (MoECLIP's CLIP ViT), which used to cost every such row its FLOPs/LMEI
    # columns for a reason that reads like a model defect.
    capabilities = model.capabilities()
    probe_size = capabilities.probe_input_size or (640, 640)
    # thop's hooks run a real forward through the live module on the device.
    with model_execution():
        flops_g = compute_model_flops(
            raw_module, input_size=probe_size,
            input_style=capabilities.export_input_style, device=device,
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
    with model_execution():
        predictions = model.predict(samples, artifact)
    return evaluator_for_task(dataset_task).evaluate(samples, predictions)


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
    sample (test split only — the benchmark tab never trains). Every model is
    scored by its own worker process (`application.benchmark_worker`), pinned to
    one of the detected devices; a CUDA host runs up to one model per device in
    parallel, and a single-device host runs them one after another. Yields
    `(columns, rows, status)` after every completed model so the leaderboard
    fills in live instead of appearing all at once; `columns` is the superset of
    metric names produced by any model evaluated so far, so every row stays
    padded to the same shape. `status` names the models still running, because
    with several models in flight "12/19 scored" cannot say whether the rest are
    queued, running or stuck.

    Why a process per model: `torch.export` patches module dispatch for the
    whole process while it traces, and an illegal memory access poisons a
    process's CUDA context for good — either one used to cost every model that
    ran afterwards in the shared process, not just the model that hit it.

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
    notes: list[str] = []
    sample_count: int | None = None
    total = len(model_labels)
    yield [], [], tr(lang, "bench_starting", total=total), []

    def model_payload(index: int, model_label: str, device: str) -> dict[str, Any]:
        """Everything the worker process needs to produce this model's row."""

        return {
            "index": index,
            "model_label": model_label,
            "dataset_label": dataset_label,
            "texture": texture_label,
            "shot_mode": shot_mode,
            "device": device,
            "include_profiling": include_profiling,
            "include_resolution_sweep": include_resolution_sweep,
            "cross_domain_dataset_label": cross_domain_dataset_label,
            "run_log_path": run_log_path,
            "lang": lang,
        }

    # Model labels currently on a GPU, by index, for the status line below.
    running: dict[int, str] = {}

    def evaluate_model(
        index: int, model_label: str, device: str,
    ) -> tuple[int, str, dict[str, Any] | None, list[str], list[str], int | None]:
        """Score one model in its own process.

        Isolation is the point (see `application.benchmark_worker`): a worker
        that dies — tracing corruption, a poisoned CUDA context, an out-of-memory
        kill — costs its own row and nothing else, instead of every model that
        would have run after it in a shared process.
        """

        running[index] = model_label.split(" · ")[0]
        try:
            report = _run_model_worker(model_payload(index, model_label, device))
        finally:
            running.pop(index, None)

        if report.get("error"):
            error = str(report["error"])
            _record_worker_failure(model_label, error, report.get("traceback"))
            return (
                index, model_label, None,
                [f"{model_label}: {error}{_poisoned_cuda_hint(error)}"], [], None,
            )
        return (
            index, model_label, report.get("row"),
            list(report.get("warnings") or []), list(report.get("notes") or []),
            report.get("count"),
        )

    scheduled = list(enumerate(model_labels, start=1))
    finished = 0

    def status_line(index: int, model_label: str) -> str:
        """What finished, and what is still on the GPUs.

        One model per process means the parent no longer watches a shared
        progress bar, so it reports the in-flight set itself: a 19-model run on
        7 GPUs spends most of its wall clock with several models running, and
        "12/19" alone cannot say whether the rest are queued, running or stuck.
        """

        if len(devices) == 1:
            return tr(lang, "bench_progress", index=index, total=total, model=model_label)
        # `dict(running)` first: worker threads add and remove entries while this
        # runs, and iterating the live view can raise "dictionary changed size
        # during iteration".
        names = "、".join(dict.fromkeys(dict(running).values()))
        return tr(
            lang, "bench_progress_running", done=finished, total=total,
            running=names or tr(lang, "value_none"),
        )

    if len(devices) == 1:
        completed = (
            evaluate_model(index, model_label, devices[0])
            for index, model_label in scheduled
        )
    else:
        def parallel_completed():
            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                remaining = iter(scheduled)
                futures = {}
                for device in devices:
                    try:
                        index, model_label = next(remaining)
                    except StopIteration:
                        break
                    futures[executor.submit(evaluate_model, index, model_label, device)] = device

                while futures:
                    future = next(as_completed(futures))
                    device = futures.pop(future)
                    yield future.result()
                    try:
                        index, model_label = next(remaining)
                    except StopIteration:
                        continue
                    futures[executor.submit(evaluate_model, index, model_label, device)] = device

        completed = parallel_completed()

    for index, model_label, row, warnings, model_notes, count in completed:
        if row is not None:
            rows.append(row)
        errors.extend(warnings)
        notes.extend(model_notes)
        if sample_count is None and count is not None:
            sample_count = count
        finished += 1
        yield _render(
            rows, sample_count, shot_mode, errors, status_line(index, model_label), lang,
            technical_weight, overhead_weight, notes,
        )

    yield _render(
        rows, sample_count, shot_mode, errors, lang=lang, technical_weight=technical_weight,
        overhead_weight=overhead_weight, notes=notes,
    )


def _render(
    rows: list[dict[str, Any]],
    sample_count: int | None,
    shot_mode: str,
    errors: list[str],
    status: str | None = None,
    lang: str = DEFAULT_LANGUAGE,
    technical_weight: float = 0.5,
    overhead_weight: float = 0.5,
    notes: list[str] | None = None,
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
    if notes:
        # Metrics this model cannot produce (fixed-shape export, no
        # profiler-compatible export) are information, not failures — the
        # panel used to print them under the same warning triangle as a crash.
        status += " ℹ️ " + "; ".join(notes)
    return columns, table, status, scored


def _display_value(value: Any, column: str, score_columns: list[str]) -> Any:
    if column in score_columns and isinstance(value, (int, float)):
        return round(value, 1)
    return value


def _slug(label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in label).strip("-")
