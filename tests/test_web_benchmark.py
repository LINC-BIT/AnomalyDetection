"""Tests for `web/benchmark.py`'s Gradio-facing leaderboard engine: the
opt-in profiling pass, run-log persistence, and composite scoring added on
top of the plain accuracy leaderboard.

Uses its own uniquely-named fake dataset/model registration
("*-webbench" suffix), mirroring `test_benchmark.py`'s pattern for the same
`core.registry` duplicate-name reason explained there -- registered once at
module import time (not per test), since `register_model` raises on a
second registration of the same name.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from fabric_defect_hub.core.registry import register_dataset, register_model
from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.models.base import ExportedArtifact, ModelAdapter, ModelCapabilities
from fabric_defect_hub.application import benchmark as web_benchmark

MODEL_LABEL = "Fake Model"


@register_dataset("fake-fabric-webbench")
class _FakeWebBenchDataset(DatasetAdapter):
    name = "fake-fabric-webbench"

    def load_samples(self) -> list[Sample]:
        return [
            Sample(
                id=f"sample-{i:04d}", image_path=f"{self.root}/{i:04d}.jpg", task="anomaly",
                annotations=Annotations(is_anomalous=bool(i % 2)),
            )
            for i in range(4)
        ]


class _TinyModule(nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2, 3))


@register_model("fake-backend-webbench")
class _FakeWebBenchModel(ModelAdapter):
    name = "fake-backend-webbench-model"
    backend = "fake-backend-webbench"

    def capabilities(self):
        return ModelCapabilities(
            tasks=("anomaly",),
            prediction_fields=("anomaly_score",),
            export_targets=("torchscript",),
        )

    def train(self, config):
        return None

    def predict(self, samples, artifact=None, output_dir=None, config=None):
        return [Prediction(sample_id=s.id, anomaly_score=0.9) for s in samples]

    def raw_module(self):
        # A real Conv2d (unlike `_TinyModule`'s parameter-free `.mean()`)
        # so `profiling.flops.compute_model_flops`'s hook-based counter has
        # something nonzero to count -- see test_run_benchmark_with_profiling
        # _adds_flops_and_lmei.
        return nn.Conv2d(3, 4, 3, bias=False)

    def export(self, artifact, target, config=None):
        assert target == "torchscript"
        fd, path = tempfile.mkstemp(suffix=f".{target}")
        os.close(fd)
        torch.jit.save(torch.jit.script(_TinyModule()), path)
        return ExportedArtifact(path=path, target=target)


@register_dataset("fake-fabric-webbench-target")
class _FakeWebBenchTargetDataset(DatasetAdapter):
    """A second, differently-named domain for cross-domain-degradation
    tests -- same shape as `_FakeWebBenchDataset`, registered separately
    since `register_dataset` rejects a second registration of the same
    name (see module docstring)."""

    name = "fake-fabric-webbench-target"

    def load_samples(self) -> list[Sample]:
        return [
            Sample(
                id=f"target-{i:04d}", image_path=f"{self.root}/{i:04d}.jpg", task="anomaly",
                annotations=Annotations(is_anomalous=bool(i % 2)),
            )
            for i in range(4)
        ]


def _install_fake_catalog(monkeypatch, tmp_path):
    # The UI reads "which tasks can this dataset be scored on" from
    # `core.dataset_capabilities`, not from its own catalog, so a fake
    # dataset has to declare capabilities exactly like a real one does.
    # That is the point of the arrangement -- the fixture models the
    # production requirement instead of restating the answer.
    from fabric_defect_hub.core.dataset_capabilities import (
        all_capabilities,
        register_capabilities,
    )

    already = all_capabilities()
    for name in ("fake-fabric-webbench", "fake-fabric-webbench-target"):
        # `register_capabilities` refuses a re-registration (a real dataset
        # declaring itself twice is a bug); this fixture runs once per test.
        if name not in already:
            register_capabilities(name, default_root=str(tmp_path), roles=set(), tasks=("anomaly",))

    dataset_catalog = {
        "Fake Dataset": {
            "name": "fake-fabric-webbench",
            "slice_kwarg": None,
        },
        "Fake Target Dataset": {
            "name": "fake-fabric-webbench-target",
            "slice_kwarg": None,
        },
    }
    model_catalog = {
        MODEL_LABEL: {
            "backend": "fake-backend-webbench",
            "name": "fake-backend-webbench-model",
            "checkpoint": str(tmp_path / "fake.ckpt"),
            "task": "anomaly",
            "metadata": {},
        },
    }
    monkeypatch.setattr(web_benchmark, "DATASET_CATALOG", dataset_catalog)
    monkeypatch.setattr(web_benchmark, "MODEL_CATALOG", model_catalog)
    monkeypatch.setattr(web_benchmark, "default_dataset_root", lambda label: str(tmp_path))
    monkeypatch.setattr(web_benchmark, "_detect_devices", lambda: ["cpu"])


@pytest.fixture(autouse=True)
def _worker_runs_in_process(request, monkeypatch):
    """Score each model here, not in a spawned interpreter.

    `run_benchmark` gives every model its own process so a crash (tracing
    corruption, a poisoned CUDA context, an OOM kill) can only cost that model.
    These tests register fake backends that a fresh interpreter would not know
    about, so the spawn is replaced by a direct call: the payload building and
    result merging the parent owns are still exercised, and the worker launcher
    itself is covered by the tests marked `spawn_worker`.
    """

    if "spawn_worker" in request.keywords:
        return
    from fabric_defect_hub.application import benchmark_worker

    monkeypatch.setattr(web_benchmark, "_run_model_worker", benchmark_worker.evaluate)


def test_run_benchmark_basic_leaderboard_has_no_score_columns_without_metrics(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL], run_log_path=None,
    )

    assert "composite_score" in columns
    assert rows[0][columns.index("model")] == MODEL_LABEL
    # No profiling metrics were requested, so overhead_score has nothing to
    # average and composite falls back to the technical (accuracy) score.
    assert rows[0][columns.index("composite_score")] == rows[0][columns.index("technical_score")]
    # The chart-facing payload carries the same run as name-keyed dicts.
    assert scored[0]["model"] == MODEL_LABEL
    assert "composite_score" in scored[0]


def test_run_benchmark_with_profiling_adds_overhead_metrics_and_scores(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        include_profiling=True, run_log_path=None,
    )

    assert "fps" in columns
    assert "latency_ms_mean" in columns
    row = rows[0]
    assert row[columns.index("fps")] > 0
    assert row[columns.index("overhead_score")] != ""
    assert row[columns.index("composite_score")] != ""


def test_run_benchmark_keeps_accuracy_and_measures_natively_when_export_is_unsupported(monkeypatch, tmp_path):
    """A backend whose export a profiler cannot drive — this fake declares ONNX,
    the same shape of problem as WinCLIP's unsupported ONNX operator — still gets
    overhead numbers: the adapter is measured directly, and the accuracy row is
    untouched either way."""

    _install_fake_catalog(monkeypatch, tmp_path)
    monkeypatch.setattr(
        _FakeWebBenchModel,
        "capabilities",
        lambda self: ModelCapabilities(
            tasks=("anomaly",), prediction_fields=("anomaly_score",), export_targets=("onnx",),
        ),
    )

    *_, (columns, rows, status, _) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        include_profiling=True, run_log_path=None,
    )

    assert rows[0][columns.index("model")] == MODEL_LABEL
    assert "image_auroc" in columns
    # Either a profiler drove the ONNX export or the native path measured the
    # adapter; what must not happen is losing the columns to a bare warning.
    assert "profiling skipped" not in status


def test_run_benchmark_with_profiling_adds_flops_and_allocator_lmei(monkeypatch, tmp_path):
    pytest.importorskip("thop")
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        include_profiling=True, run_log_path=None,
    )

    assert "flops_g" in columns
    assert "params_m" in columns
    row = rows[0]
    assert row[columns.index("flops_g")] >= 0
    assert row[columns.index("params_m")] >= 0
    # LMEI is defined using accelerator allocator memory.  CPU profiling
    # reports process RSS, which is useful telemetry but is not a comparable
    # substitute for VRAM and therefore must not produce a misleading score.
    memory_kind = row[columns.index("memory_measurement_kind")]
    if memory_kind == "device_allocator":
        assert "lmei" in columns
        assert row[columns.index("lmei")] != ""
    else:
        assert "lmei" not in columns


def test_run_benchmark_with_resolution_sweep_adds_slope_columns(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        include_resolution_sweep=True, run_log_path=None,
    )

    assert "resolution_slope_beta" in columns
    assert "resolution_slope_alpha" in columns
    row = rows[0]
    assert isinstance(row[columns.index("resolution_slope_beta")], float)


def test_run_benchmark_with_cross_domain_dataset_adds_degradation_column(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        cross_domain_dataset_label="Fake Target Dataset", run_log_path=None,
    )

    assert "cross_domain_delta_acc_pct" in columns
    row = rows[0]
    assert isinstance(row[columns.index("cross_domain_delta_acc_pct")], float)


def test_run_benchmark_cross_domain_skips_column_for_incompatible_target(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        cross_domain_dataset_label="Nonexistent Dataset", run_log_path=None,
    )

    assert "cross_domain_delta_acc_pct" not in columns
    assert rows  # the row itself still succeeds, just without the extra column


def test_run_benchmark_appends_to_run_log(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)
    log_path = tmp_path / "log.jsonl"

    list(web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL], run_log_path=str(log_path),
    ))

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["model"]["backend"] == "fake-backend-webbench"


def test_run_benchmark_custom_preset_uses_custom_weight(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        score_preset="custom", custom_technical_weight=0.9, run_log_path=None,
    )

    assert rows[0][columns.index("composite_score")] != ""


class _SizedProbeModel(_FakeWebBenchModel):
    """`_FakeWebBenchModel`, declaring what side length it can be probed at."""

    def __init__(self, probe_input_size):
        self._probe_input_size = probe_input_size

    def capabilities(self):
        return ModelCapabilities(
            tasks=("anomaly",),
            prediction_fields=("anomaly_score",),
            probe_input_size=self._probe_input_size,
        )


def test_flops_probe_uses_the_size_the_model_declares(monkeypatch):
    """The shared 640x640 probe is wrong for every patch-based backbone: the
    declared size is what keeps a Dinomaly/MoECLIP row from losing its
    FLOPs/LMEI columns to a shape error that reads like a model defect.
    """

    seen: dict[str, tuple[int, int]] = {}

    def fake_flops(module, input_size, input_style="batched", device="cpu", batch_size=1):
        seen["input_size"] = tuple(input_size)
        return 1.25

    monkeypatch.setattr("fabric_defect_hub.profiling.flops.compute_model_flops", fake_flops)
    monkeypatch.setattr(
        "fabric_defect_hub.model_statistics.parameter_counts",
        lambda module: {"parameter_count": 2_000_000},
    )
    monkeypatch.setattr(
        "fabric_defect_hub.evaluation.lmei_profiler.calculate_lmei", lambda **kwargs: 0.75
    )

    declared = web_benchmark._flops_and_lmei(
        _SizedProbeModel((448, 448)), {}, "cpu",
        fps=10.0, vram_mb=100.0, memory_kind="device_allocator",
    )
    assert seen["input_size"] == (448, 448)
    assert declared == {"flops_g": 1.25, "params_m": 2.0, "lmei": 0.75}

    web_benchmark._flops_and_lmei(
        _SizedProbeModel(None), {}, "cpu",
        fps=10.0, vram_mb=100.0, memory_kind="device_allocator",
    )
    assert seen["input_size"] == (640, 640)


def test_benchmark_failure_log_keeps_the_stack_the_panel_cannot_show(monkeypatch, tmp_path):
    """The status line says `IndexError: pop from empty list`, which names no
    frame; the log has to carry enough to act on.
    """

    log = tmp_path / "nested" / "failures.log"
    monkeypatch.setattr(web_benchmark, "BENCHMARK_FAILURE_LOG", log)

    try:
        raise IndexError("pop from empty list")
    except IndexError as exc:
        web_benchmark._record_benchmark_failure("Some Model", exc)

    text = log.read_text(encoding="utf-8")
    assert "Some Model" in text
    assert "Traceback (most recent call last)" in text
    assert "IndexError: pop from empty list" in text


def test_benchmark_failure_logging_never_breaks_the_batch(monkeypatch):
    """It runs inside the `except` whose whole job is to keep one model's
    failure from taking the batch down, so an unwritable path must be inert.
    """

    monkeypatch.setattr(
        web_benchmark, "BENCHMARK_FAILURE_LOG", Path("/proc/definitely-not-writable/failures.log")
    )
    web_benchmark._record_benchmark_failure("Some Model", ValueError("boom"))


def test_every_model_is_scored_by_its_own_worker(monkeypatch, tmp_path):
    """One model, one process — the parent hands over a payload and merges the
    report back, and never loads a model itself."""

    _install_fake_catalog(monkeypatch, tmp_path)
    monkeypatch.setattr(web_benchmark, "_detect_devices", lambda: ["cuda:3"])
    seen: list[dict] = []

    def fake_worker(payload):
        seen.append(payload)
        return {
            "row": {"model": payload["model_label"], "runtime_s": 1.0},
            "warnings": [], "count": 4, "error": None, "traceback": None,
        }

    monkeypatch.setattr(web_benchmark, "_run_model_worker", fake_worker)
    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL], run_log_path=None,
    )

    assert len(seen) == 1
    payload = seen[0]
    assert payload["model_label"] == MODEL_LABEL
    assert payload["dataset_label"] == "Fake Dataset"
    assert payload["shot_mode"] == "Full-shot"
    assert payload["device"] == "cuda:3"
    assert payload["run_log_path"] is None
    assert any(MODEL_LABEL in str(cell) for cell in rows[0])


def test_a_worker_error_reaches_the_panel_and_the_failure_log(monkeypatch, tmp_path):
    """A crash in another process still has to say what went wrong, and where
    the stack is: the worker sends its own traceback back for the log."""

    _install_fake_catalog(monkeypatch, tmp_path)
    log = tmp_path / "failures.log"
    monkeypatch.setattr(web_benchmark, "BENCHMARK_FAILURE_LOG", log)
    monkeypatch.setattr(
        web_benchmark, "_run_model_worker",
        lambda payload: {
            "row": None, "warnings": [], "count": None,
            "error": "RuntimeError: boom",
            "traceback": "Traceback (most recent call last):\nRuntimeError: boom\n",
        },
    )

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL], run_log_path=None,
    )

    assert rows == []
    assert "RuntimeError: boom" in status
    assert "RuntimeError: boom" in log.read_text(encoding="utf-8")


def test_parallel_status_line_names_the_models_still_running(monkeypatch, tmp_path):
    """"12/19 scored" cannot say whether the rest are queued or running, so the
    line has to name what is on the GPUs right now."""

    import time as _time

    _install_fake_catalog(monkeypatch, tmp_path)
    monkeypatch.setattr(web_benchmark, "_detect_devices", lambda: ["cuda:0", "cuda:1"])

    def slow_worker(payload):
        _time.sleep(0.05)
        return {
            "row": {"model": payload["model_label"], "runtime_s": 0.1},
            "warnings": [], "count": 4, "error": None, "traceback": None,
        }

    monkeypatch.setattr(web_benchmark, "_run_model_worker", slow_worker)
    statuses = [
        status
        for _columns, _rows, status, _scored in web_benchmark.run_benchmark(
            "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL] * 3, run_log_path=None,
        )
    ]

    assert any("on the GPUs now" in status for status in statuses), statuses


class _FakeProcess:
    """Stands in for a worker process: `wait()` behaves, nothing else matters."""

    def __init__(self, returncode=-9, raises_timeout=False):
        self.returncode = returncode
        self._raises_timeout = raises_timeout

    def wait(self, timeout=None):
        if self._raises_timeout:
            raise web_benchmark.subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
        return self.returncode

    def poll(self):
        return self.returncode


@pytest.mark.spawn_worker
def test_worker_that_dies_without_a_report_is_reported_not_swallowed(monkeypatch):
    """An OOM kill, or quitting the UI mid-run, leaves no report file; the
    parent has to say so instead of quietly producing no row."""

    monkeypatch.setattr(
        web_benchmark, "_spawn_worker", lambda payload, results_path, env: _FakeProcess(-9)
    )
    report = web_benchmark._run_model_worker({"model_label": "Some Model"})

    assert report["row"] is None
    assert "without writing a report" in report["error"]
    assert "-9" in report["error"]


@pytest.mark.spawn_worker
def test_worker_that_hangs_is_given_up_on(monkeypatch):
    """A wedged worker holds a GPU the rest of the batch could use."""

    monkeypatch.setattr(
        web_benchmark, "_spawn_worker",
        lambda payload, results_path, env: _FakeProcess(raises_timeout=True),
    )
    report = web_benchmark._run_model_worker({"model_label": "Some Model"})

    assert "timed out" in report["error"]


@pytest.mark.spawn_worker
def test_spawned_worker_gets_the_payload_on_stdin(monkeypatch, tmp_path):
    """The payload travels on stdin and the child's stdout stays inherited, so
    per-image progress still reaches the terminal the UI was started from."""

    recorded: dict = {}

    class _RecordingProcess(_FakeProcess):
        def __init__(self):
            super().__init__(returncode=0)
            self.stdin = SimpleNamespace(write=lambda text: recorded.setdefault("payload", text),
                                         close=lambda: recorded.setdefault("closed", True))

    def fake_popen(command, **kwargs):
        recorded["command"] = command
        recorded["kwargs"] = kwargs
        return _RecordingProcess()

    monkeypatch.setattr(web_benchmark.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(web_benchmark, "_LIVE_WORKERS", set())
    process = web_benchmark._spawn_worker(
        {"model_label": "M"}, tmp_path / "result.json", {"PATH": "/usr/bin"}
    )
    try:
        assert recorded["command"][1:] == ["-m", "fabric_defect_hub.application.benchmark_worker"]
        assert "stdout" not in recorded["kwargs"] and "stderr" not in recorded["kwargs"]
        assert json.loads(recorded["payload"])["model_label"] == "M"
        assert recorded["closed"] is True
        if sys.platform.startswith("linux"):
            assert recorded["kwargs"]["preexec_fn"] is web_benchmark._child_setup
        assert process in web_benchmark._LIVE_WORKERS
    finally:
        web_benchmark._LIVE_WORKERS.discard(process)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="uses /bin/sleep")
def test_terminating_the_batch_stops_in_flight_workers():
    """A stranded worker keeps a GPU until it finishes on its own."""

    sleeper = web_benchmark.subprocess.Popen(["sleep", "60"])
    with web_benchmark._LIVE_WORKERS_LOCK:
        web_benchmark._LIVE_WORKERS.add(sleeper)
    try:
        stopped = web_benchmark.terminate_benchmark_workers(grace_seconds=5.0)

        assert stopped == 1
        assert sleeper.poll() is not None
        assert sleeper not in web_benchmark._LIVE_WORKERS
    finally:
        sleeper.kill()
        with web_benchmark._LIVE_WORKERS_LOCK:
            web_benchmark._LIVE_WORKERS.discard(sleeper)


def test_fast_exit_stops_workers_before_uvicorn_shuts_down(monkeypatch):
    """The hang being fixed here: uvicorn's graceful shutdown waits for the
    benchmark request, and that request waits for a model that can take
    minutes — so the workers have to die first."""

    order: list[str] = []
    fake_uvicorn = SimpleNamespace(Server=type("Server", (), {"handle_exit": lambda self, sig, frame: order.append("uvicorn")}))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(web_benchmark, "_fast_exit_installed", False)
    monkeypatch.setattr(web_benchmark, "terminate_benchmark_workers",
                        lambda *args, **kwargs: order.append("workers") or 2)

    assert web_benchmark.install_fast_exit() is True
    fake_uvicorn.Server.handle_exit(object(), web_benchmark.signal.SIGINT, None)

    assert order == ["workers", "uvicorn"]


def test_fast_exit_wraps_the_handler_uvicorn_actually_registers(monkeypatch):
    """The wrapper has to land on the real `uvicorn.Server.handle_exit`, because
    that is the callable uvicorn installs as its SIGINT handler."""

    uvicorn = pytest.importorskip("uvicorn")
    original = uvicorn.Server.handle_exit
    monkeypatch.setattr(web_benchmark, "_fast_exit_installed", False)
    monkeypatch.setattr(web_benchmark, "terminate_benchmark_workers", lambda *a, **k: 0)
    monkeypatch.setattr(uvicorn.Server, "handle_exit", original)
    try:
        assert web_benchmark.install_fast_exit() is True
        assert uvicorn.Server.handle_exit is not original

        server = SimpleNamespace(should_exit=False, force_exit=False, _captured_signals=[])
        uvicorn.Server.handle_exit(server, web_benchmark.signal.SIGINT, None)

        # Ctrl+C still does what uvicorn does with it (start the graceful path).
        assert server.should_exit is True
        assert server._captured_signals == [web_benchmark.signal.SIGINT]
    finally:
        uvicorn.Server.handle_exit = original


def test_device_detection_covers_cuda_mps_and_cpu(monkeypatch):
    """One worker per accelerator when there are several, one device otherwise:
    the same scheduling path has to serve an 8-GPU box, an Apple laptop and a
    CPU-only host."""

    monkeypatch.setattr(web_benchmark, "_idle_cuda_device_indices", lambda count: [])

    def fake_torch(*, cuda_available=False, count=0, mps_available=False):
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda_available, device_count=lambda: count),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
        )

    assert web_benchmark._detect_devices(fake_torch(cuda_available=True, count=3)) == [
        "cuda:0", "cuda:1", "cuda:2",
    ]
    assert web_benchmark._detect_devices(fake_torch(cuda_available=True, count=1)) == ["cuda:0"]
    assert web_benchmark._detect_devices(fake_torch(mps_available=True)) == ["mps"]
    assert web_benchmark._detect_devices(fake_torch()) == ["cpu"]


def test_profiler_prefers_the_export_that_detection_models_can_actually_produce():
    """`torch.export` rejects Faster/Cascade/Mask R-CNN's heads, so preferring it
    cost those rows their profiling and their sweep. `torchscript` goes first."""

    class _BothTargets(_FakeWebBenchModel):
        def capabilities(self):
            return ModelCapabilities(
                tasks=("anomaly",), prediction_fields=("anomaly_score",),
                export_targets=("torchscript", "exported_program"),
            )

    setup = web_benchmark._profile_setup(_BothTargets(), "cpu")

    assert setup is not None
    assert setup[2] == "torchscript"


def test_profiling_falls_back_to_native_when_the_export_fails(monkeypatch):
    """WinCLIP's ONNX export hits an operator the exporter cannot represent;
    measuring the adapter directly beats dropping the row's overhead columns."""

    class _Unexportable(_FakeWebBenchModel):
        def export(self, artifact, target, config=None):
            raise RuntimeError("operators not yet supported by torch.export")

    monkeypatch.setattr(
        web_benchmark, "_native_profile_model",
        lambda model, artifact, samples, device: {"profiling_mode": "native", "fps": 12.5},
    )

    metrics = web_benchmark._profile_model(_Unexportable(), None, "cpu", samples=[])

    assert metrics == {"profiling_mode": "native", "fps": 12.5}


def test_a_fixed_shape_export_is_not_applicable_rather_than_failed(monkeypatch):
    """A model with no profiler-compatible export, or a fixed-shape one, cannot
    have a resolution slope — that is information, not a failure."""

    monkeypatch.setattr(web_benchmark, "_profile_setup", lambda model, device: None)
    with pytest.raises(web_benchmark.MetricNotApplicable):
        web_benchmark._resolution_sweep(_FakeWebBenchModel(), None, "cpu")

    class _Export:
        path = "/tmp/does-not-matter"
        metadata: dict = {}

    class _FixedShape(_FakeWebBenchModel):
        def export(self, artifact, target, config=None):
            return _Export()

    monkeypatch.setattr(
        web_benchmark, "_profile_setup",
        lambda model, device: (SimpleNamespace(profile=lambda *a, **k: {}), web_benchmark.ProfileConfig(
            device=device, engine="pytorch", precision="fp32", input_size=(64, 64),
            input_style="batched", warmup_runs=1, measured_runs=1, power_mode="disabled",
        ), "torchscript"),
    )
    monkeypatch.setattr("fabric_defect_hub.profiling.sweeps.resolution_scaling", lambda *a, **k: {})
    with pytest.raises(web_benchmark.MetricNotApplicable) as caught:
        web_benchmark._resolution_sweep(_FixedShape(), None, "cpu")
    assert "fixed-shape" in str(caught.value)


def test_a_metric_that_cannot_exist_is_a_note_not_a_warning(monkeypatch, tmp_path):
    """The panel used to print "not applicable" under the same ⚠️ as a crash."""

    from fabric_defect_hub.application import benchmark_worker

    _install_fake_catalog(monkeypatch, tmp_path)

    def not_applicable(*args, **kwargs):
        raise web_benchmark.MetricNotApplicable("its export is fixed-shape")

    monkeypatch.setattr(benchmark_worker, "_resolution_sweep", not_applicable)
    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        include_resolution_sweep=True, run_log_path=None,
    )

    assert rows, "the model is still scored"
    assert "ℹ️" in status and "fixed-shape" in status
    assert "⚠️" not in status
