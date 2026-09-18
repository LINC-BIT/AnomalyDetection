import json
import sys

import pytest

from fabric_defect_hub.training_runs import BatchRunTracker, run_parallel_batch


def test_batch_tracker_skips_completed_models_when_resuming(tmp_path):
    models = [{"key": "yolo", "backend": "ultralytics", "variant": "yolov8n", "config": "x.yaml"}]
    tracker = BatchRunTracker(tmp_path, "run-1", models)
    tracker.begin("yolo")
    tracker.finish("yolo", succeeded=True, detail="published/yolo.pt")

    resumed = BatchRunTracker(tmp_path, "run-1", models, resume=True)
    assert resumed.should_run("yolo") is False
    assert json.loads((tmp_path / "run-1" / "state.json").read_text())["models"]["yolo"]["status"] == "succeeded"


def test_batch_tracker_retries_interrupted_model(tmp_path):
    models = [{"key": "faster", "backend": "torchvision", "variant": "fasterrcnn", "config": "x.yaml"}]
    tracker = BatchRunTracker(tmp_path, "run-2", models)
    tracker.begin("faster")
    tracker.interrupt("faster")

    resumed = BatchRunTracker(tmp_path, "run-2", models, resume=True)
    assert resumed.should_run("faster") is True


# --------------------------------------------------------------------------- #
# Parallel batch: one child process and one device per model
# --------------------------------------------------------------------------- #
PLAN = [
    {"key": key, "backend": "fake", "variant": "fake", "config": "fake.yaml"}
    for key in ("alpha", "beta", "gamma")
]


def _write_report(report_path, published):
    """A stand-in `adh train` child: prints a marker, writes its JSON report."""

    script = (
        "import json, pathlib, sys;"
        f"print('training {published}', flush=True);"
        f"pathlib.Path(sys.argv[1]).write_text(json.dumps({{'published_path': '{published}',"
        " 'metrics': {'image_auroc': 0.5}}))"
    )
    return [sys.executable, "-c", script, report_path]


def _model(key):
    return type("M", (), {"key": key})()


def test_parallel_batch_trains_models_concurrently_and_records_devices(tmp_path):
    """One process and one device per model; the parent is the only place the
    physical card is recorded, because a pinned child only ever sees `cuda:0`."""

    tracker = BatchRunTracker(tmp_path, "batch-1", PLAN)
    results = run_parallel_batch(
        [_model(key) for key in ("alpha", "beta", "gamma")],
        tracker,
        jobs=2,
        devices=["cuda:0", "cuda:1"],
        command_for=lambda model, report_path, device: _write_report(report_path, f"p-{model.key}"),
        stream=lambda *args, **kwargs: None,
    )

    by_key = {result["model"]: result for result in results}
    assert set(by_key) == {"alpha", "beta", "gamma"}
    assert all(result["status"] == "succeeded" for result in results)
    assert by_key["alpha"]["published_path"] == "p-alpha"
    assert by_key["alpha"]["metrics"] == {"image_auroc": 0.5}

    state = json.loads((tmp_path / "batch-1" / "state.json").read_text())
    assert state["models"]["alpha"]["status"] == "succeeded"
    # Round-robin over the devices the parent assigned, in submission order.
    assert state["models"]["alpha"]["device"] == "cuda:0"
    assert state["models"]["beta"]["device"] == "cuda:1"
    assert state["models"]["gamma"]["device"] == "cuda:0"
    # Each model keeps its own log, with the child's own output in it.
    log = (tmp_path / "batch-1" / "logs" / "alpha.log").read_text()
    assert "training p-alpha" in log and "cuda:0" in log


def test_one_failing_model_does_not_take_the_batch_down(tmp_path):
    """The reason for a process per model: a backend that dies — an OOM kill, an
    illegal CUDA access — costs its own row, not the whole batch."""

    tracker = BatchRunTracker(tmp_path, "batch-2", PLAN)
    results = run_parallel_batch(
        [_model(key) for key in ("alpha", "beta", "gamma")],
        tracker,
        jobs=3,
        devices=["cpu"],
        command_for=lambda model, report_path, device: (
            [sys.executable, "-c", "import sys; sys.exit(3)"]
            if model.key == "beta"
            else _write_report(report_path, f"p-{model.key}")
        ),
        stream=lambda *args, **kwargs: None,
    )

    by_key = {result["model"]: result for result in results}
    assert by_key["beta"]["status"] == "failed"
    assert "exit code 3" in by_key["beta"]["detail"]
    assert by_key["alpha"]["status"] == "succeeded"
    assert by_key["gamma"]["status"] == "succeeded"

    state = json.loads((tmp_path / "batch-2" / "state.json").read_text())
    assert state["models"]["beta"]["status"] == "failed"
    assert state["models"]["alpha"]["status"] == "succeeded"


def test_a_model_that_already_succeeded_is_skipped_on_resume(tmp_path):
    """`--resume` has to mean the same thing in the parallel path: never retrain
    a model this batch already finished."""

    tracker = BatchRunTracker(tmp_path, "batch-3", PLAN)
    tracker.begin("alpha")
    tracker.finish("alpha", succeeded=True, detail="published")

    spawned: list[str] = []

    def command_for(model, report_path, device):
        spawned.append(model.key)
        return _write_report(report_path, f"p-{model.key}")

    results = run_parallel_batch(
        [_model(key) for key in ("alpha", "beta")],
        tracker,
        jobs=2,
        devices=["cpu"],
        command_for=command_for,
        stream=lambda *args, **kwargs: None,
    )

    by_key = {result["model"]: result for result in results}
    assert by_key["alpha"]["status"] == "skipped"
    assert spawned == ["beta"]


def test_available_devices_follow_what_the_host_has(monkeypatch):
    """`--jobs` defaults to this list, so an 8-GPU box fans out, and a laptop or
    a CPU host stays sequential without the caller having to know which it is."""

    import sys
    from types import SimpleNamespace

    from fabric_defect_hub import runtime_device

    def fake_torch(*, cuda=False, count=0, mps=False):
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: count),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        )

    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=True, count=4))
    assert runtime_device.available_torch_devices() == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps=True))
    assert runtime_device.available_torch_devices() == ["mps"]

    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    assert runtime_device.available_torch_devices() == ["cpu"]
