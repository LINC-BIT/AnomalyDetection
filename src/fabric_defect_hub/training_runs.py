"""Persistent bookkeeping for multi-model training runs."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from fabric_defect_hub.core.processes import WorkerRegistry


def default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class BatchRunTracker:
    """Atomically persist batch status, events, and per-model log paths."""

    def __init__(
        self,
        root: str | Path,
        run_id: str,
        models: list[dict[str, str]],
        *,
        resume: bool = False,
    ) -> None:
        self.directory = Path(root) / run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logs_directory = self.directory / "logs"
        self.logs_directory.mkdir(exist_ok=True)
        self.state_path = self.directory / "state.json"
        self.events_path = self.directory / "events.jsonl"
        # Parallel batches update the state from several threads at once; a
        # read-modify-write of `state.json` is only atomic under this lock.
        self._lock = threading.Lock()

        if resume:
            if not self.state_path.is_file():
                raise FileNotFoundError(
                    f"cannot resume batch run {run_id!r}: missing {self.state_path}"
                )
            self.state = json.loads(self.state_path.read_text())
        else:
            if self.state_path.exists():
                raise FileExistsError(
                    f"batch run {run_id!r} already exists; pass --resume to continue it or choose --run-id"
                )
            self.state: dict[str, Any] = {
                "run_id": run_id,
                "created_at": _timestamp(),
                "updated_at": _timestamp(),
                "models": {
                    model["key"]: {
                        **model,
                        "status": "pending",
                        "attempts": 0,
                        "log_path": str(self.log_path(model["key"])),
                    }
                    for model in models
                },
            }
            self._write_state()
            self.record_event("batch_created", {"model_count": len(models)})

    def log_path(self, model_key: str) -> Path:
        return self.logs_directory / f"{model_key}.log"

    def should_run(self, model_key: str) -> bool:
        return self.state["models"][model_key]["status"] != "succeeded"

    def begin(self, model_key: str, device: str | None = None) -> None:
        with self._lock:
            model = self.state["models"][model_key]
            model["status"] = "running"
            model["attempts"] += 1
            model["started_at"] = _timestamp()
            if device is not None:
                # The batch scheduler pins each worker to a device; the child
                # only ever sees its own, so the parent is the one place the
                # physical card is recorded.
                model["device"] = device
            model.pop("finished_at", None)
            model.pop("detail", None)
            self._write_state()
            self.record_event("model_started", {"key": model_key, "attempt": model["attempts"], "device": device})

    def finish(self, model_key: str, *, succeeded: bool, detail: str) -> None:
        with self._lock:
            model = self.state["models"][model_key]
            model["status"] = "succeeded" if succeeded else "failed"
            model["finished_at"] = _timestamp()
            model["detail"] = detail
            self._write_state()
            self.record_event("model_finished", {"key": model_key, "status": model["status"], "detail": detail})

    def interrupt(self, model_key: str) -> None:
        with self._lock:
            model = self.state["models"][model_key]
            model["status"] = "interrupted"
            model["finished_at"] = _timestamp()
            model["detail"] = "interrupted by the parent process"
            self._write_state()
            self.record_event("model_interrupted", {"key": model_key})

    def record_event(self, event: str, payload: dict[str, Any]) -> None:
        entry = {"timestamp": _timestamp(), "event": event, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _write_state(self) -> None:
        self.state["updated_at"] = _timestamp()
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.state_path)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def run_parallel_batch(
    models: list[Any],
    tracker: BatchRunTracker,
    *,
    jobs: int,
    devices: list[str],
    command_for: Callable[[Any, str, str], list[str]],
    encode: Callable[[str], bytes] | None = None,
    stream: Callable[..., None] = print,
) -> list[dict[str, Any]]:
    """Train several models at once, one child process per model.

    Process isolation is the same argument as `application.benchmark`'s (see
    `core.processes`): a backend that dies — an out-of-memory kill, an illegal
    CUDA access — costs its own model instead of the whole batch, and every
    child owns a CUDA context that is pinned to one device, so two jobs never
    fight over the same card by accident.

    `command_for(model, report_path, device)` returns the argv for one child; it
    is expected to be the same call a single `adh train` makes (the CLI's
    `--json-report` is how the parent gets the published path and metrics back).
    `stream` receives each child's output line so the terminal keeps showing
    progress, and the same text goes to the model's log file.
    """

    registry = WorkerRegistry("train-all")
    registry.install_exit_hooks()
    results: list[dict[str, Any]] = []
    pending: list[Any] = []
    for model in models:
        if tracker.should_run(model.key):
            pending.append(model)
        else:
            stream(f"SKIP {model.key}: already succeeded in this batch")
            results.append({"model": model.key, "status": "skipped"})

    def run_one(index: int, model: Any) -> dict[str, Any]:
        device = devices[index % len(devices)]
        report_path = tracker.directory / f"{model.key}.report.json"
        report_path.unlink(missing_ok=True)
        tracker.begin(model.key, device=device)
        stream(f"\n{'=' * 70}\n>>> {model.key} on {device} ({jobs} model(s) at a time)\n{'=' * 70}")
        started = time.monotonic()
        with tracker.log_path(model.key).open("w", encoding="utf-8") as log:
            log.write(f"# model: {model.key}\n# device: {device}\n\n")
            process = registry.spawn(
                command_for(model, str(report_path), device),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                env=WorkerRegistry.device_environ(device),
            )
            assert process.stdout is not None  # noqa: S101 - guaranteed by stdout=PIPE
            for line in process.stdout:
                stream(line, end="")
                log.write(line)
            returncode = process.wait()
            registry.forget(process)
        elapsed = time.monotonic() - started
        report = _read_report(report_path)
        detail = _batch_detail(report, returncode)
        succeeded = returncode == 0 and report.get("error") is None
        if not succeeded:
            stream(f"FAIL {model.key} after {elapsed:.0f}s: {detail}")
            with tracker.log_path(model.key).open("a", encoding="utf-8") as log:
                log.write(f"\n{detail}\n")
            tracker.finish(model.key, succeeded=False, detail=detail)
            return {"model": model.key, "status": "failed", "detail": detail, "device": device}
        stream(f"OK   {model.key} in {elapsed:.0f}s -> {detail}")
        tracker.finish(model.key, succeeded=True, detail=detail)
        return {
            "model": model.key,
            "status": "succeeded",
            "device": device,
            "published_path": report.get("published_path"),
            "metrics": report.get("metrics") or {},
        }

    try:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
            futures = {
                executor.submit(run_one, index, model): model
                for index, model in enumerate(pending)
            }
            for future in as_completed(futures):
                results.append(future.result())
    except BaseException:
        # Ctrl+C, or anything else taking the batch down: kill the children
        # before the executor's exit joins their threads, or quitting waits on
        # whichever model is mid-epoch.
        stopped = registry.terminate_all()
        if stopped:
            stream(f"interrupted: stopped {stopped} training process(es)")
        raise
    return results


def _read_report(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _batch_detail(report: dict[str, Any], returncode: int) -> str:
    if report.get("error"):
        return str(report["error"])
    published = report.get("published_path") or ""
    if published:
        return str(published)
    return f"exit code {returncode}"
