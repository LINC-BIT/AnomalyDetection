"""Persistent bookkeeping for multi-model training runs."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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

    def begin(self, model_key: str) -> None:
        model = self.state["models"][model_key]
        model["status"] = "running"
        model["attempts"] += 1
        model["started_at"] = _timestamp()
        model.pop("finished_at", None)
        model.pop("detail", None)
        self._write_state()
        self.record_event("model_started", {"key": model_key, "attempt": model["attempts"]})

    def finish(self, model_key: str, *, succeeded: bool, detail: str) -> None:
        model = self.state["models"][model_key]
        model["status"] = "succeeded" if succeeded else "failed"
        model["finished_at"] = _timestamp()
        model["detail"] = detail
        self._write_state()
        self.record_event("model_finished", {"key": model_key, "status": model["status"], "detail": detail})

    def interrupt(self, model_key: str) -> None:
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
