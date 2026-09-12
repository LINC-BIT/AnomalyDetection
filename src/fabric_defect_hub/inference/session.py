"""Cross-platform lifecycle management for interactive model inference."""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.i18n import DEFAULT_LANGUAGE, tr
from fabric_defect_hub.loader import load_model
from fabric_defect_hub.models.base import Artifact, ModelAdapter


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested for a model outside the active session."""


@dataclass
class LoadedModel:
    model_id: str
    adapter: ModelAdapter
    artifact: Artifact
    task: str
    loaded_at: float
    model_memory_bytes: int | None


class InferenceSessionManager:
    """Own exactly one resident model and expose portable load/predict/unload calls."""

    def __init__(self, model_loader: Callable[..., ModelAdapter] = load_model):
        self._model_loader = model_loader
        self._active: LoadedModel | None = None
        self._lock = threading.RLock()

    def load(self, model_id: str, spec: dict[str, Any], artifact: Artifact) -> dict[str, Any]:
        """Load a selected artifact once, evicting any previously resident model."""

        with self._lock:
            if self._active is not None and self._active.model_id == model_id:
                return self.status()
            self._unload_active()
            started_at = time.perf_counter()
            adapter = self._model_loader(spec["backend"], spec["name"])
            self._load_artifact(adapter, artifact)
            _move_adapter_to_device(adapter, _runtime_memory()["device"])
            self._active = LoadedModel(
                model_id=model_id,
                adapter=adapter,
                artifact=artifact,
                task=str(spec["task"]),
                loaded_at=time.time(),
                model_memory_bytes=_model_memory_bytes(adapter),
            )
            status = self.status()
            status["load_time_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
            return status

    def unload(self) -> dict[str, Any]:
        """Release the active adapter and clear accelerator allocator caches."""

        with self._lock:
            self._unload_active()
            return self.status()

    def predict(self, model_id: str, samples: list[Sample], **kwargs) -> list[Prediction]:
        """Run inference only through the explicitly loaded model."""

        with self._lock:
            if self._active is None:
                raise ModelNotLoadedError("No model is loaded. Select a model and choose Load model first.")
            if self._active.model_id != model_id:
                raise ModelNotLoadedError(
                    f"{self._active.model_id!r} is loaded; load {model_id!r} before running inference."
                )
            return self._active.adapter.predict(samples, self._active.artifact, **kwargs)

    def capabilities(self, model_id: str):
        """What the resident model can produce (`ModelCapabilities`), or None
        when `model_id` isn't the loaded one.

        Lets a caller answer "is a pixel-level anomaly map even possible for
        this model" (`caps.fills("anomaly_map")`) without constructing a
        second adapter — the resident one already knows, and for the
        image-level-only models (`anomalib.presets.IMAGE_LEVEL_ONLY`) the
        answer is no.
        """

        with self._lock:
            if self._active is None or self._active.model_id != model_id:
                return None
            return self._active.adapter.capabilities()

    def status(self) -> dict[str, Any]:
        """Return JSON-safe resident-model and process/device memory metrics."""

        with self._lock:
            memory = _runtime_memory()
            active = self._active
            return {
                "loaded": active is not None,
                "model_id": active.model_id if active else None,
                "task": active.task if active else None,
                "model_memory_bytes": active.model_memory_bytes if active else 0,
                **memory,
            }

    def _unload_active(self) -> None:
        if self._active is not None:
            unload = getattr(self._active.adapter, "unload", None)
            if callable(unload):
                unload()
            self._active = None
        gc.collect()
        clear_accelerator_cache()

    @staticmethod
    def _load_artifact(adapter: ModelAdapter, artifact: Artifact) -> None:
        loader = getattr(adapter, "load_trained_model", None)
        if callable(loader):
            loader(artifact)
            return
        raise TypeError(f"{type(adapter).__name__} does not implement load_trained_model().")


def format_session_status(status: dict[str, Any], lang: str = DEFAULT_LANGUAGE) -> str:
    """Render portable runtime metrics for a lightweight UI status panel."""

    model = status["model_id"] or tr(lang, "value_none")
    model_memory = _format_mib(status.get("model_memory_bytes"), lang)
    process_memory = _format_mib(status.get("process_rss_bytes"), lang)
    lines = [
        tr(lang, "runtime_active_model", model=model),
        tr(lang, "runtime_device", device=status["device"]),
        tr(lang, "runtime_params", value=model_memory),
        tr(lang, "runtime_rss", value=process_memory),
    ]
    if status.get("cuda_allocated_bytes") is not None:
        lines.append(tr(
            lang, "runtime_cuda",
            alloc=_format_mib(status["cuda_allocated_bytes"], lang),
            reserved=_format_mib(status["cuda_reserved_bytes"], lang),
        ))
    if status.get("mps_allocated_bytes") is not None:
        lines.append(tr(lang, "runtime_mps", value=_format_mib(status["mps_allocated_bytes"], lang)))
    if "load_time_ms" in status:
        lines.append(tr(lang, "runtime_load_time", ms=status["load_time_ms"]))
    return "  \n".join(lines)


def _model_memory_bytes(adapter: ModelAdapter) -> int | None:
    model = getattr(adapter, "_model", None)
    if model is None:
        return None
    native = getattr(model, "model", model)
    parameters = getattr(native, "parameters", None)
    buffers = getattr(native, "buffers", None)
    if not callable(parameters) or not callable(buffers):
        return None
    try:
        values = list(parameters()) + list(buffers())
        return sum(value.numel() * value.element_size() for value in values)
    except (AttributeError, RuntimeError):
        return None


def _runtime_memory() -> dict[str, Any]:
    report: dict[str, Any] = {
        "device": "cpu",
        "process_rss_bytes": _process_rss_bytes(),
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "mps_allocated_bytes": None,
    }
    try:
        import torch
    except ImportError:
        return report
    if torch.cuda.is_available():
        report.update(
            device="cuda:0",
            cuda_allocated_bytes=int(torch.cuda.memory_allocated()),
            cuda_reserved_bytes=int(torch.cuda.memory_reserved()),
        )
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        report.update(device="mps", mps_allocated_bytes=int(torch.mps.current_allocated_memory()))
    return report


def _move_adapter_to_device(adapter: ModelAdapter, device: str) -> None:
    model = getattr(adapter, "_model", None)
    native = getattr(model, "model", model)
    move = getattr(native, "to", None)
    if callable(move):
        move(device)


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def clear_accelerator_cache() -> None:
    """Release cached CUDA/MPS allocator memory. Shared with the Benchmark
    tab's model-cycling loop (`web/benchmark.py`), which has the same
    load-one-model-at-a-time-then-free-it lifecycle as this session manager
    but doesn't go through it (it evaluates a batch, not a resident session)."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        return


def _format_mib(value: int | None, lang: str = DEFAULT_LANGUAGE) -> str:
    return tr(lang, "value_unavailable") if value is None else f"{value / (1024 * 1024):.1f} MiB"
