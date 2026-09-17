"""Mutual exclusion between model *tracing* and model *execution*.

`torch.export` — and the `torch.fx.symbolic_trace` machinery underneath it —
patches `nn.Module.__call__` **process-wide** for the duration of a trace. In a
benchmark that runs one worker per GPU (`application/benchmark.py`) that has two
consequences:

* a plain forward running in *another* worker gets captured by the tracer and
  registers its modules in that tracer's namespace, so the two disagree about a
  generated name — `AssertionError: Unexpected key relu@1, expected relu` — and
  the model that was merely predicting is the one reported as failed
  (Mask R-CNN profiling vs DeepLabV3+ predicting, 2026-09-17);
* two traces at once corrupt the same state.

Tracing is therefore a *writer* and everything that runs a model is a *reader*:
readers stay concurrent (that is what makes one worker per GPU worth it),
writers are serialized and exclude readers. Writers take priority, so a stream
of evaluations cannot starve an export.

Two limits worth stating rather than discovering later: the guard is not
reentrant (a writer taken inside a reader would wait for itself), and it only
covers the hazard this project can actually observe — it is not a claim that
the frameworks are thread-safe in general.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Iterator

__all__ = ["model_execution", "model_tracing"]


class _ModelExecutionGuard:
    """Writer-preferring readers/writer lock over model execution."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextlib.contextmanager
    def reading(self) -> Iterator[None]:
        with self._condition:
            # A writer that is already waiting goes first: otherwise a steady
            # supply of evaluations could keep an export out forever.
            while self._writer or self._writers_waiting:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if not self._readers:
                    self._condition.notify_all()

    @contextlib.contextmanager
    def writing(self) -> Iterator[None]:
        with self._condition:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
            finally:
                self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


_GUARD = _ModelExecutionGuard()


def model_execution() -> contextlib.AbstractContextManager[None]:
    """Marks a section that runs a model's forward pass.

    Overlaps freely with other `model_execution()` sections; excludes
    `model_tracing()`.
    """

    return _GUARD.reading()


def model_tracing() -> contextlib.AbstractContextManager[None]:
    """Marks a section that traces or exports a model.

    Runs alone: no forward pass and no other trace may be in flight, because
    the tracer patches module dispatch for the whole process.
    """

    return _GUARD.writing()
