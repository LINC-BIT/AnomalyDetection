"""Tests for the tracing/execution guard (`core/execution.py`).

The guard exists because `torch.export` patches `nn.Module.__call__`
process-wide while it traces, so a benchmark worker that is merely predicting
can be captured by a sibling worker's export and fail with
`AssertionError: Unexpected key relu@1, expected relu` (observed live on
2026-09-17). These tests pin the two properties that fix it -- forwards overlap
each other, a trace excludes them -- without importing torch.
"""

from __future__ import annotations

import threading
import time

from fabric_defect_hub.core.execution import model_execution, model_tracing


def test_forward_passes_overlap_each_other():
    """One worker per GPU is only worth it if readers are concurrent."""

    both_inside = threading.Barrier(2)
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            with model_execution():
                both_inside.wait(timeout=5)
        except BaseException as exc:  # pragma: no cover - the assertion below says why
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # A serializing guard would time the barrier out instead.
    assert not errors


def test_a_forward_cannot_start_while_a_trace_is_active():
    trace_active = threading.Event()
    forward_attempted = threading.Event()
    forward_ran = threading.Event()
    release_trace = threading.Event()

    def tracer() -> None:
        with model_tracing():
            trace_active.set()
            release_trace.wait(timeout=10)

    def forward() -> None:
        trace_active.wait(timeout=10)
        forward_attempted.set()
        with model_execution():
            forward_ran.set()

    tracer_thread = threading.Thread(target=tracer)
    forward_thread = threading.Thread(target=forward)
    tracer_thread.start()
    forward_thread.start()

    assert forward_attempted.wait(timeout=10)
    time.sleep(0.2)  # give a broken guard every chance to let the forward through
    assert not forward_ran.is_set()

    release_trace.set()
    forward_thread.join(timeout=10)
    tracer_thread.join(timeout=10)
    assert forward_ran.is_set()


def test_a_forward_waits_for_an_active_trace_and_then_runs():
    """The mirror image of the test above: the trace finishes first, so the
    forward must not be blocked forever."""

    order: list[str] = []

    def tracer() -> None:
        with model_tracing():
            time.sleep(0.1)
            order.append("trace")

    thread = threading.Thread(target=tracer)
    thread.start()
    time.sleep(0.01)
    with model_execution():
        order.append("forward")
    thread.join(timeout=10)

    assert order == ["trace", "forward"]
