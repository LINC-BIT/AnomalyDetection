"""A registry of child processes that a parent must not leave behind.

Two long-running fan-outs in this project spawn one child per unit of work —
`application.benchmark` one per model evaluation, `training_runs` one per model
training — and both need the same three guarantees:

* **A child dies with its parent.** On Linux `PR_SET_PDEATHSIG` asks the kernel
  to signal it, so `kill -9` on the host (or a crashed UI) cannot strand a
  process holding a GPU and gigabytes of VRAM.
* **Quitting stops them promptly.** SIGTERM first, SIGKILL only for one that
  ignores it: a worker inside a CUDA call does not always react to a polite
  signal, and waiting for it is exactly the "quitting takes minutes" behaviour
  these fan-outs used to have.
* **A clean exit kills them too** (`atexit`), so nothing outlives the parent
  when it exits on its own.

The registry is deliberately small: it tracks processes, it kills them, and it
knows nothing about what they compute.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Sequence

__all__ = ["WorkerRegistry", "child_dies_with_parent", "terminate_process"]


def child_dies_with_parent() -> None:  # pragma: no cover - runs in the forked child
    """`preexec_fn`: ask the kernel to SIGTERM this child when its parent dies.

    Runs between fork and exec, so it must not allocate or import anything
    heavy — hence the local imports and the blanket `except`. Linux only;
    elsewhere the registry's `atexit` cleanup is the only guard, which is why it
    is always registered.
    """

    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG = 1
    except Exception:
        return


def terminate_process(process: subprocess.Popen, grace_seconds: float = 3.0) -> None:
    """SIGTERM one child, SIGKILL it if it will not go."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=max(0.0, grace_seconds))
    except subprocess.TimeoutExpired:
        process.kill()


class WorkerRegistry:
    """The children one parent process has running right now."""

    def __init__(self, name: str = "worker") -> None:
        self._name = name
        self._processes: set[subprocess.Popen] = set()
        self._lock = threading.Lock()
        self._hooks_installed = False

    # -- tracking -------------------------------------------------------- #
    def spawn(self, command: Sequence[str], **kwargs: Any) -> subprocess.Popen:
        """Start a child, remember it, and let it die with this process."""

        if sys.platform.startswith("linux"):
            kwargs.setdefault("preexec_fn", child_dies_with_parent)
        process = subprocess.Popen(list(command), **kwargs)
        self.track(process)
        return process

    def track(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.add(process)

    def forget(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)

    def live(self) -> list[subprocess.Popen]:
        with self._lock:
            return list(self._processes)

    # -- stopping -------------------------------------------------------- #
    def terminate_all(self, grace_seconds: float = 3.0) -> int:
        """Stop every child still running; return how many were asked to stop."""

        processes = self.live()
        for process in processes:
            terminate_process(process, grace_seconds)
            self.forget(process)
        return len(processes)

    def install_exit_hooks(self, *, also_uvicorn: bool = False) -> bool:
        """Kill the children on a normal exit, and (for the UI) on the first
        Ctrl+C.

        `atexit` alone is not enough for the UI: uvicorn's graceful shutdown
        waits for the request that is waiting for a child, so the child has to
        die *before* uvicorn's handler runs. `install_exit_hooks` therefore
        wraps `uvicorn.Server.handle_exit` — uvicorn registers that method as
        its signal handler when it starts, so this must be called before
        `launch()`. Returns whether the uvicorn hook was installed.
        """

        atexit.register(self.terminate_all)
        if not also_uvicorn or self._hooks_installed:
            return self._hooks_installed
        try:
            import uvicorn
        except ImportError:  # pragma: no cover - the UI extra always brings uvicorn
            return False

        original = uvicorn.Server.handle_exit
        registry = self

        def handle_exit(self, sig, frame):  # type: ignore[no-untyped-def]
            stopped = registry.terminate_all()
            if stopped:
                print(
                    f"[{registry._name}] {signal.Signals(sig).name}: stopped {stopped} worker "
                    f"process(es) before shutting down",
                    file=sys.stderr, flush=True,
                )
            return original(self, sig, frame)

        uvicorn.Server.handle_exit = handle_exit  # type: ignore[method-assign]
        self._hooks_installed = True
        return True

    # -- helpers --------------------------------------------------------- #
    @staticmethod
    def device_environ(device: str) -> dict[str, str]:
        """Environment for a child pinned to `device`.

        CUDA workers get `CUDA_VISIBLE_DEVICES`, so the child sees one GPU as
        `cuda:0`: every backend's own auto-detection then lands on the assigned
        card without this project having to thread a device through each
        training config. CPU/MPS need no pinning, and the caller records the
        logical device it assigned either way.
        """

        env = dict(os.environ)
        if device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.partition(":")[2]
        return env


def wait_for(process: subprocess.Popen, timeout: float | None = None) -> int:
    """`Popen.wait`, but with a sentence a human can act on when it expires."""

    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        terminate_process(process)
        raise TimeoutError(f"worker exceeded its {timeout:.0f}s timeout") from exc
