"""Human-visible progress for the loops this project writes itself.

Four backends run their own training loop (torchvision, dinomaly, moeclip,
mambaad) and three of those also run their own inference loop. Until now
none of them printed anything between "started" and "finished": on a long
cloud run the only evidence of life was `history.csv` growing, and MambaAD
did not even write that. Anomalib (Lightning) and Ultralytics bring their
own progress bars, so this module deliberately covers only the loops we own
— it is not a wrapper around theirs.

**Plain lines, not a redrawing bar.** A `tqdm`-style bar needs a TTY and a
carriage return; the two places this output actually has to survive are a
pipe (`fdh train-all` tees every model's output into
`artifacts/training_runs/<id>/logs/<model>.log`) and a notebook cell. Both
mangle carriage returns and neither gives the child a TTY. One newline-
terminated, immediately flushed line every few seconds reads correctly in a
terminal, in `tail -f`, in a log file, and in Jupyter — the same text in all
four, which is worth more here than a bar that looks nice in exactly one.

Rate limited by wall-clock time (not by iteration count) so a fast loop
prints as rarely as a slow one, and always emitting the final line so the
last thing in the log is a complete count rather than wherever the timer
happened to land.

Environment:
    FDH_PROGRESS=0            silence it (any of 0/false/no/off)
    FDH_PROGRESS_INTERVAL=10  seconds between lines (default 5)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, TextIO

__all__ = ["ProgressReporter", "format_duration", "note", "progress_enabled"]

DEFAULT_INTERVAL_SECONDS = 5.0
_PREFIX = "[fdh]"
_FALSE = {"0", "false", "no", "off"}


def progress_enabled() -> bool:
    """Whether progress lines are wanted at all (`FDH_PROGRESS`)."""

    return os.environ.get("FDH_PROGRESS", "1").strip().lower() not in _FALSE


def _interval_seconds() -> float:
    raw = os.environ.get("FDH_PROGRESS_INTERVAL")
    if not raw:
        return DEFAULT_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_INTERVAL_SECONDS
    return value if value > 0 else DEFAULT_INTERVAL_SECONDS


def format_duration(seconds: float) -> str:
    """`M:SS` under an hour, `H:MM:SS` above it. `--` for unknown."""

    if seconds is None or seconds != seconds or seconds < 0:  # NaN-safe
        return "--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def note(message: str, *, stream: TextIO | None = None) -> None:
    """One-off line in the same style as the reporter's own output."""

    if not progress_enabled():
        return
    target = stream if stream is not None else sys.stdout
    print(f"{_PREFIX} {message}", file=target, flush=True)


class ProgressReporter:
    """Counts steps and prints a rate-limited line about them.

    Use as a context manager so the closing line is emitted even when the
    loop raises:

        with ProgressReporter("dinomaly train", total=total_iters) as progress:
            for ...:
                ...
                progress.update(loss=loss_value)
    """

    def __init__(
        self,
        label: str,
        total: int | None = None,
        *,
        unit: str = "it",
        interval: float | None = None,
        stream: TextIO | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.label = label
        self.total = total if total and total > 0 else None
        self.unit = unit
        self.interval = interval if interval is not None else _interval_seconds()
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = progress_enabled() if enabled is None else enabled
        self.count = 0
        self._started = time.monotonic()
        self._last_emit = 0.0
        self._closed = False
        if self.enabled:
            total_text = f"{self.total} {self.unit}" if self.total else f"unknown number of {self.unit}"
            self._write(f"{self.label}: starting ({total_text})")

    # -- counting ---------------------------------------------------------

    def update(self, step: int = 1, **metrics: Any) -> None:
        """Advance by `step` and print if the interval has elapsed.

        The *first* update always prints, whatever the interval: it is the
        line that separates "still setting up / stuck loading data" from
        "running, just slowly", which on a fresh host is the question being
        asked. After that the interval applies.

        Extra keyword arguments are appended to the line as `name value`
        pairs (floats to four decimals) — pass whatever the loop already
        has to hand, typically `loss=` and `lr=`.
        """

        self.count += step
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_emit < self.interval:
            return
        self._last_emit = now
        self._write(self._line(now, metrics))

    def close(self, **metrics: Any) -> None:
        """Final line, with the elapsed total. Idempotent."""

        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return
        elapsed = time.monotonic() - self._started
        parts = [
            f"{self.label}: done",
            f"{self.count} {self.unit}",
            f"in {format_duration(elapsed)}",
        ]
        parts.extend(self._metric_parts(metrics))
        self._write(" | ".join(parts))

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- formatting -------------------------------------------------------

    def _line(self, now: float, metrics: dict[str, Any]) -> str:
        elapsed = now - self._started
        rate = self.count / elapsed if elapsed > 0 else 0.0
        if self.total:
            percent = 100.0 * self.count / self.total
            head = f"{self.label}: {self.count}/{self.total} {self.unit} ({percent:.1f}%)"
        else:
            head = f"{self.label}: {self.count} {self.unit}"
        parts = [head, f"{rate:.2f} {self.unit}/s", f"elapsed {format_duration(elapsed)}"]
        if self.total and rate > 0:
            parts.append(f"eta {format_duration((self.total - self.count) / rate)}")
        parts.extend(self._metric_parts(metrics))
        return " | ".join(parts)

    @staticmethod
    def _metric_parts(metrics: dict[str, Any]) -> list[str]:
        parts = []
        for name, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, float):
                parts.append(f"{name} {value:.4f}")
            else:
                parts.append(f"{name} {value}")
        return parts

    def _write(self, message: str) -> None:
        print(f"{_PREFIX} {message}", file=self.stream, flush=True)
