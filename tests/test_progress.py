"""`core.progress`: the rate limiter, the off switch, and the line format.

The reason these are worth pinning: the whole point of this module is that
its output survives a *pipe* (`fdh train-all` tees every model's output
into a log file) and a notebook cell. That rules out a carriage-return bar
and makes "one newline-terminated line, flushed, not too often" the actual
contract -- so that is what is asserted here, rather than the text itself.
"""

from __future__ import annotations

import io

import pytest

from fabric_defect_hub.core.progress import ProgressReporter, format_duration, note, progress_enabled


def _lines(stream: io.StringIO) -> list[str]:
    return [line for line in stream.getvalue().splitlines() if line]


def test_every_line_is_newline_terminated_and_prefixed():
    stream = io.StringIO()
    with ProgressReporter("demo", total=2, interval=0, stream=stream, enabled=True) as progress:
        progress.update()
        progress.update()

    text = stream.getvalue()
    assert text.endswith("\n")
    assert "\r" not in text  # a redrawing bar would need one; a log file cannot use it
    assert all(line.startswith("[fdh] ") for line in _lines(stream))


def test_updates_are_rate_limited_but_the_first_and_last_are_always_emitted():
    stream = io.StringIO()
    # A one-hour interval means no *periodic* line can fire during the test,
    # so what is left is exactly the two lines that ignore the interval.
    progress = ProgressReporter("demo", total=100, interval=3600, stream=stream, enabled=True)
    for _ in range(50):
        progress.update()
    assert len(_lines(stream)) == 2  # "starting", then the first iteration

    progress.close()
    lines = _lines(stream)
    assert len(lines) == 3
    assert "done" in lines[-1] and "50 it" in lines[-1]


def test_close_is_idempotent():
    stream = io.StringIO()
    progress = ProgressReporter("demo", stream=stream, enabled=True)
    progress.close()
    progress.close()
    assert sum("done" in line for line in _lines(stream)) == 1


def test_the_closing_line_is_emitted_even_when_the_loop_raises():
    stream = io.StringIO()
    with pytest.raises(RuntimeError):
        with ProgressReporter("demo", total=10, stream=stream, enabled=True) as progress:
            progress.update()
            raise RuntimeError("boom")
    assert "done" in _lines(stream)[-1]


def test_a_known_total_yields_percent_and_eta():
    stream = io.StringIO()
    progress = ProgressReporter("demo", total=10, interval=0, stream=stream, enabled=True)
    progress.update(loss=0.5)
    line = _lines(stream)[-1]
    assert "1/10 it" in line and "%" in line and "eta" in line
    assert "loss 0.5000" in line


def test_an_unknown_total_says_so_instead_of_guessing_one():
    stream = io.StringIO()
    progress = ProgressReporter("demo", interval=0, stream=stream, enabled=True)
    progress.update()
    line = _lines(stream)[-1]
    assert "1 it" in line
    assert "eta" not in line and "%" not in line


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_the_env_switch_silences_everything(monkeypatch, value, capsys):
    monkeypatch.setenv("FDH_PROGRESS", value)
    assert progress_enabled() is False

    stream = io.StringIO()
    with ProgressReporter("demo", total=3, interval=0, stream=stream) as progress:
        progress.update()
    note("also silenced")
    assert stream.getvalue() == ""
    assert capsys.readouterr().out == ""


def test_a_disabled_reporter_still_counts():
    progress = ProgressReporter("demo", total=3, stream=io.StringIO(), enabled=False)
    progress.update(2)
    assert progress.count == 2


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (9.6, "0:09"), (90, "1:30"), (3600, "1:00:00"), (3725, "1:02:05"), (-1, "--")],
)
def test_duration_formatting(seconds, expected):
    assert format_duration(seconds) == expected
