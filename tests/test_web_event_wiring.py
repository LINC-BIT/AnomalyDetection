"""Every Gradio event must pass exactly as many inputs as its handler takes.

A mismatch does not fail when the app is built — it fails later, in the UI.
A *missing* input shifts the remaining values one position left, so
`load_model_handler(model_label, current_state, lang)` wired with
`inputs=[model_choice, lang_state]` receives the language code as
`current_state`, and the first `.get()` on it raises

    AttributeError: 'str' object has no attribute 'get'

for every model the user tries to load. An extra input fails the same way
from the other side (a value lands in `lang` that `tr()` cannot index).
Neither is visible until someone clicks the button, so it is checked here
instead: the wiring is read from the source with `ast`, so this test needs
no Gradio event loop and no running server.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "src" / "fabric_defect_hub" / "web" / "app.py"
EVENTS = ("click", "change", "submit", "input", "select", "load", "release")


def _positional_count(fn: object) -> int:
    return len([
        parameter
        for parameter in inspect.signature(fn).parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ])


def _declared_handlers(tree: ast.Module) -> tuple[dict[str, int], dict[str, tuple[str, str]]]:
    local: dict[str, int] = {}
    imported: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            local[node.name] = len(node.args.posonlyargs) + len(node.args.args)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported[alias.asname or alias.name] = (node.module, alias.name)
    return local, imported


def _expected_inputs(name: str, local: dict[str, int], imported: dict[str, tuple[str, str]]) -> int | None:
    if name in local:
        return local[name]
    if name in imported:
        module, attribute = imported[name]
        try:
            return _positional_count(getattr(importlib.import_module(module), attribute))
        except Exception:  # a handler whose own module needs an optional dependency
            return None
    return None


def wirings() -> list[tuple[int, str, int, int | None]]:
    """`(line, handler name, inputs passed, parameters declared)` per event."""

    tree = ast.parse(APP.read_text(encoding="utf-8"))
    local, imported = _declared_handlers(tree)
    found: list[tuple[int, str, int, int | None]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in EVENTS or not node.args:
            continue
        handler = node.args[0]
        if not isinstance(handler, ast.Name):
            continue
        inputs = next((kw.value for kw in node.keywords if kw.arg == "inputs"), None)
        if not isinstance(inputs, (ast.List, ast.Tuple)):
            continue
        found.append((node.lineno, handler.id, len(inputs.elts), _expected_inputs(handler.id, local, imported)))
    return found


def test_the_wiring_audit_finds_the_event_handlers():
    """Guard the guard: an empty or tiny harvest would make the check vacuous."""

    assert len(wirings()) >= 10


def test_every_event_passes_exactly_its_handler_inputs():
    mismatches = [
        f"line {line}: {name} gets {passed} inputs but declares {declared}"
        for line, name, passed, declared in wirings()
        if declared is not None and passed != declared
    ]
    assert mismatches == [], (
        "Gradio passes inputs positionally, so a mismatch shifts values into the wrong "
        "parameters and only fails when the user triggers the event:\n  " + "\n  ".join(mismatches)
    )
