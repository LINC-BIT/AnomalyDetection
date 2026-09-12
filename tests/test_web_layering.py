"""Keeps the Gradio front end on the interface layers.

The web package is a *consumer* of this project's contracts, not a seventh
place that knows how each backend behaves. That distinction kept eroding in
small, individually-reasonable steps, and every erosion was a fact stated
twice:

* `_ANOMALY_MAP_BACKENDS = {"anomalib", "dinomaly", "moeclip", "mambaad"}`
  duplicated in `web/single_image.py` and `predict.py` to decide who gets an
  `output_dir` — and wrong within a backend, since GANomaly is an anomalib
  model with no pixel map at all.
* `_BACKEND_PROBE_MODULE = {"dinomaly": "timm", ...}` reimplementing
  `core.availability.backend_is_importable` with a weaker question.
* `from fabric_defect_hub.models.anomalib.checkpoint import inspect_checkpoint`
  plus `if spec["backend"] != "anomalib"` — which showed four of the six
  backends a message claiming their weights were "native Ultralytics
  artifacts".
* `"list" if backend == "torchvision" and task in (...)` deciding the shape
  of an exported module's input, a fact only the backend knows.

Each is now answered by the layer that owns it: `ModelCapabilities`
(`fills` / `export_input_style`), `core.availability`, `core.checkpoint`.
The two rules below are what stop them coming back — they are mechanical, so
they fail on the commit that reintroduces the coupling rather than on the
next person to read the code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import fabric_defect_hub.web as web_package

BACKEND_NAMES = {"ultralytics", "torchvision", "anomalib", "dinomaly", "moeclip", "mambaad"}

# The only `models.*` module the UI may import: the contract itself
# (`Artifact`, `ModelAdapter`, `ModelCapabilities`). Anything deeper is a
# specific backend's implementation.
ALLOWED_MODELS_IMPORTS = {"fabric_defect_hub.models.base"}

WEB_MODULES = sorted(Path(web_package.__file__).parent.glob("*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """`id()`s of every docstring constant, so the literal scan below reads
    code rather than prose. The comments and docstrings in these modules
    legitimately name backends — that is where an explanation belongs.
    """

    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                found.add(id(body[0].value))
    return found


@pytest.mark.architecture
@pytest.mark.parametrize("path", WEB_MODULES, ids=lambda p: p.name)
def test_web_module_does_not_import_a_specific_backend(path: Path):
    """`web/*` may import the model contract, never a backend's package."""

    offenders = sorted(
        node.module
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("fabric_defect_hub.models.")
        and node.module not in ALLOWED_MODELS_IMPORTS
    )
    assert not offenders, (
        f"web/{path.name} imports backend internals {offenders}. Ask the contract instead — "
        f"ModelCapabilities for what a model produces, core.availability for whether a backend "
        f"is installed, core.checkpoint for weight provenance."
    )


@pytest.mark.architecture
@pytest.mark.parametrize("path", WEB_MODULES, ids=lambda p: p.name)
def test_web_module_has_no_backend_names_in_its_code(path: Path):
    """No backend name may appear as a *value* in the web package.

    Checked as a string-literal scan rather than as "does it contain an if":
    a set membership test, a dict keyed by backend, and an if-chain are the
    same coupling written three ways. Comments and docstrings are exempt —
    naming a backend while explaining a decision is the opposite of the
    problem.
    """

    tree = _parse(path)
    docstrings = _docstring_nodes(tree)
    offenders = sorted(
        {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in BACKEND_NAMES
            and id(node) not in docstrings
        }
    )
    assert not offenders, (
        f"web/{path.name} hard-codes backend name(s) {offenders}. Per-backend behaviour belongs "
        f"in that backend's adapter and reaches the UI through ModelCapabilities."
    )


@pytest.mark.architecture
def test_dataset_catalog_holds_presentation_only():
    """`DATASET_CATALOG` maps a display label to how the *page* should behave
    — nothing about what a dataset is.

    It used to carry `tasks` (which tasks the dataset has ground truth for)
    and `dir` (where it lives), both already declared in
    `core.dataset_capabilities`. The two copies agreed, which is exactly when
    a duplicated fact is most dangerous: nothing was enforcing it, so the
    first divergence would have silently mis-scored a leaderboard row.
    """

    from fabric_defect_hub.web.single_image import DATASET_CATALOG

    presentation_only = {"name", "env", "slice_kwarg", "task"}
    for label, spec in DATASET_CATALOG.items():
        extra = sorted(set(spec) - presentation_only)
        assert not extra, (
            f"DATASET_CATALOG[{label!r}] carries {extra}, which is dataset *capability*, not "
            f"presentation. Declare it in core.dataset_capabilities and read it back."
        )


def test_every_ui_dataset_declares_capabilities():
    """The flip side: the UI can only read `tasks`/`default_root` back if the
    dataset actually declared them, so an entry pointing at an undeclared
    dataset has to fail here rather than at the first click.
    """

    import fabric_defect_hub.datasets  # noqa: F401 -- triggers @register_dataset
    from fabric_defect_hub.core.dataset_capabilities import all_capabilities
    from fabric_defect_hub.web.single_image import DATASET_CATALOG, dataset_tasks

    declared = all_capabilities()
    for label, spec in DATASET_CATALOG.items():
        name = spec["name"]
        assert name in declared, f"{label!r} names dataset {name!r}, which declares no capabilities"
        assert declared[name].default_root, f"{name!r} declares no default_root; the UI cannot locate it"
        assert dataset_tasks(name), f"{name!r} declares no tasks; nothing could be benchmarked on it"
        assert spec["task"] in dataset_tasks(name), (
            f"{label!r}'s gallery default task {spec['task']!r} is not one of the tasks "
            f"{name!r} declares ({dataset_tasks(name)})"
        )


def test_the_contract_can_answer_what_the_ui_used_to_hard_code():
    """The rules above are only enforceable because the contract answers the
    questions the UI was answering for itself. Pin that: if one of these
    accessors disappears, the UI has nowhere to ask and the coupling comes
    back by necessity rather than by neglect.
    """

    from fabric_defect_hub.core.availability import backend_is_importable
    from fabric_defect_hub.core.checkpoint import inspect_checkpoint
    from fabric_defect_hub.models.base import ModelCapabilities

    caps = ModelCapabilities(tasks=("anomaly",), prediction_fields=("anomaly_score",))
    assert caps.fills("anomaly_map") is False          # -> no output_dir, no heatmap chip
    assert caps.export_input_style == "batched"        # -> ProfileConfig.input_style
    assert callable(backend_is_importable)             # -> model_status
    assert callable(inspect_checkpoint)                # -> checkpoint_diagnostic
