"""Guards the one property that makes `fabric_defect_hub.api` safe to have:
it is a delegation layer, not a second implementation.

The risk this file exists for is specific. A flat, friendly surface
(`fdh.train(...)`, `fdh.predict(...)`) is exactly the place where
"just this one special case for anomalib" accumulates. Once it does, the
project has two pipelines: the contract-tested one under `core/` and
`models/`, and an untested shadow of it in the convenience layer -- and
`test_pipeline_contract.py` / `test_adapter_contract.py` cannot see the
second one, because those tests check adapters and pipelines, not this
module.

So the rule enforced here is mechanical: **no per-backend knowledge in
api.py**. Backend names may appear only in the module-level `_PRESET_MODULES`
table (a lookup, not a branch). Anything that needs to differ per backend
belongs in the layer that owns it -- e.g. `training.RUN_LENGTH_KEYS`, which
api.py looks up rather than re-deriving.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from fabric_defect_hub import api

BACKEND_NAMES = {"ultralytics", "torchvision", "anomalib", "dinomaly", "moeclip", "mambaad"}

# The contract-layer modules api.py is allowed to delegate into. A public
# facade function that calls into none of them is not delegating -- it is
# doing the work itself, which is the thing this file exists to prevent.
DELEGATION_TARGETS = {
    "fabric_defect_hub.training",
    "fabric_defect_hub.inference.runner",
    "fabric_defect_hub.loader",
    "fabric_defect_hub.catalog",
    "fabric_defect_hub.core.registry",
    "fabric_defect_hub.models.base",
    # Measurement + its grouping. `benchmark()` is a facade over the sweep
    # exactly as `train()` is over `run_train`, and the taxonomy is what
    # keeps the UI's tables from being a second opinion on where a metric
    # belongs.
    "fabric_defect_hub.metric_sweep",
    "fabric_defect_hub.metrics_taxonomy",
}

_SOURCE = Path(inspect.getfile(api)).read_text()
_TREE = ast.parse(_SOURCE)


def _public_functions() -> list[ast.FunctionDef]:
    return [
        node
        for node in _TREE.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def test_the_facade_exposes_the_documented_entry_points():
    # If one of these disappears, `docs/interface.md` and the package docstring
    # are lying to whoever read them first.
    exposed = {node.name for node in _public_functions()}
    assert {
        "load_config", "from_pretrained", "list_pretrained",
        "list_models", "list_datasets", "train", "predict", "evaluate",
    } <= exposed


@pytest.mark.architecture
@pytest.mark.parametrize("function", _public_functions(), ids=lambda node: node.name)
def test_no_public_facade_function_branches_on_a_backend(function: ast.FunctionDef):
    """No backend name may appear inside a facade function at all.

    Checked as a string-constant search rather than as "does it contain an
    `if`": a dict lookup keyed by a hard-coded backend name is the same
    coupling as an if-chain, just spelled differently, and both would drift
    silently when a seventh backend is added.
    """

    found = sorted(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and node.value in BACKEND_NAMES
    )
    assert not found, (
        f"api.{function.name} hard-codes backend name(s) {found}. Per-backend knowledge "
        "belongs in the layer that owns it (e.g. training.RUN_LENGTH_KEYS), looked up here."
    )


@pytest.mark.architecture
def test_backend_names_appear_only_in_the_module_level_lookup_table():
    """The one permitted place: `_PRESET_MODULES`, which maps backend ->
    preset module. That is a table the facade reads, not behaviour it
    implements.
    """

    for node in _TREE.body:
        # The declaration is annotated (`_PRESET_MODULES: dict[str, str] = {...}`),
        # so it parses as AnnAssign; accept a bare Assign too rather than
        # letting this guard go silently dead if the annotation is dropped.
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == "_PRESET_MODULES" for target in targets):
            keys = {
                key.value for key in node.value.keys  # type: ignore[union-attr]
                if isinstance(key, ast.Constant)
            }
            assert keys == BACKEND_NAMES, "every backend needs a preset module entry"
            return
    pytest.fail("_PRESET_MODULES table not found in api.py")


@pytest.mark.architecture
@pytest.mark.parametrize("function", _public_functions(), ids=lambda node: node.name)
def test_every_public_facade_function_delegates(function: ast.FunctionDef):
    """Each public function must import from a contract-layer module.

    api.py imports lazily inside functions (so `import fabric_defect_hub`
    stays cheap and framework-free), which makes those imports a reliable
    signal of what each function actually delegates to.
    """

    imported = {
        node.module
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    # `importlib` alone counts: list_models/list_datasets reach the preset
    # modules and the dataset registry through it.
    uses_importlib = any(
        isinstance(node, ast.Import) and any(alias.name == "importlib" for alias in node.names)
        for node in ast.walk(function)
    )
    assert imported & DELEGATION_TARGETS or uses_importlib, (
        f"api.{function.name} imports nothing from the contract layer {sorted(DELEGATION_TARGETS)}; "
        "a facade function that delegates to nothing is doing the work itself."
    )


def test_importing_the_package_pulls_in_no_deep_learning_framework():
    """`import fabric_defect_hub` must stay usable for discovery
    (`list_models`, `load_config`) on a machine with no backend installed --
    which is what makes the facade safe to put in `__init__.py` at all. All
    of api.py's own imports are function-local for this reason.
    """

    module_level = {
        node.module for node in _TREE.body if isinstance(node, ast.ImportFrom) and node.module
    }
    heavy = {name for name in module_level if name.split(".")[0] in {"torch", "anomalib", "lightning", "timm"}}
    assert not heavy, f"api.py imports {sorted(heavy)} at module level"


def test_facade_functions_survive_importing_every_submodule():
    """`fdh.predict` must still be the function after anything imports a
    submodule.

    Python binds a submodule as an attribute of its package on import, so a
    module named `fabric_defect_hub.predict` silently overwrote the facade's
    `predict` function — and the very first `fdh.predict(...)` call did the
    overwriting itself, since it imports `run_predict` from that module.
    Calling it twice in one session raised `TypeError: 'module' object is not
    callable`. The module now lives at `fabric_defect_hub.inference.runner`;
    this test is what keeps any future top-level module from taking one of
    these names back.
    """

    import importlib
    import pkgutil

    import fabric_defect_hub as fdh

    for module in pkgutil.walk_packages(fdh.__path__, f"{fdh.__name__}."):
        if module.name.endswith("__main__"):  # would run the CLI
            continue
        try:
            importlib.import_module(module.name)
        except ImportError:
            continue  # an optional backend that isn't installed here

    # `recipes` is deliberately a module export (the config-profile package
    # itself); everything else in `__all__` is a function or a class.
    intentionally_a_module = {"recipes"}
    shadowed = sorted(
        name
        for name in fdh.__all__
        if name not in intentionally_a_module
        and isinstance(getattr(fdh, name), type(importlib))
    )
    assert not shadowed, (
        f"{shadowed} in fabric_defect_hub.__all__ resolve to submodules, not to the exported "
        f"objects — a submodule of the same name shadows them once imported."
    )


def test_run_config_carries_exactly_what_run_train_accepts():
    """`train(cfg)` is a call, not a translation -- which only holds if every
    field `RunConfig` forwards is a real `run_train` parameter. If `run_train`
    renames one, this fails here instead of at someone's next training run.
    """

    from fabric_defect_hub.training import run_train

    accepted = set(inspect.signature(run_train).parameters)
    forwarded = {"backend", "variant", "overrides", "set_overrides", "publish"}
    assert forwarded <= accepted, sorted(forwarded - accepted)


def test_train_routes_config_dir_to_both_resolution_and_the_run(tmp_path):
    """`config_dir` has to reach the keyword resolution *and* the training
    call. Left in `**kwargs` it reached only the latter, so a custom config
    directory would have its backend inferred from the default directory's
    config and then trained from its own — a silent mismatch, not an error.
    """

    from fabric_defect_hub import api

    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "anomalib_example.yaml").write_text("model:\n  name: PaDiM\n")

    captured = {}

    def fake_run_train(model, **kwargs):
        captured.update(model=model, **kwargs)
        return "ran"

    import fabric_defect_hub.training as training_module

    original = training_module.run_train
    training_module.run_train = fake_run_train
    try:
        assert api.train("padim", config_dir=directory) == "ran"
    finally:
        training_module.run_train = original

    assert captured["config_dir"] == directory
    assert captured["backend"] == "anomalib"
