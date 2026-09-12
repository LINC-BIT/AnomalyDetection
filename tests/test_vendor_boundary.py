"""The vendor boundary: upstream module names live in `vendor.py` and nowhere
else, and importing a vendored backend leaves no trace in the process.

Vendored research repos ship flat top-level modules under generic names —
Dinomaly and MoECLIP both define a `utils` and a `dataset`. The Benchmark tab
runs every canonical model back to back in one process, so a repo left
occupying `sys.modules["utils"]` silently hands its code to the next backend
that asks. A third vendored repo makes that near-certain.

Two rules, checked two ways:

  1. **Static** — no module outside a backend's `vendor.py` may name an
     upstream module. Parsed with `ast`, not grepped, so a name inside a
     string or a comment is not a false positive and `from utils import x`
     inside a function body is not a false negative.
  2. **Behavioural** — after importing (and using) a vendored backend,
     `sys.modules` and `sys.path` are exactly as they were.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "fabric_defect_hub"

# Top-level names each vendored checkout occupies. Kept here deliberately
# rather than imported from the `vendor.py` modules: this test is the
# independent statement of the rule, and reading the rule from the code it
# polices would let both drift together.
VENDORED_ROOTS = {
    "dinomaly": ("utils", "dataset", "models", "optimizers", "dinov1", "dinov2", "beit"),
    "moeclip": ("utils", "dataset", "model", "forward_utils"),
}

ALL_VENDORED_ROOTS = {name for roots in VENDORED_ROOTS.values() for name in roots}

# The only files allowed to name them.
VENDOR_MODULES = {
    SRC_ROOT / "models" / "dinomaly" / "vendor.py",
    SRC_ROOT / "models" / "moeclip" / "vendor.py",
    SRC_ROOT / "core" / "vendor.py",
}


def _upstream_imports(path: Path) -> list[tuple[int, str]]:
    """(line, module) for every import of a vendored top-level name.

    Only absolute imports count: `from .models import x` and
    `from fabric_defect_hub.models import x` are this project's own `models`
    package, not Dinomaly's.
    """

    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in ALL_VENDORED_ROOTS:
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — always ours
                continue
            root = (node.module or "").split(".")[0]
            if root in ALL_VENDORED_ROOTS:
                found.append((node.lineno, node.module or ""))

    return found


def _project_sources() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_the_scan_actually_covers_the_project():
    sources = _project_sources()
    assert len(sources) > 50, "source scan found suspiciously few files"
    assert all(p.is_file() for p in VENDOR_MODULES), "vendor modules moved; update this test"


@pytest.mark.architecture
@pytest.mark.parametrize("path", _project_sources(), ids=lambda p: str(p.relative_to(SRC_ROOT)))
def test_upstream_modules_are_named_only_inside_vendor_modules(path):
    if path in VENDOR_MODULES:
        return

    offenders = _upstream_imports(path)

    assert not offenders, (
        f"{path.relative_to(SRC_ROOT)} imports vendored upstream module(s) "
        f"{[name for _, name in offenders]} at line(s) {[line for line, _ in offenders]}. "
        "Go through the backend's `vendor.py` (`import_vendor()[...]`) instead — a bare "
        "`import utils` resolves to whichever vendored repo loaded first."
    )


@pytest.mark.architecture
def test_vendor_modules_do_not_import_upstream_at_module_scope():
    """Even inside `vendor.py`, upstream names must be resolved through
    `VendoredRepo` at call time — a module-scope `import utils` would run at
    package-import time, outside the isolation window.
    """

    for path in VENDOR_MODULES:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:  # module scope only
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                name = getattr(node, "module", None) or node.names[0].name
                assert name.split(".")[0] not in ALL_VENDORED_ROOTS, (
                    f"{path.name}:{node.lineno} imports {name} at module scope"
                )


# --------------------------------------------------------------------------- #
# Behavioural: the process is left as it was found
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", sorted(VENDORED_ROOTS))
def test_importing_a_backend_leaves_no_upstream_names_behind(backend):
    module = pytest.importorskip(f"fabric_defect_hub.models.{backend}.adapter")

    leaked = sorted(
        name
        for name in sys.modules
        if name.split(".")[0] in VENDORED_ROOTS[backend]
        and not name.startswith("fabric_defect_hub")
        and getattr(sys.modules[name], "__file__", None)
        and "components" in str(sys.modules[name].__file__)
    )

    assert not leaked, f"{module.__name__} left {leaked} occupying sys.modules"


@pytest.mark.parametrize("backend", sorted(VENDORED_ROOTS))
def test_resolving_a_checkout_restores_sys_path_and_sys_modules(backend):
    """The isolation window itself: whether or not the checkout is present,
    asking for its modules must not mutate the process.
    """

    vendor = pytest.importorskip(f"fabric_defect_hub.models.{backend}.vendor")

    modules_before = set(sys.modules)

    try:
        vendor.import_vendor()
    except (FileNotFoundError, ImportError):
        # Checkout absent or its dependencies unavailable here; the cleanup
        # runs in a `finally`, so the assertions below are still the point.
        pass

    # Specifically *our* checkout root, not `sys.path` as a whole: importing
    # the vendored code can legitimately extend `sys.path` through third
    # parties (MoECLIP's `model/tokenizer.py` imports `pkg_resources`, which
    # appends setuptools' own `_vendor` directory). What must not survive is
    # the entry this mechanism added.
    assert str(vendor.vendor_root()) not in sys.path, "vendored checkout left on sys.path"

    new_upstream = {
        name for name in set(sys.modules) - modules_before
        if name.split(".")[0] in VENDORED_ROOTS[backend]
    }
    assert not new_upstream, f"vendored modules left in sys.modules: {sorted(new_upstream)}"


def test_two_repos_sharing_a_name_do_not_shadow_each_other():
    """The failure this whole mechanism exists to prevent: Dinomaly and
    MoECLIP both ship a top-level `utils` and `dataset`, and the Benchmark tab
    runs both in one process.
    """

    shared = set(VENDORED_ROOTS["dinomaly"]) & set(VENDORED_ROOTS["moeclip"])
    assert shared, "expected these two checkouts to collide on generic names"

    dinomaly = pytest.importorskip("fabric_defect_hub.models.dinomaly.vendor")
    moeclip = pytest.importorskip("fabric_defect_hub.models.moeclip.vendor")
    if not (dinomaly.vendor_root().is_dir() and moeclip.vendor_root().is_dir()):
        pytest.skip("both vendored checkouts must be present to prove non-interference")

    try:
        first = dinomaly.import_vendor()["utils"]
        second = moeclip.import_vendor()["utils"]
        # Importing MoECLIP second must not have replaced Dinomaly's, and the
        # two must be genuinely different modules.
        assert dinomaly.import_vendor()["utils"] is first
        assert first is not second
        assert "dinomaly" in str(first.__file__)
        assert "moeclip" in str(second.__file__)
    except ImportError as exc:
        pytest.skip(f"vendored dependency unavailable here: {exc}")
