"""Importing vendored research code without letting it into the process.

Research repos are not packages. Dinomaly and MoECLIP both ship flat
top-level modules meant to be run from their own repo root — `utils`,
`dataset`, `models`/`model`, `optimizers`, `forward_utils` — with no package
prefix and no pip release. The obvious way to use them is to put the checkout
on `sys.path` and `import utils`, and that is exactly what breaks: the two
repos define *different* modules under the same generic names, the Benchmark
tab runs every canonical model back to back in one process (`web/benchmark.py`),
and whichever backend imported first would win while the second silently got
the other repo's code. Adding a third vendored repo makes a collision close to
certain.

`VendoredRepo` imports a checkout inside a window where its root is
`sys.path[0]` and any already-imported module owning one of its names is
temporarily displaced, then *takes the imported modules back out* of
`sys.modules` and keeps them in a private cache, restoring what was there
before. The module objects keep working afterwards — their cross-references
were resolved into module globals at import time — but they no longer occupy
the shared names, and `sys.path` is left exactly as it was found.

The one assumption: vendored code must not import its own top-level names
lazily from inside a function body, since by then the name is gone from
`sys.modules`. Verified against both pinned checkouts (Dinomaly's only such
imports are in `dinov2/run/*`, DINOv2's SLURM job launchers, which nothing
here touches).

Only a backend's `vendor.py` may name upstream modules. Everything else goes
through the mapping this returns, which is what
`tests/test_vendor_boundary.py` enforces.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

# Prefer the repository-level shared vendor directory after the multi-domain
# layout migration. Keep the textile-local path as a compatibility fallback.
from fabric_defect_hub.paths import COMPONENTS_ROOT


class VendoredRepo:
    """One vendored checkout, imported in isolation and cached.

    `owned_roots` are the top-level names the checkout occupies — everything
    that must be moved out of `sys.modules` afterwards, including names pulled
    in transitively. `entry_modules` are the ones callers actually ask for, in
    dependency order.
    """

    def __init__(
        self,
        name: str,
        directory: str,
        owned_roots: tuple[str, ...],
        entry_modules: tuple[str, ...],
        missing_hint: str,
    ):
        self.name = name
        self.root = COMPONENTS_ROOT / directory
        self.owned_roots = owned_roots
        self.entry_modules = entry_modules
        self.missing_hint = missing_hint
        self._cache: dict[str, ModuleType] = {}

    # ------------------------------------------------------------------ #
    def require_root(self) -> Path:
        """The checkout directory, or a `FileNotFoundError` that says how to
        get it — never a confusing `ModuleNotFoundError` from deep inside an
        import.
        """

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.name} vendor checkout not found at {self.root}. {self.missing_hint}"
            )
        return self.root

    def _owns(self, module_name: str) -> bool:
        return any(
            module_name == root or module_name.startswith(f"{root}.")
            for root in self.owned_roots
        )

    def modules(self) -> dict[str, ModuleType]:
        """Import (once) and return this checkout's modules by name, e.g.
        `repo.modules()["models.uad"].ViTill`.
        """

        if self._cache:
            return self._cache

        root = self.require_root()

        displaced = {name: module for name, module in sys.modules.items() if self._owns(name)}
        for name in displaced:
            del sys.modules[name]

        path_str = str(root)
        sys.path.insert(0, path_str)
        try:
            imported = {name: importlib.import_module(name) for name in self.entry_modules}
        finally:
            # Reclaim this repo's modules out of the shared namespace, then put
            # back whatever (another vendored repo, or a genuine third-party
            # `dataset`) was there before. Runs even on failure so a
            # half-finished import cannot leave the process in a mixed state.
            for name in [name for name in sys.modules if self._owns(name)]:
                del sys.modules[name]
            sys.modules.update(displaced)
            if path_str in sys.path:
                sys.path.remove(path_str)

        self._cache.update(imported)
        return self._cache

    def __getitem__(self, module_name: str) -> ModuleType:
        return self.modules()[module_name]
