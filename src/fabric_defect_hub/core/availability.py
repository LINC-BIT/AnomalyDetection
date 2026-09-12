"""Runtime probes for what's actually usable *on this machine*, as opposed
to `core.dataset_capabilities`/`core.registry`, which describe what the
project *knows how to use* regardless of what happens to be staged/installed
locally.

Not every machine has every dataset staged under `data/<Dataset>` (see
`training.DEFAULT_DATASET_ROOTS`'s per-machine symlink convention) or every
optional ML framework installed (anomalib, ultralytics, ...) -- a cloud
training box in particular is never expected to carry every dataset this
project knows about. This module answers "is X usable right now?" so
callers (`core.decision`, `training.apply_available_dataset`, `fdh doctor`)
can degrade gracefully -- substitute an available alternative, or explain
precisely what's missing -- instead of failing deep inside a backend's
training loop with a raw `FileNotFoundError` from Torch's dataloader.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def root_is_staged(root: str | None) -> bool:
    """Whether `root` is a real, non-empty, on-disk directory right now.

    False for `None`, an unresolved `${ENV_VAR}` placeholder (mirrors
    `training.apply_default_dataset_root`'s own "usable root" check), a path
    that doesn't exist, or an existing-but-empty directory (a staged-but-not
    -yet-populated symlink target -- functionally equivalent to "not staged").
    """

    if not root or "${" in root:
        return False
    path = Path(root).expanduser()
    if not path.is_dir():
        return False
    return any(path.iterdir())


def _default_root_map() -> dict[str, str]:
    from fabric_defect_hub.training import DEFAULT_DATASET_ROOTS

    return DEFAULT_DATASET_ROOTS


def dataset_is_staged(name: str, root_map: dict[str, str] | None = None) -> bool:
    """Whether the default root for registered dataset `name` is staged.

    `root_map` defaults to `training.DEFAULT_DATASET_ROOTS` but is looked up
    lazily and accepted as a parameter (rather than imported at module level)
    so this module has no import-time dependency on `training.py`, and so
    callers can probe an explicit root override instead.
    """

    resolved_map = root_map if root_map is not None else _default_root_map()
    return root_is_staged(resolved_map.get(name))


def staged_datasets(names: set[str] | frozenset[str], root_map: dict[str, str] | None = None) -> set[str]:
    """The subset of `names` (registered dataset names) that are actually
    staged on this machine right now."""

    resolved_map = root_map if root_map is not None else _default_root_map()
    return {name for name in names if root_is_staged(resolved_map.get(name))}


def backend_is_importable(backend: str) -> bool:
    """Whether `backend`'s optional ML framework actually imports here.

    Mirrors `loader.import_all_model_backends`'s own best-effort import
    logic (missing framework -> not importable, not a crash) without
    triggering every OTHER backend's import as a side effect.
    """

    from fabric_defect_hub.loader import _MODEL_BACKEND_MODULES

    module_path = _MODEL_BACKEND_MODULES.get(backend)
    if module_path is None:
        return False
    try:
        importlib.import_module(module_path)
    except ImportError:
        return False
    return True
