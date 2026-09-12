"""The vendored `components/dinomaly` checkout, imported in isolation.

Dinomaly ships as flat top-level modules (`models.uad`, `dinov1`, `dinov2`,
`beit`, `optimizers`, `utils`, `dataset`) meant to be run from its own repo
root; there is no `pip install dinomaly`. This module used to just append the
checkout to `sys.path` permanently and let the adapter `from utils import ...`
— which left `utils`, `dataset` and `models` occupied process-wide by
Dinomaly's versions. MoECLIP vendors modules under the same generic names, and
the Benchmark tab runs both in one process, so whichever loaded first won.

`core.vendor.VendoredRepo` now imports the checkout inside a window and takes
its modules back out of `sys.modules` afterwards. See that module for the
mechanism and its one assumption. This file is the only place in the project
allowed to name Dinomaly's upstream modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

from fabric_defect_hub.core.vendor import VendoredRepo

_REPO = VendoredRepo(
    name="Dinomaly",
    directory="dinomaly",
    # Every top-level name the checkout occupies, including ones pulled in
    # transitively (`models.vit_encoder` imports `dinov2`/`beit`).
    owned_roots=("utils", "dataset", "models", "optimizers", "dinov1", "dinov2", "beit"),
    entry_modules=(
        "utils",
        "dataset",
        "optimizers",
        "dinov1.utils",
        "models",
        "models.vit_encoder",
        "models.uad",
        "models.vision_transformer",
    ),
    missing_hint=(
        "Expected the unmodified upstream repo under the repository-level components/dinomaly "
        "(see components/README.md)."
    ),
)


def vendor_root() -> Path:
    return _REPO.root


def import_vendor() -> dict[str, ModuleType]:
    """Import (once) and return Dinomaly's vendored modules by name, e.g.
    `import_vendor()["models.uad"].ViTill`.
    """

    # Upstream creates ~/.cache/dinomaly during import. Keep downloaded
    # backbone files inside the checkout so local/sandboxed deployments do
    # not depend on a writable home directory.
    os.environ.setdefault("DINOMALY_WEIGHTS_DIR", str(_REPO.root / ".cache" / "weights"))
    return _REPO.modules()
