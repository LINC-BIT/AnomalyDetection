"""The vendored `components/moeclip` checkout, imported in isolation.

MoECLIP ships as flat top-level modules meant to be run from its own repo root
-- `utils`, `dataset`, `model`, `forward_utils` -- with no package prefix and
no pip release. Those names collide with Dinomaly's (`utils`, `dataset`), and
the Benchmark tab runs every canonical model back to back in one process (see
`web/benchmark.py`), so neither repo may be left occupying them.

`core.vendor.VendoredRepo` handles that; see it for the mechanism and its one
assumption. This file is the only place in the project allowed to name
MoECLIP's upstream modules.

MoECLIP also hardcodes `torch.device("cuda:0")` when allocating its LoRA
expert weights (`model/moe_adapter.py::SimpleLoraExpert.__init__`), which
makes the model unconstructable on any CPU/MPS machine. Until that is patched
on the fork (the proper fix -- see `components/README.md`),
`cuda_free_module_init()` supplies a scoped compatibility shim.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

from fabric_defect_hub.core.vendor import VendoredRepo

_REPO = VendoredRepo(
    name="MoECLIP",
    directory="moeclip",
    # `model` and `dataset` are packages, so their submodules
    # (`model.clip`, `dataset.constants`, ...) move in and out along with them.
    owned_roots=("utils", "dataset", "model", "forward_utils"),
    entry_modules=(
        "utils",
        "dataset.constants",
        "model.clip",
        "model.moe_adapter",
        "model.tokenizer",
        "forward_utils",
    ),
    missing_hint=(
        "Expected the checkout at the repository-level components/moeclip "
        "directory (or textile/components for legacy trees). Restore the "
        "vendor checkout before using MoECLIP."
    ),
)


def vendor_root() -> Path:
    return _REPO.root


def import_vendor() -> dict[str, ModuleType]:
    """Import (once) and return MoECLIP's vendored modules by name, e.g.
    `import_vendor()["model.moe_adapter"].MoECLIP`.
    """

    return _REPO.modules()


@contextmanager
def cuda_free_module_init(device) -> Iterator[None]:
    """Make `nn.Linear(..., device='cuda:0')` fall back to `device`.

    `SimpleLoraExpert.__init__` allocates its LoRA `A`/`B` matrices with a
    hardcoded `torch.device("cuda:0")`, so building `MoECLIP` raises on any
    machine without CUDA -- including this project's dev laptop and CI.
    Inside this context (and only when CUDA really is unavailable), any
    `nn.Linear` asked for a CUDA device is built on `device` instead;
    everything else is untouched, and the patch is reverted on exit.

    The subclass is transparent to `isinstance`, `state_dict()` keys and
    checkpoint round-trips, so a model built here loads a checkpoint
    trained on a CUDA box and vice versa.
    """

    import torch

    if torch.cuda.is_available():
        yield
        return

    original = torch.nn.Linear
    device_str = str(device)

    class _DeviceFallbackLinear(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, device=None, **kwargs):
            if device is not None and torch.device(device).type == "cuda":
                device = torch.device(device_str)
            super().__init__(*args, device=device, **kwargs)

    torch.nn.Linear = _DeviceFallbackLinear
    try:
        yield
    finally:
        torch.nn.Linear = original
