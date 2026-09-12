"""DataAdapter: the `Sample` -> backend batch contract.

Every backend converts this project's unified `Sample` objects into whatever
its own training loop consumes, and those conversions genuinely differ —
MambaAD wants a bare ImageNet-normalized image tensor, MoECLIP wants a dict
with a jointly-augmented image+mask pair and a prompt class, torchvision
detection wants `(image, target_dict)` with box/mask tv_tensors. Forcing all
of them through one shared `DataLoader` was tried and reverted; it produced a
pipeline that was wrong for most backends.

**So the implementations stay separate. What is unified is the interface.**

A `DataAdapter` fixes four things, which is everything a caller needs to drive
any backend's data without knowing which backend it holds:

  1. Construction — `(samples, **backend options)`, `samples` always first.
  2. The map-style protocol — `__len__` / `__getitem__(int)`.
  3. `batch_spec()` — a *declaration* of what `__getitem__` returns: item
     shape, image layout/dtype, normalization statistics, mask semantics.
     Without it, "what does a batch look like here" is only answerable by
     reading each backend's source, and the normalization constants a run
     used never make it into the run log.
  4. `collate_fn` — the batching callable, since detection targets and
     variable-length masks cannot use torch's default collate.

`build_dataloader()` then works identically for every backend.

This module stays importable without torch on purpose (torch is imported
lazily inside `build_dataloader`), matching the two backend data modules that
already avoid subclassing `torch.utils.data.Dataset` for the same reason.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from fabric_defect_hub.core.types import Sample

# What `__getitem__` returns. A backend declares one of these rather than
# leaving callers to discover it by indexing and looking at the result.
ITEM_KINDS = (
    "image",         # a single image tensor
    "image_target",  # (image, target dict)  -- torchvision detection convention
    "image_mask",    # (image, mask tensor)  -- semantic segmentation
    "mapping",       # a dict of named tensors/values
)

# How a mask, if any, is encoded. `None` means the item carries no mask.
MASK_SEMANTICS = (
    "binary_float_1hw",       # FloatTensor[1, H, W], values in {0.0, 1.0}
    "binary_uint8_nhw",       # UInt8Tensor[N, H, W], one plane per instance
)

IMAGE_LAYOUTS = ("CHW", "HWC")


@dataclass(frozen=True)
class Normalization:
    """Per-channel mean/std actually applied to the image tensor.

    Recorded rather than assumed: ImageNet statistics and CLIP statistics
    differ, and a run that silently used the wrong one is a bug no metric
    surfaces.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.mean) != len(self.std):
            raise ValueError(f"mean/std length mismatch: {self.mean} vs {self.std}")
        if any(s == 0 for s in self.std):
            raise ValueError(f"normalization std must be non-zero: {self.std}")


@dataclass(frozen=True)
class BatchSpec:
    """What one item out of a `DataAdapter` looks like.

    Every string field is validated against a fixed vocabulary, so a typo is a
    hard error at declaration time rather than a silently wrong assumption
    downstream (same rule as `models.base.ModelCapabilities`).
    """

    item_kind: str
    image_layout: str = "CHW"
    image_dtype: str = "float32"
    normalization: Normalization | None = None
    mask_semantics: str | None = None
    image_size: tuple[int, int] | None = None  # None = images keep native resolution

    def __post_init__(self) -> None:
        if self.item_kind not in ITEM_KINDS:
            raise ValueError(f"unknown item_kind {self.item_kind!r}; expected one of {ITEM_KINDS}")
        if self.image_layout not in IMAGE_LAYOUTS:
            raise ValueError(
                f"unknown image_layout {self.image_layout!r}; expected one of {IMAGE_LAYOUTS}"
            )
        if self.mask_semantics is not None and self.mask_semantics not in MASK_SEMANTICS:
            raise ValueError(
                f"unknown mask_semantics {self.mask_semantics!r}; expected one of {MASK_SEMANTICS}"
            )

    def as_run_metadata(self) -> dict[str, Any]:
        """Flat, JSON-safe description for the run log — so a result row can
        say which preprocessing produced it.
        """

        return {
            "item_kind": self.item_kind,
            "image_layout": self.image_layout,
            "image_dtype": self.image_dtype,
            "image_size": list(self.image_size) if self.image_size else None,
            "mask_semantics": self.mask_semantics,
            "normalization_mean": list(self.normalization.mean) if self.normalization else None,
            "normalization_std": list(self.normalization.std) if self.normalization else None,
        }


class DataAdapter(ABC):
    """Base class for every `Sample` -> backend-batch conversion."""

    def __init__(self, samples: list[Sample]):
        self.samples = samples

    @abstractmethod
    def batch_spec(self) -> BatchSpec:
        """Declare what `__getitem__` returns. See `BatchSpec`."""

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """One converted item, matching `batch_spec()`."""

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def collate_fn(self) -> Callable | None:
        """The batching callable, or `None` to use torch's default collate.

        Detection targets vary in box count and cannot be stacked, so those
        adapters override this.
        """

        return None

    def build_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
    ):
        """A `torch.utils.data.DataLoader` over this adapter, wired with the
        right collate — identical call for every backend.
        """

        from torch.utils.data import DataLoader

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
        )
