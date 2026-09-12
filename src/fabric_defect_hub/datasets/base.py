"""DatasetAdapter: turns an arbitrary dataset on disk into a list of unified
`Sample` objects, without forcing every task into the same label format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fabric_defect_hub.core.types import Sample


# What counts as an image file on disk. Shared because it is a fact about
# image files, not about any one dataset's layout -- five adapters kept
# identical private copies, so adding a format meant finding all five.
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")

class DatasetAdapter(ABC):
    """Base class every concrete dataset (public or in-house) implements."""

    name: str

    def __init__(self, root: str, split: str = "test", **kwargs):
        self.root = root
        self.split = split
        self.options = kwargs

    @abstractmethod
    def load_samples(self) -> list[Sample]:
        """Return the unified `Sample` list for `self.split`."""

    def __len__(self) -> int:
        return len(self.load_samples())
