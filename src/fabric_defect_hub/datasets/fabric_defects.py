"""`DatasetAdapter` for the "Fabric Defects Dataset", a fabric-texture defect
dataset laid out as flat class folders one level below the linked root:
`Fabric Defect Dataset/defect free/` (normal) plus `hole/`, `Vertical/`,
`horizontal/`, `lines/`, `stain/` (defect types). No author-provided
train/test split, so this reuses `FlatFolderAnomalyDataset`'s synthesized,
leak-free split.

Unlike TILDA-400, this dataset is **not** purely image-level: the `hole`,
`Vertical`, and `horizontal` defect folders additionally ship binary
segmentation masks named `<stem>_processed (<n>).<ext>` alongside each base
photo `<stem>.<ext>` (`lines`/`stain` have no masks at all). These are NOT
independent defect photos — visually they are black frames with small white
speckle regions marking the defect location, and there can be 1-3 per base
image (multiple annotated regions/passes). This adapter therefore:

1. Excludes every `_processed (<n>)` file from the image pool so it is never
   counted as its own `Sample`.
2. Attaches the first (sorted) mask to its base image's
   `Annotations.anomaly_mask`, and — when `task="segmentation"` — all of them
   to `Annotations.masks`.

In-domain fabric, so it is a *training-eligible* source (see
`training.ANOMALY_TRAINABLE_DATASETS` and the `fabric-train` composite), and
also usable on its own for anomaly evaluation (image-level everywhere,
pixel-level where a mask exists).
"""

from __future__ import annotations

import re
from pathlib import Path

from fabric_defect_hub.core.registry import register_dataset
from fabric_defect_hub.core.types import Sample
from fabric_defect_hub.datasets.base import IMAGE_SUFFIXES
from fabric_defect_hub.datasets.flat_folder import FlatFolderAnomalyDataset

# Matches the mask-file naming convention: "<stem>_processed (<n>)" (before
# the extension) -- the space before "(" is inconsistent in the raw dataset
# (both "10_processed (1)" and "15_processed(2)" occur), hence "\s*".
_PROCESSED_MASK_RE = re.compile(r"^(?P<stem>.+)_processed\s*\(\d+\)$")

# Masks are images; the accepted formats are the same set.
_MASK_SUFFIXES = IMAGE_SUFFIXES


@register_dataset("fabric-defects")
class FabricDefectsDataset(FlatFolderAnomalyDataset):
    name = "fabric-defects"
    NORMAL_DIRNAME = "defect free"
    # Images sit under "<root>/Fabric Defect Dataset/<class>/", not directly
    # under the linked root.
    ROOT_SUBDIR = "Fabric Defect Dataset"

    @staticmethod
    def _images(directory: Path) -> list[Path]:
        """Same discovery as the base class, minus mask files — a
        `_processed (<n>)` file must never be treated as its own photo."""

        images = FlatFolderAnomalyDataset._images(directory)
        return [path for path in images if _PROCESSED_MASK_RE.match(path.stem) is None]

    @staticmethod
    def _masks_for(image_path: Path) -> list[Path]:
        """Every `_processed (<n>)` mask belonging to `image_path`'s stem, in
        the same directory, sorted for a deterministic "first mask" pick.
        Matched via `_PROCESSED_MASK_RE` (not a fixed prefix) so both the
        "_processed (1)" and "_processed(1)" spacing variants are found.
        """

        return sorted(
            path
            for path in image_path.parent.iterdir()
            if path.suffix.lower() in _MASK_SUFFIXES
            and (match := _PROCESSED_MASK_RE.match(path.stem)) is not None
            and match.group("stem") == image_path.stem
        )

    def _build_defect_sample(self, defect_type: str, image_path: Path) -> Sample:
        sample = super()._build_defect_sample(defect_type, image_path)
        masks = self._masks_for(image_path)
        if masks:
            sample.annotations.anomaly_mask = str(masks[0])
            if self.task == "segmentation":
                sample.annotations.masks = [str(mask) for mask in masks]
        return sample
