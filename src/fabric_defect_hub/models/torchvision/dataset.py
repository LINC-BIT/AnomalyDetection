"""Bridges our unified `Sample` list straight into a `torch.utils.data.Dataset`
that `torchvision.models.detection.*` can train/evaluate on directly.

Unlike the Ultralytics (`datasets/yolo_bbox.py`) and Anomalib
(`datasets/anomalib_folder.py`) backends, torchvision's detection models take
Python objects, not files on disk — there is no on-disk staging step here at
all: `Sample.image_path` is read straight into a tensor, `Sample.annotations`
straight into the `target` dict torchvision expects. Nothing is ever
symlinked or converted to an intermediate format.

Target dict convention (matches `torchvision.models.detection.{faster,mask}_rcnn`
exactly, verified against torchvision 0.27 — see module docstring in
`adapter.py` for how it's produced/consumed):

    {
        "boxes":  FloatTensor[N, 4]   absolute-pixel xyxy,
        "labels": Int64Tensor[N]      1..num_fg_classes (0 is reserved for background),
        "masks":  UInt8Tensor[N,H,W]  only when instance segmentation masks are used,
        "image_id": Int64Tensor[1],
        "area":   FloatTensor[N],
        "iscrowd": Int64Tensor[N],
    }
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from fabric_defect_hub.core.data_adapter import BatchSpec, DataAdapter
from fabric_defect_hub.core.types import Sample


def build_class_map(samples: list[Sample], class_names: list[str] | None = None) -> dict[str, int]:
    """Assign a stable foreground class id (>=1; 0 is background) to every
    label seen across `samples`. Pass `class_names` to fix the id order
    explicitly (e.g. to match a previously trained checkpoint).
    """

    if class_names is not None:
        return {name: idx + 1 for idx, name in enumerate(class_names)}

    labels: set[str] = set()
    for sample in samples:
        if sample.annotations.labels:
            labels.update(sample.annotations.labels)
    return {name: idx + 1 for idx, name in enumerate(sorted(labels))} or {"defect": 1}


class SampleDetectionDataset(DataAdapter, Dataset):
    """`Sample` list -> (image_tensor, target_dict) for torchvision detection models.

    `with_masks=True` also populates `target["masks"]` from
    `Sample.annotations.masks` (segmentation) or, failing that,
    `Sample.annotations.anomaly_mask` (a single binary mask reused for every
    box in the image — ZJU-Leaper's mask convention) — required for Mask
    R-CNN, ignored by Faster R-CNN.

    `transforms`: an optional `torchvision.transforms.v2`-style callable
    applied to `(image, target)` jointly (so box/mask-aware augmentations —
    flips, crops — stay consistent with the image). See
    `presets.build_transforms()` for the fabric-tailored default.
    """

    def __init__(
        self,
        samples: list[Sample],
        class_map: dict[str, int] | None = None,
        with_masks: bool = False,
        transforms=None,
    ):
        super().__init__(samples)
        self.class_map = class_map or build_class_map(samples)
        self.with_masks = with_masks
        self.transforms = transforms

    def batch_spec(self) -> BatchSpec:
        return BatchSpec(
            item_kind="image_target",
            # torchvision detection models take unnormalized [0, 1] images and
            # do their own normalization inside `GeneralizedRCNNTransform`.
            normalization=None,
            mask_semantics="binary_uint8_nhw" if self.with_masks else None,
            image_size=None,  # native resolution; the model resizes internally
        )

    @property
    def collate_fn(self):
        return detection_collate_fn

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        from PIL import Image
        from torchvision import tv_tensors
        from torchvision.transforms.v2 import functional as F

        sample = self.samples[index]
        with Image.open(sample.image_path) as img:
            pil_image = img.convert("RGB")
        width, height = pil_image.size

        boxes = sample.annotations.boxes or []
        labels = sample.annotations.labels or []
        label_ids = [self.class_map[label] for label in labels if label in self.class_map]
        kept_boxes = [box for box, label in zip(boxes, labels) if label in self.class_map]

        boxes_tensor = torch.as_tensor(kept_boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(label_ids, dtype=torch.int64)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if len(kept_boxes) > 0
            else torch.zeros((0,), dtype=torch.float32)
        )

        target: dict[str, Any] = {
            "boxes": tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(height, width)),
            "labels": labels_tensor,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": torch.zeros((len(kept_boxes),), dtype=torch.int64),
        }

        if self.with_masks:
            target["masks"] = tv_tensors.Mask(self._build_masks(sample, kept_boxes, height, width))

        image = tv_tensors.Image(F.pil_to_tensor(pil_image))
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = F.to_dtype(image, torch.float32, scale=True)

        return image, target

    @staticmethod
    def _build_masks(sample: Sample, kept_boxes: list, height: int, width: int) -> torch.Tensor:
        """Per-instance binary masks, shape [N, H, W]."""

        n = len(kept_boxes)
        if n == 0:
            return torch.zeros((0, height, width), dtype=torch.uint8)

        mask_path = sample.annotations.anomaly_mask
        if mask_path is None:
            # No pixel mask available: fall back to filled bounding boxes so
            # Mask R-CNN still has a supervision signal (coarser than a real
            # mask, but valid — matches the box exactly).
            masks = torch.zeros((n, height, width), dtype=torch.uint8)
            for i, (x1, y1, x2, y2) in enumerate(kept_boxes):
                masks[i, int(y1) : int(y2), int(x1) : int(x2)] = 1
            return masks

        import numpy as np
        from PIL import Image as PILImage

        with PILImage.open(mask_path) as m:
            arr = np.array(m.convert("L"))
        mask_arr = torch.from_numpy(arr > 0) if arr.shape == (height, width) else None
        if mask_arr is None:
            # Mask resolution doesn't match the image (shouldn't happen for
            # ZJU-Leaper, but stay safe rather than crash training).
            masks = torch.zeros((n, height, width), dtype=torch.uint8)
            for i, (x1, y1, x2, y2) in enumerate(kept_boxes):
                masks[i, int(y1) : int(y2), int(x1) : int(x2)] = 1
            return masks

        # ZJU-Leaper ships one whole-image defect mask, not per-box instance
        # masks; reuse it (cropped to each box) for every box in the image.
        return mask_arr.unsqueeze(0).repeat(n, 1, 1).to(torch.uint8)


def detection_collate_fn(batch: list[tuple[Any, dict[str, Any]]]):
    """Detection targets vary in box count, so the default collate (which
    tries to stack tensors) doesn't work; batch as a tuple of lists instead
    — this is torchvision's own documented collate convention for detection.
    """

    return tuple(zip(*batch))


class SampleSegmentationDataset(DataAdapter, Dataset):
    """`Sample` list -> (image_tensor, mask_tensor) for semantic segmentation.
    Returns:
        image: FloatTensor[3, H, W] normalized to [0, 1]
        mask: FloatTensor[1, H, W] with binary values (0.0 or 1.0)
    """

    def __init__(self, samples: list[Sample], transforms=None):
        super().__init__(samples)
        self.transforms = transforms

    def batch_spec(self) -> BatchSpec:
        return BatchSpec(
            item_kind="image_mask",
            normalization=None,
            mask_semantics="binary_float_1hw",
            image_size=None,
        )

    @property
    def collate_fn(self):
        return segmentation_collate_fn

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image
        from torchvision import tv_tensors
        from torchvision.transforms.v2 import functional as F
        import numpy as np

        sample = self.samples[index]
        with Image.open(sample.image_path) as img:
            pil_image = img.convert("RGB")
        width, height = pil_image.size

        mask_path = sample.annotations.anomaly_mask
        if mask_path is None and sample.annotations.masks:
            mask_path = sample.annotations.masks[0]

        if mask_path is not None:
            with Image.open(mask_path) as m:
                mask_arr = np.array(m.convert("L")) > 0
            mask_tensor = torch.from_numpy(mask_arr).to(torch.float32).unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, height, width), dtype=torch.float32)

        image = tv_tensors.Image(F.pil_to_tensor(pil_image))
        mask = tv_tensors.Mask(mask_tensor)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        else:
            image = F.to_dtype(image, torch.float32, scale=True)

        return image, mask


def segmentation_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """ Collate function for segmentation: returns list of image tensors and list of mask tensors. """
    return tuple(zip(*batch))
