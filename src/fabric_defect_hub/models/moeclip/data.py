"""`Sample` -> MoECLIP batch adapter.

Upstream reads its data through `dataset/__init__.py::BaseDataset`, which
is driven by per-dataset `.jsonl` metadata files shipped in the repo plus
a hardcoded `BASE_PATH` in `dataset/constants.py` — there is no way to
hand it a list of images. Rather than materialize a fake metadata file per
run (the staging trick the anomalib/Dinomaly adapters use, which works
there because those backends read a *folder layout*), this reimplements
`BaseDataset.__getitem__` directly over this project's `Sample` objects.

The transform pipeline is deliberately identical to upstream's
`BaseDataset(..., text=True)` — the variant `train.py` actually uses:
bicubic resize + CLIP normalization, and the geometric augmentations
(rotation / translation / h+v flip) applied to the image and its mask
*jointly*, by concatenating them into one 4-channel tensor. The colour
jitters in `BaseDataset` only apply to `text=False`, which upstream never
constructs, so they are not reproduced here.
"""

from __future__ import annotations

import math
from typing import Callable

from fabric_defect_hub.core.data_adapter import BatchSpec, DataAdapter, Normalization
from fabric_defect_hub.core.types import Sample

# CLIP's image statistics, as upstream hardcodes them.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(img_size: int, train: bool):
    """(image transform, mask transform, joint geometric augmentation)."""

    from PIL import Image
    from torchvision import transforms

    image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), Image.NEAREST),
            transforms.ToTensor(),
        ]
    )
    if not train:
        return image_transform, mask_transform, None

    joint = transforms.Compose(
        [
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=math.degrees(math.pi / 6))], p=0.5
            ),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )
    return image_transform, mask_transform, joint


class SampleDataset(DataAdapter):
    """A map-style dataset over `Sample`s, yielding the exact dict
    upstream's training/eval loops consume: `image`, `mask`, `label`,
    `file_name`, `class_name`.

    Not a `torch.utils.data.Dataset` subclass on purpose — `DataLoader`
    only needs `__len__`/`__getitem__` from a map-style dataset, and not
    subclassing keeps this module importable (and unit-testable) without
    torch installed.

    `class_name_fn` maps a sample to its prompt class (see
    `presets.class_name_for`); the batch's class names decide which text
    embeddings the loss is computed against.
    """

    def __init__(
        self,
        samples: list[Sample],
        img_size: int,
        class_name_fn: Callable[[Sample], str],
        train: bool = True,
    ):
        import torch

        super().__init__(samples)
        self._torch = torch
        self.img_size = img_size
        self.class_name_fn = class_name_fn
        self.train = train
        self.transform_x, self.transform_mask, self.joint = build_transforms(img_size, train)

    def batch_spec(self) -> BatchSpec:
        return BatchSpec(
            item_kind="mapping",
            # CLIP's own image statistics, not ImageNet's -- the distinction
            # this declaration exists to make explicit.
            normalization=Normalization(mean=CLIP_MEAN, std=CLIP_STD),
            mask_semantics="binary_float_1hw",
            image_size=(self.img_size, self.img_size),
        )

    def __getitem__(self, index: int):
        from PIL import Image

        torch = self._torch
        sample = self.samples[index]
        image = self.transform_x(Image.open(sample.image_path).convert("RGB"))

        mask_path = sample.annotations.anomaly_mask
        label = 1 if sample.annotations.is_anomalous else 0
        if label and mask_path:
            mask = self.transform_mask(Image.open(mask_path).convert("L"))
            mask = (mask != 0).float()
        else:
            # Normal images have an all-zero mask by definition; defective
            # ones without pixel ground truth are filtered out before
            # training (see `adapter._select_train_samples`), so an
            # all-zero mask here only ever means "normal" at train time.
            mask = torch.zeros([1, self.img_size, self.img_size])

        if self.joint is not None:
            stacked = self.joint(torch.cat([image, mask], dim=0))
            image, mask = stacked[0:3], stacked[3:4]

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(label).to(torch.int64),
            "file_name": sample.image_path,
            "class_name": self.class_name_fn(sample),
        }
