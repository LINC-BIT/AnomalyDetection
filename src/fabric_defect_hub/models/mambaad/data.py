"""`Sample` -> MambaAD batch adapter. One-class training needs only the
image (no mask, no label) -- the same ImageNet-normalized resize+center-crop
preprocessing upstream's `DefaultAD` dataset config declares
(`configs/mambaad/mambaad_mvtec.py`'s `data.train_transforms`), reimplemented
directly over `Sample` the same way `models/dinomaly` and `models/moeclip`
each have their own thin `Sample` -> tensor bridge rather than a shared one
(the three backends' native transform pipelines genuinely differ).
"""

from __future__ import annotations

from typing import Callable

from fabric_defect_hub.core.data_adapter import BatchSpec, DataAdapter, Normalization
from fabric_defect_hub.core.types import Sample

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int) -> Callable:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ImageOnlyDataset(DataAdapter):
    """Map-style dataset over `Sample`s, yielding just the transformed
    image tensor -- one-class training needs nothing else. Not a
    `torch.utils.data.Dataset` subclass so this module stays importable
    without torch (same reasoning as `models/moeclip/data.py::SampleDataset`);
    `DataAdapter` is torch-free for that reason too.
    """

    def __init__(self, samples: list[Sample], image_size: int):
        super().__init__(samples)
        self.image_size = image_size
        self.transform = build_transform(image_size)

    def batch_spec(self) -> BatchSpec:
        return BatchSpec(
            item_kind="image",
            normalization=Normalization(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            image_size=(self.image_size, self.image_size),
        )

    def __getitem__(self, index: int):
        from PIL import Image

        image = Image.open(self.samples[index].image_path).convert("RGB")
        return self.transform(image)
