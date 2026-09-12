"""`DatasetAdapter` for VisA (Visual Anomaly, Zou et al., ECCV 2022) — 12
non-fabric object categories (candle, capsules, pcb1-4, ...), each laid out
as:

    <category>/Data/Images/Normal/<id>.JPG
    <category>/Data/Images/Anomaly/<id>.JPG
    <category>/Data/Masks/Anomaly/<id>.png       # pixel ground truth

Like MVTec AD, VisA is registered here as a *cross-domain, eval-only*
benchmark: run a fabric-trained anomaly model against a completely different
image distribution. It is deliberately NOT in
`training.ANOMALY_TRAINABLE_DATASETS` — training on it is rejected.

VisA ships no on-disk train/test partition per category (just Normal/Anomaly
folders), so — like the flat-folder fabric datasets — the normal images are
split deterministically by sorted position (`train_ratio`) into a train pool
and a test pool; anomalous images always go to test. This keeps a would-be
`split="train"` normal-only slice disjoint from `split="test"`, though the
primary use here is `split="test"` for zero-shot evaluation.

Homepage: https://github.com/amazon-science/spot-diff
"""

from __future__ import annotations

import random
from pathlib import Path

from fabric_defect_hub.core.registry import register_dataset
from fabric_defect_hub.core.types import Annotations, Sample, Task
from fabric_defect_hub.datasets.base import IMAGE_SUFFIXES, DatasetAdapter



@register_dataset("visa")
class VisADataset(DatasetAdapter):
    """Configurable adapter over VisA's per-category Normal/Anomaly layout.

    Parameters (keyword), matching the other anomaly adapters:

    root: dataset root containing one subdirectory per category, each with
        `Data/Images/{Normal,Anomaly}` and `Data/Masks/Anomaly`.
    split: "train" (normal-only front slice) or "test" (normal back slice +
        all anomalies).
    category: a single category (e.g. "candle"). None aggregates every
        category found under `root`.
    num_samples: total images to load. None = everything in the selection.
    use_defect: include anomalous images (test split only).
    defect_ratio: fraction of the loaded set that is anomalous, applied only
        when `num_samples` is set and `use_defect` is True.
    task: drives `Sample.task`; pixel masks are attached when available.
    seed: RNG seed for reproducible subsampling.
    train_ratio: fraction of normal images assigned to the train pool.
    """

    name = "visa"

    def __init__(
        self,
        root: str,
        split: str = "test",
        category: str | None = None,
        num_samples: int | None = None,
        use_defect: bool = True,
        defect_ratio: float = 0.5,
        task: Task = "anomaly",
        seed: int = 0,
        train_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(root=root, split=split, **kwargs)
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if not 0.0 <= defect_ratio <= 1.0:
            raise ValueError(f"defect_ratio must be in [0, 1], got {defect_ratio}")
        if not 0.0 <= train_ratio <= 1.0:
            raise ValueError(f"train_ratio must be in [0, 1], got {train_ratio}")

        self.root_path = Path(root)
        self.category = category
        self.num_samples = num_samples
        self.use_defect = use_defect
        self.defect_ratio = defect_ratio
        self.task = task
        self.seed = seed
        self.train_ratio = train_ratio

    # ------------------------------------------------------------------ #
    # Category discovery
    # ------------------------------------------------------------------ #
    def _available_categories(self) -> list[str]:
        if not self.root_path.is_dir():
            return []
        return sorted(
            path.name
            for path in self.root_path.iterdir()
            if path.is_dir() and (path / "Data" / "Images" / "Normal").is_dir()
        )

    def _categories(self) -> list[str]:
        if self.category is None:
            return self._available_categories()
        available = self._available_categories()
        if self.category not in available:
            raise ValueError(
                f"unknown VisA category {self.category!r}; available: {', '.join(available) or '<none found>'}"
            )
        return [self.category]

    @staticmethod
    def _images(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        return sorted(
            path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _normal_pool(self) -> list[tuple[str, Path]]:
        pool: list[tuple[str, Path]] = []
        for category in self._categories():
            normals = self._images(self.root_path / category / "Data" / "Images" / "Normal")
            cut = round(len(normals) * self.train_ratio)
            selected = normals[:cut] if self.split == "train" else normals[cut:]
            pool.extend((category, path) for path in selected)
        return pool

    def _defect_pool(self) -> list[tuple[str, Path]]:
        if self.split != "test" or not self.use_defect:
            return []
        pool: list[tuple[str, Path]] = []
        for category in self._categories():
            anomaly_dir = self.root_path / category / "Data" / "Images" / "Anomaly"
            pool.extend((category, path) for path in self._images(anomaly_dir))
        return pool

    def _select(self):
        normal_pool = self._normal_pool()
        defect_pool = self._defect_pool()

        rng = random.Random(self.seed)
        rng.shuffle(normal_pool)
        rng.shuffle(defect_pool)

        if self.num_samples is None:
            return normal_pool, defect_pool
        if not defect_pool:
            return normal_pool[: self.num_samples], []

        n_defect = min(round(self.num_samples * self.defect_ratio), len(defect_pool))
        n_normal = min(self.num_samples - n_defect, len(normal_pool))
        return normal_pool[:n_normal], defect_pool[:n_defect]

    # ------------------------------------------------------------------ #
    # Sample building
    # ------------------------------------------------------------------ #
    def _build_normal_sample(self, category: str, image_path: Path) -> Sample:
        return Sample(
            id=f"{category}/Normal/{image_path.stem}",
            image_path=str(image_path),
            task=self.task,
            annotations=Annotations(is_anomalous=False),
            metadata={"category": category, "defect_type": "good"},
        )

    def _build_defect_sample(self, category: str, image_path: Path) -> Sample:
        mask_path = self.root_path / category / "Data" / "Masks" / "Anomaly" / f"{image_path.stem}.png"
        annotations = Annotations(is_anomalous=True)
        if mask_path.is_file():
            annotations.anomaly_mask = str(mask_path)
            if self.task == "segmentation":
                annotations.masks = [str(mask_path)]
        return Sample(
            id=f"{category}/Anomaly/{image_path.stem}",
            image_path=str(image_path),
            task=self.task,
            annotations=annotations,
            metadata={"category": category, "defect_type": "anomaly"},
        )

    def load_samples(self) -> list[Sample]:
        normal_pool, defect_pool = self._select()
        samples = [self._build_normal_sample(cat, path) for cat, path in normal_pool]
        samples += [self._build_defect_sample(cat, path) for cat, path in defect_pool]
        return samples
