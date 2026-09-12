"""Shared base for *flat-folder, image-level* anomaly datasets: a single
directory of normal images plus one or more directories of defective images,
with no pixel masks and — crucially — no author-provided train/test split.

TILDA-400 and the "Fabric Defects Dataset" both look like this. Unlike
MVTec-AD / RAW_FABRID / ZJU-Leaper (which ship their own `train/`/`test/`
partition on disk), these are just class folders. So this base *synthesizes*
a deterministic, leak-free split: the normal images are sorted and cut at
`train_ratio`; the front slice is the train pool, the back slice is the test
pool, and every defective image goes to test. The cut is by sorted position,
not by RNG, so `split="train"` and `split="test"` are always disjoint
regardless of the sampling `seed` — a normal image is never both trained on
and evaluated against.

Concrete datasets subclass this and set `NORMAL_DIRNAME` (+ optionally
`ROOT_SUBDIR` when the images live one level below the linked root) and
register themselves with `@register_dataset`.
"""

from __future__ import annotations

import random
from pathlib import Path

from fabric_defect_hub.core.types import Annotations, Sample, Task
from fabric_defect_hub.datasets.base import IMAGE_SUFFIXES, DatasetAdapter



class FlatFolderAnomalyDataset(DatasetAdapter):
    """Base adapter for a normal folder + N defect folders, image-level only.

    Subclass contract:
    * `name`            -- registry name (also set via `@register_dataset`).
    * `NORMAL_DIRNAME`  -- the folder holding normal images (e.g. "good",
                           "defect free").
    * `ROOT_SUBDIR`     -- optional extra path segment between the linked
                           root and the class folders (e.g. the "Fabric
                           Defect Dataset" nesting); "" when the class
                           folders sit directly under `root`.

    Parameters (keyword), matching every other fabric adapter so the loader
    /pipeline can treat them interchangeably:

    split: "train" (normal-only front slice) or "test" (normal back slice +
        all defects).
    num_samples: total images to load. None = everything in the selection.
    use_defect: include defective images (test split only). False = normal
        only.
    defect_ratio: fraction of the loaded set that is defective, applied only
        when `num_samples` is set and `use_defect` is True.
    task: drives `Sample.task`. No masks exist, so "segmentation" yields no
        pixel ground truth here.
    seed: RNG seed for reproducible subsampling (not for the train/test cut,
        which is deterministic by sorted position).
    train_ratio: fraction of normal images assigned to the train pool.
    """

    NORMAL_DIRNAME: str = "good"
    ROOT_SUBDIR: str = ""

    def __init__(
        self,
        root: str,
        split: str = "test",
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

        self.base_path = Path(root) / self.ROOT_SUBDIR if self.ROOT_SUBDIR else Path(root)
        self.num_samples = num_samples
        self.use_defect = use_defect
        self.defect_ratio = defect_ratio
        self.task = task
        self.seed = seed
        self.train_ratio = train_ratio

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    @staticmethod
    def _images(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    def _defect_dirnames(self) -> list[str]:
        if not self.base_path.is_dir():
            return []
        return sorted(
            path.name
            for path in self.base_path.iterdir()
            if path.is_dir() and path.name != self.NORMAL_DIRNAME
        )

    def _normal_split(self) -> list[Path]:
        """Deterministic, disjoint normal partition (see module docstring)."""

        normals = self._images(self.base_path / self.NORMAL_DIRNAME)
        cut = round(len(normals) * self.train_ratio)
        return normals[:cut] if self.split == "train" else normals[cut:]

    def _defect_images(self) -> list[tuple[str, Path]]:
        if self.split != "test" or not self.use_defect:
            return []
        pool: list[tuple[str, Path]] = []
        for defect_type in self._defect_dirnames():
            pool.extend((defect_type, path) for path in self._images(self.base_path / defect_type))
        return pool

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _select(self):
        normal_pool = self._normal_split()
        defect_pool = self._defect_images()

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
    def _build_normal_sample(self, image_path: Path) -> Sample:
        return Sample(
            id=f"{self.NORMAL_DIRNAME}/{image_path.stem}",
            image_path=str(image_path),
            task=self.task,
            annotations=Annotations(is_anomalous=False),
            metadata={"defect_type": "good"},
        )

    def _build_defect_sample(self, defect_type: str, image_path: Path) -> Sample:
        return Sample(
            id=f"{defect_type}/{image_path.stem}",
            image_path=str(image_path),
            task=self.task,
            annotations=Annotations(is_anomalous=True),
            metadata={"defect_type": defect_type},
        )

    def load_samples(self) -> list[Sample]:
        normal_pool, defect_pool = self._select()
        samples = [self._build_normal_sample(path) for path in normal_pool]
        samples += [self._build_defect_sample(dtype, path) for dtype, path in defect_pool]
        return samples
