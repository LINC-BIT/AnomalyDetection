"""`DatasetAdapter` for MVTec LOCO AD (Bergmann et al., IJCV 2022) — the
"logical constraints" anomaly benchmark: 5 object categories (breakfast_box,
juice_bottle, pushpins, screw_bag, splicing_connectors), each with both
*structural* anomalies (scratches, dents — local) and *logical* anomalies
(missing/extra/misplaced parts — global).

Layout is MVTec-AD-like but with two differences this adapter handles:

    <category>/train/good/<id>.png
    <category>/test/{good,logical_anomalies,structural_anomalies}/<id>.png
    <category>/ground_truth/<anomaly_type>/<id>/<k>.png   # per-image mask dir

i.e. the ground truth for one anomalous image is a *directory* of one or
more mask PNGs (logical anomalies can mark several disconnected regions),
not a single `<id>_mask.png` file as in MVTec AD. There is also a
`validation/` split on disk, which this adapter ignores (this project's
split contract is train/test only).

Registered as a *cross-domain, eval-only* benchmark (like MVTec AD / VisA):
NOT in `training.ANOMALY_TRAINABLE_DATASETS`.

Homepage: https://www.mvtec.com/company/research/datasets/mvtec-loco
"""

from __future__ import annotations

import random
from pathlib import Path

from fabric_defect_hub.core.registry import register_dataset
from fabric_defect_hub.core.types import Annotations, Sample, Task
from fabric_defect_hub.datasets.base import IMAGE_SUFFIXES, DatasetAdapter



@register_dataset("mvtec-loco")
class MVTecLOCODataset(DatasetAdapter):
    """Configurable adapter over the MVTec LOCO per-category folder layout.

    Parameters mirror `MVTecADDataset` exactly (root, split, category,
    num_samples, use_defect, defect_ratio, task, seed) so the two are
    interchangeable from the loader/UI's point of view. "train" yields
    normal samples only (`<category>/train/good`).
    """

    name = "mvtec-loco"

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
        **kwargs,
    ):
        super().__init__(root=root, split=split, **kwargs)
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if not 0.0 <= defect_ratio <= 1.0:
            raise ValueError(f"defect_ratio must be in [0, 1], got {defect_ratio}")

        self.root_path = Path(root)
        self.category = category
        self.num_samples = num_samples
        self.use_defect = use_defect
        self.defect_ratio = defect_ratio
        self.task = task
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Category discovery
    # ------------------------------------------------------------------ #
    def _available_categories(self) -> list[str]:
        if not self.root_path.is_dir():
            return []
        return sorted(
            path.name
            for path in self.root_path.iterdir()
            if path.is_dir() and (path / "train" / "good").is_dir()
        )

    def _categories(self) -> list[str]:
        if self.category is None:
            return self._available_categories()
        available = self._available_categories()
        if self.category not in available:
            raise ValueError(
                f"unknown MVTec LOCO category {self.category!r}; available: {', '.join(available) or '<none found>'}"
            )
        return [self.category]

    @staticmethod
    def _images(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _normal_pool(self) -> list[tuple[str, Path]]:
        subdir = "train" if self.split == "train" else "test"
        pool: list[tuple[str, Path]] = []
        for category in self._categories():
            pool.extend((category, path) for path in self._images(self.root_path / category / subdir / "good"))
        return pool

    def _defect_pool(self) -> list[tuple[str, str, Path]]:
        if self.split != "test" or not self.use_defect:
            return []
        pool: list[tuple[str, str, Path]] = []
        for category in self._categories():
            test_dir = self.root_path / category / "test"
            if not test_dir.is_dir():
                continue
            for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"):
                pool.extend((category, defect_dir.name, path) for path in self._images(defect_dir))
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
    def _first_mask(self, category: str, defect_type: str, stem: str) -> str | None:
        """LOCO stores each anomalous image's ground truth as a *directory*
        of one or more mask PNGs. Return the first (sorted) one, or None if
        the directory is absent. (Image-level metrics don't need a mask at
        all; pixel metrics use whichever region masks exist.)
        """

        gt_dir = self.root_path / category / "ground_truth" / defect_type / stem
        masks = self._images(gt_dir)
        return str(masks[0]) if masks else None

    def _build_normal_sample(self, category: str, image_path: Path) -> Sample:
        return Sample(
            id=f"{category}/good/{image_path.stem}",
            image_path=str(image_path),
            task=self.task,
            annotations=Annotations(is_anomalous=False),
            metadata={"category": category, "defect_type": "good"},
        )

    def _build_defect_sample(self, category: str, defect_type: str, image_path: Path) -> Sample:
        annotations = Annotations(is_anomalous=True)
        mask = self._first_mask(category, defect_type, image_path.stem)
        if mask is not None:
            annotations.anomaly_mask = mask
            if self.task == "segmentation":
                annotations.masks = [mask]
        return Sample(
            id=f"{category}/{defect_type}/{image_path.stem}",
            image_path=str(image_path),
            task=self.task,
            annotations=annotations,
            metadata={"category": category, "defect_type": defect_type},
        )

    def load_samples(self) -> list[Sample]:
        normal_pool, defect_pool = self._select()
        samples = [self._build_normal_sample(category, path) for category, path in normal_pool]
        samples += [
            self._build_defect_sample(category, defect_type, path)
            for category, defect_type, path in defect_pool
        ]
        return samples
