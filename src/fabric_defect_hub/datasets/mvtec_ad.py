"""`DatasetAdapter` for the classic MVTec AD industrial anomaly-detection
benchmark (Bergmann et al., CVPR 2019).

Unlike the fabric datasets already in this project, MVTec AD is not a
fabric benchmark — it ships 15 object/texture categories (bottle, cable,
carpet, ...), each in its own subdirectory with the standard MVTec-AD
layout: `train/good`, `test/<defect_type>/`, `ground_truth/<defect_type>/`.
Each category has *multiple* defect-type subfolders (unlike RAW_FABRID's
single `defect` bucket), and ground-truth masks are named `<id>_mask.png`.

It is registered here primarily to support cross-domain zero-shot /
robustness evaluation: run a fabric-trained anomaly model against a
completely different, non-fabric image distribution.

Mirrors `ZJULeaperDataset`/`RawFabricDataset`'s sample-count control so
every dataset in this project is interchangeable from the loader/UI's
point of view:

1. Sample-count control — `num_samples` (None = use everything, i.e. the
   "few-shot / full data" regime; a small number like 300-400 = the
   "low-shot" regime).
2. Defect adaptation — `use_defect` (False = zero-shot / normal-only; True =
   include defective images).
3. Category filtering — `category` selects one MVTec AD category (e.g.
   "bottle"); None aggregates every category found under `root`.
4. Unified config — `num_samples`, `use_defect`, `defect_ratio` together fix
   how many images, whether defects are present, and the defect fraction.

Dataset homepage: https://www.mvtec.com/company/research/datasets/mvtec-ad
"""

from __future__ import annotations

import random
from pathlib import Path

from fabric_defect_hub.core.registry import register_dataset
from fabric_defect_hub.core.types import Annotations, Sample, Task
from fabric_defect_hub.datasets.base import IMAGE_SUFFIXES, DatasetAdapter



@register_dataset("mvtec-ad")
class MVTecADDataset(DatasetAdapter):
    """Configurable adapter over the MVTec AD per-category folder layout.

    Parameters (all via keyword, consumed here rather than left in
    `self.options`):

    root: dataset root containing one subdirectory per category (e.g.
        `bottle/`, `cable/`, ...), each with `train/good`,
        `test/<defect_type>/...`, `ground_truth/<defect_type>/..._mask.png`.
    split: "train" or "test". "train" only ever yields normal samples
        (`<category>/train/good`); MVTec AD's train split is one-class.
    category: a single category name (e.g. "bottle"). None = aggregate
        every category found under `root`.
    num_samples: total images to load. None = all images in the selection
        (few-shot / full-data). A small int (e.g. 350) = low-shot regime.
    use_defect: include defective images. False = zero-shot (normal only).
        Ignored for split="train" (always normal-only).
    defect_ratio: fraction of the loaded set that is defective. Only applied
        when `num_samples` is set and `use_defect` is True; ignored when
        `num_samples` is None (then all normal + all defect are used).
    task: "anomaly" (default), "detection", or "segmentation". Drives
        `Sample.task`; pixel masks are always attached when available
        regardless of `task`.
    seed: RNG seed for reproducible subsampling.
    """

    name = "mvtec-ad"

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
                f"unknown MVTec AD category {self.category!r}; available: {', '.join(available) or '<none found>'}"
            )
        return [self.category]

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _images(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    def _normal_pool(self) -> list[tuple[str, Path]]:
        subdir = "train" if self.split == "train" else "test"
        pool: list[tuple[str, Path]] = []
        for category in self._categories():
            good_dir = self.root_path / category / subdir / "good"
            pool.extend((category, path) for path in self._images(good_dir))
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
            # Few-shot / full-data: take everything in the selection.
            return normal_pool, defect_pool

        # Low-shot: honour total count and defect ratio.
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
            id=f"{category}/good/{image_path.stem}",
            image_path=str(image_path),
            task=self.task,
            annotations=Annotations(is_anomalous=False),
            metadata={"category": category, "defect_type": "good"},
        )

    def _build_defect_sample(self, category: str, defect_type: str, image_path: Path) -> Sample:
        mask_path = self.root_path / category / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
        annotations = Annotations(is_anomalous=True)
        if mask_path.is_file():
            annotations.anomaly_mask = str(mask_path)
            if self.task == "segmentation":
                annotations.masks = [str(mask_path)]
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
