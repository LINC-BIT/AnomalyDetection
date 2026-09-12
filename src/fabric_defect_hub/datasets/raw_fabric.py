"""`DatasetAdapter` for the RAW_FABRID raw-fabric anomaly-detection dataset.

RAW_FABRID ships its images pre-cropped into an MVTec-AD-style folder tree
(`MVTec/train/good`, `MVTec/test/good`, `MVTec/test/defect`,
`MVTec/ground_truth/defect`) alongside the original high-resolution images,
COCO annotations and CSV metadata at the dataset root. The MVTec tree is
already exactly the `Sample`/anomaly-task shape this project needs — the
same layout `anomalib_folder.py` stages *to* for other datasets — so this
adapter reads it directly instead of reparsing the COCO/CSV originals.

Mirrors `ZJULeaperDataset`'s sample-count control so both datasets are
interchangeable from the loader/UI's point of view:

1. Sample-count control — `num_samples` (None = use everything, i.e. the
   "few-shot / full data" regime; a small number like 300-400 = the
   "low-shot" regime).
2. Defect adaptation — `use_defect` (False = zero-shot / normal-only; True =
   include defective images). Only meaningful for `split="test"`; the MVTec
   `train` split is one-class (normal only) by construction.
3. Unified config — `num_samples`, `use_defect`, `defect_ratio` together fix
   how many images, whether defects are present, and the defect fraction.

RAW_FABRID has no texture/pattern subdivision (unlike ZJU-Leaper), so there
is no `pattern` parameter here.
"""

from __future__ import annotations

import random
from pathlib import Path

from fabric_defect_hub.core.registry import register_dataset
from fabric_defect_hub.core.types import Annotations, Sample, Task
from fabric_defect_hub.datasets.base import IMAGE_SUFFIXES, DatasetAdapter



@register_dataset("raw-fabric")
class RawFabricDataset(DatasetAdapter):
    """Configurable adapter over the RAW_FABRID `MVTec/` folder layout.

    Parameters (all via keyword, consumed here rather than left in
    `self.options`):

    root: dataset root containing the `MVTec/` subdirectory (i.e. the
        `RAW_FABRID` folder itself, whether reached directly or through a
        symlink — callers should pass an already-resolved path).
    split: "train" or "test". "train" only ever yields normal samples
        (`MVTec/train/good`); RAW_FABRID's MVTec tree has no train-time
        defect images.
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

    name = "raw-fabric"

    def __init__(
        self,
        root: str,
        split: str = "test",
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
        self.num_samples = num_samples
        self.use_defect = use_defect
        self.defect_ratio = defect_ratio
        self.task = task
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    @property
    def _mvtec_root(self) -> Path:
        return self.root_path / "MVTec"

    def _normal_dir(self) -> Path:
        return self._mvtec_root / ("train" if self.split == "train" else "test") / "good"

    def _defect_dir(self) -> Path:
        return self._mvtec_root / "test" / "defect"

    def _mask_dir(self) -> Path:
        return self._mvtec_root / "ground_truth" / "defect"

    @staticmethod
    def _list_images(directory: Path) -> list[str]:
        if not directory.is_dir():
            return []
        return sorted(
            path.stem for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _select_ids(self) -> tuple[list[str], list[str]]:
        """Return (normal_ids, defect_ids) chosen per the count/ratio config."""

        normal_pool = self._list_images(self._normal_dir())
        defect_pool = (
            self._list_images(self._defect_dir())
            if self.use_defect and self.split == "test"
            else []
        )

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
    def _build_sample(self, image_id: str, is_defect: bool) -> Sample:
        directory = self._defect_dir() if is_defect else self._normal_dir()
        image_path = next(
            (path for path in directory.glob(f"{image_id}.*") if path.suffix.lower() in IMAGE_SUFFIXES),
            directory / f"{image_id}.png",
        )

        annotations = Annotations(is_anomalous=is_defect)
        if is_defect:
            mask_path = next(
                (path for path in self._mask_dir().glob(f"{image_id}.*") if path.is_file()),
                None,
            )
            if mask_path is not None:
                annotations.anomaly_mask = str(mask_path)
                if self.task == "segmentation":
                    annotations.masks = [str(mask_path)]

        return Sample(
            id=image_id,
            image_path=str(image_path),
            task=self.task,
            annotations=annotations,
            metadata={"source": "RAW_FABRID/MVTec"},
        )

    def load_samples(self) -> list[Sample]:
        normal_ids, defect_ids = self._select_ids()
        samples = [self._build_sample(i, is_defect=False) for i in normal_ids]
        samples += [self._build_sample(i, is_defect=True) for i in defect_ids]
        return samples
