"""`DatasetAdapter` for the Tianchi Guangdong Fabric Defect Detection
Challenge (2019), a bbox-labelled fabric defect corpus.

Source layout (three labelled collections, unioned by this adapter):
  guangdong1_round1_train1_20190828/partA/{defect_Images,normal_Images,Annotations/anno_train.json}
  guangdong1_round1_train1_20190828/partB/{...}
  guangdong1_round1_train2_20190828/{...}
Each collection separates `defect_Images/` from `normal_Images/` and lists
per-box annotations (`name`, `defect_name`, `bbox` in absolute xyxy) for the
defect images only, in `Annotations/anno_train.json` -- see the dataset's own
`README.md` for the format and the defect-name -> category-id table.

`guangdong1_round1_testA_20190818/` and `..._testB_20190919/` are the
competition's *unlabelled* submission images (no ground truth of any kind)
and are deliberately never touched by this adapter -- their images are not
guaranteed normal, so folding them into the normal pool would silently
corrupt anomaly training.

Two roles this dataset serves, both native to what's on disk (see
`core.dataset_capabilities` for how these are declared to the rest of the
project):
1. Detection training (`task="detection"`) -- every box + its `defect_name`
   feeds `yolo_bbox`/torchvision detectors directly, the same `Sample` shape
   any other `DatasetAdapter`'s boxes/labels output takes.
2. Anomaly (one-class) training/eval (`task="anomaly"`) -- `normal_Images`
   supplies genuine good-sample images; pass `use_defect=False` to draw only
   from that pool. This is also how `fabric-train` pulls Tianchi's normal
   images into the project-wide anomaly training composite.

Unlike ZJU-Leaper/RAW_FABRID, Tianchi ships no author train/test split --
the "train1"/"train2" naming refers to release batches, not a split. This
adapter synthesizes one instead: the normal and defect pools are each sorted
by image filename and cut at `train_ratio`, independently per part, so
`split` selection is deterministic and leak-free regardless of `seed` while
still giving *both* splits real, bboxed defect images -- unlike the
flat-folder one-class datasets (`datasets/flat_folder.py`), whose "all
defects go to test" rule assumes no native detection use.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from fabric_defect_hub.core.registry import register_dataset
from fabric_defect_hub.core.types import Annotations, Sample, Task
from fabric_defect_hub.datasets.base import DatasetAdapter

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# (part key, subdirectory under `root`). Each holds its own
# defect_Images/normal_Images/Annotations/anno_train.json triplet.
_PARTS: tuple[tuple[str, str], ...] = (
    ("train1-partA", "guangdong1_round1_train1_20190828/partA"),
    ("train1-partB", "guangdong1_round1_train1_20190828/partB"),
    ("train2", "guangdong1_round1_train2_20190828"),
)


@register_dataset("tianchi")
class TianchiDataset(DatasetAdapter):
    """Configurable adapter over the Tianchi Guangdong fabric defect corpus.

    root: dataset root containing the `guangdong1_round1_*` directories.
    split: "train" or "test" -- a synthesized per-part, per-class split (see
        module docstring); unrelated to the competition's own
        train1/train2/testA/testB naming (testA/testB are never read here).
    parts: which labelled collection(s) to draw from -- None/"all" (default)
        pools every part; a string or list of strings from
        {"train1-partA", "train1-partB", "train2"} restricts to those.
    num_samples: total images to load. None = everything in the selection.
    use_defect: include defective (bboxed) images. False = normal-only, the
        mode `fabric-train` uses to add Tianchi's good samples to the
        anomaly training corpus.
    defect_ratio: fraction of the loaded set that is defective. Only applied
        when `num_samples` is set and `use_defect` is True.
    task: "detection" (default -- this dataset's native format) or "anomaly".
        Boxes/labels are always attached to defective samples when present;
        `task` only picks `Sample.task`.
    train_ratio: fraction of each part's normal/defect pool assigned to the
        train split (independently per pool, see module docstring).
    seed: RNG seed for reproducible subsampling.
    """

    name = "tianchi"

    def __init__(
        self,
        root: str,
        split: str = "test",
        parts: str | list[str] | None = None,
        num_samples: int | None = None,
        use_defect: bool = True,
        defect_ratio: float = 0.5,
        task: Task = "detection",
        train_ratio: float = 0.8,
        seed: int = 0,
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
        self.parts = parts
        self.num_samples = num_samples
        self.use_defect = use_defect
        self.defect_ratio = defect_ratio
        self.task = task
        self.train_ratio = train_ratio
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Part resolution
    # ------------------------------------------------------------------ #
    def _selected_parts(self) -> list[tuple[str, str]]:
        if self.parts is None or self.parts == "all":
            return list(_PARTS)
        wanted = {self.parts} if isinstance(self.parts, str) else set(self.parts)
        by_key = dict(_PARTS)
        unknown = wanted - by_key.keys()
        if unknown:
            raise ValueError(
                f"unknown tianchi part(s) {sorted(unknown)}; expected a subset of {sorted(by_key)}"
            )
        return [(key, by_key[key]) for key in wanted]

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    @staticmethod
    def _images(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)

    @staticmethod
    def _load_annotations(part_root: Path) -> dict[str, list[tuple[str, list[float]]]]:
        """Map image filename -> [(defect_name, [xmin, ymin, xmax, ymax]), ...]."""

        anno_path = part_root / "Annotations" / "anno_train.json"
        if not anno_path.is_file():
            return {}
        with open(anno_path, encoding="utf-8") as fh:
            entries = json.load(fh)
        by_name: dict[str, list[tuple[str, list[float]]]] = {}
        for entry in entries:
            name = entry.get("name")
            bbox = entry.get("bbox")
            if not name or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            defect_name = str(entry.get("defect_name", "defect"))
            by_name.setdefault(name, []).append((defect_name, [float(v) for v in bbox]))
        return by_name

    def _split_pool(self, pool: list[Path]) -> list[Path]:
        """Deterministic, disjoint partition by sorted image id (see module
        docstring) -- independent of `seed`, so train/test never leak."""

        cut = round(len(pool) * self.train_ratio)
        return pool[:cut] if self.split == "train" else pool[cut:]

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _select(
        self,
    ) -> tuple[list[tuple[str, Path]], list[tuple[str, Path, list[tuple[str, list[float]]]]]]:
        normal_pool: list[tuple[str, Path]] = []
        defect_pool: list[tuple[str, Path, list[tuple[str, list[float]]]]] = []

        for part_key, subdir in self._selected_parts():
            part_root = self.root_path / subdir
            normal_images = self._split_pool(self._images(part_root / "normal_Images"))
            normal_pool.extend((part_key, path) for path in normal_images)

            if not self.use_defect:
                continue
            annotations = self._load_annotations(part_root)
            defect_images = self._split_pool(self._images(part_root / "defect_Images"))
            defect_pool.extend(
                (part_key, path, annotations.get(path.name, [])) for path in defect_images
            )

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
    def _build_normal_sample(self, part_key: str, image_path: Path) -> Sample:
        return Sample(
            id=f"{part_key}/normal/{image_path.stem}",
            image_path=str(image_path),
            task=self.task,
            annotations=Annotations(is_anomalous=False),
            metadata={"part": part_key},
        )

    def _build_defect_sample(
        self, part_key: str, image_path: Path, boxes: list[tuple[str, list[float]]]
    ) -> Sample:
        annotations = Annotations(is_anomalous=True)
        if boxes:
            annotations.boxes = [box for _, box in boxes]
            annotations.labels = [name for name, _ in boxes]
        return Sample(
            id=f"{part_key}/defect/{image_path.stem}",
            image_path=str(image_path),
            task=self.task,
            annotations=annotations,
            metadata={"part": part_key, "defect_names": sorted({name for name, _ in boxes})},
        )

    def load_samples(self) -> list[Sample]:
        normal_pool, defect_pool = self._select()
        samples = [self._build_normal_sample(part, path) for part, path in normal_pool]
        samples += [self._build_defect_sample(part, path, boxes) for part, path, boxes in defect_pool]
        return samples
