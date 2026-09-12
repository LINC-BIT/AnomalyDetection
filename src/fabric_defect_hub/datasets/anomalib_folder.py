"""On-the-fly conversion from `Sample` selections (post `DatasetAdapter`
selection — e.g. `ZJULeaperDataset.load_samples()`) to the MVTec-style
folder layout `anomalib.data.Folder` requires. Mirrors `yolo_bbox.py`'s
staging pattern for the Ultralytics backend.

`anomalib.data.Folder` has no Python-object ingestion path — PatchCore,
PaDiM, RD4AD, EfficientAD and SuperSimpleNet all read `train/good`,
`test/good`, `test/defect` and (optionally) `ground_truth/defect` off
disk. Materializing a full converted copy of a dataset the size of
ZJU-Leaper just to point `Folder` at it would duplicate tens of thousands
of images for no reason. So, exactly like `yolo_bbox`, this symlinks each
selected image into a temporary directory (no pixel data is copied) and
deletes the whole thing again once training/prediction has consumed it.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from fabric_defect_hub.core.types import Sample


@dataclass
class FolderLayout:
    """Ready-to-use kwargs for `anomalib.data.Folder(**layout.as_kwargs())`."""

    root: Path
    normal_dir: str = "train/good"
    normal_test_dir: str = "test/good"
    abnormal_dir: str = "test/defect"
    mask_dir: str | None = "ground_truth/defect"

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "root": str(self.root),
            "normal_dir": self.normal_dir,
            "normal_test_dir": self.normal_test_dir,
            "abnormal_dir": self.abnormal_dir,
        }
        if self.mask_dir is not None:
            kwargs["mask_dir"] = self.mask_dir
        return kwargs


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src.resolve())


@contextmanager
def anomalib_folder_staging_dir(
    train_samples: list[Sample],
    test_samples: list[Sample],
    tmp_root: str | None = None,
    mask_suffix: str | None = None,
    image_suffix: str | None = None,
) -> Iterator[FolderLayout]:
    """Stage `train_samples` (must be all-normal — PatchCore/PaDiM/RD4AD/
    EfficientAD/SuperSimpleNet train one-class on normal data only, e.g.
    load them with `use_defect=False`) and `test_samples` (mixed
    normal/defect, split by `Sample.annotations.is_anomalous`) as an
    MVTec-style folder tree.

    Pixel masks (`Sample.annotations.anomaly_mask`) are included only when
    *every* defective test sample has one — anomalib requires a 1:1 match
    between abnormal images and ground-truth masks, so a partial set would
    fail at `Folder` construction; falling back to image-level-only
    (`mask_dir=None`) is safer than guessing. `mask_suffix` and
    `image_suffix` can normalize staged filenames for a backend with fixed
    extension conventions.

    Deleted automatically when the `with` block exits, success or failure.
    """

    bad_train = [s for s in train_samples if s.annotations.is_anomalous]
    if bad_train:
        raise ValueError(
            f"train_samples must be all-normal (one-class training); got "
            f"{len(bad_train)} defective samples, e.g. {bad_train[0].id!r}. "
            "Load them with use_defect=False."
        )

    root = Path(tempfile.mkdtemp(prefix="fdh_anomalib_", dir=tmp_root))
    try:
        for sample in train_samples:
            src = Path(sample.image_path)
            suffix = image_suffix or src.suffix
            _link(src, root / "train" / "good" / f"{sample.id}{suffix}")

        defect_samples = [s for s in test_samples if s.annotations.is_anomalous]
        for sample in test_samples:
            src = Path(sample.image_path)
            split_dir = "test/defect" if sample.annotations.is_anomalous else "test/good"
            suffix = image_suffix or src.suffix
            _link(src, root / split_dir / f"{sample.id}{suffix}")

        masked = [s for s in defect_samples if s.annotations.anomaly_mask]
        use_masks = bool(defect_samples) and len(masked) == len(defect_samples)
        if use_masks:
            for sample in defect_samples:
                mask_src = Path(sample.annotations.anomaly_mask)
                suffix = mask_suffix or mask_src.suffix
                _link(mask_src, root / "ground_truth" / "defect" / f"{sample.id}{suffix}")

        yield FolderLayout(root=root, mask_dir="ground_truth/defect" if use_masks else None)
    finally:
        shutil.rmtree(root, ignore_errors=True)
