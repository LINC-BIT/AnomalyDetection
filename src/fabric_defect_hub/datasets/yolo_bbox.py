"""On-the-fly conversion from our internal `Sample` annotations (COCO-style:
absolute-pixel `xyxy` boxes) to the Darknet/YOLO bbox format Ultralytics
requires (one `.txt` per image, lines of `class cx cy w h`, all normalized
to [0, 1]).

Ultralytics' trainer needs real files on disk — an `images/` + `labels/`
folder pair (or a `data.yaml` pointing at them). But writing out a full
converted copy of a dataset the size of ZJU-Leaper (94k+ images) ahead of
time, just to train on a few hundred of them, wastes disk space for no
reason. So this module does the conversion lazily, at the moment training
actually starts, on exactly the `Sample`s a `DatasetAdapter` already
selected for you (few-shot, low-shot, defect-only, a hand-labeled subset,
...) — see `ZJULeaperDataset.load_samples()`.

`yolo_staging_dir()` is a context manager: it symlinks each selected image
into a temporary directory (no pixel data is copied — a symlink costs one
inode, not the image size) and writes only the small per-image label
`.txt`, then deletes the whole temporary directory again once the `with`
block exits (i.e. once training/export has consumed it). Nothing persists
on disk before or after.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fabric_defect_hub.core.types import Sample

DEFAULT_CLASS_MAP = {"defect": 0}


def build_class_map(samples: list[Sample], class_names: list[str] | None = None) -> dict[str, int]:
    """Assign a stable class id to every label seen across `samples`.

    Pass `class_names` to fix the id order explicitly (e.g. to match a
    checkpoint trained earlier); otherwise ids are assigned by sorted label
    name, so the same sample set always yields the same mapping.
    """

    if class_names is not None:
        return {name: idx for idx, name in enumerate(class_names)}

    labels: set[str] = set()
    for sample in samples:
        if sample.annotations.labels:
            labels.update(sample.annotations.labels)
    return {name: idx for idx, name in enumerate(sorted(labels))} or dict(DEFAULT_CLASS_MAP)


def xyxy_to_yolo_line(box: list[float], class_id: int, img_w: int, img_h: int) -> str:
    """Convert one absolute-pixel `[xmin, ymin, xmax, ymax]` box to a Darknet label line."""

    xmin, ymin, xmax, ymax = box
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    # Clamp: source annotations occasionally sit exactly on the image edge,
    # which floating point can push a hair outside [0, 1].
    cx, cy, w, h = (min(max(v, 0.0), 1.0) for v in (cx, cy, w, h))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _sample_to_label_text(sample: Sample, class_map: dict[str, int]) -> str:
    boxes = sample.annotations.boxes
    labels = sample.annotations.labels
    if not boxes:
        return ""  # normal/background image: a valid, empty YOLO label

    from PIL import Image

    with Image.open(sample.image_path) as img:
        img_w, img_h = img.size

    lines = [
        xyxy_to_yolo_line(box, class_map[label], img_w, img_h)
        for box, label in zip(boxes, labels or [])
        if label in class_map
    ]
    return "\n".join(lines)


def _stage_split(root: Path, split: str, samples: list[Sample], class_map: dict[str, int], *, tiling: bool, tile_size: tuple[int, int], overlap: float) -> None:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        if tiling:
            _stage_tiled_sample(images_dir, labels_dir, sample, class_map, tile_size, overlap)
            continue
        src = Path(sample.image_path).resolve()
        dst = images_dir / f"{sample.id}{src.suffix}"
        if not dst.exists():
            dst.symlink_to(src)

        label_text = _sample_to_label_text(sample, class_map)
        (labels_dir / f"{sample.id}.txt").write_text(label_text)


def _stage_tiled_sample(images_dir: Path, labels_dir: Path, sample: Sample, class_map: dict[str, int], tile_size: tuple[int, int], overlap: float) -> None:
    """Write overlapping image tiles and clip YOLO labels into each tile."""
    from PIL import Image

    tile_w, tile_h = tile_size
    if tile_w <= 0 or tile_h <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile_size must be positive and overlap must be in [0, 1)")
    with Image.open(sample.image_path) as image:
        width, height = image.size
        step_x, step_y = max(1, round(tile_w * (1 - overlap))), max(1, round(tile_h * (1 - overlap)))
        x_starts = _tile_starts(width, tile_w, step_x)
        y_starts = _tile_starts(height, tile_h, step_y)
        for row, top in enumerate(y_starts):
            for col, left in enumerate(x_starts):
                right, bottom = min(left + tile_w, width), min(top + tile_h, height)
                tile_id = f"{sample.id}_r{row}_c{col}"
                image.crop((left, top, right, bottom)).save(images_dir / f"{tile_id}.jpg")
                lines = []
                for box, label in zip(sample.annotations.boxes, sample.annotations.labels or []):
                    xmin, ymin, xmax, ymax = box
                    clipped = [max(xmin, left) - left, max(ymin, top) - top, min(xmax, right) - left, min(ymax, bottom) - top]
                    if clipped[2] > clipped[0] and clipped[3] > clipped[1] and label in class_map:
                        lines.append(xyxy_to_yolo_line(clipped, class_map[label], right - left, bottom - top))
                (labels_dir / f"{tile_id}.txt").write_text("\n".join(lines))


def _tile_starts(length: int, tile_length: int, step: int) -> list[int]:
    if length <= tile_length:
        return [0]
    starts = list(range(0, length - tile_length + 1, step))
    if starts[-1] != length - tile_length:
        starts.append(length - tile_length)
    return starts


@contextmanager
def yolo_staging_dir(
    splits: list[Sample] | dict[str, list[Sample]],
    class_map: dict[str, int] | None = None,
    class_names: list[str] | None = None,
    tmp_root: str | None = None,
    tiling: bool = False,
    tile_size: tuple[int, int] = (256, 256),
    overlap: float = 0.25,
) -> Iterator[Path]:
    """Stage `splits` as a Darknet/YOLO dataset and yield the `data.yaml` path.

    `splits` is either a flat `Sample` list (staged as a single 'train'
    split, reused for 'val') or a `{split_name: samples}` dict (typically
    `{'train': ..., 'val': ...}`) — pass in whatever `DatasetAdapter`
    already handed you post-selection.

    The staging directory (and everything in it — symlinks + label .txt
    files) is deleted when the `with` block exits, success or failure.
    """

    if isinstance(splits, list):
        splits = {"train": splits, "val": splits}

    all_samples = [s for group in splits.values() for s in group]
    resolved_class_map = class_map or build_class_map(all_samples, class_names)

    root = Path(tempfile.mkdtemp(prefix="fdh_yolo_", dir=tmp_root))
    try:
        for split_name, samples in splits.items():
            _stage_split(root, split_name, samples, resolved_class_map, tiling=tiling, tile_size=tile_size, overlap=overlap)

        data_yaml = root / "data.yaml"
        names_by_id = {idx: name for name, idx in resolved_class_map.items()}
        _write_data_yaml(data_yaml, root, list(splits.keys()), names_by_id)
        yield data_yaml
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_data_yaml(path: Path, root: Path, split_names: list[str], names_by_id: dict[int, str]) -> None:
    import yaml

    config = {
        "path": str(root),
        "names": names_by_id,
    }
    for split_name in split_names:
        config[split_name] = f"images/{split_name}"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
