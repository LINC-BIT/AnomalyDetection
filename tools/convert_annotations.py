#!/usr/bin/env python3
"""Convert COCO detection annotations into FabricDefectHub Sample JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabric_defect_hub.core.serialization import save_samples
from fabric_defect_hub.core.types import Annotations, Sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", help="COCO instances JSON path")
    parser.add_argument("output", help="output Sample JSON path")
    parser.add_argument("--image-root", default="", help="directory joined with each COCO file_name")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    samples = coco_detection_samples(data, Path(args.image_root))
    save_samples(samples, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")
    return 0


def coco_detection_samples(data: dict, image_root: Path) -> list[Sample]:
    required = {"images", "annotations", "categories"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"COCO JSON is missing keys: {sorted(missing)}")
    categories = {item["id"]: item["name"] for item in data["categories"]}
    annotations_by_image: dict[object, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        annotations_by_image[annotation["image_id"]].append(annotation)

    samples: list[Sample] = []
    for image in data["images"]:
        image_id = image["id"]
        boxes: list[list[float]] = []
        labels: list[str] = []
        for annotation in annotations_by_image[image_id]:
            bbox = annotation.get("bbox")
            category = categories.get(annotation.get("category_id"))
            if not isinstance(bbox, list) or len(bbox) != 4 or category is None:
                raise ValueError(f"Invalid annotation for image id {image_id!r}: {annotation!r}")
            x, y, width, height = (float(value) for value in bbox)
            if width <= 0 or height <= 0:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(category)
        samples.append(
            Sample(
                id=str(image_id),
                image_path=str(image_root / image["file_name"]),
                task="detection",
                annotations=Annotations(
                    boxes=boxes or None,
                    labels=labels or None,
                    is_anomalous=bool(boxes),
                ),
                metadata={"source_format": "coco", "width": image.get("width"), "height": image.get("height")},
            )
        )
    return samples


if __name__ == "__main__":
    raise SystemExit(main())
