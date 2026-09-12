#!/usr/bin/env python3
"""Render unified Prediction boxes, masks, and anomaly maps onto images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabric_defect_hub.core.serialization import load_predictions, load_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", help="Sample JSON path")
    parser.add_argument("predictions", help="Prediction JSON path")
    parser.add_argument("output_dir", help="directory for rendered images")
    args = parser.parse_args(argv)

    predictions = {prediction.sample_id: prediction for prediction in load_predictions(args.predictions)}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for sample in load_samples(args.samples):
        prediction = predictions.get(sample.id)
        if prediction is None:
            continue
        render_prediction(sample.image_path, prediction, output_dir / f"{sample.id}.png")
        rendered += 1
    print(f"Rendered {rendered} predictions to {output_dir}")
    return 0


def render_prediction(image_path: str, prediction, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGBA")
    image = _overlay_masks(image, prediction.masks)
    image = _overlay_anomaly_map(image, prediction.anomaly_map)
    draw = ImageDraw.Draw(image)
    for index, box in enumerate(prediction.boxes or []):
        label = (prediction.labels or ["defect"] * len(prediction.boxes or []))[index]
        score = (prediction.scores or [None] * len(prediction.boxes or []))[index]
        text = label if score is None else f"{label} {score:.3f}"
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], max(0, box[1] - 14)), text, fill="red")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def _overlay_masks(image, masks):
    if not masks:
        return image
    import numpy as np
    from PIL import Image

    union = np.asarray(masks, dtype=bool)
    if union.ndim == 3:
        union = union.any(axis=0)
    alpha = Image.fromarray((union * 96).astype("uint8")).resize(image.size)
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    overlay.putalpha(alpha)
    return Image.alpha_composite(image, overlay)


def _overlay_anomaly_map(image, map_path: str | None):
    if map_path is None:
        return image
    import numpy as np
    from PIL import Image

    values = np.asarray(np.load(map_path), dtype=float)
    values = values - values.min()
    maximum = values.max()
    if maximum > 0:
        values /= maximum
    red = (values * 255).astype("uint8")
    overlay = Image.fromarray(red, mode="L").resize(image.size).convert("RGBA")
    overlay.putalpha(Image.fromarray((values * 128).astype("uint8")).resize(image.size))
    return Image.alpha_composite(image, overlay)


if __name__ == "__main__":
    raise SystemExit(main())
