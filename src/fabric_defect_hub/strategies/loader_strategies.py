"""Strategy-Driven Loading & Protocol Engine (SDLP).

Provides paper-grade loading and evaluation strategies including sparse proportionate subsampling,
sliding-window dynamic tiling & stitching, Test-Time Augmentation (TTA), and BatchNorm calibration.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from fabric_defect_hub.core.types import Prediction, Sample


class SparseSubsampler:
    """Proportionate & Stratified Sparse Subsampling Strategy."""

    @staticmethod
    def apply_sparse_ratio(samples: List[Sample], sparse_ratio: float, seed: int = 42) -> List[Sample]:
        """Subsamples dataset to a precise sparse ratio (e.g., 0.1 for 10% few-shot data)."""
        if not (0.0 < sparse_ratio <= 1.0):
            raise ValueError(f"sparse_ratio must be in range (0.0, 1.0], got {sparse_ratio}")

        if sparse_ratio == 1.0 or not samples:
            return samples

        num_target = max(1, int(len(samples) * sparse_ratio))
        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        return shuffled[:num_target]

    @staticmethod
    def apply_stratified_pattern(samples: List[Sample], sparse_ratio: float = 0.1) -> List[Sample]:
        """Applies pattern-wise or category-wise stratified subsampling across fabric groups."""
        grouped: Dict[str, List[Sample]] = {}
        for s in samples:
            key = s.metadata.get("pattern") or s.metadata.get("category") or "default"
            grouped.setdefault(key, []).append(s)

        stratified_samples: List[Sample] = []
        for key, group in grouped.items():
            num_keep = max(1, int(len(group) * sparse_ratio))
            stratified_samples.extend(group[:num_keep])

        return stratified_samples


class SlidingWindowTiler:
    """Sliding-Window Dynamic Tiling and Mask Stitching for 4K High-Res Fabric Images."""

    def __init__(self, tile_size: Tuple[int, int] = (256, 256), overlap: float = 0.25) -> None:
        self.tile_h, self.tile_w = tile_size
        self.overlap = overlap
        self.stride_h = int(self.tile_h * (1.0 - overlap))
        self.stride_w = int(self.tile_w * (1.0 - overlap))

    def split_sample(self, sample: Sample) -> Tuple[List[Sample], Dict[str, Any]]:
        """Splits a single high-resolution Sample into overlapping tile Samples."""
        img = getattr(sample, "image", None) or getattr(sample, "image_path", None)
        if img is None:
            return [sample], {"tiled": False}

        if hasattr(img, "size"):  # PIL Image (width, height)
            w, h = img.size
        elif isinstance(img, np.ndarray):
            h, w = img.shape[:2]
        else:
            return [sample], {"tiled": False}

        if h <= self.tile_h and w <= self.tile_w:
            return [sample], {"tiled": False}

        tiles: List[Sample] = []
        tile_coords: List[Tuple[int, int, int, int]] = []

        y = 0
        while y < h:
            x = 0
            y_end = min(y + self.tile_h, h)
            y_start = max(0, y_end - self.tile_h)

            while x < w:
                x_end = min(x + self.tile_w, w)
                x_start = max(0, x_end - self.tile_w)

                if hasattr(img, "crop"):
                    tile_img = img.crop((x_start, y_start, x_end, y_end))
                else:
                    tile_img = img[y_start:y_end, x_start:x_end]

                tile_sample = Sample(
                    id=f"{sample.id}_tile_{y_start}_{x_start}",
                    image_path=sample.image_path if isinstance(sample.image_path, str) else "",
                    task=sample.task,
                    annotations=sample.annotations,
                    metadata={**sample.metadata, "tile_box": (y_start, x_start, y_end, x_end)},
                )
                setattr(tile_sample, "image", tile_img)
                tiles.append(tile_sample)
                tile_coords.append((y_start, x_start, y_end, x_end))

                if x_end == w:
                    break
                x += self.stride_w

            if y_end == h:
                break
            y += self.stride_h

        meta_info = {
            "tiled": True,
            "orig_size": (h, w),
            "tile_coords": tile_coords,
            "sample_id": sample.id,
        }
        return tiles, meta_info

    def stitch_predictions(self, tile_preds: List[Prediction], meta_info: Dict[str, Any]) -> Prediction:
        """Stitches predictions from individual tiles back into a full-resolution prediction."""
        if not meta_info.get("tiled", False):
            return tile_preds[0] if tile_preds else None

        h, w = meta_info["orig_size"]
        full_anomaly_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        max_score = 0.0
        for pred in tile_preds:
            if pred.anomaly_score is not None:
                max_score = max(max_score, pred.anomaly_score)

            tile_box = getattr(pred, "metadata", {}).get("tile_box") if hasattr(pred, "metadata") else None
            if tile_box is None and hasattr(pred, "metadata") and pred.metadata is not None:
                tile_box = pred.metadata.get("tile_box")

            amap = pred.anomaly_map
            if tile_box and amap is not None and isinstance(amap, np.ndarray):
                y_start, x_start, y_end, x_end = tile_box
                if amap.shape != (self.tile_h, self.tile_w):
                    from scipy.ndimage import zoom
                    zh = (y_end - y_start) / amap.shape[0]
                    zw = (x_end - x_start) / amap.shape[1]
                    amap = zoom(amap, (zh, zw), order=1)

                full_anomaly_map[y_start:y_end, x_start:x_end] += amap
                count_map[y_start:y_end, x_start:x_end] += 1.0

        count_map[count_map == 0] = 1.0
        full_anomaly_map /= count_map

        return Prediction(
            sample_id=meta_info["sample_id"],
            anomaly_score=float(max_score),
            anomaly_map=full_anomaly_map,
        )


class TTAInferenceWrapper:
    """Test-Time Augmentation (TTA) Wrapper for robust multi-scale and flip prediction merging."""

    def __init__(self, model_adapter: Any, tta_mode: str = "flip_multiscale") -> None:
        self.model_adapter = model_adapter
        self.tta_mode = tta_mode

    def predict(self, samples: List[Sample], artifact: Any = None, **kwargs) -> List[Prediction]:
        """Runs TTA augmented predictions and averages anomaly maps / scores."""
        if not self.tta_mode or self.tta_mode == "none":
            return self.model_adapter.predict(samples, artifact, **kwargs)

        base_preds = self.model_adapter.predict(samples, artifact, **kwargs)

        # Apply Horizontal Flip TTA Pass
        flipped_samples = []
        for s in samples:
            img = getattr(s, "image", None) or getattr(s, "image_path", None)
            if hasattr(img, "transpose"):  # PIL Image
                from PIL import Image
                f_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif isinstance(img, np.ndarray):
                f_img = np.fliplr(img)
            else:
                f_img = img

            fs = Sample(
                id=f"{s.id}_flip",
                image_path=s.image_path if isinstance(s.image_path, str) else "",
                task=s.task,
                annotations=s.annotations,
                metadata=s.metadata,
            )
            if hasattr(s, "image"):
                setattr(fs, "image", f_img)
            flipped_samples.append(fs)

        flip_preds = self.model_adapter.predict(flipped_samples, artifact, **kwargs)

        # Merge base and flipped predictions
        merged_preds: List[Prediction] = []
        for bp, fp in zip(base_preds, flip_preds):
            merged_score = (bp.anomaly_score + fp.anomaly_score) / 2.0 if (bp.anomaly_score is not None and fp.anomaly_score is not None) else bp.anomaly_score

            merged_map = bp.anomaly_map
            if bp.anomaly_map is not None and fp.anomaly_map is not None and isinstance(bp.anomaly_map, np.ndarray) and isinstance(fp.anomaly_map, np.ndarray):
                unflipped_map = np.fliplr(fp.anomaly_map)
                merged_map = (bp.anomaly_map + unflipped_map) / 2.0

            merged_preds.append(
                Prediction(
                    sample_id=bp.sample_id,
                    boxes=bp.boxes,
                    labels=bp.labels,
                    scores=bp.scores,
                    masks=bp.masks,
                    anomaly_score=merged_score,
                    anomaly_map=merged_map,
                )
            )

        return merged_preds


class BatchNormCalibrator:
    """BatchNorm & LayerNorm Running Statistics Calibration on Normal Fabric Images."""

    @staticmethod
    def calibrate(model: nn.Module, normal_samples: List[Sample], num_steps: int = 16) -> None:
        """Runs forward passes on normal fabric samples in training mode to calibrate BN running_mean/var."""
        if not hasattr(model, "train"):
            return

        model.train()
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

        with torch.no_grad():
            for i, s in enumerate(normal_samples[:num_steps]):
                img = getattr(s, "image", None) or getattr(s, "image_path", None)
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                    arr = np.array(img).transpose(2, 0, 1) / 255.0
                    tensor = torch.from_numpy(arr).float().unsqueeze(0).to(device)
                elif isinstance(img, np.ndarray):
                    arr = img.transpose(2, 0, 1) / 255.0 if img.ndim == 3 else img[np.newaxis, ...] / 255.0
                    tensor = torch.from_numpy(arr).float().unsqueeze(0).to(device)
                else:
                    continue

                try:
                    model(tensor)
                except Exception:
                    break

        model.eval()
