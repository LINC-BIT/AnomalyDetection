#!/usr/bin/env python3
"""Plot image-level and pixel-level ROC/PR curves from saved anomaly predictions."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "datasets/textile/ZJU-Leaper"
PREDICTION_ROOT = ROOT / "artifacts/runtime/anomaly_maps/benchmark"
OUTPUT_ROOT = ROOT / "artifacts/local_benchmark_plots"
MAX_PIXELS_PER_MODEL = 20_000
MAX_IMAGES_PER_MODEL = 5
# Width of the display-only Gaussian smoothing kernel, in axis fraction units.
SMOOTH_SIGMA = 0.02
# Curves are separated by hue and by line style. tab10 is used rather than tab20:
# tab20 pairs a dark and a light variant of the same hue, which reads as one colour.
CURVE_PALETTE = "tab10"
LINE_STYLES = ("-", "--", "-.", ":")

MODELS = {
    "moeclip---mvtec-ad-adapter": "MoECLIP",
    "patchcore---zju-leaper--normal-only": "PatchCore",
    "dinomaly---zju-leaper--normal-only": "Dinomaly",
    "stfpm---zju-leaper--normal-only": "STFPM",
    "padim---zju-leaper--normal-only": "PaDiM",
    "winclip---laion-400m-zero-shot": "WinCLIP",
    "efficientad---zju-leaper--normal-only": "EfficientAD",
    "reverse-distillation---zju-leaper--normal-only": "Reverse Distillation",
    "supersimplenet---zju-leaper--normal-only": "SuperSimpleNet",
    "ganomaly---zju-leaper--normal-only": "GANomaly",
}

BBOX_MODELS = {
    "yolov8n---zju-leaper": "YOLOv8n",
    "yolov8s---zju-leaper": "YOLOv8s",
    "yolo11n---zju-leaper": "YOLO11n",
    "faster-r-cnn---zju-leaper": "Faster R-CNN",
    "cascade-r-cnn---zju-leaper": "Cascade R-CNN",
    "detr---zju-leaper": "DETR",
}


def image_label(sample_id: str) -> int:
    root = ET.parse(DATASET_ROOT / "Annotations/xmls" / f"{sample_id}.xml").getroot()
    return int(root.findtext("defective", "0"))


def load_predictions(key: str) -> list[dict]:
    path = PREDICTION_ROOT / key / f"benchmark-{key}/predictions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def image_pairs(predictions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for row in predictions:
        if row.get("anomaly_score") is not None:
            score = float(row["anomaly_score"])
        else:
            score = max((float(value) for value in (row.get("scores") or [])), default=0.0)
        pairs.append((image_label(row["sample_id"]), score))
    labels, scores = zip(*pairs)
    return np.asarray(labels), np.asarray(scores, dtype=float)


def pixel_pairs(predictions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    if len(predictions) > MAX_IMAGES_PER_MODEL:
        predictions = [predictions[index] for index in sorted(rng.choice(len(predictions), size=MAX_IMAGES_PER_MODEL, replace=False))]
    for row in predictions:
        map_path = row.get("anomaly_map")
        mask_path = DATASET_ROOT / "Annotations/masks" / f"{row['sample_id']}.png"
        if not map_path or not mask_path.exists():
            continue
        score_path = ROOT / map_path
        if not score_path.exists():
            score_path = PREDICTION_ROOT / Path(map_path).name
        if not score_path.exists():
            continue
        pred_map = np.load(score_path)
        with Image.open(mask_path) as image:
            truth = np.asarray(image.convert("L")) > 0
        if pred_map.ndim > 2:
            pred_map = np.squeeze(pred_map)
        if pred_map.shape != truth.shape:
            resized = Image.fromarray(pred_map.astype(np.float32), mode="F").resize((truth.shape[1], truth.shape[0]), Image.Resampling.BILINEAR)
            pred_map = np.asarray(resized)
        truth_flat = truth.ravel()
        score_flat = np.asarray(pred_map, dtype=float).ravel()
        if len(score_flat) > MAX_PIXELS_PER_MODEL:
            selected = rng.choice(len(score_flat), size=MAX_PIXELS_PER_MODEL, replace=False)
            truth_flat = truth_flat[selected]
            score_flat = score_flat[selected]
        labels.append(truth_flat)
        scores.append(score_flat)
    if not labels:
        raise ValueError("No usable anomaly maps and masks were found.")
    return np.concatenate(labels).astype(np.uint8), np.concatenate(scores)


def roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores, kind="mergesort")
    truth = labels[order].astype(bool)
    positives = max(int(truth.sum()), 1)
    negatives = max(int((~truth).sum()), 1)
    thresholds = np.r_[np.inf, scores[order]]
    predicted = np.arange(len(truth) + 1)
    cumulative_true = np.r_[0, np.cumsum(truth)]
    cumulative_false = predicted - cumulative_true
    tpr = cumulative_true / positives
    fpr = cumulative_false / negatives
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def pr_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores, kind="mergesort")
    truth = labels[order].astype(bool)
    true_positive = np.cumsum(truth)
    false_positive = np.cumsum(~truth)
    positives = max(int(truth.sum()), 1)
    recall = true_positive / positives
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    unique_recall, inverse = np.unique(recall, return_inverse=True)
    envelope = np.zeros_like(unique_recall, dtype=float)
    for index in range(len(unique_recall)):
        envelope[index] = np.max(precision[inverse == index])
    precision = np.maximum.accumulate(envelope[::-1])[::-1]
    recall = np.r_[0.0, unique_recall, 1.0]
    precision = np.r_[1.0, precision, precision[-1] if len(precision) else 0.0]
    return recall, precision, float(np.trapezoid(precision, recall))


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _smooth_curve(
    x: np.ndarray,
    y: np.ndarray,
    direction: str,
    samples: int = 600,
    sigma_fraction: float = SMOOTH_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a staircase ROC/PR trace for display only.

    The empirical traces are step functions, so with few evaluation samples the
    staircase dominates the figure. Evaluating the steps on a dense grid and
    convolving them with a Gaussian kernel removes the visible corners, and a
    monotonicity pass with pinned endpoints keeps the curve shape valid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique, indices = np.unique(x, return_index=True)
    x, y = unique, y[indices]
    dense = np.linspace(0.0, 1.0, samples)
    if len(x) < 2:
        return dense, np.zeros_like(dense)
    positions = np.clip(np.searchsorted(x, dense, side="right") - 1, 0, len(x) - 1)
    values = y[positions]
    kernel = _gaussian_kernel(max(sigma_fraction * samples, 1.0))
    pad = len(kernel) // 2
    smooth = np.convolve(np.pad(values, pad, mode="edge"), kernel, mode="valid")
    if direction == "increasing":
        smooth = np.maximum.accumulate(smooth)
        smooth[0], smooth[-1] = 0.0, 1.0
    else:
        smooth = np.minimum.accumulate(smooth)
        smooth[0] = 1.0
    return dense, np.clip(smooth, 0.0, 1.0)


def collect_curves(pairs: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[str, object]]:
    """Compute both PR and ROC traces once so both panels share models and colors."""
    curves: dict[str, dict[str, object]] = {}
    for name, (true_labels, scores) in pairs.items():
        if len(np.unique(true_labels)) < 2:
            continue
        recall, precision, average_precision = pr_points(true_labels, scores)
        fpr, tpr, auroc = roc_points(true_labels, scores)
        curves[name] = {
            "recall": recall,
            "precision": precision,
            "ap": average_precision,
            "fpr": fpr,
            "tpr": tpr,
            "auroc": auroc,
            "prevalence": float(np.mean(true_labels)),
        }
    return curves


def _panel_curves(curve_kind: str, curves: dict[str, dict[str, object]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Smooth every curve of one panel for display."""
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, curve in curves.items():
        if curve_kind == "roc":
            traces[name] = _smooth_curve(curve["fpr"], curve["tpr"], "increasing")
        else:
            traces[name] = _smooth_curve(curve["recall"], curve["precision"], "decreasing")
    return traces


def render_curve_panel(
    ax,
    curve_kind: str,
    curves: dict[str, dict[str, object]],
    styles: dict[str, tuple[tuple, str]],
    caption: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    traces = _panel_curves(curve_kind, curves)
    for name, (dense_x, smooth_y) in traces.items():
        color, linestyle = styles[name]
        ax.plot(dense_x, smooth_y, linewidth=1.8, color=color, linestyle=linestyle)
    if curve_kind == "roc":
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("False positive rate", fontsize=12)
        ax.set_ylabel("True positive rate", fontsize=12)
    else:
        prevalence = float(np.mean([float(curve["prevalence"]) for curve in curves.values()])) if curves else 0.0
        ax.axhline(prevalence, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
    # Panel labels belong under the panel, as in a published figure caption.
    ax.text(0.5, -0.145, caption, transform=ax.transAxes, ha="center", va="top", fontsize=14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=11)


def render_figure(
    curves: dict[str, dict[str, object]],
    title: str,
    output: Path,
    stem: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Draw the PR and ROC panels with one shared legend on the far right."""
    if not curves:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    palette = matplotlib.colormaps[CURVE_PALETTE]
    styles = {
        name: (tuple(palette(index % palette.N)), LINE_STYLES[index % len(LINE_STYLES)])
        for index, name in enumerate(curves)
    }
    render_curve_panel(axes[0], "pr", curves, styles, "(a) Precision-Recall", xlim, ylim)
    render_curve_panel(axes[1], "roc", curves, styles, "(b) ROC", xlim, ylim)
    handles = [
        plt.Line2D([], [], color=styles[name][0], linestyle=styles[name][1], linewidth=2.2, label=name)
        for name in curves
    ]
    legend = axes[1].legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=True,
        fancybox=True,
        fontsize=13,
        labelspacing=0.75,
        handlelength=2.4,
        borderpad=1.1,
    )
    frame = legend.get_frame()
    frame.set_boxstyle("round,pad=0.5,rounding_size=0.8")
    frame.set_edgecolor("0.55")
    frame.set_linewidth(1.2)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# Figure id -> (output stem, human label). The ids run on from the metric bar
# figures so a driver can ask for any figure in the report by one handle
# (`--only 10`), instead of naming a script and hoping it draws just that one.
FIGURES = {
    "11": ("11_anomaly_image_level_pr_roc", "image-level PR/ROC, anomaly models"),
    "12": ("12_detection_image_level_pr_roc", "image-level PR/ROC, detection models"),
    "13": ("13_anomaly_pixel_level_pr_roc", "pixel-level PR/ROC, anomaly-map models"),
}


def select_figures(only: list[str] | None, missing: bool, output: Path) -> list[str]:
    ids = list(FIGURES)
    if only:
        unknown = sorted(set(only) - set(FIGURES))
        if unknown:
            raise SystemExit(f"unknown figure id(s): {', '.join(unknown)}. Known: {', '.join(ids)}")
        ids = [figure_id for figure_id in ids if figure_id in set(only)]
    if missing:
        ids = [figure_id for figure_id in ids if not (output / f"{FIGURES[figure_id][0]}.png").exists()]
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--xlim", type=float, nargs=2, default=(0.0, 1.0), metavar=("XMIN", "XMAX"), help="display range of the horizontal axis (default: 0 1)")
    parser.add_argument("--ylim", type=float, nargs=2, default=(0.0, 1.02), metavar=("YMIN", "YMAX"), help="display range of the vertical axis (default: 0 1.02)")
    parser.add_argument("--only", nargs="+", metavar="ID", help="render only these figure ids (see --list)")
    parser.add_argument("--missing", action="store_true", help="render only figures whose PNG is not on disk yet")
    parser.add_argument("--list", action="store_true", help="list the figure ids and their output state, then exit")
    args = parser.parse_args()
    xlim = (float(args.xlim[0]), float(args.xlim[1]))
    ylim = (float(args.ylim[0]), float(args.ylim[1]))

    if args.list:
        for figure_id, (stem, label) in FIGURES.items():
            state = "present" if (args.output_dir / f"{stem}.png").exists() else "missing"
            print(f"{figure_id}  {stem:<34} {state}  {label}")
        return 0

    selected = select_figures(args.only, args.missing, args.output_dir)

    image_anomaly: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    image_detection: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, name in BBOX_MODELS.items():
        try:
            image_detection[name] = image_pairs(load_predictions(key))
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"skip bbox {name}: {exc}", file=sys.stderr)
    pixel: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, name in MODELS.items():
        try:
            predictions = load_predictions(key)
            image_anomaly[name] = image_pairs(predictions)
            if any(row.get("anomaly_map") for row in predictions):
                pixel[name] = pixel_pairs(predictions)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"skip {name}: {exc}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_sets = {
        "11": (collect_curves(image_anomaly), "Image-level anomaly detection: Precision-Recall and ROC"),
        "12": (collect_curves(image_detection), "Image-level defect detection: Precision-Recall and ROC"),
        "13": (collect_curves(pixel), "Pixel-level anomaly detection: Precision-Recall and ROC"),
    }
    for figure_id in selected:
        curves, title = curve_sets[figure_id]
        stem, label = FIGURES[figure_id]
        print(f"rendering {figure_id} -> {stem} ({len(curves)} models: {label})")
        render_figure(curves, title, args.output_dir, stem, xlim, ylim)
    print(f"rendered: {', '.join(selected) if selected else '<none>'}")
    print(f"outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
