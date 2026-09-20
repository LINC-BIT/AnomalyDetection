# Figure captions

The descriptions live here, not inside the PNGs: a heading burned into an image cannot be
reworded, translated or renumbered afterwards. Every figure is written to
`artifacts/local_benchmark_plots/` as PNG + PDF.

Machines: **Small** = RTX 4090 24 GB, **Full** = A100 80 GB. Dataset: ZJU-Leaper test split,
350 samples, 640 × 640, fp32. 19 configurations, unless a caption says otherwise.

---

## 01 · `01_image_level` — Image-level metrics

Per-image accuracy for every model that reports an image-level score: 10 anomaly models and 6
detectors, whose score is derived from the highest-confidence box per image. One group of four
bars per model; a bar is absent where a model reports nothing for that metric, never zero-filled.

| Bar | Metric |
| --- | --- |
| 1 (grey solid) | Image AUROC |
| 2 (red `/`) | Image F1 |
| 3 (blue `/`) | Image Precision |
| 4 (grey `/`) | Image Recall |

DETR's F1 / precision / recall are 0.000 — not missing. It scores no box above the evaluation
floor, so the derived score is 0 for every image and the three bars have no height. The model is
kept in this table because the zero is the finding.

## 02 · `02_pixel_level` — Pixel-level metrics

Pixel-level accuracy for the 12 models whose output is a mask or a heatmap. The chart deliberately
mixes two metric families, and a group carries only the family its output supports.

- **Threshold-free ranking** — Pixel AUROC, AUPRO, IAP. These sweep a threshold over a
  *continuous* score map, so a model that persists only a binarised mask cannot report them.
- **Thresholded overlap** — Pixel F1, mIoU. One binary mask is enough.

| Bar | Metric |
| --- | --- |
| 1 (grey solid) | Pixel AUROC |
| 2 (red `/`) | AUPRO |
| 3 (blue `/`) | IAP |
| 4 (grey `/`) | Pixel F1 |
| 5 (amber `\`) | mIoU |

`dice` is not plotted: for a binary mask it *is* `pixel_f1` (the evaluator's `_dice` and
`_pixel_f1` are the same formula, and the two columns agree to every quoted digit).

## 03 · `03_instance_level` — Instance-level detection metrics

Box quality for the detection backends.

| Bar | Metric |
| --- | --- |
| 1 (grey solid) | mAP@[.5:.95] |
| 2 (red `/`) | mAP@0.5 |
| 3 (blue `/`) | mAP@0.75 |
| 4 (grey `/`) | F1@0.5 |

DETR is excluded: its mAP is 0.0011, so its bar has no visible height in some panels and flattens
every other model in the rest. The value is still in `snapshot_audit.csv` and the model still
appears in figures 01 and 12.

## 04 · `04_compute_throughput_latency` — Compute cost

Four panels, **(a)** throughput **(b)** mean latency **(c)** p95 latency **(d)** whole-split wall
time. FPS is linear; latency and wall time are log-scaled because the backends span orders of
magnitude.

## 05 · `05_parameters_vs_flops` — Model size and compute complexity

Parameters against FLOPs, both log-scaled, one point per model, annotated. Colour is the family:
anomaly (teal) or detection (blue).

## 06 · `06_quality_vs_latency` — Quality versus latency

Three panels by metric granularity, not by model family: **(a)** image level (Image AUROC),
**(b)** pixel level (Pixel F1), **(c)** instance level (mAP@0.5). Mean latency on a log axis.
Anomaly and segmentation models share panel (b), which is the only quality metric they both report.

## 07 · `07_memory_by_measurement_kind` — Peak memory by instrument

**(a)** whole-process RSS, **(b)** CUDA allocator peak. **The two panels are not two views of one
number and must not be compared across:** RSS counts the entire Python process (weights, CUDA
context, host copies, framework overhead), the allocator counts GPU tensor memory only. A model
can show 1.8 GB RSS and 0.05 GB of allocator memory because its weights live in host memory. Each
panel has its own y-axis for that reason.

## 08 · `08_instance_size_breakdown` — Size-bucketed AP and AR

The detection report's size breakdown, which the four headline metrics in 03 do not cover.

| Bar | Metric |
| --- | --- |
| 1 (grey solid) | mAP small |
| 2 (red `/`) | mAP medium |
| 3 (blue `/`) | mAP large |
| 4 (grey `/`) | mAR@1 |
| 5 (amber `\`) | mAR@10 |
| 6 (teal `\`) | mAR@100 |

## 09 · `09_instance_counts` — Detections against ground truth

**(a)** boxes at the evaluation confidence floor, **(b)** the precision / recall / F1 of those same
boxes. Panel (a) is what mAP does not say: whether a model found the defects at all. DETR has no
bars — it emits no box above the floor.

## 10 · `10_ranking` — Three-dimension rank

**(a)** anomaly detection / zero-shot, **(b)** supervised defect detection. Split because an
anomaly model and a detector were never measured on the same metrics.

Each model gets three bars — its rank within its own panel on **Technical** (the image + pixel +
instance accuracy tables), **Memory** and **Compute** (the two overhead tables). Every metric is
scored in the direction `metrics_taxonomy` declares for it (`higher` / `lower`; `neutral` and
non-numeric columns are skipped), min-max normalized across the panel, and a model's dimension
score is the average of the metrics it actually reports — so reporting fewer metrics is neither
punished nor rewarded. All three bars are ranks (1 = best) on one axis. Models are ordered by their
mean rank; that mean is not drawn, because a fourth series on a rank axis reads as a fourth
measurement rather than as a summary.

## 11 · `11_anomaly_image_level_pr_roc` — PR/ROC, anomaly models

**(a)** precision-recall, **(b)** ROC, image level, 10 anomaly models. Curves are smoothed for
display only (Gaussian on a dense resampling of the empirical staircase); the AP and AUROC in the
legend are computed from the unsmoothed data.

## 12 · `12_detection_image_level_pr_roc` — PR/ROC, detection models

Same two panels for the 6 detection backends. DETR is included: its AUROC (0.65) and AP (0.50) are
real, readable numbers even though its box metrics are degenerate.

## 13 · `13_anomaly_pixel_level_pr_roc` — PR/ROC, pixel level

Same two panels for the 9 anomaly models that persist a continuous anomaly map. The three mask
models (UNet++, DeepLabV3+, Mask R-CNN) cannot appear: a binarised mask has no score to sweep, so
it yields one operating point, not a curve. Their pixel quality is in figure 02 instead.

---

## Conventions shared by every figure

- **No title inside the image.** The report writes its heading. Set `FIGURE_TITLES = True` in
  `plot_local_benchmark.py` for a standalone copy.
- **One legend per figure when the panels share their bar styles** (10), one per panel when they do
  not (09). Never the same legend twice.
- **Multi-panel figures are lettered** `(a)`, `(b)`, … above each panel.
- **A missing measurement is an absent bar, never a zero-height one.**
- **DETR is dropped from figures 03–10** (`EXCLUDED_MODELS`); it stays in the audit table, in 01
  and in 12.
