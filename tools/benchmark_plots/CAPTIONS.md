# Figure captions

The descriptions live here, not inside the PNGs: a heading burned into an image cannot be
reworded, translated or renumbered afterwards. Every figure is written to
`artifacts/local_benchmark_plots/` as PNG + PDF + SVG. The SVG keeps text
editable, so a label can be reworded in a vector editor; PNG and PDF are the
self-contained copies.

Machines: **Small** = RTX 4090 24 GB, **Full** = A100 80 GB. Dataset: ZJU-Leaper test split,
350 samples, 640 × 640, fp32. 19 configurations, unless a caption says otherwise.

---

## 01 · `01_image_level` — Image-level metrics, split by purpose

Two panels, because an anomaly detector's image score and a supervised detector's image score are
not the same measurement. Both panels carry the **same metrics in the same order**, so a metric has
one bar style across the whole figure. A bar is absent where a model reports nothing for that
metric, never zero-filled.

| Bar | Metric |
| --- | --- |
| 1 (light grey solid) | Image AUROC |
| 2 (red `.`) | Image AP |
| 3 (blue `/`) | Image F1 |
| 4 (grey `/`) | Image Precision |
| 5 (amber `\`) | Image Recall |

**(a) Anomaly detection** — the ten unsupervised / zero-shot backends, on their continuous
normality score. **(b) Defect detection** — the supervised detectors, on `max(box confidence)`.
Panel titles stay that short on purpose: what each panel is *about* belongs in this caption, not in
the image. `mAP@0.5` is deliberately not here — mAP is an instance-level metric and already has
`03_instance_level`.

**The two panels are not a ranking of one another.** A detector's image score is `max(box
confidence)`, which is exactly `0` whenever it finds nothing: `YOLOv8n` scores 157 of 245 normal
images at 0 while no defective image scores 0, so its AUROC is 0.997 even though its recall at its
own threshold is 0.571 (`YOLOv8s`: AUROC 0.993, recall 0.362). The detector ranks well and fires at
a threshold that misses most defects. Within a panel the bars compare like with like; across the two
panels they do not, which is why the families are drawn apart instead of on one axis.

`Image AP` is the area under panel (a) of the PR/ROC curve figures, which is what aligns this
figure with them on the precision-recall dimension. It is produced by both evaluators as of
2026-09-21, so it is absent from a snapshot taken before that and appears after the next
benchmark re-run.

`DETR` is absent: `EXCLUDED_MODELS` drops it from every bar figure. Its F1 / precision / recall are
0.000 — not missing — and its AUROC of 0.65 is still readable in
`12_detection_image_level_pr_roc`. Segmentation models (`DeepLabV3+`, `UNet++`, `Mask R-CNN`)
report no image-level score at all and appear in `02_pixel_level` instead.

## 02 · `02_pixel_level` — Pixel-level metrics

Pixel-level accuracy for the 12 models whose output is a mask or a heatmap. The chart mixes two
metric families, and a group carries only the family its output supports.

- **Threshold-free ranking** — Pixel AUROC, AUPRO, IAP. These sweep a threshold over a
  *continuous* score map, so a model that persists only a binarised mask cannot report them.
- **Thresholded overlap** — Pixel F1.

| Bar | Metric |
| --- | --- |
| 1 (light grey solid) | Pixel AUROC |
| 2 (red `.`) | AUPRO |
| 3 (blue `/`) | IAP |
| 4 (grey `/`) | Pixel F1 |

All twelve groups carry the same four bars — on this snapshot the three mask models report the
ranking metrics too, so no group is short a bar.

`dice` is not plotted: for a binary mask it *is* `pixel_f1` (the evaluator's `_dice` and
`_pixel_f1` are the same formula, and the two columns agree to every quoted digit).

`miou` is not plotted either, and for the same class of reason: only the three segmentation models
report it, so it was a fifth bar on three of twelve groups and nothing on the other nine, which
reads as a missing measurement rather than an inapplicable one. For a single binary mask per image
IoU is also a monotone function of F1 (`IoU = F1 / (2 - F1)`), so the bar would repeat `pixel_f1`.
It remains in `snapshot_audit.csv` and in the report's segmentation tables.

## 03 · `03_instance_level` — Instance-level detection metrics

Box quality for the detection backends.

| Bar | Metric |
| --- | --- |
| 1 (light grey solid) | mAP@[.5:.95] |
| 2 (red `.`) | mAP@0.5 |
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

**(a)** "Peak memory, whole program", **(b)** "Peak memory, GPU". The panel titles are deliberately
plain — a reader is not expected to know what an RSS or a CUDA allocator is — so the precise
definitions live here.

**The two panels are not two views of one number and must not be compared across:**

- Panel (a) is measured as the **resident set size of the whole Python process**: the model
  weights, the GPU context, host copies of tensors, the framework and its libraries.
- Panel (b) is measured as **tensor memory allocated on the GPU** (`torch.cuda.max_memory_allocated`),
  which excludes the interpreter, the host copies and the CUDA context itself.

Because the two instruments answer different questions, only some models have a reading in each
panel: 11 of 19 were profiled with the whole-program instrument and 8 with the GPU-tensor one. A
model is absent from the panel that did not measure it rather than drawn at zero, and each panel
has its own y-axis because the two ranges differ by an order of magnitude. A model can show 1.8 GB
of whole-program memory and 0.05 GB of GPU-tensor memory because its weights live in host memory.
Whichever instrument a row carries is recorded as `memory_measurement_kind` in the snapshot and in
`snapshot_audit.csv`, and the report's memory section lists it per model.

## 08 · `08_instance_size_breakdown` — Size-bucketed AP and AR

The detection report's size breakdown, which the four headline metrics in 03 do not cover.

| Bar | Metric |
| --- | --- |
| 1 (light grey solid) | mAP small |
| 2 (red `.`) | mAP medium |
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
