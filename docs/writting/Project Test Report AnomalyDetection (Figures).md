# Project Test Report: AnomalyDetection

This report reproduces the examples and experiments of the AnomalyDetection project by following the step-by-step instructions in the [README](../../README.md). The testing results were reproduced on an A100 GPU server.

## Outline

- [1. Hardware and Software Specification](#1-hardware-and-software-specification)
- [2. Evaluation Reproduction](#2-evaluation-reproduction)
  - [2.1 Examples](#21-examples)
  - [2.2 Technical and Overhead Metrics](#22-technical-and-overhead-metrics)
    - [2.2.1 Technical Metrics: Image Level](#221-technical-metrics-image-level)
    - [2.2.2 Technical Metrics: Pixel Level](#222-technical-metrics-pixel-level)
    - [2.2.3 Technical Metrics: Instance Level](#223-technical-metrics-instance-level)
    - [2.2.4 Overhead Metrics: Compute](#224-overhead-metrics-compute)
    - [2.2.5 Overhead Metrics: Memory](#225-overhead-metrics-memory)
  - [2.3 Analysis](#23-analysis)
- [3. Extensibility](#3-extensibility)
  - [3.1 Adding a New Dataset](#31-adding-a-new-dataset)
  - [3.2 Adding the YOLO 26 Model](#32-adding-the-yolo-26-model)
- [4. Discussion](#4-discussion)

## 1. Hardware and Software Specification

The hardware and software configuration of the testing platform is listed in Table 1.

<p align="center"><strong>Table 1: Hardware and Software Configuration</strong></p>

<div align="center">

|      Subsystem      |            Value             |
| :---: | :---: |
|  Operating System   |          Ubuntu 22.04.5          |
|         CPU         |      2 × Intel Xeon Gold 6430       |
|    System Memory    |              256 GB              |
|     GPU & VRAM      |          NVIDIA A100 80GB          |
|       Python        |               3.12               |
|       PyTorch       |           2.14.0 (+cu130)         |
|   CUDA Toolchain    |     Driver 550.144.03, CUDA 12.4    |

</div>

<br>

## 2. Evaluation Reproduction

### 2.1 Examples

The environment was set up per [Environment Setup](../../README.md#2-environment-setup), then the test cases under [Examples](../../README.md#5-examples) were reproduced.

Following Steps 1 to 4, the target model was selected and loaded:

<p align="center">
  <img src="../../docs/images/img_1.png" alt="Example Image" width="90%" />
</p>

Following Step 5, the image to be evaluated was selected and loaded:

<p align="center">
  <img src="../../docs/images/img_2.png" alt="Example Image" width="90%" />
</p>

Following Step 6, anomaly detection was executed successfully. The resulting output confirms that the platform and its configuration are working correctly, so the remaining tests can be performed.

<p align="center">
  <img src="../../docs/images/img_3.png" alt="Example Image" width="90%" />
</p>

### 2.2 Technical and Overhead Metrics

Following the instructions in [README §4.1.3 Benchmarking](../../README.md#413-benchmarking), the complete benchmark suite was run. The resulting data are reported below as figures.

The evaluated configuration:

<div align="center">

| Item | Value |
| :---: | :---: |
| Dataset | [ZJU-Leaper](../../README.md#32-supported-datasets) test split |
| Samples | 350 in total |
| Input size | 640 × 640 |
| Precision | fp32 |
| Metrics | [Image, pixel, instance and overhead](../../README.md#33-supported-metrics) |
| Evaluated configurations | 19 |

</div>

<br>

#### 2.2.1 Technical Metrics: Image Level

This part evaluates the image-level metrics defined by [README §3.3.1](../../README.md#331-technical-metrics-image-level):

- **Image AUROC** (Area Under the Receiver Operating Characteristic curve)**:** the probability that a defective image is ranked above a normal one. 0.5 is random guessing, 1.0 is perfect, and no decision threshold is needed.
- **Image Precision:** the share of images flagged as defective that really are defective.
- **Image Recall:** the share of truly defective images that are flagged.
- **Image F1:** the balance of precision and recall in one number; it drops when either is poor.

**Key observation:** The detection models top this axis — YOLO11n and YOLOv8n both at 0.9965 — but the thresholded metrics separate them sharply: each reports precision 1.000 and recall 0.600 / 0.571, so their F1 falls to 0.750 / 0.727. Among the anomaly models PatchCore leads at 0.9932, and GANomaly, Reverse Distillation and SuperSimpleNet sit near chance at 0.65 / 0.62 / 0.59.

<p align="center"><strong>Figure 1: Image-level metrics on the A100 server</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/01_image_level.png" alt="Image-level metrics on the A100 server" width="100%" />
</p>

<sub>One group of four bars per model: (1) Image AUROC, (2) Image F1, (3) Image Precision, (4) Image Recall. An absent bar means the model reports nothing for that metric, never a zero. DETR's F1 / precision / recall are 0.000 rather than missing — it scores no box above the evaluation floor, so its derived image score is 0. All 16 models that emit an image-level score are shown: 10 anomaly and 6 detection, the latter scored from their highest-confidence box per image.</sub>

<br>

<p align="center"><strong>Figure 2: Image-level precision-recall and ROC, anomaly models</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/11_anomaly_image_level_pr_roc.png" alt="Image-level precision-recall and ROC, anomaly models" width="100%" />
</p>

<sub>**(a)** precision-recall, **(b)** ROC, over the 10 anomaly models. Curves are smoothed for display only — a Gaussian kernel over a dense resampling of the empirical staircase; the AP and AUROC in each legend are computed from the unsmoothed data. The x- and y-ranges can be changed with `--xlim` / `--ylim`.</sub>

<br>

<p align="center"><strong>Figure 3: Image-level precision-recall and ROC, detection models</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/12_detection_image_level_pr_roc.png" alt="Image-level precision-recall and ROC, detection models" width="100%" />
</p>

<sub>**(a)** precision-recall, **(b)** ROC, over the 6 detection backends. DETR appears here: its AUROC (0.65) and AP (0.50) are real, readable numbers even though its box metrics are degenerate.</sub>

<br>


#### 2.2.2 Technical Metrics: Pixel Level

This part evaluates the pixel-level metrics defined by [README §3.3.2](../../README.md#332-technical-metrics-pixel-level):

- **Pixel AUROC:** the probability that a defective pixel is ranked above a normal one, pooled over the pixels of all images. 0.5 is random guessing.
- **Pixel AUPRO:** the share of each defective region that is covered before the false-positive rate reaches a fixed limit. Every region counts equally, so a small defect is not hidden by a large one.
- **Pixel F1:** the match between predicted and true defect pixels at a fixed pixel threshold, penalised by missed pixels and by extra pixels.
- **Pixel IoU:** the intersection over union between the predicted defect area and the true defect area.
- **IAP (Instance Average Precision):** the average precision over connected defect regions, with every region weighted equally.

**Key observation:** MoECLIP leads localization by a wide margin (0.9857 AUROC / 0.9701 AUPRO) and SuperSimpleNet is last (0.4547 / 0.5033). Pixel F1 and mIoU stay low across the board — the best is MoECLIP at 0.6733 — because the pixel threshold is not calibrated on this run; AUROC and AUPRO need no threshold and are therefore the more informative columns here.

<p align="center"><strong>Figure 4: Pixel-level metrics on the A100 server</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/02_pixel_level.png" alt="Pixel-level metrics on the A100 server" width="100%" />
</p>

<sub>The chart mixes two metric families and a group carries only the family its output supports. Bars per model: (1) Pixel AUROC, (2) AUPRO, (3) IAP, (4) Pixel F1, (5) mIoU. Pixel AUROC, AUPRO and IAP sweep a threshold over a *continuous* score map; Pixel F1 and mIoU need only one binary mask, which is why the three mask models report two bars and the anomaly-map models four. `dice` is not plotted: for a binary mask it is numerically identical to `pixel_f1`.</sub>

<br>

<sub>All 9 models that emit a per-pixel anomaly map.</sub>

<br>

<p align="center"><strong>Figure 5: Pixel-level precision-recall and ROC, anomaly-map models</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/13_anomaly_pixel_level_pr_roc.png" alt="Pixel-level precision-recall and ROC, anomaly-map models" width="100%" />
</p>

<sub>**(a)** precision-recall, **(b)** ROC, over the 9 anomaly models that persist a continuous anomaly map. The three mask models (UNet++, DeepLabV3+, Mask R-CNN) cannot appear: a binarised mask has no score to sweep, so it yields a single operating point rather than a curve. Their pixel quality is in Figure 4.</sub>

<br>


The 3 segmentation models are the other pixel-level producers; their Pixel F1 and mIoU are the last two bar styles in Figure 4. They report no threshold-free pixel metric, because their output is a binary mask rather than a score map.

#### 2.2.3 Technical Metrics: Instance Level

This part evaluates the instance-level metrics defined by [README §3.3.3](../../README.md#333-technical-metrics-instance-level):

- **AP:** box precision–recall quality averaged over IoU thresholds 0.50–0.95 (COCO style); higher is better.
- **AP50 / AP75:** AP at a single IoU threshold. 0.50 accepts a loosely placed box, 0.75 demands a tight one, where IoU is the overlap between a predicted box and the true box.
- **Precision:** the share of reported boxes that overlap a real defect (IoU ≥ 0.50).
- **Recall:** the share of real defects covered by a reported box.
- **F1:** the balance of box precision and recall in one number.
- **TP / FP / FN:** the underlying counts, namely defects found, spurious boxes and defects missed.

**Key observation:** The detectors divide into two groups. The YOLO variants are the most precise (0.857–0.929) and the least complete (recall 0.214–0.363); the two R-CNN detectors are the reverse (precision 0.550 / 0.633, recall 0.692 / 0.681), and Cascade R-CNN takes the best AP50 (0.6428) and F1 (0.6561). DETR finds nothing at all — TP 0, FP 0, FN 182, the whole test split.

<p align="center"><strong>Figure 6: Instance-level detection metrics on the A100 server</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/03_instance_level.png" alt="Instance-level detection metrics on the A100 server" width="100%" />
</p>

<sub>Bars per model: (1) mAP@[.5:.95], (2) mAP@0.5, (3) mAP@0.75, (4) F1@0.5. DETR is excluded: its mAP is 0.0011, so its bar has no visible height and it flattens every other model. The value is still in `snapshot_audit.csv` and the model still appears in Figures 1 and 3.</sub>

<br>

<p align="center"><strong>Figure 7: Size-bucketed AP and AR</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/08_instance_size_breakdown.png" alt="Size-bucketed AP and AR" width="100%" />
</p>

<sub>The size breakdown the four headline metrics above do not cover. Bars per model: (1) mAP small, (2) mAP medium, (3) mAP large, (4) mAR@1, (5) mAR@10, (6) mAR@100.</sub>

<br>

<p align="center"><strong>Figure 8: Detections against ground truth</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/09_instance_counts.png" alt="Detections against ground truth" width="100%" />
</p>

<sub>**(a)** boxes kept at the evaluation confidence floor, **(b)** the precision / recall / F1 of those same boxes. Panel (a) is what mAP does not say: whether a model found the defects at all. DETR has no bars — it emits no box above the floor.</sub>

<br>

- Precision, Recall, F1 and TP/FP/FN are taken at confidence threshold 0.25; AP75 is the third bar style in Figure 6, and mAP@[.5:.95] is the first.
- A standalone `adh evaluate` run on the DETR checkpoint independently confirmed `TP=0` and `FP=0`.

#### 2.2.4 Overhead Metrics: Compute

This part evaluates the compute metrics defined by [README §3.3.5](../../README.md#335-overhead-metrics-compute):

- **FPS:** images processed per second.
- **Latency mean / p95 / p99:** the time to process one image, in milliseconds; p95 and p99 describe the slowest frames rather than the average.
- **FLOPs (G):** the arithmetic work of one forward pass, in billions of operations. Fixed by the architecture, so identical on any machine.
- **Wall-time:** the total seconds the scored run took, accuracy pass included.

**Key observation:** Throughput spans more than three orders of magnitude on one machine: YOLOv8n reaches 432 FPS and Cascade R-CNN 0.82 FPS. FLOPs do not predict it — YOLOv8s does less arithmetic than YOLOv8n (6.99 G against 14.90 G) and still runs slower (355 against 432 FPS), so architecture and memory traffic matter as much as the operation count.

<p align="center"><strong>Figure 9: Compute cost on the A100 server</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/04_compute_throughput_latency.png" alt="Compute cost on the A100 server" width="100%" />
</p>

<sub>Four panels: **(a)** throughput (FPS), **(b)** mean latency, **(c)** p95 latency, **(d)** whole-split wall time. FPS is linear; latency and wall time are log-scaled because the backends span orders of magnitude. All 19 configurations are shown.</sub>

<br>

<p align="center"><strong>Figure 10: Quality versus latency, by metric granularity</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/06_quality_vs_latency.png" alt="Quality versus latency, by metric granularity" width="100%" />
</p>

<sub>Three panels by metric granularity rather than by model family: **(a)** image level (Image AUROC), **(b)** pixel level (Pixel F1), **(c)** instance level (mAP@0.5); mean latency is on a log axis. Anomaly and segmentation models share panel (b), which is the only quality metric they both report.</sub>

<br>


#### 2.2.5 Overhead Metrics: Memory

This part evaluates the memory metrics defined by [README §3.3.6](../../README.md#336-overhead-metrics-memory):

- **Parameters (M):** the number of learned weights, in millions. Fixed by the architecture, so identical on any machine.
- **Peak memory:** the most memory used at once during the run.
- **Allocator peak:** the same peak measured by the CUDA allocator, which excludes the Python interpreter and its libraries.
- **Retained / extra:** the memory still held after the run, which a long-running service keeps resident.

**Key observation:** Peak memory spans two orders of magnitude on one run — YOLOv8n at 46.5 MB against PatchCore at 4505 MB. The two instruments are not interchangeable: whole-process RSS includes the weights, the CUDA context and host copies, while the CUDA allocator counts GPU tensor memory only, which is why Figure 11 gives them separate axes.

<p align="center"><strong>Figure 11: Peak memory by measurement instrument</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/07_memory_by_measurement_kind.png" alt="Peak memory by measurement instrument" width="100%" />
</p>

<sub>**(a)** whole-process RSS and **(b)** CUDA allocator peak. The two panels are **not two views of one number** and must not be compared across: RSS counts the entire Python process (weights, CUDA context, host copies, framework overhead) while the allocator counts GPU tensor memory only — a model can show 1.8 GB of RSS and 0.05 GB of allocator memory because its weights live in host memory. Each panel therefore has its own y-axis. Peak memory is a property of this host and should not be carried to another machine.</sub>

<br>

<p align="center"><strong>Figure 12: Model size and compute complexity</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/05_parameters_vs_flops.png" alt="Model size and compute complexity" width="100%" />
</p>

<sub>Parameters against FLOPs, both log-scaled and annotated. Both are fixed by the architecture, so unlike FPS, latency and memory they transfer to any machine. Colour marks the family: anomaly (teal) or detection (blue).</sub>

<br>


### 2.3 Analysis

The 19 configurations split into two paradigms: **supervised Defect Detection (DD)** and **unsupervised / zero-shot Anomaly Detection (AD)**. Both are scored on the metric families of [README §3.3](../../README.md#33-supported-metrics) — image level, pixel level, instance level and overhead — and the rankings below use the results (Figures 1, 4 and 6).

Because the dimensions have different scales, each model is reduced to three independent ranks — accuracy, memory and compute — as Figure 13 shows. Ranks are computed within a paradigm, so a model is only ever compared with models measured on the same metrics.

<p align="center"><strong>Figure 13: Three-dimension rank</strong></p>

<p align="center">
  <img src="../../artifacts/local_benchmark_plots/10_ranking.png" alt="Three-dimension rank" width="100%" />
</p>

<sub>**(a)** anomaly detection / zero-shot, **(b)** supervised defect detection — split because an anomaly model and a detector were never measured on the same metrics. Each model gets three bars: its rank within its own panel on **Technical** (the image + pixel + instance accuracy tables), **Memory** and **Compute** (the two overhead tables). Every metric is scored in the direction the project's metric taxonomy declares for it (`higher` / `lower`; `neutral` and non-numeric columns are skipped), min-max normalized across the panel, and a model's dimension score is the average of the metrics it actually reports — so reporting fewer metrics is neither punished nor rewarded. All three bars are ranks (1 = best) on one axis; models are ordered by their mean rank, which is not drawn, because a fourth series on a rank axis reads as a fourth measurement rather than as a summary. Panel (a) includes the three segmentation models, which the project classifies as pixel-level anomaly detection, so it ranks 13 models rather than the 10 anomaly backends alone.</sub>

<br>


- **Anomaly and segmentation (AD).** Mask R-CNN leads the combined order at 3.33, but only on the strength of its technical rank: 1st on accuracy, 3rd on memory, 6th on compute. The three dimensions disagree sharply for most models — MoECLIP is 2nd on technical and 12th on memory, PaDiM is 1st on compute and 10th on technical, and WinCLIP and Reverse Distillation come last overall. A single blended position would hide exactly that, which is why Figure 13 shows one bar per dimension rather than one number.
- **Detection (DD).** YOLO11n leads at 2.33, ahead of YOLOv8n (2.67), YOLOv8s (3.00), Faster R-CNN (3.33) and Cascade R-CNN (3.67). Cascade R-CNN is the most accurate of the five and the slowest; YOLOv8s is the reverse. DETR is absent from the panel because its instance metrics are degenerate — see Figure 6.
- **Segmentation (SEG).** Mask R-CNN 0.7242 > UNet++ 0.6833 > DeepLabV3+ 0.6477 (Pixel F1).
- **Image level.** PatchCore 0.9932, MoECLIP 0.9877, Dinomaly 0.9797 and WinCLIP 0.9772 lead the anomaly models; the YOLO11n / YOLOv8n pair (0.9965) tops the whole run.
- **Pixel level.** MoECLIP leads localization (0.9857 AUROC / 0.9701 AUPRO). Pixel F1 and mIoU stay low because the pixel threshold is not calibrated.
- **Overhead.** 0.82–432 FPS across 19 models: YOLO real-time, Cascade R-CNN and WinCLIP ≈ 1 FPS.

**Scope.** Every number here describes this host. Parameter counts and FLOPs are fixed by the architecture and transfer to any machine; FPS, latency and peak memory do not.

## 3. Extensibility

### 3.1 Adding a New Dataset

Following [README §6.1](../../README.md#61-example-add-a-small-anomaly-dataset), the `bottle` category of MVTec AD was copied into a new dataset root, `datasets/general/Bottle`, and PatchCore was trained and tested on it.

<p align="center"><strong>Table 15: Extensibility validation — PatchCore on the newly added Bottle dataset</strong></p>

<div align="center">

| Image AUROC | Pixel AUROC | Pixel AUPRO | IAP |
| :---: | :---: | :---: | :---: |
| 1.0000 | 0.9935 | 0.9934 | 0.8461 |

</div>

<br>

The results are usable, confirming that the new dataset was added successfully and that PatchCore trained on it.

### 3.2 Adding the YOLO 26 Model

Following [README §6.2](../../README.md#62-example-add-a-yolo-26-model), YOLO 26 was added to the `ultralytics` backend:

- **Preset entry** — `yolo26n` variant and alias in `presets.py`
- **Model configuration** — `configs/models/ultralytics_yolo26_example.yaml`
- **Registry row** — one `yolo26n` entry in `configs/registry/models.yaml`
- **Training run** — ZJU-Leaper, test mode (a pipeline check, not an accuracy run): 1 epoch, batch 2, 640 × 640 input, 8 training and 8 validation samples

The run produced a new model, registered as `textile/artifacts/models/yolo26n_yolo26n_zju_leaper.pt` with 2.50 M parameters, confirming that YOLO 26 was added successfully.

## 4. Discussion

- **Training budgets are short.** Several entries were trained for only a few steps — the anomaly smoke configuration is 1 epoch on 8 images — so the weaker results in §2.3 reflect the run budget as much as the architecture.
- **Overhead metrics are host-defined.** FPS, latency and peak memory are properties of the machine that measured them — this run is an A100 server — so they should not be carried to another host. Only FLOPs and parameter counts are host-independent.
- **Coverage gaps.** MambaAD untested; MoECLIP scored on MVTec AD rather than ZJU-Leaper; cross-domain ([README §3.3.4](../../README.md#334-technical-metrics-cross-domain)) not measured on this run; `LMEI`, `Max streams @budget`, `1-stream latency`, `resolution slope` and power/energy not produced.
