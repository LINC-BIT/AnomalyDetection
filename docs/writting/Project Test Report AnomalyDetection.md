# Project Test Report: AnomalyDetection

In this report, we reproduce all the exmaples and expaeriments in the project of AnomalyDetection, following the step-by-step instructions in [README](../../README.md). The test outcomes were evaluated on both a GPU workstation (RTX 4090) and a high-performance GPU server to verify its reproducibility.

## Outline

- [1. Hardware and Software Specifications](#1-hardware-and-software-specifications)
- [2. Evaluation Reproduction](#2-evaluation-reproduction)
  - [2.1 Examples](#21-examples)
  - [2.2 Technical and Overhead Metrics](#22-technical-and-overhead-metrics)
    - [2.2.1 Technical Metrics: Image Level](#221-technical-metrics-image-level)
    - [2.2.2 Technical Metrics: Pixel Level](#222-technical-metrics-pixel-level)
    - [2.2.3 Technical Metrics: Instance Level](#223-technical-metrics-instance-level)
    - [2.2.4 Overhead Metrics: Compute](#224-overhead-metrics-compute)
    - [2.2.5 Overhead Metrics: Memory](#225-overhead-metrics-memory)
- [3. Analysis](#3-analysis)
- [4. Observations and Conclusion](#4-observations-and-conclusion)

## 1. Hardware and Software Specifications

This report presents a comparison of configurations and experimental results between two platforms:

- Small Machine: RTX 4090 workstation
- Full Machine: A100 server

The hardware and software information for the two platforms are shown in the table below:

<p align="center"><strong>Table 1: Hardware and Software Configuration</strong></p>

<div align="center">

|      Subsystem      |            Small Machine             |              Full Machine               |
| :---: | :---: | :---: |
|  Operating System   |          Ubuntu 22.04 LTS            |       Ubuntu 22.04.5     |
|         CPU         |                  Intel Core i9-13900K                   | 2 × Intel Xeon Gold 6430    |
|    System Memory    |                  32 GB                   |                  256 GB                   |
|     GPU & VRAM      |       NVIDIA RTX 4090 24 GB          |            NVIDIA A100 80GB               |
|       Python        |                 3.12                 |                   3.12                    |
|       PyTorch       |                  2.14.0 (+cu130)                   |              2.14.0 (+cu130)              |
|   CUDA Toolchain    |             Driver 535.54.03, CUDA 12.8                |        Driver 550.144.03, CUDA 12.4       |

</div>

<br>

## 2. Evaluation Reproduction

### 2.1 Examples

This section starts with setting up the environment following the [Environment Setup](../../README.md#2-environment-setup) guide, followed by reproducing the test cases listed under the [Examples](../../README.md#5-examples) section of the README.

This process presents only the execution results obtained on the Full Machine.

[TODO] add the images and data.

### 2.2 Technical and Overhead Metrics

Both platforms were evaluated using the following identical configuration:

<div align="center">

| Item | Value |
| :---: | :---: |
| Dataset | [ZJU-Leaper](../../README.md#32-supported-datasets), On the test split |
| Sample Size | 350 samples in total |
| Input size | 640 × 640 |
| Precision | fp32 |
| Test Metrics | [Image Level, Pixel Level, Instance Level, Overhead](../../README.md#33-supported-metrics) |
| Total Evaluated Model Configurations | 19 |

</div>

<br>

Representative metrics are reported below. The count is 19 unique model configurations, not 19 entries in every table: 10 anomaly models, 6 detection models, and 3 segmentation models. Each configuration appears in the table(s) for metrics its output supports; for example, GANomaly is included in image-level, compute, and memory results but cannot appear in a pixel-localization table because it has no spatial anomaly map. For full details, see [results for Small Machine](./results_small_machine.md) and [results for Full Machine](./results_full_gpu_server.md).

#### 2.2.1 Technical Metrics: Image Level

This part evaluates the image-level metrics defined by [README §3.3.1](../../README.md#331-technical-metrics-image-level):

- **Image AUROC:** how often a defective image scores above a normal one. 0.5 is random guessing, 1.0 is perfect, and it needs no decision threshold.
- **Image Precision:** of the images flagged as defective, the share that really are defective.
- **Image Recall:** of the truly defective images, the share that is flagged.
- **Image F1:** one score that balances Precision against Recall, and drops if either of them is poor.

**Key observation:** The two machines agree on Image AUROC to within **<span style="color:#0070C0">0.2 points</span>** for all ten anomaly detectors, while the thresholded metrics, which depend on the calibration of the decision threshold, drift by up to 4.4 points.

<p align="center"><strong>Table 2: Image-level testing results on Small Machine</strong></p>

<div align="center">

| Model | Image AUROC | Image F1 | Image Precision | Image Recall |
| :---: | :---: | :---: | :---: | :---: |
| Dinomaly | 0.9797 | 0.9320 | 0.9505 | 0.9143 |
| EfficientAD | 0.9502 | 0.8586 | 0.9535 | 0.7810 |
| GANomaly | 0.6548 | 0.5159 | 0.3876 | 0.7714 |
| MoECLIP | 0.9877 | 0.9314 | 0.9596 | 0.9048 |
| PaDiM | 0.8282 | 0.7138 | 0.5854 | 0.9143 |
| PatchCore | 0.9932 | 0.9474 | 0.9519 | 0.9429 |
| Reverse Distillation | 0.6204 | 0.5455 | 0.3941 | 0.8857 |
| STFPM | 0.9497 | 0.8531 | 0.8491 | 0.8571 |
| SuperSimpleNet | 0.5847 | 0.5000 | 0.3699 | 0.7714 |
| WinCLIP | 0.9772 | 0.9026 | 0.9778 | 0.8381 |

</div>

<br>

<p align="center"><strong>Table 3: Image-level testing results on Full Machine</strong></p>

<div align="center">

| Model | Image AUROC | Image F1 | Image Precision | Image Recall |
| :---: | :---: | :---: | :---: | :---: |
| Dinomaly | 0.9797 | 0.9151 | 0.9065 | 0.9238 |
| EfficientAD | 0.9501 | 0.8557 | 0.9326 | 0.7905 |
| GANomaly | 0.6548 | 0.4802 | 0.3244 | 0.9238 |
| MoECLIP | 0.9877 | 0.9209 | 0.9000 | 0.9429 |
| PaDiM | 0.8284 | 0.7138 | 0.5854 | 0.9143 |
| PatchCore | 0.9932 | 0.9469 | 0.9608 | 0.9333 |
| Reverse Distillation | 0.6222 | 0.5378 | 0.3810 | 0.9143 |
| STFPM | 0.9497 | 0.8458 | 0.8854 | 0.8095 |
| SuperSimpleNet | 0.5852 | 0.4598 | 0.3003 | 0.9810 |
| WinCLIP | 0.9772 | 0.8584 | 0.8017 | 0.9238 |

</div>

<br>

#### 2.2.2 Technical Metrics: Pixel Level

This part evaluates the pixel-level metrics defined by [README §3.3.2](../../README.md#332-technical-metrics-pixel-level):

- **Pixel AUROC:** how often a defective pixel scores above a normal pixel, pooled over the pixels of all images. 0.5 is random guessing.
- **Pixel AUPRO / Pixel PRO:** how much of each defective region is covered before the false-positive rate passes a fixed limit. Every connected region counts equally, so a small defect is not hidden by a large one.
- **Pixel F1:** how well the predicted defect pixels match the true ones at a fixed pixel threshold, penalised both by missed pixels and by extra pixels.
- **Pixel IoU:** overlap between the predicted defect area and the true defect area, as intersection over union.
- **Image AP:** how well defect regions are found, averaged over connected regions with each region weighted equally.

**Key observation:** Both machines rank MoECLIP first and SuperSimpleNet last, but their absolute pixel-level values do not overlap; see [3. Analysis](#3-analysis).

<p align="center"><strong>Table 4: Pixel-level testing results on Small Machine</strong></p>

<div align="center">

| Model | Pixel AUROC | Pixel AUPRO | Pixel F1 | Image AP |
| :---: | :---: | :---: | :---: | :---: |
| Dinomaly | 0.7452 | 0.6029 | 0.1470 | 0.0574 |
| EfficientAD | 0.6072 | 0.4814 | 0.1035 | 0.0287 |
| MoECLIP | 0.9039 | 0.8374 | 0.5447 | 0.3983 |
| PaDiM | 0.7559 | 0.7117 | 0.1282 | 0.0528 |
| PatchCore | 0.6730 | 0.5925 | 0.0841 | 0.0373 |
| Reverse Distillation | 0.3529 | 0.5415 | 0.4321 | 0.2120 |
| STFPM | 0.6807 | 0.6353 | 0.0954 | 0.0546 |
| SuperSimpleNet | 0.4645 | 0.4108 | 0.0548 | 0.0184 |
| WinCLIP | 0.7483 | 0.7399 | 0.1482 | 0.1240 |

</div>

<br>

<p align="center"><strong>Table 5: Pixel-level testing results on Full Machine</strong></p>

<div align="center">

| Model | Pixel AUROC | Pixel PRO | Pixel F1 | Pixel IoU |
| :---: | :---: | :---: | :---: | :---: |
| Dinomaly | 0.9805 | 0.8895 | 0.4247 | 0.6293 |
| EfficientAD | 0.7863 | 0.5437 | 0.4814 | 0.5549 |
| MoECLIP | 0.9857 | 0.9701 | 0.6733 | 0.7006 |
| PaDiM | 0.9521 | 0.9192 | 0.1658 | 0.3531 |
| PatchCore | 0.9833 | 0.9136 | 0.4005 | 0.6259 |
| Reverse Distillation | 0.6986 | 0.7003 | 0.0263 | 0.0868 |
| STFPM | 0.9570 | 0.8744 | 0.3461 | 0.5682 |
| SuperSimpleNet | 0.4547 | 0.5033 | 0.0252 | 0.0792 |
| WinCLIP | 0.9248 | 0.9140 | 0.1080 | 0.3078 |

</div>

<br>

`Pixel AUPRO` and `Pixel PRO` are the same metric under two names, and the Small Machine run did not measure `Pixel IoU`. These tables contain the 9 anomaly models that emit per-pixel maps; GANomaly remains part of the 19 evaluated configurations and is reported in the image-level and overhead tables, but is omitted here because it produces no per-pixel anomaly map.

#### 2.2.3 Technical Metrics: Instance Level

This part evaluates the instance-level metrics defined by [README §3.3.3](../../README.md#333-technical-metrics-instance-level):

- **AP:** how well the detected boxes follow the precision–recall trade-off, averaged over IoU thresholds 0.50–0.95 (COCO style). Higher is better.
- **AP50 / AP75:** the same measure at one IoU threshold. 0.50 accepts a loosely placed box, 0.75 demands a tight one. IoU is the overlap between a predicted box and the true box.
- **Precision:** of the boxes reported as defects, the share that overlaps a real defect (IoU ≥ 0.50).
- **Recall:** of the real defects, the share that a reported box covers.
- **F1:** one score that balances box Precision against box Recall.
- **TP / FP / FN:** the counts behind those numbers — correctly found defects, spurious boxes, and missed defects.

**Key observation:** The thresholded **<span style="color:#0070C0">F1 reproduces exactly</span>** across the two machines, while AP50 is higher on the Full Machine.

<p align="center"><strong>Table 6: Instance-level testing results on Small Machine</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | AP50 | AP | Precision | Recall | F1 | TP | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cascade R-CNN | 0.6232 | 0.3267 | 0.6327 | 0.6813 | 0.6561 | 124 | 72 | 58 |
| DETR | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 182 |
| Faster R-CNN | 0.6247 | 0.3185 | 0.5478 | 0.6923 | 0.6117 | 126 | 104 | 56 |
| YOLO11n | 0.4555 | 0.2316 | 0.9091 | 0.3297 | 0.4839 | 60 | 6 | 122 |
| YOLOv8n | 0.4595 | 0.2227 | 0.8571 | 0.3626 | 0.5097 | 66 | 11 | 116 |
| YOLOv8s | 0.4332 | 0.1991 | 0.9286 | 0.2143 | 0.3482 | 39 | 3 | 143 |

</div>

<br>

<p align="center"><strong>Table 7: Instance-level testing results on Full Machine</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | AP50 | AP | AP75 | Precision | Recall | F1 | TP | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cascade R-CNN | 0.6428 | 0.3333 | 0.3042 | 0.6327 | 0.6813 | 0.6561 | 124 | 72 | 58 |
| DETR | 0.0022 | 0.0011 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 182 |
| Faster R-CNN | 0.6623 | 0.3321 | 0.3014 | 0.5502 | 0.6923 | 0.6131 | 126 | 103 | 56 |
| YOLO11n | 0.5754 | 0.2846 | 0.2583 | 0.9091 | 0.3297 | 0.4839 | 60 | 6 | 122 |
| YOLOv8n | 0.5861 | 0.2721 | 0.2130 | 0.8571 | 0.3626 | 0.5097 | 66 | 11 | 116 |
| YOLOv8s | 0.5380 | 0.2512 | 0.1866 | 0.9286 | 0.2143 | 0.3482 | 39 | 3 | 143 |

</div>

<br>

Precision, Recall, F1 and TP/FP/FN are taken at confidence threshold 0.25. The Small Machine did not measure AP75, so that column is omitted. A standalone `adh evaluate` run on the available DETR checkpoint independently confirmed `TP=0` and `FP=0`. Supervised segmentation reproduces in the same way: on the Small Machine the masks score UNet++ 0.6833 / DeepLabV3+ 0.6479 / Mask R-CNN 0.7539 (Dice), and on the Full Machine the same models score 0.6833 / 0.6477 / 0.7242 (Mask AP50), i.e. the same metric under two names.

#### 2.2.4 Overhead Metrics: Compute

This part evaluates the compute metrics defined by [README §3.3.5](../../README.md#335-overhead-metrics-compute):

- **FPS (mean / p95):** images processed per second. The p95 is the speed that 95% of frames still reach, so it exposes occasional slow frames.
- **Latency mean / p95 / p99:** time to process one image, in milliseconds. p95 and p99 describe the slowest frames rather than the average.
- **FLOPs (G):** arithmetic work of one forward pass, in billions of operations. It is fixed by the architecture, so it is the same on any machine.
- **Wall-time:** total seconds the scored run took, accuracy pass included.
- **Max concurrent streams:** how many video streams can run at once while every frame still meets the latency budget.

**Key observation:** **<span style="color:#0070C0">FLOPs reproduce</span>** for every model measured twice, while FPS and latency are host-specific and differ by 1.1×–31×.

<p align="center"><strong>Table 8: Compute overhead testing results on Small Machine</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | FPS | Latency mean | FLOPs (G) | Max Concurrent Streams |
| :---: | :---: | :---: | :---: | :---: |
| EfficientAD | 3.37 | 296.57 ms | 75.91 | 0 |
| Faster R-CNN | 7.81 | 128.08 ms | 268.98 | 0 |
| GANomaly | 10.63 | 94.08 ms | 16.54 | 0 |
| PaDiM | 31.69 | 31.56 ms | 3.69 | 2 |
| PatchCore | 1.39 | 719.00 ms | 24.14 | 0 |
| STFPM | 22.00 | 45.44 ms | 7.38 | 1 |
| SuperSimpleNet | 2.40 | 417.01 ms | 96.70 | 0 |
| UNet++ | 119.22 | 100.19 ms | 250.23 | 0 |
| YOLO11n | 20.83 | 48.01 ms | 6.44 | 0 |
| YOLOv8n | 181.69 | 5.50 ms | 8.19 | 0 |
| YOLOv8s | 11.51 | 86.89 ms | 28.65 | 0 |

</div>

<br>

<p align="center"><strong>Table 9: Compute overhead testing results on Full Machine</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | FPS (mean) | FPS (p95) | Latency mean | Latency p95 | Latency p99 | FLOPs (G) | Wall-time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EfficientAD | 3.72 | 3.85 | 268.96 ms | 324.86 ms | 354.60 ms | 75.9065 | 60.7 s |
| Faster R-CNN | 60.42 | 60.67 | 16.55 ms | 18.27 ms | 19.72 ms | 268.9822 | 21.6 s |
| GANomaly | 34.85 | 34.92 | 28.69 ms | 30.37 ms | 33.46 ms | 33.0444 | 51.4 s |
| PaDiM | 77.32 | 77.41 | 12.93 ms | 13.51 ms | 13.82 ms | 3.6884 | 43.3 s |
| PatchCore | 4.42 | 4.54 | 226.31 ms | 309.07 ms | 312.48 ms | 24.1408 | 88.3 s |
| STFPM | 74.77 | 81.69 | 13.37 ms | 21.89 ms | 24.02 ms | 7.3767 | 88.5 s |
| SuperSimpleNet | 17.38 | 17.44 | 57.53 ms | 61.21 ms | 71.03 ms | 96.7028 | 93.0 s |
| UNet++ | 79.54 | 90.78 | 12.57 ms | 19.50 ms | 30.95 ms | 250.2320 | 68.4 s |
| YOLO11n | 308.48 | 310.60 | 3.24 ms | 3.45 ms | 3.47 ms | 6.4406 | 11.9 s |
| YOLOv8n | 432.21 | 437.37 | 2.31 ms | 2.64 ms | 2.68 ms | 14.9008 | 11.4 s |
| YOLOv8s | 355.33 | 355.92 | 2.81 ms | 2.95 ms | 3.24 ms | 6.9886 | 11.9 s |

</div>

<br>

The two runs report different compute metric sets: the Small Machine measured `Max Concurrent Streams` but not the percentiles or the wall time.

#### 2.2.5 Overhead Metrics: Memory

This part evaluates the memory metrics defined by [README §3.3.6](../../README.md#336-overhead-metrics-memory):

- **Parameters (M):** number of learned weights, in millions. Fixed by the architecture, so it is the same on any machine.
- **Peak memory:** the most memory used at once during the run.
- **Allocator peak:** the same peak as seen by the CUDA allocator, which leaves out the Python interpreter and its libraries.
- **Retained / extra:** memory still held after the run, which is what a long-running service keeps resident.
- **Memory backend:** the counter that produced the peak — whole-process resident set (`process_rss`) or CUDA allocator (`device_allocator`). The two are not on the same scale.

**Key observation:** **<span style="color:#0070C0">Parameters reproduce</span>** for every model measured twice, while peak memory is reported by two different backends and is not comparable across the two machines.

<p align="center"><strong>Table 10: Memory overhead testing results on Small Machine</strong></p>

<div align="center">

| Model | Parameters (M) | Peak Memory (MB) |
| :---: | :---: | :---: |
| EfficientAD | 8.06 | 1001.19 |
| Faster R-CNN | 41.35 | 610.70 |
| GANomaly | 188.69 | 2664.77 |
| PaDiM | 2.78 | 748.84 |
| PatchCore | 24.86 | 2498.00 |
| STFPM | 5.57 | 2767.25 |
| SuperSimpleNet | 33.72 | 1035.28 |
| UNet++ | 53.97 | 1190.22 |
| YOLO11n | 2.59 | 2176.80 |
| YOLOv8n | 3.01 | 1095.68 |
| YOLOv8s | 11.14 | 2056.80 |

</div>

<br>

<p align="center"><strong>Table 11: Memory overhead testing results on Full Machine</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | Parameters (M) | Peak memory (MB) | Allocator peak (MB) | Retained / extra (MB) | Memory backend |
| :---: | :---: | :---: | :---: | :---: | :---: |
| EfficientAD | 8.0586 | 1869.2 | 1869.2 | 71.72 | process_rss |
| Faster R-CNN | 41.3523 | 588.9 | 588.9 | 314.74 | device_allocator |
| GANomaly | 188.6895 | 3044.3 | 3044.3 | 2159.62 | process_rss |
| PaDiM | 2.7828 | 2247.4 | 2247.4 | 168.49 | process_rss |
| PatchCore | 24.8625 | 4505.1 | 4505.0 | 455.11 | process_rss |
| STFPM | 5.5656 | 1879.1 | 1879.0 | 31.97 | process_rss |
| SuperSimpleNet | 33.7194 | 2365.4 | 2365.4 | 196.50 | process_rss |
| UNet++ | 26.1934 | 535.0 | 535.0 | 200.05 | device_allocator |
| YOLO11n | 2.5900 | 65.5 | 65.5 | 5.23 | device_allocator |
| YOLOv8n | 3.0110 | 46.5 | 46.5 | 5.97 | device_allocator |
| YOLOv8s | 11.1360 | 121.5 | 121.5 | 21.48 | device_allocator |

</div>

<br>

The Small Machine run reported only the parameter count and the whole-process peak, so it has no allocator, retention or backend columns.

## 3. Analysis

**Core Accuracy:** Detection accuracy metrics (e.g., Image AUROC, parameter counts) are fully reproduced across both platforms.

**Efficiency Overhead:** Memory and compute overheads (FPS, latency, peak memory) naturally vary with hardware performance and profiling configurations, reflecting platform-specific behavior.

## 4. Observations and Conclusion

All 19 configurations produced a result, including the zero-shot (WinCLIP) and foundation-model (MoECLIP, Dinomaly) entries and the three YOLO variants. The measurements support the following reading.

1. **Accuracy tiers are clean.**
   - *Image level:* PatchCore (0.9932), MoECLIP (0.9877), Dinomaly (0.9797), WinCLIP (0.9772). GANomaly, Reverse Distillation and SuperSimpleNet are not usable at this level.
   - *Pixel level:* MoECLIP (0.9857 AUROC, 0.9701 PRO) leads, with PatchCore and Dinomaly close behind. For mask-based localization, Mask R-CNN beats UNet++ and DeepLabV3+.
   - *Instance level:* Faster R-CNN and Cascade R-CNN are the best localizers; YOLO is the precision-oriented option; DETR is unusable on this dataset.
2. **Efficiency spans three orders of magnitude.** YOLOv8n and YOLO11n are real-time; Faster R-CNN, Mask R-CNN, PaDiM, DeepLabV3+, UNet++ and STFPM sit at 59–80 FPS; PatchCore, EfficientAD, MoECLIP and WinCLIP sit at 1–5 FPS with 1.8–4.5 GB of memory. MoECLIP is the heaviest model by a wide margin.
3. **A practical split follows from the tables.** For real-time screening, YOLOv8n/YOLO11n or PaDiM; for maximum image-level sensitivity, PatchCore or MoECLIP; for pixel-accurate localization, MoECLIP or Mask R-CNN; for bounding-box localization with a low false-alarm budget, Faster R-CNN or Cascade R-CNN.

**Limitations.** MambaAD has not yet been tested; MoECLIP was scored on MVTec AD rather than ZJU-Leaper; cross-domain transfer ([README §3.3.4](../../README.md#334-technical-metrics-cross-domain)) is covered by the Small Machine results only, and communication overhead is not covered at all; the compute tables omit `LMEI`, `Max streams @budget`, `1-stream latency`, `resolution slope` and power/energy, which need the resolution sweep and a power-readable host.
