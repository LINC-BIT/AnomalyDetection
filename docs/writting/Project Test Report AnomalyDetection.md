# Project Test Report: AnomalyDetection

This report reproduces the examples and experiments of the AnomalyDetection project by following the step-by-step instructions in the [README](../../README.md). Every outcome was measured on two platforms — an RTX 4090 workstation and an A100 server — to verify reproducibility.

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
  - [2.3 Analysis](#23-analysis)
- [3. Extensibility](#3-extensibility)
  - [3.1 Adding a New Dataset](#31-adding-a-new-dataset)
  - [3.2 Adding the YOLO 26 Model](#32-adding-the-yolo-26-model)
- [4. Discussion](#4-discussion)

## 1. Hardware and Software Specifications

Configurations and results are compared across two platforms:

- Small Machine: RTX 4090 workstation
- Full Machine: A100 server

Table 1 lists their hardware and software.

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

The environment was set up per [Environment Setup](../../README.md#2-environment-setup), then the test cases under [Examples](../../README.md#5-examples) were reproduced. Only the Full Machine results are shown.

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

Following the instructions in [README §4.1.3 Benchmarking](../../README.md#413-benchmarking), the complete benchmark suite was run on both machines. The resulting data are reported below.

Both platforms used the same configuration:

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

19 configurations: 10 anomaly (AD), 6 detection (DD), 3 segmentation (SEG). Each appears in every table its output supports; the Full Machine tables carry all of them — image 16, pixel 12, instance 6, compute and memory 19. Full data: [Small Machine](./results_small_machine.md), [Full Machine](./results_full_gpu_server.md).

#### 2.2.1 Technical Metrics: Image Level

This part evaluates the image-level metrics defined by [README §3.3.1](../../README.md#331-technical-metrics-image-level):

- **Image AUROC:** the probability that a defective image is ranked above a normal one. 0.5 is random guessing, 1.0 is perfect, and no decision threshold is needed.
- **Image Precision:** the share of images flagged as defective that really are defective.
- **Image Recall:** the share of truly defective images that are flagged.
- **Image F1:** the balance of precision and recall in one number; it drops when either is poor.

**Key observation:** Image AUROC agrees within **<span style="color:#0070C0">0.2 points</span>** across the two machines for all ten anomaly models; the thresholded metrics drift by up to 4.4 points. The Full Machine's detection models rank highest on this axis — YOLO11n / YOLOv8n 0.9965.

**Read Table 3 by paradigm, not down the AUROC column.** It lists both families together, but an anomaly model reports a continuous normality score for every image while a detector reports the highest-confidence box it found, which is exactly `0` when it found nothing. That difference makes the two AUROCs different measurements. On the Full Machine, YOLOv8n scores 157 of 245 normal images at exactly 0 and no defective image at 0, which is why its AUROC is 0.997 while its recall at its own threshold is only 0.571 (YOLOv8s: AUROC 0.993, recall 0.362). The detector ranks images well and still misses most defects at the operating point it would actually use. Figure 1 in the figures report therefore draws the two families as two panels rather than one axis, and within a panel the bars compare like with like.

From this revision the evaluators also record **Image AP** (`image_ap`, the area under the image-level precision-recall curve) beside Image AUROC. It is not in Tables 2 and 3 because those runs predate the change; the next benchmark run fills it in, and it is the number to compare against the PR curves in Figures 2 and 3.

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
| Cascade R-CNN | 0.9598 | 0.9510 | 0.9798 | 0.9238 |
| DETR | 0.6489 | 0.0000 | 0.0000 | 0.0000 |
| Dinomaly | 0.9797 | 0.9151 | 0.9065 | 0.9238 |
| EfficientAD | 0.9501 | 0.8557 | 0.9326 | 0.7905 |
| Faster R-CNN | 0.9919 | 0.9561 | 0.9800 | 0.9333 |
| GANomaly | 0.6548 | 0.4802 | 0.3244 | 0.9238 |
| MoECLIP | 0.9877 | 0.9209 | 0.9000 | 0.9429 |
| PaDiM | 0.8284 | 0.7138 | 0.5854 | 0.9143 |
| PatchCore | 0.9932 | 0.9469 | 0.9608 | 0.9333 |
| Reverse Distillation | 0.6222 | 0.5378 | 0.3810 | 0.9143 |
| STFPM | 0.9497 | 0.8458 | 0.8854 | 0.8095 |
| SuperSimpleNet | 0.5852 | 0.4598 | 0.3003 | 0.9810 |
| WinCLIP | 0.9772 | 0.8584 | 0.8017 | 0.9238 |
| YOLO11n | 0.9965 | 0.7500 | 1.0000 | 0.6000 |
| YOLOv8n | 0.9965 | 0.7273 | 1.0000 | 0.5714 |
| YOLOv8s | 0.9934 | 0.5315 | 1.0000 | 0.3619 |

</div>

<br>

Table 3 lists **all 16 Full Machine models that emit an image-level score**: 10 anomaly + 6 detection.

#### 2.2.2 Technical Metrics: Pixel Level

This part evaluates the pixel-level metrics defined by [README §3.3.2](../../README.md#332-technical-metrics-pixel-level):

- **Pixel AUROC:** the probability that a defective pixel is ranked above a normal one, pooled over the pixels of all images. 0.5 is random guessing.
- **Pixel AUPRO:** the share of each defective region that is covered before the false-positive rate reaches a fixed limit. Every region counts equally, so a small defect is not hidden by a large one.
- **Pixel F1:** the match between predicted and true defect pixels at a fixed pixel threshold, penalised by missed pixels and by extra pixels.
- **Pixel IoU:** the intersection over union between the predicted defect area and the true defect area.
- **IAP (Instance Average Precision):** the average precision over connected defect regions, with every region weighted equally.

**Key observation:** Both machines rank MoECLIP first and SuperSimpleNet last, but their absolute pixel-level values do not overlap; see [§2.3](#23-analysis).

<p align="center"><strong>Table 4: Pixel-level testing results on Small Machine</strong></p>

<div align="center">

| Model | Pixel AUROC | Pixel AUPRO | Pixel F1 | IAP |
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

| Model | Pixel AUROC | Pixel AUPRO | Pixel F1 | Pixel IoU |
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

All 9 Full Machine models that emit a per-pixel anomaly map.

<p align="center"><strong>Table 6: Pixel-level segmentation mask metrics (both machines)</strong></p>

<div align="center">

| Model | Dice (Small) | mIoU (Small) | Dice (Full) | mIoU (Full) |
| :---: | :---: | :---: | :---: | :---: |
| Mask R-CNN | 0.7539 | 0.6285 | 0.7242 | 0.5993 |
| UNet++ | 0.6833 | 0.5614 | 0.6833 | 0.5613 |
| DeepLabV3+ | 0.6479 | 0.5133 | 0.6477 | 0.5132 |

</div>

<br>

The 3 segmentation models — the other pixel-level producers.

#### 2.2.3 Technical Metrics: Instance Level

This part evaluates the instance-level metrics defined by [README §3.3.3](../../README.md#333-technical-metrics-instance-level):

- **AP:** box precision–recall quality averaged over IoU thresholds 0.50–0.95 (COCO style); higher is better.
- **AP50 / AP75:** AP at a single IoU threshold. 0.50 accepts a loosely placed box, 0.75 demands a tight one, where IoU is the overlap between a predicted box and the true box.
- **Precision:** the share of reported boxes that overlap a real defect (IoU ≥ 0.50).
- **Recall:** the share of real defects covered by a reported box.
- **F1:** the balance of box precision and recall in one number.
- **TP / FP / FN:** the underlying counts, namely defects found, spurious boxes and defects missed.

**Key observation:** The thresholded **<span style="color:#0070C0">F1 reproduces exactly</span>** across the two machines, while AP50 is higher on the Full Machine.

<p align="center"><strong>Table 7: Instance-level testing results on Small Machine</strong></p>

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

<p align="center"><strong>Table 8: Instance-level testing results on Full Machine</strong></p>

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

- Precision, Recall, F1 and TP/FP/FN are taken at confidence threshold 0.25. The Small Machine did not measure AP75, so Table 7 omits that column.
- A standalone `adh evaluate` run on the DETR checkpoint independently confirmed `TP=0` and `FP=0`.

#### 2.2.4 Overhead Metrics: Compute

This part evaluates the compute metrics defined by [README §3.3.5](../../README.md#335-overhead-metrics-compute):

- **FPS:** images processed per second.
- **Latency mean / p95 / p99:** the time to process one image, in milliseconds; p95 and p99 describe the slowest frames rather than the average.
- **FLOPs (G):** the arithmetic work of one forward pass, in billions of operations. Fixed by the architecture, so identical on any machine.
- **Wall-time:** the total seconds the scored run took, accuracy pass included.

**Key observation:** **<span style="color:#0070C0">FLOPs reproduce</span>** for every model measured twice, while FPS and latency are host-specific and differ by 1.1×–31×.

<p align="center"><strong>Table 9: Compute overhead testing results on Small Machine</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | FPS | Latency mean | FLOPs (G) |
| :---: | :---: | :---: | :---: |
| EfficientAD | 3.37 | 296.57 ms | 75.91 |
| Faster R-CNN | 7.81 | 128.08 ms | 268.98 |
| GANomaly | 10.63 | 94.08 ms | 16.54 |
| PaDiM | 31.69 | 31.56 ms | 3.69 |
| PatchCore | 1.39 | 719.00 ms | 24.14 |
| STFPM | 22.00 | 45.44 ms | 7.38 |
| SuperSimpleNet | 2.40 | 417.01 ms | 96.70 |
| UNet++ | 119.22 | 100.19 ms | 250.23 |
| YOLO11n | 20.83 | 48.01 ms | 6.44 |
| YOLOv8n | 181.69 | 5.50 ms | 8.19 |
| YOLOv8s | 11.51 | 86.89 ms | 28.65 |

</div>

<br>

<p align="center"><strong>Table 10: Compute overhead testing results on Full Machine</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | FPS | Latency mean | Latency p50 | Latency p95 | Latency p99 | FLOPs (G) | Wall-time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cascade R-CNN | 0.82 | 1222.08 ms | 1223.76 ms | 1444.89 ms | 1444.89 ms | 408.0209 | 44.4 s |
| DETR | 56.58 | 17.67 ms | 17.27 ms | 19.12 ms | 19.18 ms | 186.0145 | 46.0 s |
| DeepLabV3+ | 80.34 | 12.45 ms | 10.80 ms | 11.66 ms | 40.82 ms | 572.1205 | 66.4 s |
| Dinomaly | 29.74 | 33.63 ms | 30.01 ms | 63.34 ms | 63.34 ms | 273.0566 | 37.4 s |
| EfficientAD | 3.72 | 268.96 ms | 277.75 ms | 324.86 ms | 354.60 ms | 75.9065 | 60.7 s |
| Faster R-CNN | 60.42 | 16.55 ms | 16.06 ms | 18.27 ms | 19.72 ms | 268.9822 | 21.6 s |
| GANomaly | 34.85 | 28.69 ms | 28.32 ms | 30.37 ms | 33.46 ms | 33.0444 | 51.4 s |
| Mask R-CNN | 59.08 | 16.93 ms | 17.61 ms | 18.46 ms | 18.75 ms | 277.0023 | 512.4 s |
| MoECLIP | 5.02 | 199.15 ms | 211.75 ms | 221.62 ms | 221.62 ms | 2241.5497 | 131.4 s |
| PaDiM | 77.32 | 12.93 ms | 12.96 ms | 13.51 ms | 13.82 ms | 3.6884 | 43.3 s |
| PatchCore | 4.42 | 226.31 ms | 227.70 ms | 309.07 ms | 312.48 ms | 24.1408 | 88.3 s |
| Reverse Distillation | 7.90 | 126.64 ms | 145.05 ms | 215.47 ms | 266.32 ms | 67.8974 | 63.3 s |
| STFPM | 74.77 | 13.37 ms | 13.56 ms | 21.89 ms | 24.02 ms | 7.3767 | 88.5 s |
| SuperSimpleNet | 17.38 | 57.53 ms | 56.90 ms | 61.21 ms | 71.03 ms | 96.7028 | 93.0 s |
| UNet++ | 79.54 | 12.57 ms | 10.22 ms | 19.50 ms | 30.95 ms | 250.2320 | 68.4 s |
| WinCLIP | 1.00 | 1004.65 ms | 1015.57 ms | 1080.09 ms | 1080.09 ms | 446.7010 | 590.6 s |
| YOLO11n | 308.48 | 3.24 ms | 3.36 ms | 3.45 ms | 3.47 ms | 6.4406 | 11.9 s |
| YOLOv8n | 432.21 | 2.31 ms | 2.49 ms | 2.64 ms | 2.68 ms | 14.9008 | 11.4 s |
| YOLOv8s | 355.33 | 2.81 ms | 2.79 ms | 2.95 ms | 3.24 ms | 6.9886 | 11.9 s |

</div>

<br>

All 19 Full Machine configurations. Table 9 (Small) lists the 11 models that machine profiled; Table 11 adds the memory columns for the same runs.

#### 2.2.5 Overhead Metrics: Memory

This part evaluates the memory metrics defined by [README §3.3.6](../../README.md#336-overhead-metrics-memory):

- **Parameters (M):** the number of learned weights, in millions. Fixed by the architecture, so identical on any machine.
- **Peak memory:** the most memory used at once during the run.
- **Measured as:** which of two instruments produced that peak. **A model's row is only comparable with another row that used the same instrument**, and Table 12 therefore names the instrument on every row.
- **Average memory:** the mean of the memory samples taken during the measured runs; for a model measured on the card it equals the peak, because the allocator reading is already a high-water mark.
- **Artifact size:** the checkpoint file on disk — weights, plus optimizer state for a fine-tuned model. This is storage, not run-time memory, and it is why a checkpoint can be larger than the memory the model uses.

**Key observation:** **<span style="color:#0070C0">Parameters reproduce</span>** for every model measured twice, while peak memory is not comparable across the two machines — and, on the Full Machine, not even across models within one table. **Two different instruments produced the peak-memory column, and which one a model got depends on how that model could be profiled, not on how good or how large it is.**

- 8 of the 19 models were exported to a TorchScript or ONNX graph and profiled through it. The profiler then reads **tensor memory live on the graphics card**, the closest available figure to true VRAM.
- The other 11 could not be exported — the anomaly backends, plus Cascade R-CNN, whose custom layers no exporter accepts — so they were profiled natively by running the model directly. The only instrument available on that path is the **resident memory of the whole Python process**: the weights, the graphics-card context, host copies of tensors, the framework and its libraries.

The two quantities are not two views of one number. A model can report 1.8 GB of whole-program memory and 0.05 GB of tensor memory because its weights live in host memory. Reading down the "Peak memory" column as if it were one ranking therefore compares an instrument with a model: Cascade R-CNN's 2281 MB is whole-program memory, DETR's 690 MB is card-tensor memory, and neither number says the other model is smaller. Figure 11 in the figures report draws the two instruments on separate axes for the same reason, and the instrument is recorded per row as `memory_measurement_kind` in `snapshot_audit.csv`.

<p align="center"><strong>Table 11: Memory overhead testing results on Small Machine</strong></p>

<div align="center">

| Model | Parameters (M) | Peak memory (MB) |
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

All 11 models were profiled natively on that machine, so their peak memory is whole-program memory throughout. These rows are comparable with each other, and with the "whole program" rows of Table 12 — not with its "GPU tensors" rows.

<br>

<p align="center"><strong>Table 12: Memory overhead testing results on Full Machine</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | Parameters (M) | Peak memory (MB) | Measured as | Average memory (MB) | Artifact size (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Cascade R-CNN | 69.1641 | 2281.1 | whole program | 2265.3 | 526.94 |
| DETR | 44.3878 | 690.0 | GPU tensors | 690.0 | 329.89 |
| DeepLabV3+ | 40.3470 | 603.9 | GPU tensors | 603.9 | 308.18 |
| Dinomaly | 148.0090 | 1755.4 | whole program | 1755.4 | 564.71 |
| EfficientAD | 8.0586 | 1869.2 | whole program | 1869.2 | 71.72 |
| Faster R-CNN | 41.3523 | 588.9 | GPU tensors | 588.9 | 314.74 |
| GANomaly | 188.6895 | 3044.3 | whole program | 3044.3 | 2159.62 |
| Mask R-CNN | 43.9755 | 608.9 | GPU tensors | 608.9 | 334.76 |
| MoECLIP | 433.5619 | 3538.5 | whole program | 3538.4 | 18.35 |
| PaDiM | 2.7828 | 2247.4 | whole program | 2247.4 | 168.49 |
| PatchCore | 24.8625 | 4505.1 | whole program | 4505.0 | 455.11 |
| Reverse Distillation | 89.0023 | 3004.1 | whole program | 3004.1 | 765.57 |
| STFPM | 5.5656 | 1879.1 | whole program | 1879.0 | 31.97 |
| SuperSimpleNet | 33.7194 | 2365.4 | whole program | 2365.4 | 196.50 |
| UNet++ | 26.1934 | 535.0 | GPU tensors | 535.0 | 200.05 |
| WinCLIP | 208.3773 | 4254.8 | whole program | 4248.6 | 0.00 |
| YOLO11n | 2.5900 | 65.5 | GPU tensors | 65.5 | 5.23 |
| YOLOv8n | 3.0110 | 46.5 | GPU tensors | 46.5 | 5.97 |
| YOLOv8s | 11.1360 | 121.5 | GPU tensors | 121.5 | 21.48 |

</div>

<br>

All 19 Full Machine configurations. Table 11 lists the 11 models the Small Machine profiled. Compare the "Peak memory" column only between rows that share a "Measured as" value; the trailing columns are the average of the memory samples and the checkpoint's size on disk, neither of which is a run-time peak.

### 2.3 Analysis

The 19 configurations split into two paradigms: **supervised Defect Detection (DD)** and **unsupervised / zero-shot Anomaly Detection (AD)**. Both are scored on the metric families of [README §3.3](../../README.md#33-supported-metrics) — image level, pixel level, instance level and overhead — and the rankings below use the Full Machine tables (Tables 3, 5 and 8).

Because the metrics have different scales, each model is reduced to one composite figure: its **mean rank** across every technical metric it reports, where 1 is best. The mean score column averages the same raw values.

<p align="center"><strong>Table 13: Anomaly detection — mean rank across the eight image- and pixel-level metrics</strong></p>

<div align="center">

| Model | Mean rank | Mean score |
| :---: | :---: | :---: |
| MoECLIP | 1.75 | 0.8851 |
| PatchCore | 2.38 | 0.8447 |
| Dinomaly | 3.50 | 0.8311 |
| WinCLIP | 5.12 | 0.7270 |
| EfficientAD | 5.50 | 0.7369 |
| STFPM | 5.62 | 0.7795 |
| PaDiM | 6.00 | 0.6790 |
| GANomaly | 7.75 | 0.5958 |
| Reverse Distillation | 7.88 | 0.4959 |
| SuperSimpleNet | 8.38 | 0.4236 |

</div>

<br>

<p align="center"><strong>Table 14: Supervised detection — mean rank across the six instance-level metrics</strong></p>

<div align="center">

| Model | Mean rank | Mean score |
| :---: | :---: | :---: |
| Cascade R-CNN | 1.83 | 0.5417 |
| Faster R-CNN | 2.17 | 0.5252 |
| YOLO11n | 3.33 | 0.4735 |
| YOLOv8n | 3.33 | 0.4668 |
| YOLOv8s | 4.33 | 0.4112 |
| DETR | 6.00 | 0.0006 |

</div>

<br>

- **Anomaly (AD).** Three tiers: MoECLIP 1.75, PatchCore 2.38, Dinomaly 3.50 usable; WinCLIP–PaDiM 5.12–6.00 middle; GANomaly, Reverse Distillation, SuperSimpleNet 7.75–8.38 near chance. GANomaly is ranked on image metrics only.
- **Detection (DD).** Cascade R-CNN 1.83 and Faster R-CNN 2.17 lead; YOLO11n / YOLOv8n tie at 3.33, YOLOv8s 4.33; DETR degenerate at 6.00.
- **Segmentation (SEG).** Mask R-CNN 0.7242 > UNet++ 0.6833 > DeepLabV3+ 0.6477 (`Dice`).
- **Image level.** PatchCore 0.9932, MoECLIP 0.9877, Dinomaly 0.9797 and WinCLIP 0.9772 lead the anomaly models; the Full Machine's YOLO11n / YOLOv8n (0.9965) top the whole run.
- **Pixel level.** MoECLIP leads localization (0.9857 AUROC / 0.9701 AUPRO); Pixel F1 and IoU stay low because the pixel threshold is not calibrated.
- **Overhead.** 0.82–432 FPS across 19 models: YOLO real-time, Cascade R-CNN and WinCLIP ≈ 1 FPS.

**Reproducibility.** Image AUROC within 0.2 points, parameter counts and FLOPs exact. FPS and latency are host-specific (1.1×–31×). Peak memory is not comparable across machines, and on the Full Machine not even across models: two instruments produced it (Table 12), so only rows sharing a `Measured as` value may be compared.

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
- **Overhead metrics are host-defined, and peak memory is also instrument-defined.** FPS, latency and peak memory are measured per machine, so their values differ between the RTX 4090 and the A100 runs; only FLOPs and parameter counts are host-independent. Peak memory additionally depends on whether a model could be profiled through an exported graph (GPU-tensor memory) or had to be profiled natively (whole-program memory), which is why Table 12 names the instrument per row and Figure 11 gives the two separate axes.
- **Coverage gaps.** MambaAD untested; MoECLIP scored on MVTec AD rather than ZJU-Leaper; cross-domain ([README §3.3.4](../../README.md#334-technical-metrics-cross-domain)) covered on the Small Machine only; `LMEI`, `Max streams @budget`, `1-stream latency`, `resolution slope` and power/energy not produced.
