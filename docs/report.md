# AnomalyDetection Academic Benchmark Report: Multi-Dimensional & Cross-Domain Evaluation

## Measurement contract

- `fps` is aggregate throughput: total frames divided by total measured inference time.
  `instantaneous_fps_*` describes the distribution of reciprocal per-iteration latency;
  it is deliberately reported separately because its arithmetic mean is not aggregate FPS.
- Formal profiling requires an explicit power policy. Set `FDH_POWER_MODE=required` to
  fail when a valid sensor is unavailable, or `FDH_POWER_MODE=disabled` to record that
  power was intentionally not measured. Performance and power use separate passes, so
  sensor sampling does not contaminate latency/FPS measurements.
- Every runtime sweep row contains an `instrumentation` object describing timing,
  memory, power, and FLOPs scope. PyTorch allocator memory, process RSS, and TensorRT
  I/O-buffer lower bounds are not cross-engine comparable. Raw memory is excluded from
  composite rankings, and LMEI is only produced for device-allocator memory.
- FLOPs are a THOP hook-based static estimate using two FLOPs per MAC. They are retained
  for inspection but are not treated as directly comparable across unsupported/custom
  operators or different eager backend implementations.

This report presents a comprehensive academic benchmark evaluation of anomaly detection, object detection, and semantic segmentation models on the **ZJU-Leaper** industrial fabric defect dataset. All experiments were conducted under identical protocol configurations across two primary domain settings:

1. **In-Domain Evaluation (Pattern 1–4):** Models were trained on four fabric background patterns and evaluated on the same textures.
2. **Cross-Domain / Full-Set Evaluation (Pattern 1–19):** Models trained on Pattern 1–4 were evaluated on all 19 patterns, simulating production-line texture changes and out-of-distribution conditions.

---

## 1. Master Evaluation Summary Table

| Model Category | Model Name | Backbone / Recipe | Domain Scope | Image AUROC (%) | Image F1 (%) | Image Prec (%) | Image Rec (%) | mAP@50 (%) | Bbox Prec (%) | Bbox Rec (%) | Bbox F1 (%) | mIoU (%) | Dice (%) | Pixel F1 (%) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Anomaly Det.** | **PatchCore** | Wide-ResNet50 | Pattern 1~4 | **99.32** | **94.74** | **95.19** | 94.29 | — | — | — | — | — | — | — |
| **Anomaly Det.** | **PatchCore** | Wide-ResNet50 | Pattern 1~19 | 58.22 | 51.56 | 35.48 | 94.29 | — | — | — | — | — | — | — |
| **Anomaly Det.** | **Dinomaly** | DINOv2-Reg-ViT-B | Pattern 1~4 | **97.97** | **93.20** | **95.05** | 91.43 | — | — | — | — | — | — | — |
| **Anomaly Det.** | **Dinomaly** | DINOv2-Reg-ViT-B | Pattern 1~19 | 62.17 | 50.98 | 34.32 | **99.05** | — | — | — | — | — | — | — |
| **Anomaly Det.** | **PaDiM** | ResNet18 | Pattern 1~4 | 82.82 | 71.38 | 58.54 | 91.43 | — | — | — | — | — | — | — |
| **Anomaly Det.** | **PaDiM** | ResNet18 | Pattern 1~19 | 59.21 | 49.64 | 33.23 | 98.10 | — | — | — | — | — | — | — |
| **Anomaly Det.** | **RD4AD** | ResNet18 | Pattern 1~4 | 62.04 | 54.55 | 39.41 | 88.57 | — | — | — | — | — | — | — |
| **Anomaly Det.** | **RD4AD** | ResNet18 | Pattern 1~19 | 52.04 | 48.58 | 32.29 | 98.10 | — | — | — | — | — | — | — |
| **Anomaly Det.** | **STFPM** | ResNet18 | Pattern 1~19 | 60.63 | 50.81 | 35.47 | 89.52 | — | — | — | — | — | — | — |
| **Anomaly Det.** | **GANomaly** | Conv-AE | Pattern 1~19 | 51.27 | 47.57 | 31.92 | 93.33 | — | — | — | — | — | — | — |
| **Anomaly Det.** | **SuperSimpleNet** | ResNet18 | Pattern 1~19 | 50.20 | 48.18 | 32.35 | 94.28 | — | — | — | — | — | — | — |
| **Object Det.** | **Faster R-CNN** | ResNet50-FPN | Pattern 1~4 | — | — | — | — | **62.47** | 54.78 | **69.23** | **61.17** | — | — | — |
| **Object Det.** | **Faster R-CNN** | ResNet50-FPN | Pattern 1~19 | — | — | — | — | 2.69 | 2.20 | 15.86 | 3.86 | — | — | — |
| **Object Det.** | **YOLOv8n** | YOLOv8-Nano | Pattern 1~4 | — | — | — | — | 45.95 | 85.71 | 36.26 | 50.97 | — | — | — |
| **Object Det.** | **YOLOv8n** | YOLOv8-Nano | Pattern 1~19 | — | — | — | — | 2.85 | 9.88 | 5.52 | 7.08 | — | — | — |
| **Object Det.** | **YOLO11n** | YOLO11-Nano | Pattern 1~4 | — | — | — | — | 45.55 | **90.91** | 32.97 | 48.39 | — | — | — |
| **Object Det.** | **YOLO11n** | YOLO11-Nano | Pattern 1~19 | — | — | — | — | 4.18 | 11.86 | 4.83 | 6.86 | — | — | — |
| **Object Det.** | **YOLOv8s** | YOLOv8-Small | Pattern 1~19 | — | — | — | — | 2.58 | 24.14 | 4.83 | 8.05 | — | — | — |
| **Object Det.** | **Cascade R-CNN** | ResNet50-FPN | Pattern 1~19 | — | — | — | — | 4.20 | 2.43 | 14.48 | 4.16 | — | — | — |
| **Object Det.** | **DETR** | ResNet50 | Pattern 1~19 | — | — | — | — | 0.00 | 0.00 | 0.00 | 0.00 | — | — | — |
| **Segmentation** | **Mask R-CNN** | ResNet50-FPN | Pattern 1~19 | — | — | — | — | — | — | — | — | **16.82** | **23.93** | **23.93** |
| **Segmentation** | **UNet++** | ResNet34 | Pattern 1~19 | — | — | — | — | — | — | — | — | 12.89 | 18.85 | 18.85 |
| **Segmentation** | **DeepLabV3+** | ResNet50 | Pattern 1~19 | — | — | — | — | — | — | — | — | 12.54 | 19.19 | 19.19 |

---

## 2. Granularity-Specific Academic Evaluations

### 2.1 🖼️ Image-Level Unsupervised Anomaly Detection

Unsupervised anomaly detection models are trained using **only normal (defect-free) fabric images**. They output an anomaly map and image-level score.

| Model | Domain Scope | Image AUROC (%) | Image F1 (%) | Precision (%) | Recall (%) | Decision Threshold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PatchCore** | Pattern 1~4 (In-Domain) | **99.32** | **94.74** | 95.19 | 94.29 | 0.5213 |
| **Dinomaly** | Pattern 1~4 (In-Domain) | 97.97 | 93.20 | **95.05** | 91.43 | 0.1076 |
| **PaDiM** | Pattern 1~4 (In-Domain) | 82.82 | 71.38 | 58.54 | 91.43 | 0.4975 |
| **RD4AD** | Pattern 1~4 (In-Domain) | 62.04 | 54.55 | 39.41 | 88.57 | 0.4610 |
| **Dinomaly** | Pattern 1~19 (Full Set) | **62.17** | 50.98 | 34.32 | **99.05** | 0.1263 |
| **STFPM** | Pattern 1~19 (Full Set) | 60.63 | 50.81 | 35.47 | 89.52 | 0.5704 |
| **PaDiM** | Pattern 1~19 (Full Set) | 59.21 | 49.64 | 33.23 | 98.10 | 0.4146 |
| **PatchCore** | Pattern 1~19 (Full Set) | 58.22 | **51.56** | **35.48** | 94.29 | 0.6534 |
| **RD4AD** | Pattern 1~19 (Full Set) | 52.04 | 48.58 | 32.29 | 98.10 | 0.3574 |
| **GANomaly** | Pattern 1~19 (Full Set) | 51.27 | 47.57 | 31.92 | 93.33 | 0.5058 |
| **SuperSimpleNet** | Pattern 1~19 (Full Set) | 50.20 | 48.18 | 32.35 | 94.28 | 0.2241 |

#### Academic Insights (Anomaly Detection):
- **In-Domain Excellence**: On known textures (Pattern 1–4), memory-bank and foundation-feature architectures achieve remarkable performance (**PatchCore: 99.32% AUROC**, **Dinomaly: 97.97% AUROC**).
- **Out-of-Distribution Sensitivity**: When subjected to unseen texture patterns (Pattern 5–19), nearest-neighbor memory-bank methods suffer severe false alarm spikes because unseen normal textures are misclassified as anomalies (**AUROC drops to ~58–62%**).

---

### 2.2 📍 Instance-Level Bounding Box Object Detection

Supervised object detection models are trained using **annotated defect bounding boxes**. Performance is evaluated via COCO mAP and classification metrics at a confidence threshold of $0.25$.

| Model | Domain Scope | mAP@50 (%) | mAP@50-95 (%) | Precision (%) | Recall (%) | F1-Score (%) | TP | FP | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faster R-CNN** | Pattern 1~4 (In-Domain) | **62.47** | **31.85** | 54.78 | **69.23** | **61.17** | 126 | 104 | 56 |
| **YOLOv8n** | Pattern 1~4 (In-Domain) | 45.95 | 22.27 | 85.71 | 36.26 | 50.97 | 66 | 11 | 116 |
| **YOLO11n** | Pattern 1~4 (In-Domain) | 45.55 | 23.16 | **90.91** | 32.97 | 48.39 | 60 | 6 | 122 |
| **Cascade R-CNN** | Pattern 1~19 (Full Set) | **4.20** | 2.64 | 2.43 | 14.48 | 4.16 | 21 | 843 | 124 |
| **YOLO11n** | Pattern 1~19 (Full Set) | 4.18 | 1.95 | 11.86 | 4.83 | 6.86 | 7 | 52 | 138 |
| **YOLOv8n** | Pattern 1~19 (Full Set) | 2.85 | 1.09 | 9.88 | 5.52 | 7.08 | 8 | 73 | 137 |
| **Faster R-CNN** | Pattern 1~19 (Full Set) | 2.69 | 1.86 | 2.20 | **15.86** | 3.86 | 23 | 1024 | 122 |
| **YOLOv8s** | Pattern 1~19 (Full Set) | 2.58 | 1.10 | **24.14** | 4.83 | 8.05 | 7 | 22 | 138 |
| **DETR** | Pattern 1~19 (Full Set) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 145 |

#### Academic Insights (Object Detection):
- **Precision vs. Recall Tradeoff**:
  - **YOLO Series (YOLO11n / YOLOv8n)**: High precision (**85.7%–90.9%** on in-domain test data) with very few false positives ($FP \le 11$), but lower recall.
  - **Two-Stage Detectors (Faster R-CNN)**: Higher recall (**69.23%** in-domain) capturing subtle defect instances, achieving the top **62.47% mAP@50**.
- **Domain Shift Impact**: When testing across all 19 patterns, supervised object detectors fail to locate defects on unseen backgrounds ($mAP@50 < 4.2\%$).

---

### 2.3 🔍 Pixel-Level Defect Segmentation & Mask Evaluation

Defect segmentation models evaluate exact pixel boundaries for fabric flaws.

| Model | Model Task | Domain Scope | mIoU (%) | Dice Score (%) | Pixel-F1 (%) | Evaluated Samples |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Mask R-CNN** | Instance Segmentation | Pattern 1~19 (Full Set) | **16.82** | **23.93** | **23.93** | 80 |
| **UNet++** | Semantic Segmentation | Pattern 1~19 (Full Set) | 12.89 | 18.85 | 18.85 | 105 |
| **DeepLabV3+** | Semantic Segmentation | Pattern 1~19 (Full Set) | 12.54 | 19.19 | 19.19 | 105 |

---

## 3. Production Line Texture Change Analysis

The dataset split comparison illustrates the industrial challenge of fabric line changes:

```
[In-Domain Test (Pattern 1-4)]                   [Cross-Domain Full Test (Pattern 1-19)]
PatchCore AUROC: 99.32%  ---------------------->  PatchCore AUROC: 58.22%  (Drop: -41.10%)
Dinomaly AUROC:  97.97%  ---------------------->  Dinomaly AUROC:  62.17%  (Drop: -35.80%)
Faster R-CNN mAP: 62.47% ---------------------->  Faster R-CNN mAP: 2.69%  (Drop: -59.78%)
YOLO11n Prec:    90.91%  ---------------------->  YOLO11n Prec:    11.86%  (Drop: -79.05%)
```

### Key Key Takeaways:
1. **Memory-Bank Anomaly Detectors**: PatchCore and PaDiM store feature embeddings of normal textures. When deployed to unseen textures (Pattern 5+), normal features deviate from the stored memory bank, causing extensive false positives.
2. **Supervised Object Detectors**: YOLO and Faster R-CNN learn strong feature representations tied to specific background textures. OOD background shifts disrupt feature activation, resulting in high false negative rates ($FN > 120$).
3. **Recommendation**: For production environments with frequent fabric texture switching, model training must incorporate **multi-texture few-shot fine-tuning** or **self-supervised texture-disentangled feature extractors**.
