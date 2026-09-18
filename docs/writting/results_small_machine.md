# Complete Benchmark Results — Small Machine

These tables are the complete data behind [Project Test Report AnomalyDetection.md](./Project%20Test%20Report%20AnomalyDetection.md).

Platform: GPU workstation, NVIDIA RTX 4090 24 GB, CUDA 12.8, PyTorch mixed precision (FP16/AMP).
Dataset: [ZJU-Leaper](../../README.md#32-supported-datasets), `test` split.
Metric names follow [README §3.3.1–§3.3.6](../../README.md#33-supported-metrics).

---

<p align="center"><strong>Table 1: Image-level anomaly recognition</strong></p>

<div align="center">

| Model | Image AUROC | Image F1 | Image Precision | Image Recall | Decision Threshold |
| :---: | :---: | :---: | :---: | :---: | :---: |
| PatchCore | 0.9932 | 0.9474 | 0.9519 | 0.9429 | 0.5213 |
| MoECLIP | 0.9877 | 0.9314 | 0.9596 | 0.9048 | 0.4228 |
| Dinomaly | 0.9797 | 0.9320 | 0.9505 | 0.9143 | 0.1076 |
| WinClip | 0.9772 | 0.9026 | 0.9778 | 0.8381 | 0.4341 |
| EfficientAD | 0.9502 | 0.8586 | 0.9535 | 0.7810 | 0.5002 |
| STFPM | 0.9497 | 0.8531 | 0.8491 | 0.8571 | 0.5205 |
| PaDiM | 0.8282 | 0.7138 | 0.5854 | 0.9143 | 0.4975 |
| GANomaly | 0.6548 | 0.5159 | 0.3876 | 0.7714 | 0.5025 |
| RD4AD | 0.6204 | 0.5455 | 0.3941 | 0.8857 | 0.4610 |
| SuperSimpleNet | 0.5847 | 0.5000 | 0.3699 | 0.7714 | 0.5863 |

</div>

<br>

---

<p align="center"><strong>Table 2: Pixel-level anomaly localization</strong></p>

<div align="center">

| Model | Pixel AUROC | Pixel AUPRO | Pixel F1 | Image AP |
| :---: | :---: | :---: | :---: | :---: |
| MoECLIP | 0.9039 | 0.8374 | 0.5447 | 0.3983 |
| PaDiM | 0.7559 | 0.7117 | 0.1282 | 0.0528 |
| WinClip | 0.7483 | 0.7399 | 0.1482 | 0.1240 |
| Dinomaly | 0.7452 | 0.6029 | 0.1470 | 0.0574 |
| STFPM | 0.6807 | 0.6353 | 0.0954 | 0.0546 |
| PatchCore | 0.6730 | 0.5925 | 0.0841 | 0.0373 |
| EfficientAD | 0.6072 | 0.4814 | 0.1035 | 0.0287 |
| SuperSimpleNet | 0.4645 | 0.4108 | 0.0548 | 0.0184 |
| RD4AD | 0.3529 | 0.5415 | 0.4321 | 0.2120 |
| GANomaly | — | — | — | — |

</div>

<br>

`Pixel AUPRO` is the same metric the Full Machine results call `Pixel PRO`.

---

<p align="center"><strong>Table 3: Instance-level supervised detection</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | mAP@50 | mAP | F1 | Precision | Recall | TP | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Faster R-CNN | 0.6247 | 0.3185 | 0.6117 | 0.5478 | 0.6923 | 126 | 104 | 56 |
| Cascade R-CNN | 0.6232 | 0.3267 | 0.6561 | 0.6327 | 0.6813 | 124 | 72 | 58 |
| YOLOv8n | 0.4595 | 0.2227 | 0.5097 | 0.8571 | 0.3626 | 66 | 11 | 116 |
| YOLO11n | 0.4555 | 0.2316 | 0.4839 | 0.9091 | 0.3297 | 60 | 6 | 122 |
| YOLOv8s | 0.4332 | 0.1991 | 0.3482 | 0.9286 | 0.2143 | 39 | 3 | 143 |
| DETR | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 182 |

</div>

<br>

Precision, Recall, F1 and TP/FP/FN are taken at confidence threshold 0.25.

---

<p align="center"><strong>Table 4: Supervised segmentation</strong></p>

<div align="center">

| Model | Dice | mIoU | Pixel F1 | Evaluated Images |
| :---: | :---: | :---: | :---: | :---: |
| Mask R-CNN | 0.7539 | 0.6285 | 0.7539 | 98 |
| UNet++ | 0.6833 | 0.5614 | 0.6833 | 105 |
| DeepLabV3+ | 0.6479 | 0.5133 | 0.6479 | 105 |

</div>

<br>

`Dice` is the metric the Full Machine results call `Mask AP50` for UNet++ and DeepLabV3+.

---

<p align="center"><strong>Table 5: Compute and memory overhead</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | FPS | Latency mean | Peak Memory (MB) | Parameters (M) | FLOPs (G) | Max Concurrent Streams |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv8n | 181.69 | 5.50 ms | 1095.68 | 3.01 | 8.19 | 0 |
| PaDiM | 31.69 | 31.56 ms | 748.84 | 2.78 | 3.69 | 2 |
| STFPM | 22.00 | 45.44 ms | 2767.25 | 5.57 | 7.38 | 1 |
| YOLO11n | 20.83 | 48.01 ms | 2176.80 | — | — | 0 |
| GANomaly | 10.63 | 94.08 ms | 2664.77 | 188.69 | 16.54 | 0 |
| YOLOv8s | 11.51 | 86.89 ms | 2056.80 | — | — | 0 |
| UNet++ | 119.22 | 100.19 ms | 1190.22 | 53.97 | 250.23 | 0 |
| Faster R-CNN | 7.81 | 128.08 ms | 610.70 | 41.35 | 268.98 | 0 |
| SuperSimpleNet | 2.40 | 417.01 ms | 1035.28 | 33.72 | 96.70 | 0 |
| EfficientAD | 3.37 | 296.57 ms | 1001.19 | 8.06 | 75.91 | 0 |
| PatchCore | 1.39 | 719.00 ms | 2498.00 | 24.86 | 24.14 | 0 |

</div>

<br>

`Peak Memory` is whole-process resident set, which is why it is on a different scale from the Full Machine's allocator-based figures.

---

<p align="center"><strong>Table 6: Cross-domain generalization</strong></p>

Patterns 1–4 (regular striped) → Patterns 5–19 (complex patterned/floral), same weights.

<div align="center">

| Category | Model | In-domain | Cross-domain | Relative Drop | Cross-domain Precision |
| :---: | :---: | :---: | :---: | :---: | :---: |
| AD | Dinomaly | 0.9797 (AUROC) | 0.6217 (AUROC) | −35.80% | 0.3432 |
| AD | PatchCore | 0.9932 (AUROC) | 0.5822 (AUROC) | −41.10% | 0.3548 |
| AD | PaDiM | 0.8282 (AUROC) | 0.5921 (AUROC) | −23.61% | 0.3323 |
| AD | RD4AD | 0.6204 (AUROC) | 0.5204 (AUROC) | −10.00% | 0.3229 |
| DD | Faster R-CNN | 0.6247 (mAP@50) | 0.0269 (mAP@50) | −59.78% | 0.0220 |
| DD | YOLO11n | 0.4555 (mAP@50) | 0.0418 (mAP@50) | −41.37% | 0.1186 |
| DD | YOLOv8n | 0.4595 (mAP@50) | 0.0285 (mAP@50) | −43.10% | 0.0988 |

</div>

<br>
