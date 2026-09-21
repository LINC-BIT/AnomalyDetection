# Complete Test Running Results on Full Machine

These tables are the complete data behind [Project Test Report AnomalyDetection.md](./Project%20Test%20Report%20AnomalyDetection.md).

Platform: 
- 2 × Intel Xeon Gold 6430 (128 threads), 256 GB
- NVIDIA A100 80 GB, CUDA 12.4, fp32.
- Dataset: [ZJU-Leaper](../../README.md#32-supported-datasets), `test` split.

---

<p align="center"><strong>Table 1: Image-level anomaly recognition</strong></p>

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

`F1 / Precision / Recall` are thresholded; the threshold must be the one fitted by **Calibrate decision thresholds on the training split** (README §3.3.1).

---

<p align="center"><strong>Table 2: Pixel-level anomaly localization</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

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

All 9 anomaly models that emit a per-pixel map.

---

<p align="center"><strong>Table 2b: Segmentation mask metrics</strong></p>

<div align="center">

| Model | Dice | mIoU |
| :---: | :---: | :---: |
| Mask R-CNN | 0.7242 | 0.5993 |
| UNet++ | 0.6833 | 0.5613 |
| DeepLabV3+ | 0.6477 | 0.5132 |

</div>

<br>

The 3 segmentation models, the other pixel-level producers of the run.

---

<p align="center"><strong>Table 3: Instance-level defect detection</strong></p>

<div align="center" style="width: 100%; overflow-x: auto;">

| Model | AP50 | AP | AP75 | APs | APm | APl | AR@1 | AR@10 | AR@100 | ARs | ARm | ARl | Precision | Recall | F1 | TP | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cascade R-CNN | 0.6428 | 0.3333 | 0.3042 | 0.0103 | 0.2175 | 0.5009 | 0.2808 | 0.4478 | 0.4703 | 0.1286 | 0.3618 | 0.6269 | 0.6327 | 0.6813 | 0.6561 | 124 | 72 | 58 |
| DETR | 0.0022 | 0.0011 | 0.0001 | 0.0000 | 0.0000 | 0.0021 | 0.0148 | 0.0159 | 0.0165 | 0.0000 | 0.0000 | 0.0323 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 182 |
| Faster R-CNN | 0.6623 | 0.3321 | 0.3014 | 0.0339 | 0.2216 | 0.4758 | 0.2736 | 0.4489 | 0.4549 | 0.1000 | 0.3632 | 0.6022 | 0.5502 | 0.6923 | 0.6131 | 126 | 103 | 56 |
| YOLO11n | 0.5754 | 0.2846 | 0.2583 | 0.0453 | 0.1694 | 0.4272 | 0.2357 | 0.4357 | 0.5385 | 0.1476 | 0.5132 | 0.6452 | 0.9091 | 0.3297 | 0.4839 | 60 | 6 | 122 |
| YOLOv8n | 0.5861 | 0.2721 | 0.2130 | 0.0245 | 0.1505 | 0.4131 | 0.2181 | 0.4280 | 0.5269 | 0.0810 | 0.5059 | 0.6430 | 0.8571 | 0.3626 | 0.5097 | 66 | 11 | 116 |
| YOLOv8s | 0.5380 | 0.2512 | 0.1866 | 0.0481 | 0.1443 | 0.3852 | 0.2099 | 0.4154 | 0.5198 | 0.0619 | 0.4971 | 0.6398 | 0.9286 | 0.2143 | 0.3482 | 39 | 3 | 143 |

</div>

<br>

Precision, Recall, F1 and TP/FP/FN are taken at confidence threshold 0.25.

---

<p align="center"><strong>Table 4: Compute overhead</strong></p>

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

All 19 configurations. `LMEI`, `Max streams @budget`, `1-stream latency` and `resolution slope` were not produced (resolution sweep disabled).

---

<p align="center"><strong>Table 5: Memory overhead</strong></p>

<div align="center">

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

All 19 configurations. **Peak memory was produced by two different instruments and the column is not a single ranking.** 8 models were exported to a TorchScript/ONNX graph and profiled through it, so the reading is tensor memory live on the graphics card (`device_allocator`, "GPU tensors" below). The other 11 could not be exported — the anomaly backends, plus Cascade R-CNN, whose custom layers no exporter accepts — and were profiled natively, where the only available instrument is the resident memory of the whole Python process (`process_rss`, "whole program" below). The two are not two views of one number: whole-program memory includes the weights, the graphics-card context, host tensor copies and the framework's libraries. Compare rows only within the same "Measured as" value. The Small Machine's figures are whole-program throughout, so they are comparable with the "whole program" rows here and not with the "GPU tensors" rows. The instrument is also recorded per row as `memory_measurement_kind` in `snapshot_audit.csv`.

---

## Corrections applied to the raw export

<div align="center">

| # | Change | Evidence |
| :---: | :---: | :---: |
| 1 | Instance `mAP50` → `F1 @0.25` | The value equals `2·TP/(2·TP+FP+FN)` exactly for all five scored models: Cascade 248/378=0.6561, Faster 252/411=0.6131, YOLO11n 120/248=0.4839, YOLOv8n 132/259=0.5097, YOLOv8s 78/224=0.3482. It also equals the Small Machine `F1-Score` column verbatim. |
| 2 | Instance `TP` YOLO11n 6 → 60 | Its own `P = 0.9091 = 60/66`; the Small Machine results record TP 60 / FP 6 / FN 122. The `6` is a dropped digit. |
| 3 | Instance `mAP50-95` dropped | 0.0000 for every row — not written by the exporter, not a measurement. |
| 4 | Instance `F1` column dropped | Not consistent with its own `P`/`R` (Faster R-CNN: 2·0.5502·0.6923/(1.2425)=0.6131, table said 0.7326) and matches nothing else in the run. Source unverified. |
| 5 | Compute `Params (M)` → `FLOPs (G)` | Values equal the Small Machine FLOPs column: Faster 268.9822=268.98, SuperSimpleNet 96.7028=96.70, PatchCore 24.1408=24.14, STFPM 7.3767=7.38, PaDiM 3.6884=3.69, YOLOv8n 8.1942=8.19, EfficientAD 75.9065=75.91, UNet++ 250.2320=250.23. |
| 6 | Compute `FLOPs (G)` column dropped | Values (0.2775 … 14.9008) match neither the model's parameters nor its known FLOPs (e.g. DETR 186 G, YOLOv8n 8.19 G). Unverified. |
| 7 | Compute `Peak GPU Mem (GB)` dropped | Mostly `—`, the rest `-0.0002 / 0.0000 / -0.0000` — noise, and memory belongs to Table 5. |
| 8 | Memory `Benchmark peak memory (MB)` → `Parameters (M)` | Confirmed by building each architecture and counting: Cascade 69.16 M, DETR 44.39 M, DeepLabV3+ 40.35 M, Faster 41.35 M, Mask 43.98 M, UNet++ 26.19 M; and against the Small Machine results for PatchCore 24.86, PaDiM 2.78, STFPM 5.57, EfficientAD 8.06, GANomaly 188.69, SuperSimpleNet 33.72, YOLOv8n 3.01. All 19 match to 4 decimals. |
| 9 | Image `显存/处理耗时` dropped | Mixed/uninterpretable; all detection rows are a constant 0.5 and the rest disagree with the latency in Table 4. Latency now appears only in Table 4. |
| 10 | Pixel-level `PRO` → `AUPRO` | Same metric; `AUPRO` is the README name. |

</div>

<br>

## Still open

1. **Threshold calibration** — whether `Calibrate decision thresholds on the training split` was on. Without it, Table 1's F1/P/R and Table 2's Pixel F1 are unmeasured (README §3.3.1/§3.3.2). The Small Machine run had it on (it reports a decision threshold per model); this run's thresholds are unknown.
2. **MoECLIP row** is labelled `MVTec AD adapter`, i.e. a different dataset from every other row.
3. **MambaAD** is declared in README §3.1 but absent here.
4. **Cross-domain (README §3.3.4)** and **communication overhead** are absent; the Small Machine results carry a cross-domain table instead.
5. **Compute metrics not produced**: `LMEI`, `Max streams @budget`, `1-stream latency`, `resolution slope`, `power/energy`, `model transfer`.
6. **Pixel IAP** and instance `Recall (small <10px)` / `Recall (normal)` are absent.
