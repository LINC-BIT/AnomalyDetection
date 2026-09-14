<h1 align="center">AnomalyDetection</h1>

---

## Outline

- [Outline](#outline)
- [1. Introduction](#1-introduction)
- [2. Environment Setup](#2-environment-setup)
  - [2.1 Requirements](#21-requirements)
    - [2.1.1 Hardware Requirements](#211-hardware-requirements)
    - [2.1.2 Software Requirements](#212-software-requirements)
  - [2.2 Installation](#22-installation)
    - [2.2.1 Clone the Repository](#221-clone-the-repository)
    - [2.2.2 Install Python Environment](#222-install-python-environment)
    - [2.2.3 Install Dependencies](#223-install-dependencies)
    - [2.2.4 Verify Installation](#224-verify-installation)
    - [2.2.5 Download Datasets](#225-download-datasets)
    - [2.2.6 Download Checkpoints](#226-download-checkpoints)
- [3. Models, Datasets, and Metrics](#3-models-datasets-and-metrics)
  - [3.1 Supported Models](#31-supported-models)
  - [3.2 Supported Datasets](#32-supported-datasets)
  - [3.3 Supported Metrics](#33-supported-metrics)
    - [3.3.1 Technical Metrics: Image level](#331-technical-metrics-image-level)
    - [3.3.2 Technical Metrics: Pixel level](#332-technical-metrics-pixel-level)
    - [3.3.3 Technical Metrics: Instance level](#333-technical-metrics-instance-level)
    - [3.3.4 Technical Metrics: Cross-domain](#334-technical-metrics-cross-domain)
    - [3.3.5 Overhead Metrics: Compute](#335-overhead-metrics-compute)
    - [3.3.6  Overhead Metrics: Memory](#336--overhead-metrics-memory)
- [4. Workflows](#4-workflows)
  - [4.1 Web front-end](#41-web-front-end)
    - [4.1.1 Launch](#411-launch)
    - [4.1.2 Model session](#412-model-session)
    - [4.1.3 Benchmark](#413-benchmark)
    - [4.1.4 Run history](#414-run-history)
    - [4.1.5 Recorded demonstrations](#415-recorded-demonstrations)
  - [4.2 Command line](#42-command-line)
    - [4.2.1 Model configuration resolution](#421-model-configuration-resolution)
    - [4.2.2 Training](#422-training)
    - [4.2.3 Inference](#423-inference)
    - [4.2.4 Evaluation](#424-evaluation)
    - [4.2.5 Benchmarking](#425-benchmarking)
    - [4.2.6 Batch training](#426-batch-training)
    - [4.2.7 Catalogue and diagnostic commands](#427-catalogue-and-diagnostic-commands)
- [5. Examples](#5-examples)
  - [5.1 Start the web front-end](#51-start-the-web-front-end)
  - [5.2 Run Anomaly Detection Example](#52-run-anomaly-detection-example)
- [6. Extensibility](#6-extensibility)


---

## 1. Introduction

AnomalyDetection is an **extensible, multi-domain platform** that collects **18 classical methods**, multiple datasets for benchmarking, and a **Web front-end**.

Industrial inspection demands high-speed, high-accuracy anomaly detection. However, existing methods remain fragmented with disparate formats and lack joint assessments of accuracy and deployment cost. To address this, we present a unified benchmarking platform that integrates SOTA methods to systematically evaluate both predictive performance and computational overhead.

## 2. Environment Setup

### 2.1 Requirements

#### 2.1.1 Hardware Requirements

<table align="center">
  <thead>
    <tr>
      <th>Workflow</th>
      <th>CPU / RAM</th>
      <th>GPU</th>
      <th>Disk Requirements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Web front-end</td>
      <td>8 cores, 16 GB</td>
      <td>Optional</td>
      <td>At least 10 GB(depends on downloaded checkpoints and datasets)</td>
    </tr>
    <tr>
      <td>Full Training</td>
      <td>16 cores, 32 GB</td>
      <td>NVIDIA recommended</td>
      <td>40 GB (Full dataset) + 5 GB (Checkpoints)</td>
    </tr>
    <tr>
      <td>Full Benchmarking</td>
      <td>16 cores, 32 GB</td>
      <td>Recommended</td>
      <td>At least 50 GB (Scales with tested checkpoints and evaluated datasets)</td>
    </tr>
  </tbody>
</table>

#### 2.1.2 Software Requirements

<table align="center">
  <thead>
    <tr>
      <th>Item</th>
      <th>Requirement</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>Python</strong></td><td><code>≥3.10</code></td></tr>
    <tr><td><strong>Environment manager</strong></td><td>Conda</td></tr>
    <tr><td><strong>Operating system</strong></td><td>Linux, Windows, macOS</td></tr>
    <tr><td><strong>GPU</strong></td><td>optional for inference; NVIDIA/CUDA for training</td></tr>
    <tr><td><strong>CUDA</strong></td><td>only for NVIDIA acceleration</td></tr>

  </tbody>
</table>

### 2.2 Installation

#### 2.2.1 Clone the Repository

```bash
git clone --recurse-submodules https://github.com/LINC-BIT/AnomalyDetection.git
cd AnomalyDetection
```

Do not clone submodules manually. If submodule initialization fails or is executed incorrectly, re-run `git submodule update --init --recursive` to restore the submodules.

#### 2.2.2 Install Python Environment

Create and activate the Python environment:

```bash
conda create -n anomalib_env python=3.12 -y
conda activate anomalib_env
python --version
```

#### 2.2.3 Install Dependencies

Choose the scenario according to the deployment; **Full** is recommended.

<table align="center">
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Installation command</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Full</td><td><code>python -m pip install -e ".[all]"</code></td></tr>
    <tr><td>Frontend</td><td><code>python -m pip install -e ".[ui]"</code></td></tr>
    <tr><td>Training</td><td><code>python -m pip install -e ".[train]"</code></td></tr>
    <tr><td>Evaluation</td><td><code>python -m pip install -e ".[eval]"</code></td></tr>
  </tbody>
</table>

#### 2.2.4 Verify Installation

Use the following commands to verify the installation, no dataset and no weight is required:

```bash
adh doctor
adh list
```

Expected running output:

```json
{
  "backends": {
    "anomalib": {
      "framework_installed": true,
      "dataset_kind": "one-class",
      "dataset": "fabric-defects",
      "trainable_now": true,
      "reason": "picked 'fabric-defects' (staged). Other staged alternatives: fabric-train, mvtec-ad, mvtec-loco, raw-fabric, tianchi, tilda-400, visa, zju-leaper."
    }
  }
}
```

```json
{
  "datasets": ["fabric-defects", "fabric-train", "mvtec-ad", "mvtec-loco", "raw-fabric", "tianchi", "tilda-400", "visa", "zju-leaper"],
  "model_backends": {
    "known": ["anomalib", "dinomaly", "mambaad", "moeclip", "torchvision", "ultralytics"],
    "available": ["anomalib", "dinomaly", "mambaad", "moeclip", "torchvision", "ultralytics"]
  },
  "evaluators": ["anomaly", "detection", "industrial", "segmentation"],
  "profilers": ["onnxruntime", "pytorch", "tensorrt"]
}
```

Other catalogue and diagnostic commands work the same way:
- `adh inventory` prints the machine-readable model and dataset inventory, 
- `adh models` lists the model variants of each backend, 
- `adh recipes` lists the registered optimization recipes,
- `adh train --list` lists the model configurations a training run can resolve.

#### 2.2.5 Download Datasets

Datasets are staged with the download tool, which takes the registered dataset identifier and the declared root:

<table align="center">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Download command</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>MVTec AD</td><td><code>python tools/download_datasets.py mvtec-ad --root "datasets/general/MVTec AD"</code></td></tr>
    <tr><td>VisA</td><td><code>python tools/download_datasets.py visa --root datasets/general/VisA</code></td></tr>
    <tr><td>ZJU-Leaper</td><td><code>python tools/download_datasets.py zju-leaper --root datasets/textile/ZJU-Leaper</code></td></tr>
  </tbody>
</table>

These three datasets are downloaded in full. To download a single category instead, add `--category <name>`; the available category names are listed on each dataset's own page — [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff), [ZJU-Leaper](https://huggingface.co/datasets/AnupamaBandara/ZLU_Leaper).

#### 2.2.6 Download Checkpoints

We provide pretrained checkpoints for demonstration:

- **Textile domain** — the detectors `yolov8n`, `yolov8s`, `yolo11n`, `fasterrcnn_resnet50_fpn`, `cascadercnn_resnet50_fpn`, and `detr_resnet50`; the segmentation models `maskrcnn_resnet50_fpn`, `unetplusplus_resnet34`, and `deeplabv3plus_resnet50`; and the normal-only anomaly models `PatchCore`, `PaDiM`, `RD4AD`, `EfficientAD`, `SuperSimpleNet`, `STFPM`, `GANomaly`, and `Dinomaly`.
- **General domain** — the zero-shot `WinCLIP` and the auxiliary-trained `MoECLIP`.


**Step 1: Install the Hugging Face Hub client:**

```bash
python -m pip install -U huggingface_hub
```

**Step 2: Download the checkpoints:**

Download all checkpoints:

```bash
python tools/download_weights.py --all
```

Optional: To download a single checkpoint, browse the [published checkpoints on Hugging Face](https://huggingface.co/AuroraLeeeeee/AnomalyDetection-textile-weights) and pick the one needed, for example:

```bash
python tools/download_weights.py \
  textile/artifacts/models/published/yolov8n.pt \
  --output textile/artifacts/models/published/yolov8n.pt
```

## 3. Models, Datasets, and Metrics

### 3.1 Supported Models

18 methods are supported.

<table align="center">
  <thead>
    <tr>
      <th>Method</th>
      <th>Task Type</th>
      <th>Output</th>
      <th>Learning Type</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>YOLO</td><td>Defect detection</td><td>Bounding boxes</td><td>Supervised</td></tr>
    <tr><td>Faster R-CNN</td><td>Defect detection</td><td>Bounding boxes</td><td>Supervised</td></tr>
    <tr><td>Cascade R-CNN</td><td>Defect detection</td><td>Bounding boxes</td><td>Supervised</td></tr>
    <tr><td>DETR</td><td>Defect detection</td><td>Bounding boxes</td><td>Supervised</td></tr>
    <tr><td>Mask R-CNN</td><td>Defect detection</td><td>Bounding boxes, pixel-level segmentation</td><td>Supervised</td></tr>
    <tr><td>UNet++</td><td>Defect detection</td><td>Pixel-level segmentation</td><td>Supervised</td></tr>
    <tr><td>DeepLabV3+</td><td>Defect detection</td><td>Pixel-level segmentation</td><td>Supervised</td></tr>
    <tr><td>PatchCore</td><td>Anomaly detection</td><td>Image-level score, pixel-level heat map</td><td>Normal-only (one-class)</td></tr>
    <tr><td>PaDiM</td><td>Anomaly detection</td><td>Image-level score, pixel-level heat map</td><td>Normal-only (one-class)</td></tr>
    <tr><td>Reverse Distillation</td><td>Anomaly detection</td><td>Image-level score, pixel-level heat map</td><td>Normal-only (one-class)</td></tr>
    <tr><td>EfficientAD</td><td>Anomaly detection</td><td>Image-level score, pixel-level heat map</td><td>Normal-only (one-class)</td></tr>
    <tr><td>SuperSimpleNet</td><td>Anomaly detection</td><td>Image-level score, pixel-level heat map</td><td>Normal-only (one-class)</td></tr>
    <tr><td>STFPM</td><td>Anomaly detection</td><td>Image-level score, pixel-level heat map</td><td>Normal-only (one-class)</td></tr>
    <tr><td>GANomaly</td><td>Anomaly detection</td><td>Image-level score</td><td>Normal-only (one-class)</td></tr>
    <tr><td>Dinomaly</td><td>Anomaly detection</td><td>Image-level score, pixel-level heat map</td><td>Normal-only (one-class)</td></tr>
    <tr><td>MambaAD</td><td>Anomaly detection</td><td>Image-level score, pixel-level heat map</td><td>Normal-only (one-class)</td></tr>
    <tr><td>WinCLIP</td><td>Anomaly detection</td><td>Image-level score, pixel-level heat map</td><td>Zero-shot</td></tr>
    <tr><td>MoECLIP</td><td>Anomaly detection</td><td>Image-level score, pixel-level heat map</td><td>Supervised (zero-shot transfer)</td></tr>
  </tbody>
</table>

### 3.2 Supported Datasets

Nine datasets are registered.

<table align="center">
  <thead>
    <tr>
      <th>Name</th>
      <th>Domain</th>
      <th>Images</th>
      <th>Categories</th>
      <th>Annotation</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>ZJU-Leaper</td><td>Fabric</td><td>94,833</td><td>19 patterns (5 groups)</td><td>Bounding boxes, pixel-level masks</td></tr>
    <tr><td>RAW-FABRID</td><td>Fabric</td><td>19,852 tiles (709 raw images)</td><td>1 defect bucket</td><td>Pixel-level masks</td></tr>
    <tr><td>TILDA-400</td><td>Fabric</td><td>25,600</td><td>4 defect types</td><td>Image-level label</td></tr>
    <tr><td>Fabric Defects Dataset</td><td>Fabric</td><td>2,430</td><td>5 defect types</td><td>Pixel-level masks</td></tr>
    <tr><td>Tianchi Fabric Defect</td><td>Fabric</td><td>9,576</td><td>34 defect types</td><td>Bounding boxes</td></tr>
    <tr><td>Fabric-Train (composite)</td><td>Fabric</td><td>152,291 (its 5 members combined)</td><td>5 fabric datasets combined</td><td>Bounding boxes, pixel-level masks</td></tr>
    <tr><td>MVTec AD</td><td>Industrial objects</td><td>5,354</td><td>15 categories</td><td>Pixel-level masks</td></tr>
    <tr><td>MVTec LOCO AD</td><td>Everyday objects</td><td>3,346</td><td>5 categories</td><td>Pixel-level masks</td></tr>
    <tr><td>VisA</td><td>General objects</td><td>10,821</td><td>12 categories</td><td>Pixel-level masks</td></tr>
  </tbody>
</table>

Image and category counts are measured from the staged copy under `datasets/`; `Annotation` is what the dataset's adapter exposes.

### 3.3 Supported Metrics

Metrics are divided into two parts. 
- **technical** metrics evaluate model accuracy across different levels.
- **overhead** metrics evaluate the  running cost of the model.

The following table summarizes the supported metrics for each category and scope.

<table align="center">
  <thead>
    <tr>
      <th>Category</th>
      <th>Table</th>
      <th>Scope</th>
      <th>Metrics</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Technical</td><td>Image level</td><td>The image as a whole</td><td>7</td></tr>
    <tr><td>Technical</td><td>Pixel level</td><td>Every pixel of the image</td><td>7</td></tr>
    <tr><td>Technical</td><td>Instance level</td><td>Each detected defect, as a box</td><td>20</td></tr>
    <tr><td>Technical</td><td>Cross-domain</td><td>The same weights on an unseen fabric</td><td>8</td></tr>
    <tr><td>Overhead</td><td>Compute</td><td>Time, FPS, latency, FLOPs, energy, transfer size</td><td>26</td></tr>
    <tr><td>Overhead</td><td>Memory</td><td>Peak and average memory, model size, parameters</td><td>4</td></tr>
    <tr><td>Overhead</td><td>Communication</td><td>Bandwidth saved over a deployed link; declared but not implemented</td><td>—</td></tr>
  </tbody>
</table>

The communication metric is currently set to incomplete as it requires an active transport to measure.

#### 3.3.1 Technical Metrics: Image level

Evaluates the image as a whole.

<table align="center">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Image AUROC</td><td>Ranking quality of the image-level anomaly score.</td></tr>
    <tr><td>Image F1</td><td>F1 of the thresholded image decision.</td></tr>
    <tr><td>Image Precision</td><td>Share of the images called defective that are defective.</td></tr>
    <tr><td>Image Recall</td><td>Share of the defective images that are called.</td></tr>
    <tr><td>Decision threshold</td><td>The threshold the decision was taken at. It is calibrated on a validation set.</td></tr>
    <tr><td>AUROC</td><td>The unprefixed AUROC key some runs record for the same curve.</td></tr>
    <tr><td>AP</td><td>Average precision of the same image-level ranking.</td></tr>
  </tbody>
</table>

#### 3.3.2 Technical Metrics: Pixel level

Evaluates the classification result of every pixel.

<table align="center">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Pixel AUROC</td><td>Ranking quality over individual pixels, defective vs normal.</td></tr>
    <tr><td>AUPRO</td><td>Area under the per-region overlap curve, capped at a maximum false-positive rate.</td></tr>
    <tr><td>IAP</td><td>Instance Average Precision: each connected defect region gets its own precision/recall integral, and the regions are averaged with equal weight.</td></tr>
    <tr><td>Pixel F1</td><td>F1 over pixels at the calibrated pixel threshold.</td></tr>
    <tr><td>mIoU</td><td>Mean intersection-over-union of predicted and ground-truth masks (segmentation).</td></tr>
    <tr><td>Dice</td><td>Dice coefficient of predicted and ground-truth masks.</td></tr>
    <tr><td>PRO score</td><td>Per-region overlap up to a maximum false-positive rate; strict on microscopic and connected defect regions.</td></tr>
  </tbody>
</table>

#### 3.3.3 Technical Metrics: Instance level

Evaluates each detected defect as a bounding box.

<table align="center">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>mAP@[.5:.95]</td><td>Mean average precision averaged over IoU thresholds 0.5 to 0.95 (COCO style).</td></tr>
    <tr><td>mAP@0.5</td><td>mAP at IoU 0.5.</td></tr>
    <tr><td>mAP@0.75</td><td>mAP at IoU 0.75.</td></tr>
    <tr><td>mAP small</td><td>mAP restricted to small boxes.</td></tr>
    <tr><td>mAP medium</td><td>mAP restricted to medium boxes.</td></tr>
    <tr><td>mAP large</td><td>mAP restricted to large boxes.</td></tr>
    <tr><td>mAR@1</td><td>Mean average recall with one detection per image.</td></tr>
    <tr><td>mAR@10</td><td>Mean average recall with ten detections per image.</td></tr>
    <tr><td>mAR@100</td><td>Mean average recall with a hundred detections per image.</td></tr>
    <tr><td>mAR small</td><td>mAR restricted to small boxes.</td></tr>
    <tr><td>mAR medium</td><td>mAR restricted to medium boxes.</td></tr>
    <tr><td>mAR large</td><td>mAR restricted to large boxes.</td></tr>
    <tr><td>Precision @0.5</td><td>Precision of the thresholded detections at IoU 0.5.</td></tr>
    <tr><td>Recall @0.5</td><td>Recall of the thresholded detections at IoU 0.5.</td></tr>
    <tr><td>F1 @0.5</td><td>F1 of the thresholded detections at IoU 0.5.</td></tr>
    <tr><td>Recall (small &lt;10px)</td><td>Recall on boxes whose shorter side is under 10 px. These are the broken-warp / skipped-pick cases an aggregate recall averages away.</td></tr>
    <tr><td>Recall (normal)</td><td>Recall on the remaining, larger boxes.</td></tr>
    <tr><td>TP</td><td>True positives behind the thresholded numbers.</td></tr>
    <tr><td>FP</td><td>False positives behind the thresholded numbers.</td></tr>
    <tr><td>FN</td><td>False negatives behind the thresholded numbers.</td></tr>
  </tbody>
</table>

#### 3.3.4 Technical Metrics: Cross-domain

Evaluates the same weights on fabrics they were not trained on.

<table align="center">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Source accuracy</td><td>Accuracy on the patterns the model was trained on.</td></tr>
    <tr><td>Cross-domain drop</td><td>Accuracy drop, in percent, on a single held-out dataset.</td></tr>
    <tr><td>Top-k mean drop</td><td>Mean drop over the k worst (or best) held-out patterns.</td></tr>
    <tr><td>Mean drop (all patterns)</td><td>Mean drop over every held-out pattern that could be scored.</td></tr>
    <tr><td>CI low</td><td>Lower bound of the bootstrap confidence interval, resampling patterns rather than images.</td></tr>
    <tr><td>CI high</td><td>Upper bound of the same interval.</td></tr>
    <tr><td>k used</td><td>How many patterns the top-k mean actually used.</td></tr>
    <tr><td>Patterns scored</td><td>How many held-out patterns were scorable. Unscorable patterns are skipped, never counted as zero degradation.</td></tr>
  </tbody>
</table>

#### 3.3.5 Overhead Metrics: Compute

What running it costs in time, work and power.

<table align="center">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>FPS (aggregate)</td><td>Frames per second over the profiling pass.</td></tr>
    <tr><td>Instantaneous FPS mean</td><td>Mean of the per-frame instantaneous frame rate.</td></tr>
    <tr><td>Instantaneous FPS stdev</td><td>Spread of the instantaneous frame rate.</td></tr>
    <tr><td>Instantaneous FPS CV</td><td>Coefficient of variation of the instantaneous frame rate: frame-rate jitter.</td></tr>
    <tr><td>Latency mean</td><td>Mean per-frame latency, in milliseconds.</td></tr>
    <tr><td>Latency stdev</td><td>Spread of the per-frame latency.</td></tr>
    <tr><td>Latency CV</td><td>Coefficient of variation of the per-frame latency.</td></tr>
    <tr><td>Latency p50</td><td>Median per-frame latency.</td></tr>
    <tr><td>Latency p95</td><td>95th percentile per-frame latency.</td></tr>
    <tr><td>Latency p99</td><td>99th percentile per-frame latency.</td></tr>
    <tr><td>FLOPs</td><td>Multiply-accumulates of one forward pass.</td></tr>
    <tr><td>FLOPs (G)</td><td>The same in giga-FLOPs.</td></tr>
    <tr><td>LMEI</td><td>Latency–Memory Efficiency Index: throughput divided by the log-scaled cost of FLOPs and VRAM. Higher is a better edge-deployment trade-off.</td></tr>
    <tr><td>Max streams @budget</td><td>Most concurrent streams that still meet the per-frame latency budget; the worst stream decides.</td></tr>
    <tr><td>1-stream latency</td><td>Latency at a single stream: the baseline the concurrency probe compares against.</td></tr>
    <tr><td>Throughput-resolution slope</td><td>How quickly throughput decays as input resolution grows.</td></tr>
    <tr><td>Slope difference</td><td>Difference between two runs' resolution slopes.</td></tr>
    <tr><td>Power mean</td><td>Mean power draw. NVML only, so NVIDIA hosts only.</td></tr>
    <tr><td>Power peak</td><td>Peak power draw.</td></tr>
    <tr><td>Energy</td><td>Energy used over the pass, in joules.</td></tr>
    <tr><td>Wall time</td><td>Wall-clock seconds of the scored run.</td></tr>
    <tr><td>Power samples</td><td>How many power readings the mean and peak are based on.</td></tr>
    <tr><td>Model transfer (MB)</td><td>Size of the model file, recorded as the communication proxy.</td></tr>
    <tr><td>Model transfer (bytes)</td><td>The same in bytes.</td></tr>
    <tr><td>Export transfer (MB)</td><td>Size of the exported artifact, recorded as the communication proxy.</td></tr>
    <tr><td>Export transfer (bytes)</td><td>The same in bytes.</td></tr>
  </tbody>
</table>

#### 3.3.6  Overhead Metrics: Memory

<table align="center">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Parameters (M)</td><td>Parameter count, in millions.</td></tr>
    <tr><td>Memory peak</td><td>Peak memory during the profiling pass.</td></tr>
    <tr><td>Memory average</td><td>Average memory during the profiling pass.</td></tr>
    <tr><td>Model size</td><td>Size of the trained artifact on disk.</td></tr>
  </tbody>
</table>

## 4. Workflows

<p align="center"><img src="docs/images/structure.png" alt="System architecture" width="80%"></p>

The web front-end and the command line run on the same application services and the same backend contracts (`ModelAdapter`, `DataConverter`, `ModelWrapper`, `RuntimeSupport`)

### 4.1 Web front-end

Three tabs: **Single Image Detection**, **Benchmark**, and **Run History**. The language switch toggles English and Chinese.

#### 4.1.1 Launch

```bash
conda activate anomalib_env
adh-ui
```

Default URL `http://127.0.0.1:6008`, listening on all interfaces; `GRADIO_SERVER_PORT=7860 adh-ui` uses another port. If the port is taken, `adh-ui` names the holding process. A missing checkpoint is never downloaded implicitly: the expected path is shown instead.

#### 4.1.2 Model session

Select **Task type** (`Defect detection` or `Anomaly detection`), then **Application domain** (`textile` or `general`), then **Model** (`Local trained model`); the panel states the method, training corpus, training split, published-slot status, and prediction fields. Select **Load model** before inference.

Input is either an uploaded image or a dataset slice: `Dataset`, `Texture / pattern`, `Split`, `Image selection`, **Random images** (4–12), and **Sample regime** — the few-shot control:

<table align="center">
  <thead>
    <tr>
      <th>Regime</th>
      <th>What it loads</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Full-shot</td><td>the whole selected slice</td></tr>
    <tr><td>Few-shot</td><td>a small number of random images from that slice</td></tr>
  </tbody>
</table>

**Load random images** draws the slice; **Run detection** returns a gallery with a per-image anomaly score, plus a heat map when the model reports one. One pass through the tab:

1. `Anomaly detection`, application domain `general`.
2. Model `WinCLIP · LAION-400M zero-shot`, or any identifier whose `adh inventory` `weight_status` is `file` or `symlink`.
3. **Load model**.
4. Dataset `ZJU-Leaper`, any pattern or `All textures`, split `test`, `Full-shot`, then **Load random images**.
5. **Run detection**.

#### 4.1.3 Benchmark

**Benchmark** scores one or more identifiers on one dataset: `Dataset`, `Texture / pattern`, `Sample regime`, the models, optionally **Include profiling** or **Include resolution sweep**, and optionally a **Cross-domain degradation target dataset**, then **Run benchmark**.

Results are grouped as in Section 3.3; a metric that cannot be computed is `unavailable`, never zero. Each run appends to `runs/leaderboard_log.jsonl`, and maps go under `artifacts/runtime/anomaly_maps/benchmark/`.

#### 4.1.4 Run history

**Run history** reads a saved run log: enter the path, select **Refresh**, and optionally the metric to chart. It shows a `Runs` table (timestamp, model, dataset, metric values, report path) and a metric-by-model chart using each model's most recent run. Finished work can also be reviewed through the recorded demonstrations below.

#### 4.1.5 Recorded demonstrations

<table align="center">
  <thead>
    <tr>
      <th>Demonstration</th>
      <th>Recording</th>
      <th>Contents</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Web front-end demonstration</td><td><a href="docs/videos/detection.mp4">detection.mp4</a></td><td>Task selection, application-domain selection, model selection, dataset selection, image loading, and detection output.</td></tr>
    <tr><td>Benchmark demonstration</td><td><a href="docs/videos/benchmark.mp4">benchmark.mp4</a></td><td>Benchmark execution and run-history reading.</td></tr>
  </tbody>
</table>

### 4.2 Command line

#### 4.2.1 Model configuration resolution

`train`, `predict`, and `evaluate` resolve their first positional argument three ways:

1. As a path to a model configuration, for example `configs/models/ultralytics_example.yaml`.
2. As a filename stem under `--config-dir`, for example `ultralytics_example`.
3. As a model keyword matched against the `model.variant` or `model.name` field of every model configuration under `--config-dir`, for example `yolov8n` or `patchcore`.

`--config-dir` defaults to `configs/models`. These commands list different objects:

<table align="center">
  <thead>
    <tr>
      <th>Command</th>
      <th>Lists</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>adh train --list</code></td><td>the model configurations under <code>--config-dir</code></td></tr>
    <tr><td>`adh recipes`</td><td>registered optimization recipes, with paper reference and hyperparameters</td></tr>
    <tr><td>`adh models`</td><td>model variants per backend — not the published models, which are <code>adh inventory</code></td></tr>
  </tbody>
</table>

#### 4.2.2 Training

A training run takes a model configuration, modified by `--dataset`, `--variant`, `--mode`, and `--set`. `--set` overrides by dotted path and has the highest priority.

```bash
# One-class anomaly model on a general dataset.
adh train general/configs/models/anomalib.yaml \
  --variant PatchCore \
  --dataset mvtec-ad \
  --category bottle

# Textile detection model, full sample budget.
adh train yolov8n --dataset zju-leaper --mode full

# Textile one-class model, one nested override.
adh train patchcore \
  --dataset zju-leaper \
  --mode medium \
  --set train.model_kwargs.coreset_sampling_ratio=0.05
```

`moeclip` trains on one dataset and is evaluated on another:

```bash
adh train moeclip \
  --dataset mvtec-ad \
  --test-dataset zju-leaper \
  --mode test \
  --no-publish
```

`--mode test` is an eight-image wiring check, not a usable weight; combine it with `--no-publish`.

Publishing is on by default: a successful run whose identifier is in the manifest replaces that identifier's published slot, which is the file the web front-end reads.

```bash
# Smoke run; no slot is touched.
adh train patchcore --dataset zju-leaper --mode test --no-publish
```

Result keys: `backend`, `resolved_config`, `resolved_variant`, `metrics`, `trained_artifact`, `registered_artifact`, `published_path`, `weight_manifest_path`, `exports`.

#### 4.2.3 Inference

```bash
# Single image.
adh predict patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --image /absolute/path/to/image.jpg \
  --output-dir artifacts/runtime/anomaly_maps \
  --output results/patchcore-prediction.json

# Samples from a registered dataset.
adh predict patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --dataset zju-leaper \
  --split test \
  --pattern pattern1 \
  --num-samples 8 \
  --output-dir artifacts/runtime/anomaly_maps
```

`--output-dir` writes one anomaly map per sample to `<output directory>/<sample identifier>.npy`, for models that declare `anomaly_map` only. `--output` writes the predictions as a JSON array.

#### 4.2.4 Evaluation

```bash
adh evaluate patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --dataset zju-leaper \
  --split test \
  --num-samples 32 \
  --output-dir artifacts/runtime/anomaly_maps
```

`--task` forces an evaluator instead of the sample task. `--output-dir` is required for pixel-level metrics, because they need a persisted anomaly map; without it only image-level metrics are scored.

Result keys: `backend`, `resolved_config`, `variant`, `sample_count`, `metrics`.

Cross-pattern robustness scores the same weights on held-out patterns and reduces the per-pattern accuracy drops to one number:

```bash
adh evaluate patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --dataset zju-leaper \
  --pattern pattern1-4 \
  --cross-domain-patterns 5,6,7,8 \
  --cross-domain-k 3 \
  --cross-domain-mode worst \
  --output-dir artifacts/runtime/anomaly_maps
```

`--cross-domain-metric` selects the metric (default: the task headline metric); `--cross-domain-mode` is `worst` (mean over the largest drops) or `best` (mean over the smallest). The mode is echoed in the output. Patterns that cannot be scored are skipped, never counted as zero degradation.

#### 4.2.5 Benchmarking

A benchmark configuration has the top-level keys `runs`, `output_dir`, `leaderboard`, `report_path`, and `run_log_path`; `runs` is a non-empty list with one experiment per entry.

```bash
adh benchmark configs/archive/benchmark_example.yaml
```

The example scores `fasterrcnn_resnet50_fpn` on the ZJU-Leaper `test` split, writes under `artifacts/benchmarks/example`, and reads the dataset root from `ZJU_LEAPER_ROOT`, so set that variable first. The result is a JSON array with one element per run.

#### 4.2.6 Batch training

`train-all` trains every manifest identifier in one resumable batch, with one log and one state record per identifier:

```bash
adh train-all --dry-run
adh train-all --only yolov8n PatchCore --mode test --no-publish
adh train-all --run-id <run-id> --resume
```

Result keys: `batch_state`, `succeeded`, `total`, `results`; the `batch_state` directory holds the per-model state and log.

#### 4.2.7 Catalogue and diagnostic commands

<table align="center">
  <thead>
    <tr>
      <th>Command</th>
      <th>Function</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>adh inventory</code></td><td>Machine-readable inventory of models and datasets.</td></tr>
    <tr><td><code>adh list</code></td><td>Registered datasets, backends, evaluators, profilers.</td></tr>
    <tr><td><code>adh models</code></td><td>Model variants per backend; <code>--backend</code> restricts.</td></tr>
    <tr><td><code>adh recipes</code></td><td>Registered optimization recipes.</td></tr>
    <tr><td><code>adh doctor</code></td><td>Per backend: trainable on this machine, and the selected dataset.</td></tr>
    <tr><td><code>adh train --list</code></td><td>Model configurations a training command can resolve.</td></tr>
    <tr><td><code>adh run</code></td><td>Run a model or benchmark YAML configuration; the backend is inferred unless <code>--backend</code> is given.</td></tr>
    <tr><td><code>adh export-latex</code></td><td>Convert a benchmark result file into a LaTeX table.</td></tr>
  </tbody>
</table>

## 5. Examples

To start this example, please follow the instructions in ([2. Environment Setup](#2-environment-setup)) to complete the installation.

The following example illustrates the web-based anomaly detection workflow:

### 5.1 Start the web front-end

Please open the terminal and run the following command:

```bash
cd <AnomalyDetection root directory>

conda activate anomaly-detection

adh-ui
``` 

**Open** `http://127.0.0.1:6008` in your web browser.

### 5.2 Run Anomaly Detection Example

**Step 1: Select the task type.**

Select `Anomaly detection` in Task Type dropdown:

<p align="center"><img src="docs/images/img1.png" alt="Select the task type" width="80%"></p>

**Step 2: Select the application domain.**

Select `textile` in Application Domain dropdown:

<p align="center"><img src="docs/images/img2.png" alt="Select the application domain" width="80%"></p>

**Step 3: Select the model.**

Select the `PatchCore` in Model dropdown:

<p align="center"><img src="docs/images/img3.png" alt="Select the model" width="80%"></p>

**Step 4: Select the dataset.**

Select `ZJU-Leaper`, choose **All textures**, set split to **test** mode, and use **Full-shot** as sampling regime, then click the **Load random images** button:

<p align="center"><img src="docs/images/img5.png" alt="Select the dataset and the split" width="80%"></p>

please wait for the images to load.

**Step 5: Run detection.**

Click the **Run detection** button and wait for completion.

<p align="center"><img src="docs/images/img6.png" alt="Run detection" width="80%"></p>

The corresponding anomaly heat map of the image will be displayed on the right side, and the corresponding anomaly score will be shown below.

## 6. Extensibility

[TODO]
