# AnomalyDetection

**One interface for supervised defect detection, and unsupervised anomaly detection.**

AnomalyDetection is an **extensible, multi-domain platform** that runs every backend behind **one interface**. It ships **nine dataset adapters** and **29 model entries**, and is operated through a **web UI**.

---

## Quick Links

<table align="center">
  <thead>
    <tr>
      <th>Resource</th>
      <th>Entry point</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>Requirements</strong></td><td><a href="#3-requirements">Section 3</a> — software, and hardware profiles</td></tr>
    <tr><td><strong>Install</strong></td><td><a href="#4-installation">Section 4</a> — <code>python -m pip install -r requirements-full.txt</code></td></tr>
    <tr><td><strong>Verify without data</strong></td><td><a href="#44-verify-the-installation">Section 4.4</a> — <code>adh list</code>, <code>adh inventory</code>, <code>adh doctor</code></td></tr>
    <tr><td><strong>Minimal Working Example</strong></td><td><a href="#5-minimal-working-example">Section 5</a> — a visible result on one image</td></tr>
    <tr><td><strong>Datasets</strong></td><td><a href="#8-data-preparation">Section 8</a> — nine registered datasets, three with automated download</td></tr>
    <tr><td><strong>Weights</strong></td><td><a href="#9-weight-preparation">Section 9</a> — the 19 distributed weights, and the nine trainable <code>general</code> slots</td></tr>
    <tr><td><strong>Web interface</strong></td><td><a href="#10-web-interface">Section 10</a> — <code>adh-ui</code>, default <code>http://127.0.0.1:6008</code></td></tr>
    <tr><td><strong>Command line</strong></td><td><a href="#11-command-line-workflows">Section 11</a> — <code>adh train</code>, <code>adh predict</code>, <code>adh evaluate</code>, <code>adh benchmark</code></td></tr>
    <tr><td><strong>Supported models</strong></td><td><a href="#72-supported-models">Section 7.2</a> — the 29 model identifiers of the manifest</td></tr>
    <tr><td><strong>Registered datasets</strong></td><td><a href="#73-registered-datasets">Section 7.3</a> — default roots, tasks, and training roles</td></tr>
    <tr><td><strong>Directory layout</strong></td><td><a href="#6-directory-layout">Section 6</a> — where the source, configurations, datasets, and weights live</td></tr>
    <tr><td><strong>Troubleshooting</strong></td><td><a href="#14-troubleshooting">Section 14</a> — dataset, weight, web interface, and memory problems</td></tr>
    <tr><td><strong>Recorded demonstrations</strong></td><td><a href="#105-recorded-demonstrations">Section 10.5</a> — <code>detection.mp4</code>, <code>benchmark.mp4</code></td></tr>
  </tbody>
</table>

## Outline

<a href="#1-overview">1. Overview</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#11-weights">1.1 Weights</a><br>
<a href="#2-application-domains-and-task-families">2. Application domains and task families</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#21-application-domains">2.1 Application domains</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#22-the-four-task-families">2.2 The four task families</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#23-metric-availability-rules">2.3 Metric availability rules</a><br>
<a href="#3-requirements">3. Requirements</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-software">3.1 Software</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#32-hardware-profiles">3.2 Hardware profiles</a><br>
<a href="#4-installation">4. Installation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#41-obtain-the-source-and-initialise-components">4.1 Obtain the source and initialise components</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#42-create-the-python-environment">4.2 Create the Python environment</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#43-install-dependencies">4.3 Install dependencies</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#44-verify-the-installation">4.4 Verify the installation</a><br>
<a href="#5-minimal-working-example">5. Minimal Working Example</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#51-step-1-install-the-platform">5.1 Step 1: Install the platform</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#52-step-2-verify-the-installation-without-data">5.2 Step 2: Verify the installation without data</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#53-step-3-run-inference-in-the-web-interface">5.3 Step 3: Run inference in the web interface</a><br>
<a href="#6-directory-layout">6. Directory layout</a><br>
<a href="#7-application-domains-datasets-and-models">7. Application domains, datasets, and models</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#71-application-domains">7.1 Application domains</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#72-supported-models">7.2 Supported models</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#721-textile-application-domain">7.2.1 Textile application domain</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#722-general-application-domain">7.2.2 General application domain</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#73-registered-datasets">7.3 Registered datasets</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#74-choosing-a-model">7.4 Choosing a model</a><br>
<a href="#8-data-preparation">8. Data preparation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#81-registered-datasets-and-default-roots">8.1 Registered datasets and default roots</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#82-staging-a-dataset-manually">8.2 Staging a dataset manually</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#83-automated-download">8.3 Automated download</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#84-verifying-data-availability">8.4 Verifying data availability</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#85-public-dataset-sources">8.5 Public dataset sources</a><br>
<a href="#9-weight-preparation">9. Weight preparation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#91-published-slot-layout">9.1 Published-slot layout</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#92-downloading-the-distributed-weights">9.2 Downloading the distributed weights</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#93-training-a-general-domain-weight">9.3 Training a general-domain weight</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#94-verifying-weight-availability">9.4 Verifying weight availability</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#95-the-weight-manifest">9.5 The weight manifest</a><br>
<a href="#10-web-interface">10. Web interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#101-launch">10.1 Launch</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#102-model-session">10.2 Model session</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#103-benchmark">10.3 Benchmark</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#104-run-history">10.4 Run history</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#105-recorded-demonstrations">10.5 Recorded demonstrations</a><br>
<a href="#11-command-line-workflows">11. Command-line workflows</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#111-model-configuration-resolution">11.1 Model configuration resolution</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#112-training">11.2 Training</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#113-inference">11.3 Inference</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#114-evaluation">11.4 Evaluation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#115-benchmarking">11.5 Benchmarking</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#116-batch-training">11.6 Batch training</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#117-catalogue-and-diagnostic-commands">11.7 Catalogue and diagnostic commands</a><br>
<a href="#12-configuration-and-environment-variables">12. Configuration and environment variables</a><br>
<a href="#13-outputs-and-reproducibility">13. Outputs and reproducibility</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#131-output-tree">13.1 Output tree</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#132-reproducibility-checklist">13.2 Reproducibility checklist</a><br>
<a href="#14-troubleshooting">14. Troubleshooting</a><br>
<a href="#15-tools-reference">15. Tools reference</a><br>
<a href="#16-terminology">16. Terminology</a><br>
<a href="#17-additional-documentation">17. Additional documentation</a><br>

---

## 1. Overview

- **One interface, many backends.** Every backend implements the same interface and declares its
  capabilities; the CLI, web interface, evaluator, and profiler read that declaration.
- **Model manifest.** [`configs/registry/models.yaml`](configs/registry/models.yaml) is the
  catalogue of published models (method, task, domain, training corpus, evaluation corpora,
  configuration, weight source). One new entry updates every consumer.
- **Four task families**, four evaluators, declared metric keys
  ([Section 2](#2-application-domains-and-task-families)). A metric that cannot be computed is
  reported as `unavailable`.
- **Provenance.** Each training run appends to `artifacts/models/weight_manifest.jsonl`; each
  evaluation appends to `runs/leaderboard_log.jsonl`.

### 1.1 Weights

- **Distributed:** the textile weights, WinCLIP, and MoECLIP
  ([Section 9.2](#92-downloading-the-distributed-weights)).
- **Trained by the user:** the nine `general` training slots, which declare no weight
  ([Section 9.3](#93-training-a-general-domain-weight)).

## 2. Application domains and task families

### 2.1 Application domains

Two application domains: `general` (domain-independent; most of the platform) and `textile`
([Section 7.1](#71-application-domains)); further groups such as `pcb` are added the same way.
Only `textile` provides pretrained weights; any other domain requires training
([Section 9.3](#93-training-a-general-domain-weight)).

### 2.2 The four task families

Four task families, one interface. A backend fills only the fields its capability declaration
reports.

<table align="center">
  <thead>
    <tr>
      <th>Task family</th>
      <th>Required annotation</th>
      <th>Prediction fields</th>
      <th>Evaluator</th>
      <th>Headline metric</th>
      <th>Metric keys</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Supervised defect detection</td>
      <td><code>boxes</code>, <code>labels</code></td>
      <td><code>boxes</code>, <code>labels</code>, <code>scores</code></td>
      <td><code>detection</code></td>
      <td><code>map</code></td>
      <td><code>map</code>, <code>map_50</code>, <code>map_75</code>, <code>map_small</code>, <code>map_medium</code>, <code>map_large</code>, <code>mar_1</code>, <code>mar_10</code>, <code>mar_100</code>, <code>precision_at_threshold</code>, <code>recall_at_threshold</code>, <code>f1_at_threshold</code>, <code>recall_small</code>, <code>recall_normal</code></td>
    </tr>
    <tr>
      <td>Supervised defect segmentation</td>
      <td><code>masks</code></td>
      <td><code>masks</code></td>
      <td><code>segmentation</code></td>
      <td><code>miou</code></td>
      <td><code>miou</code>, <code>dice</code>, <code>pixel_f1</code>, <code>num_evaluated</code>, <code>num_skipped_empty</code></td>
    </tr>
    <tr>
      <td>Industrial anomaly detection</td>
      <td><code>is_anomalous</code></td>
      <td><code>anomaly_score</code>, and <code>anomaly_map</code> when supported</td>
      <td><code>anomaly</code></td>
      <td><code>image_auroc</code></td>
      <td><code>image_auroc</code>, <code>image_f1</code>, <code>image_precision</code>, <code>image_recall</code>, <code>image_threshold</code>, <code>pixel_auroc</code>, <code>pixel_f1</code>, <code>pixel_aupro</code>, <code>iap</code></td>
    </tr>
    <tr>
      <td>Production-line anomaly detection</td>
      <td><code>is_anomalous</code>, and a unit length per sample</td>
      <td><code>anomaly_score</code></td>
      <td><code>industrial</code></td>
      <td>Not declared</td>
      <td><code>under_detection_rate</code>, <code>over_detection_rate</code>, <code>chosen_threshold</code>, <code>num_alarms</code>, <code>num_samples</code>, <code>alarms_per_unit_length</code></td>
    </tr>
  </tbody>
</table>

The sample task selects the evaluator; `--task` overrides it ([Section 11.4](#114-evaluation)).
`industrial` has no published model identifier and is reachable through the Python API only.

### 2.3 Metric availability rules

1. Image-level anomaly metrics: sample label + prediction score.
2. Pixel-level anomaly metrics: ground-truth mask + model-produced anomaly map. A model that does
   not declare `anomaly_map` (e.g. GANomaly) produces no map.
3. `instance_segmentation`: scored by the `segmentation` evaluator over a unioned binary mask.

---

## 3. Requirements

### 3.1 Software

<table align="center">
  <thead>
    <tr>
      <th>Item</th>
      <th>Requirement</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>Python</strong></td><td><code>≥3.10</code>; <code>3.11–3.13</code> for training; this document uses <code>3.12</code></td></tr>
    <tr><td><strong>Environment manager</strong></td><td>Conda</td></tr>
    <tr><td><strong>Operating system</strong></td><td>macOS, Linux</td></tr>
    <tr><td><strong>GPU</strong></td><td>optional for inference; NVIDIA/CUDA for training</td></tr>
    <tr><td><strong>CUDA</strong></td><td>only for NVIDIA acceleration and CUDA-only extras</td></tr>
    <tr><td><strong>Git</strong></td><td>required (3 submodules)</td></tr>
  </tbody>
</table>

### 3.2 Hardware profiles

Recommendations, not minimums. On a constrained machine lower the batch size and use
`--mode test`.

<table align="center">
  <thead>
    <tr>
      <th>Workflow</th>
      <th>CPU / RAM</th>
      <th>GPU</th>
      <th>Disk</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Catalogue inspection, test suite, web interface layout</td><td>≥4 cores, 16 GB</td><td>—</td><td>10 GB + weights</td></tr>
    <tr><td>Minimal Working Example inference</td><td>≥8 cores, 16 GB</td><td>optional</td><td>weights + dataset</td></tr>
    <tr><td>Full textile training</td><td>≥8 cores, 32 GB</td><td>NVIDIA recommended</td><td>dataset + run artifacts, tens of GB</td></tr>
    <tr><td>Full benchmark suite</td><td>≥16 cores, 32–64 GB</td><td>recommended</td><td>all datasets, weights, maps, exports, logs</td></tr>
  </tbody>
</table>

---

## 4. Installation

### 4.1 Obtain the source and initialise components

```bash
git clone --recurse-submodules https://github.com/LINC-BIT/AnomalyDetection.git
cd AnomalyDetection
```

For a clone without `--recurse-submodules`:

```bash
git submodule update --init --recursive
git submodule status
```

`git submodule status` prints three lines — `components/anomalydiffusion`, `components/dinomaly`,
`components/moeclip` — each at the pinned revision. Do not clone a component repository manually;
`.gitmodules` and the Git link pin the tested revision.

### 4.2 Create the Python environment

```bash
conda create -n anomalib_env python=3.12 -y
conda activate anomalib_env
python --version
```

Prompt prefix: `(anomalib_env)`.

### 4.3 Install dependencies

<table align="center">
  <thead>
    <tr>
      <th>Dependency set</th>
      <th>Contents</th>
      <th>Installation command</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Complete</td><td>six backends; profiling, quantization, and test dependencies; installs the project (<code>-e .</code>)</td><td><code>python -m pip install -r requirements-full.txt</code></td></tr>
    <tr><td>Lean</td><td>web interface and lightweight inference; the project install is separate</td><td><code>python -m pip install -r requirements.txt</code><br><code>python -m pip install --no-deps --no-build-isolation -e .</code></td></tr>
  </tbody>
</table>

Optional groups (`pyproject.toml`):

```bash
python -m pip install -e ".[mambaad-cuda]"           # CUDA-only fused selective-scan kernel
python -m pip install -e ".[profiling-onnxruntime]"  # ONNX Runtime profiler
python -m pip install -e ".[profiling-flops]"        # FLOPs counter
python -m pip install -e ".[profiling-power-nvidia]" # NVIDIA power draw through NVML
python -m pip install -e ".[profiling-tensorrt]"     # TensorRT profiler; NVIDIA hardware only
python -m pip install -e ".[quantization]"           # fp16 and INT8 ONNX quantization
```

`make install` and `make install-ui`: the same two dependency sets.

### 4.4 Verify the installation

No dataset and no weight required. `python examples/01_inspect_catalog.py` is in
[Section 5.2](#52-step-2-verify-the-installation-without-data); the other three commands:

```bash
adh list
```

With all six frameworks installed:

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

`known`: all supported backends; `available`: importable on this machine.

```bash
adh inventory
```

JSON object with keys `models` and `datasets`; each `models` entry is one manifest record.

```bash
adh doctor
```

JSON object keyed by backend; `trainable_now` is true when the framework and a suitable dataset
are present; `reason` names the selected dataset.

```json
{
  "backends": {
    "anomalib": {
      "framework_installed": true,
      "dataset_kind": "one-class",
      "dataset": "fabric-defects",
      "trainable_now": true,
      "reason": "No dataset was requested; picked 'fabric-defects' (staged). Other staged alternatives: fabric-train, mvtec-ad, mvtec-loco, raw-fabric, tianchi, tilda-400, visa, zju-leaper."
    }
  }
}
```

One entry per backend; the selected dataset depends on what is staged locally.

---

## 5. Minimal Working Example

### 5.1 Step 1: Install the platform

Clone URL: `https://github.com/LINC-BIT/AnomalyDetection.git` (no mirror). Full procedure:
[Section 4](#4-installation).

```bash
git clone --recurse-submodules https://github.com/LINC-BIT/AnomalyDetection.git
cd AnomalyDetection
conda create -n anomalib_env python=3.12 -y
conda activate anomalib_env
git submodule update --init --recursive
python -m pip install -r requirements-full.txt
```

### 5.2 Step 2: Verify the installation without data

No dataset and no weight required.

```bash
python examples/01_inspect_catalog.py
adh list
adh inventory
adh doctor
```

On the development machine:

```text
AnomalyDetection version: 0.2.0
Registered models: 29
Registered datasets: 9
Models: yolov8n, yolov8s, yolo11n, fasterrcnn_resnet50_fpn, cascadercnn_resnet50_fpn, detr_resnet50, maskrcnn_resnet50_fpn, unetplusplus_resnet34, deeplabv3plus_resnet50, PatchCore, PaDiM, RD4AD, EfficientAD, SuperSimpleNet, STFPM, GANomaly, WinCLIP, Dinomaly, MoECLIP, MambaAD, PatchCore_general, PaDiM_general, RD4AD_general, EfficientAD_general, SuperSimpleNet_general, STFPM_general, GANomaly_general, Dinomaly_general, MambaAD_general
Datasets: fabric-defects, fabric-train, mvtec-ad, mvtec-loco, raw-fabric, tianchi, tilda-400, visa, zju-leaper
```

Other expected outputs: [Section 4.4](#44-verify-the-installation). On failure:
[Section 14](#14-troubleshooting).

### 5.3 Step 3: Run inference in the web interface

**Launch** ([Section 10.1](#101-launch)):

```bash
adh-ui
```

**Open** `http://127.0.0.1:6008`.

**Task type:** `Anomaly detection`.

<p align="center"><img src="docs/images/img1.png" alt="Select the task type" width="80%"></p>

**Application domain:** `general` (the model below belongs to it).

<p align="center"><img src="docs/images/img2.png" alt="Select the application domain" width="80%"></p>

**Model:** `WinCLIP · LAION-400M zero-shot`; the panel must state domain `general` and slot
`general/artifacts/models/published/WinCLIP.ckpt`.

<p align="center"><img src="docs/images/img3.png" alt="Select the model" width="80%"></p>

If that slot is empty locally, substitute any identifier whose `adh inventory` `weight_status` is
`file` or `symlink` ([Section 9.4](#94-verifying-weight-availability)).

**Load model:** wait for the loaded status.

<p align="center"><img src="docs/images/img4.png" alt="Load the model" width="80%"></p>

**Dataset:** `ZJU-Leaper`, any pattern or **All textures**, split `test`, **Full-shot**, then
**Load random images**.

<p align="center"><img src="docs/images/img5.png" alt="Select the dataset and the split" width="80%"></p>

**Run detection:** wait for completion.

<p align="center"><img src="docs/images/img6.png" alt="Run detection" width="80%"></p>

Result: gallery of images with per-image anomaly scores, plus the heat map when the model reports
one.

Reference scores, runtime, and a completed-run capture: not provided.

---

## 6. Directory layout

```text
AnomalyDetection/
├── src/fabric_defect_hub/            # reusable SDK; the import name is historical
│   ├── application/                  # business services used by the CLI and the web interface
│   ├── core/                         # registries, types, provenance, configuration
│   ├── datasets/                     # DatasetAdapter implementations
│   ├── models/                       # ModelAdapter implementations, one package per backend
│   ├── evaluation/                   # backend-independent metric computation
│   ├── profiling/                    # runtime and resource measurement
│   ├── quantization/                 # post-training ONNX and TensorRT quantization
│   ├── recipes/                      # optimization recipes
│   ├── reporting/                    # run log, tables, LaTeX export, training curves
│   ├── strategies/                   # sparse subsampling, tiling, test-time augmentation
│   └── web/                          # the web interface
├── components/                       # pinned upstream research checkouts (Git submodules)
├── configs/
│   ├── registry/models.yaml          # the model manifest: the published-model source of truth
│   ├── models/                       # model configurations (textile)
│   └── training_profile.yaml         # the shared training profile
├── datasets/
│   ├── textile/                      # textile datasets
│   └── general/                      # MVTec AD, MVTec LOCO, VisA, and auxiliary data
├── general/                          # general assets and artifacts
│   ├── configs/models/               # model configurations (general)
│   └── artifacts/models/published/   # general published slots
├── textile/                          # textile assets and artifacts
│   └── artifacts/models/published/   # textile published slots
├── examples/                         # Minimal Working Example scripts
├── tools/                            # conversion, export, download, and benchmark tools
├── tests/                            # the test suite
├── schemas/                          # JSON Schemas for samples, predictions, and results
└── docs/                             # focused guides and design records
```

---

## 7. Application domains, datasets, and models

### 7.1 Application domains

Two domains:

<table align="center">
  <thead>
    <tr>
      <th>Application domain</th>
      <th>Assets</th>
      <th>Datasets</th>
      <th>Published slots</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>textile</code></td><td><code>textile/</code>, plus the shared <code>datasets/textile/</code></td><td><code>zju-leaper</code>, <code>raw-fabric</code>, <code>tilda-400</code>, <code>fabric-defects</code>, <code>tianchi</code>, <code>fabric-train</code></td><td><code>textile/artifacts/models/published/</code></td></tr>
    <tr><td><code>general</code></td><td><code>general/</code>, plus the shared <code>datasets/general/</code></td><td><code>mvtec-ad</code>, <code>mvtec-loco</code>, <code>visa</code></td><td><code>general/artifacts/models/published/</code></td></tr>
  </tbody>
</table>

The manifest `domain` field records the domain of a weight; a training run publishes into the slot
of the training dataset's domain. Never relabel a weight across domains.

### 7.2 Supported models

The model manifest is authoritative: `adh inventory` (machine-readable), `adh models` (grouped by
backend).

The manifest declares **29 identifiers**: **18** `textile`, **11** `general`. One backend+variant
may appear in two domains — the nine `general` training slots. Distributed with weights: the
textile entries and `MoECLIP`; the nine `general` slots are not
([Section 9.3](#93-training-a-general-domain-weight)).

#### 7.2.1 Textile application domain

<table align="center">
  <thead>
    <tr>
      <th>Method</th>
      <th>Backend</th>
      <th>Task</th>
      <th>Model family</th>
      <th>Trained on</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>yolov8n</code></td><td><code>ultralytics</code></td><td><code>detection</code></td><td>YOLO</td><td>textile</td></tr>
    <tr><td><code>yolov8s</code></td><td><code>ultralytics</code></td><td><code>detection</code></td><td>YOLO</td><td>textile</td></tr>
    <tr><td><code>yolo11n</code></td><td><code>ultralytics</code></td><td><code>detection</code></td><td>YOLO</td><td>textile</td></tr>
    <tr><td><code>fasterrcnn_resnet50_fpn</code></td><td><code>torchvision</code></td><td><code>detection</code></td><td>Faster R-CNN</td><td>textile</td></tr>
    <tr><td><code>cascadercnn_resnet50_fpn</code></td><td><code>torchvision</code></td><td><code>detection</code></td><td>Cascade R-CNN</td><td>textile</td></tr>
    <tr><td><code>detr_resnet50</code></td><td><code>torchvision</code></td><td><code>detection</code></td><td>DETR</td><td>textile</td></tr>
    <tr><td><code>maskrcnn_resnet50_fpn</code></td><td><code>torchvision</code></td><td><code>instance_segmentation</code></td><td>Mask R-CNN</td><td>textile</td></tr>
    <tr><td><code>unetplusplus_resnet34</code></td><td><code>torchvision</code></td><td><code>segmentation</code></td><td>UNet++</td><td>textile</td></tr>
    <tr><td><code>deeplabv3plus_resnet50</code></td><td><code>torchvision</code></td><td><code>segmentation</code></td><td>DeepLabV3+</td><td>textile</td></tr>
    <tr><td><code>PatchCore</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>PatchCore</td><td>textile, normal-only</td></tr>
    <tr><td><code>PaDiM</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>PaDiM</td><td>textile, normal-only</td></tr>
    <tr><td><code>RD4AD</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>Reverse Distillation</td><td>textile, normal-only</td></tr>
    <tr><td><code>EfficientAD</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>EfficientAD</td><td>textile, normal-only</td></tr>
    <tr><td><code>SuperSimpleNet</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>SuperSimpleNet</td><td>textile, normal-only</td></tr>
    <tr><td><code>STFPM</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>STFPM</td><td>textile, normal-only</td></tr>
    <tr><td><code>GANomaly</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>GANomaly</td><td>textile, normal-only</td></tr>
    <tr><td><code>Dinomaly</code></td><td><code>dinomaly</code></td><td><code>anomaly</code></td><td>Dinomaly</td><td>textile, normal-only</td></tr>
    <tr><td><code>MambaAD</code></td><td><code>mambaad</code></td><td><code>anomaly</code></td><td>MambaAD</td><td>textile, normal-only</td></tr>
  </tbody>
</table>

#### 7.2.2 General application domain

<table align="center">
  <thead>
    <tr>
      <th>Method</th>
      <th>Backend</th>
      <th>Task</th>
      <th>Model family</th>
      <th>Trained on</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>WinCLIP</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>WinCLIP</td><td>LAION-400M, zero-shot</td></tr>
    <tr><td><code>MoECLIP</code></td><td><code>moeclip</code></td><td><code>anomaly</code></td><td>MoECLIP</td><td>MVTec AD, auxiliary</td></tr>
    <tr><td><code>PatchCore_general</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>PatchCore</td><td>not trained</td></tr>
    <tr><td><code>PaDiM_general</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>PaDiM</td><td>not trained</td></tr>
    <tr><td><code>RD4AD_general</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>Reverse Distillation</td><td>not trained</td></tr>
    <tr><td><code>EfficientAD_general</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>EfficientAD</td><td>not trained</td></tr>
    <tr><td><code>SuperSimpleNet_general</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>SuperSimpleNet</td><td>not trained</td></tr>
    <tr><td><code>STFPM_general</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>STFPM</td><td>not trained</td></tr>
    <tr><td><code>GANomaly_general</code></td><td><code>anomalib</code></td><td><code>anomaly</code></td><td>GANomaly</td><td>not trained</td></tr>
    <tr><td><code>Dinomaly_general</code></td><td><code>dinomaly</code></td><td><code>anomaly</code></td><td>Dinomaly</td><td>not trained</td></tr>
    <tr><td><code>MambaAD_general</code></td><td><code>mambaad</code></td><td><code>anomaly</code></td><td>MambaAD</td><td>not trained</td></tr>
  </tbody>
</table>

### 7.3 Registered datasets

Nine dataset identifiers are registered.

<table align="center">
  <thead>
    <tr>
      <th>Dataset identifier</th>
      <th>Application domain</th>
      <th>Default root</th>
      <th>Tasks</th>
      <th>Training roles</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>zju-leaper</code></td><td><code>textile</code></td><td><code>datasets/textile/ZJU-Leaper</code></td><td><code>detection</code>, <code>segmentation</code>, <code>anomaly</code></td><td><code>anomaly_train</code>, <code>detection_train</code>, <code>fabric_train_member</code></td></tr>
    <tr><td><code>raw-fabric</code></td><td><code>textile</code></td><td><code>datasets/textile/RAW_FABRID</code></td><td><code>anomaly</code>, <code>segmentation</code></td><td><code>anomaly_train</code>, <code>fabric_train_member</code></td></tr>
    <tr><td><code>tilda-400</code></td><td><code>textile</code></td><td><code>datasets/textile/TILDA_400</code></td><td><code>anomaly</code></td><td><code>anomaly_train</code>, <code>fabric_train_member</code></td></tr>
    <tr><td><code>fabric-defects</code></td><td><code>textile</code></td><td><code>datasets/textile/Fabric Defects Dataset</code></td><td><code>anomaly</code>, <code>segmentation</code></td><td><code>anomaly_train</code>, <code>fabric_train_member</code></td></tr>
    <tr><td><code>tianchi</code></td><td><code>textile</code></td><td><code>datasets/textile/tianchi</code></td><td><code>detection</code>, <code>anomaly</code></td><td><code>anomaly_train</code>, <code>detection_train</code>, <code>fabric_train_member</code></td></tr>
    <tr><td><code>fabric-train</code></td><td><code>textile</code></td><td><code>datasets/textile</code></td><td><code>detection</code>, <code>segmentation</code>, <code>anomaly</code></td><td><code>anomaly_train</code></td></tr>
    <tr><td><code>mvtec-ad</code></td><td><code>general</code></td><td><code>datasets/general/MVTec AD</code></td><td><code>anomaly</code>, <code>segmentation</code></td><td><code>anomaly_train</code>, <code>zero_shot_train</code></td></tr>
    <tr><td><code>mvtec-loco</code></td><td><code>general</code></td><td><code>datasets/general/MVTec LOCO</code></td><td><code>anomaly</code>, <code>segmentation</code></td><td><code>anomaly_train</code>, <code>zero_shot_train</code></td></tr>
    <tr><td><code>visa</code></td><td><code>general</code></td><td><code>datasets/general/VisA</code></td><td><code>anomaly</code>, <code>segmentation</code></td><td><code>anomaly_train</code>, <code>zero_shot_train</code></td></tr>
  </tbody>
</table>

Roles:

1. `anomaly_train` — normal-only split; trains `anomalib`, `dinomaly`, `mambaad`.
2. `zero_shot_train` — labelled anomalies; auxiliary corpus for `moeclip` (may combine with
   `anomaly_train`).
3. `detection_train` — bounding boxes; trains `ultralytics`, `torchvision`.
4. `fabric_train_member` — contributes samples to the `fabric-train` composite.

No role = evaluable, not trainable. `fabric-train` is not a member of its own union.

### 7.4 Choosing a model

<table align="center">
  <thead>
    <tr>
      <th>Question</th>
      <th>Command or field</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Which models are published?</td><td><code>adh inventory</code>, or the tables of <a href="#72-supported-models">Section 7.2</a></td></tr>
    <tr><td>Which model variants does a backend support?</td><td><code>adh models</code>, optionally restricted with <code>--backend</code></td></tr>
    <tr><td>Which backend is trainable on this machine right now?</td><td><code>adh doctor</code>, field <code>trainable_now</code></td></tr>
    <tr><td>Which dataset would a backend pick?</td><td><code>adh doctor</code>, field <code>dataset</code> and <code>reason</code></td></tr>
    <tr><td>Which model configurations can a training command resolve?</td><td><code>adh train --list</code></td></tr>
    <tr><td>Which model has a usable weight?</td><td><code>adh inventory</code>, field <code>weight_status</code></td></tr>
  </tbody>
</table>

---

## 8. Data preparation

### 8.1 Registered datasets and default roots

A dataset is available to the CLI when its directory exists at the declared default root, relative
to the repository root (roots: [Section 7.3](#73-registered-datasets), declared in
`core/dataset_capabilities.py`). `--dataset-root` overrides the root for one command.

### 8.2 Staging a dataset manually

1. Create the declared default root.
2. Copy the dataset in, preserving the structure the adapter expects.
3. Confirm with `adh doctor`, then a training run in `test` shot mode.

```bash
mkdir -p "datasets/general/MVTec LOCO"
# Copy the extracted dataset into that directory.
adh doctor
```

Each adapter documents its expected structure in its module docstring.

<table align="center">
  <thead>
    <tr>
      <th>Dataset identifier</th>
      <th>Adapter module</th>
      <th>Automated download</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>zju-leaper</code></td><td><code>datasets/zju_leaper.py</code></td><td>Available</td></tr>
    <tr><td><code>raw-fabric</code></td><td><code>datasets/raw_fabric.py</code></td><td>Not available; stage manually</td></tr>
    <tr><td><code>tilda-400</code></td><td><code>datasets/tilda.py</code></td><td>Not available; stage manually</td></tr>
    <tr><td><code>fabric-defects</code></td><td><code>datasets/fabric_defects.py</code></td><td>Not available; stage manually</td></tr>
    <tr><td><code>tianchi</code></td><td><code>datasets/tianchi.py</code></td><td>Not available; stage manually</td></tr>
    <tr><td><code>fabric-train</code></td><td><code>datasets/fabric_train.py</code></td><td>Composite; no separate download</td></tr>
    <tr><td><code>mvtec-ad</code></td><td><code>datasets/mvtec_ad.py</code></td><td>Available</td></tr>
    <tr><td><code>mvtec-loco</code></td><td><code>datasets/mvtec_loco.py</code></td><td>Not available; stage manually</td></tr>
    <tr><td><code>visa</code></td><td><code>datasets/visa.py</code></td><td>Available</td></tr>
  </tbody>
</table>

Acquisition procedures, archive sizes, licenses, and download dates: not provided.

### 8.3 Automated download

`download_datasets.py <dataset id> --root <root> [--category <name>]`.

```bash
python tools/download_datasets.py mvtec-ad --root "datasets/general/MVTec AD" --category bottle
python tools/download_datasets.py visa --root datasets/general/VisA --category capsules
python tools/download_datasets.py zju-leaper --root datasets/textile/ZJU-Leaper
```

Pass the declared root exactly, including the space in `MVTec AD`; any other spelling is outside
the registered root and `adh doctor` reports the dataset as not staged.

`zju-leaper` also accepts `--repo-id` (default `AnupamaBandara/ZLU_Leaper`).

A downloaded `general` dataset is the first step of training a `general` model;
[Section 9.3](#93-training-a-general-domain-weight) goes through to a published weight.

### 8.4 Verifying data availability

```bash
adh doctor
```

For every backend whose dataset kind is staged: `dataset` names it and `trainable_now` is `true`.

A smaller count than below means a partial download.

<table align="center">
  <thead>
    <tr>
      <th>Dataset identifier</th>
      <th>Unit</th>
      <th>Expected count</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>mvtec-ad</code></td><td>categories</td><td>15</td></tr>
    <tr><td><code>mvtec-loco</code></td><td>categories</td><td>5</td></tr>
    <tr><td><code>visa</code></td><td>classes</td><td>12</td></tr>
    <tr><td><code>zju-leaper</code></td><td>patterns</td><td>19</td></tr>
  </tbody>
</table>

### 8.5 Public dataset sources

<table align="center">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Source</th>
      <th>Automated download</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>MVTec AD</td><td><a href="https://www.mvtec.com/company/research/datasets/mvtec-ad">official page</a></td><td><code>mvtec-ad</code></td></tr>
    <tr><td>MVTec LOCO</td><td><a href="https://www.mvtec.com/company/research/datasets/mvtec-loco">official page</a></td><td>Not available</td></tr>
    <tr><td>VisA</td><td><a href="https://github.com/amazon-science/spot-diff">official repository</a></td><td><code>visa</code></td></tr>
    <tr><td>ZJU-Leaper</td><td><a href="https://huggingface.co/datasets/AnupamaBandara/ZLU_Leaper">AnupamaBandara/ZLU_Leaper</a></td><td><code>zju-leaper</code></td></tr>
    <tr><td>RAW-FABRID</td><td><a href="https://www.mdpi.com/2306-5729/11/5/116">MDPI Data 11(5):116</a></td><td>Not available</td></tr>
    <tr><td>TILDA-400</td><td>Not provided</td><td>Not available</td></tr>
    <tr><td>Fabric Defects Dataset</td><td>Not provided</td><td>Not available</td></tr>
    <tr><td>Tianchi Guangdong fabric defect challenge</td><td>Not provided</td><td>Not available</td></tr>
  </tbody>
</table>

---

## 9. Weight preparation

Weights are immutable runtime artifacts and are not committed to Git. The manifest declares where
each weight resolves: a slot without a weight is `missing`; a weight without a conforming manifest
entry is unsupported.

19 weights are downloaded ([Section 9.2](#92-downloading-the-distributed-weights)); the nine
`general` training slots are produced by training
([Section 9.3](#93-training-a-general-domain-weight)).

### 9.1 Published-slot layout

A published slot resolves under `<application domain>/artifacts/models/published/`.

```text
textile/artifacts/models/published/<model identifier><extension>
general/artifacts/models/published/<model identifier><extension>
```

The extension is determined by the backend: `.pt` for `ultralytics` and `torchvision`, `.ckpt`
for `anomalib`, and `.pth` for `dinomaly`, `moeclip`, and `mambaad`.

`adh inventory` reports each slot in `weight_status`; usable: `file`, `symlink`.

With publishing enabled, a training run replaces the slot with a relative symlink to the trained
artifact; a tree copied from another machine holds regular files, still usable.

### 9.2 Downloading the distributed weights

The 19 distributed weights are hosted on a Hugging Face Hub repository.

The download tool takes the repository identifier and the filename inside the repository as
positional arguments and requires `--output`.

```bash
python tools/download_weights.py AuroraLeeeeee/AnomalyDetection-textile-weights \
  textile/artifacts/models/published/yolov8n.pt \
  --output textile/artifacts/models/published/yolov8n.pt
python tools/download_weights.py AuroraLeeeeee/AnomalyDetection-textile-weights \
  textile/artifacts/models/published/yolov8s.pt \
  --output textile/artifacts/models/published/yolov8s.pt
python tools/download_weights.py AuroraLeeeeee/AnomalyDetection-textile-weights \
  textile/artifacts/models/published/yolo11n.pt \
  --output textile/artifacts/models/published/yolo11n.pt
```

Repeat for every other file, using the exact `weight` filename from `adh inventory`. Do not move a
weight across domains.

Repository: `AuroraLeeeeee/AnomalyDetection-textile-weights` (holds both domains). Pinned revisions
and SHA-256 checksums: not provided.

Two identifiers:

1. `WinCLIP` — LAION-400M zero-shot, `k_shot = 0`; the slot is a metadata handle (no serialized
   weight) and the adapter reconstructs the model from it.
2. `MoECLIP` — trained on an auxiliary corpus; training and evaluation corpora are different
   datasets, both recorded in the manifest.

### 9.3 Training a general-domain weight

The nine `general` training slots ([Section 1.1](#11-weights)) declare no weight and use the same
`adh train` entry point. The example trains `PatchCore` on MVTec AD `bottle` and publishes it into
`PatchCore_general`.

**Step 1. Stage the dataset** ([Section 8.3](#83-automated-download)); pass the declared root
exactly.

```bash
python tools/download_datasets.py mvtec-ad --root "datasets/general/MVTec AD" --category bottle
```

Without `--category`: every category. VisA: `visa`, `datasets/general/VisA`.

**Step 2. Check the backend and dataset.**

```bash
adh doctor
```

The `anomalib` entry: `trainable_now: true`, dataset `mvtec-ad`.

**Step 3. Train.** Publishing is on by default (`--no-publish` disables it). `--variant` selects
the anomalib method, so one configuration file trains any of them.

```bash
adh train general/configs/models/anomalib.yaml \
  --variant PatchCore \
  --dataset mvtec-ad \
  --category bottle
```

The run writes an artifact under `artifacts/models/`, appends to the weight manifest, and points
`general/artifacts/models/published/PatchCore_general.ckpt` at it
([Section 11.2](#112-training)).

**Step 4. Check the weight.**

```bash
adh inventory
```

`PatchCore_general`: `weight_status: symlink` (or `file`); `missing` before step 3.

**Step 5. Use it.** In `adh-ui`: task type `Anomaly detection`, domain `general`, model
`PatchCore · General`, then load the model
([Section 5.3](#53-step-3-run-inference-in-the-web-interface)). From the command line:

```bash
adh evaluate general/configs/models/anomalib.yaml \
  --weights general/artifacts/models/published/PatchCore_general.ckpt \
  --dataset mvtec-ad \
  --category bottle \
  --split test \
  --output-dir artifacts/runtime/anomaly_maps
```

**Other methods and datasets.** `PatchCore` and `PaDiM` are feature-based and fast; `RD4AD`,
`EfficientAD`, `SuperSimpleNet`, `STFPM`, `GANomaly`, `Dinomaly`, and `MambaAD` train a network and
need a CUDA GPU for practical runtimes.

```bash
# The same corpus, a different method.
adh train general/configs/models/anomalib.yaml --variant PaDiM --dataset mvtec-ad --category bottle

# A different general dataset.
adh train general/configs/models/anomalib.yaml --variant PatchCore --dataset visa --category capsules

# Dinomaly and MambaAD have their own general configurations.
adh train general/configs/models/dinomaly.yaml --dataset mvtec-ad --category bottle
adh train general/configs/models/mambaad.yaml --dataset visa --category capsules
```

`--mode test --no-publish`: eight-image wiring check, no usable weight, no slot touched.

Reference wall-clock time and image AUROC: not provided.

### 9.4 Verifying weight availability

```bash
adh inventory
```

On the development machine: 19 slots hold a weight, 10 are empty.

<table align="center">
  <thead>
    <tr>
      <th>Published slot</th>
      <th>State on the development machine</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>The 17 textile slots <code>yolov8n</code>, <code>yolov8s</code>, <code>yolo11n</code>, <code>fasterrcnn_resnet50_fpn</code>, <code>cascadercnn_resnet50_fpn</code>, <code>detr_resnet50</code>, <code>maskrcnn_resnet50_fpn</code>, <code>unetplusplus_resnet34</code>, <code>deeplabv3plus_resnet50</code>, <code>PatchCore</code>, <code>PaDiM</code>, <code>RD4AD</code>, <code>EfficientAD</code>, <code>SuperSimpleNet</code>, <code>STFPM</code>, <code>GANomaly</code>, and <code>Dinomaly</code></td><td><code>file</code></td></tr>
    <tr><td><code>general/artifacts/models/published/WinCLIP.ckpt</code></td><td><code>file</code>; a metadata handle</td></tr>
    <tr><td><code>general/artifacts/models/published/MoECLIP.pth</code></td><td><code>file</code></td></tr>
    <tr><td><code>textile/artifacts/models/published/MambaAD.pth</code></td><td><code>missing</code></td></tr>
    <tr><td>The nine general training slots</td><td><code>missing</code></td></tr>
  </tbody>
</table>

Resolve a missing weight by training the identifier, or place the weight at the exact path from
`adh inventory`. Never rename a weight of a different architecture.

### 9.5 The weight manifest

Each training run that produces an artifact appends a provenance record to
`artifacts/models/weight_manifest.jsonl` and writes the resolved configuration to
`artifacts/models/records/<record identifier>.config.json`. The record keeps both the artifact and
the slot, so replacing a slot never destroys provenance
([Section 13.2](#132-reproducibility-checklist)).

---

## 10. Web interface

### 10.1 Launch

```bash
conda activate anomalib_env
adh-ui
```

Default URL `http://127.0.0.1:6008`; the server listens on all interfaces.

```bash
GRADIO_SERVER_PORT=7860 adh-ui
```

If the port is in use, `adh-ui` names the holding process and offers reuse or another port. A
missing weight is never downloaded implicitly; the exact expected path is shown.

### 10.2 Model session

Order:

1. **Task type** — `Defect detection` or `Anomaly detection`.
2. **Application domain** — `textile` or `general`.
3. **Model** — identifiers declaring the selected task and domain.

The panel states method, training corpus, training split, slot status, and the declared prediction
fields. Select **Load model** before inference
([Section 5.3](#53-step-3-run-inference-in-the-web-interface)).

### 10.3 Benchmark

**Benchmark** evaluates one or more identifiers on one dataset: select dataset, category or
pattern, shot mode, identifiers, optionally the profiling or resolution sweep, then **Run
benchmark**.

Two result categories:

<table align="center">
  <thead>
    <tr>
      <th>Category</th>
      <th>Table</th>
      <th>Contents</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>technical</code></td><td><code>image_level</code></td><td>Image AUROC, image F1, image precision, and image recall.</td></tr>
    <tr><td><code>technical</code></td><td><code>pixel_level</code></td><td>Pixel AUROC, AUPRO, and IAP, when a ground-truth mask and an anomaly map are available.</td></tr>
    <tr><td><code>technical</code></td><td><code>instance_level</code></td><td>Detection boxes and instance metrics, for the models that produce boxes.</td></tr>
    <tr><td><code>technical</code></td><td><code>cross_domain</code></td><td>Scores on a held-out dataset or pattern.</td></tr>
    <tr><td><code>overhead</code></td><td><code>compute</code></td><td>Wall-clock time, frames per second, latency, FLOPs, and the edge-deployment index when the selected profiler provides it.</td></tr>
    <tr><td><code>overhead</code></td><td><code>memory</code></td><td>Peak and average memory, when the selected profiler provides them.</td></tr>
    <tr><td><code>overhead</code></td><td><code>communication</code></td><td>Exported-model transfer-size proxy. Declared but not implemented.</td></tr>
  </tbody>
</table>

Uncomputable metrics: `unavailable`. Each run appends to `runs/leaderboard_log.jsonl`; maps go
under `artifacts/runtime/anomaly_maps/benchmark/`.

### 10.4 Run history

**Run history** reads a saved JSON/JSONL report: enter the path, select **Refresh**, optionally
filter by metric. Columns: timestamp, model identifier, dataset identifier, metric values, report
path. Reads existing files only.

### 10.5 Recorded demonstrations

<table align="center">
  <thead>
    <tr>
      <th>Demonstration</th>
      <th>Recording</th>
      <th>Contents</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Web interface demonstration</td><td><a href="docs/videos/detection.mp4">detection.mp4</a></td><td>Task selection, application-domain selection, model selection, dataset selection, image loading, and detection output.</td></tr>
    <tr><td>Benchmark demonstration</td><td><a href="docs/videos/benchmark.mp4">benchmark.mp4</a></td><td>Benchmark execution and run-history reading.</td></tr>
  </tbody>
</table>

Recorded 2026-07-20. Commit revision: not provided.

---

## 11. Command-line workflows

### 11.1 Model configuration resolution

`train`, `predict`, and `evaluate` resolve their first positional argument three ways:

1. As a path to a model configuration, for example `configs/models/ultralytics_example.yaml`.
2. As a filename stem under `--config-dir`, for example `ultralytics_example`.
3. As a model keyword matched against the `model.variant` field or the `model.name` field of
   every model configuration under `--config-dir`, for example `yolov8n` or `patchcore`.

`--config-dir` default: `configs/models`. These commands list different objects:

<table align="center">
  <thead>
    <tr>
      <th>Command</th>
      <th>Lists</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>adh train --list</code></td><td>The 13 model configurations under <code>--config-dir</code>.</td></tr>
    <tr><td><code>adh recipes</code></td><td>Registered optimization recipes with paper reference and hyperparameters.</td></tr>
    <tr><td><code>adh models</code></td><td>Model variants per backend; not the published models (<code>adh inventory</code>).</td></tr>
  </tbody>
</table>

### 11.2 Training

A training run takes a model configuration, modified by `--dataset`, `--variant`, `--mode`, and
`--set`. `--set` overrides by dotted path with the highest priority.

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

`moeclip` uses one dataset for auxiliary training and another for evaluation.

```bash
adh train moeclip \
  --dataset mvtec-ad \
  --test-dataset zju-leaper \
  --mode test \
  --no-publish
```

`--mode test`: eight-image wiring check, not a usable weight; combine with `--no-publish`.

Publishing is on by default: a successful run whose identifier is in the manifest replaces that
identifier's published slot, which the web interface reads.

```bash
# Smoke run; no slot touched.
adh train patchcore --dataset zju-leaper --mode test --no-publish
```

Result keys: `backend`, `resolved_config`, `resolved_variant`, `metrics`, `trained_artifact`,
`registered_artifact`, `published_path`, `weight_manifest_path`, `exports`.

### 11.3 Inference

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

`--output-dir` writes one anomaly map per sample to
`<output directory>/<sample identifier>.npy`, only for models that declare `anomaly_map`.
`--output` writes predictions as a JSON array (`schemas/prediction.schema.json`).

Single-image result:

```json
{
  "backend": "anomalib",
  "resolved_config": "configs/models/patchcore_textile.yaml",
  "variant": "patchcore",
  "num_predictions": 1,
  "predictions": [
    {
      "sample_id": "image",
      "boxes": null,
      "labels": null,
      "scores": null,
      "masks": null,
      "anomaly_score": 0.0,
      "anomaly_map": "artifacts/runtime/anomaly_maps/image.npy"
    }
  ]
}
```

Values are illustrative.

### 11.4 Evaluation

```bash
adh evaluate patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --dataset zju-leaper \
  --split test \
  --num-samples 32 \
  --output-dir artifacts/runtime/anomaly_maps
```

`--task` forces an evaluator instead of the sample task. `--output-dir` is required for
pixel-level metrics (they need a persisted anomaly map); without it, only image-level metrics are
scored.

Result keys: `backend`, `resolved_config`, `variant`, `sample_count`, `metrics`; `metrics` keys
are those of [Section 2.2](#22-the-four-task-families).

Cross-pattern robustness: score the same weight on held-out patterns, reduce the per-pattern
accuracy drops to one number.

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

`--cross-domain-metric` selects the metric (default: the task headline metric).
`--cross-domain-mode`: `worst` = mean over the largest drops, `best` = over the smallest; the mode
is echoed in the output.

Result: the direct-evaluation JSON plus `cross_domain`. Unscorable patterns are skipped, not
counted as zero degradation.

### 11.5 Benchmarking

A benchmark configuration has top-level keys `runs`, `output_dir`, `leaderboard`, `report_path`,
and `run_log_path`; `runs` is a non-empty list, one experiment per entry.

```bash
adh benchmark configs/archive/benchmark_example.yaml
```

The example scores `fasterrcnn_resnet50_fpn` on the ZJU-Leaper `test` split, writes under
`artifacts/benchmarks/example`, and reads the dataset root from `ZJU_LEAPER_ROOT` (set it first).

Result: JSON array, one element per run.

Expected leaderboard: not provided.

### 11.6 Batch training

`train-all` trains every manifest identifier in one resumable batch: one log and one state record
per identifier.

```bash
adh train-all --dry-run
adh train-all --only yolov8n PatchCore --mode test --no-publish
adh train-all --run-id <run-id> --resume
```

Result keys: `batch_state`, `succeeded`, `total`, `results`; the `batch_state` directory holds the
per-model state and log.

### 11.7 Catalogue and diagnostic commands

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
    <tr><td><code>adh run</code></td><td>Run a model or benchmark YAML configuration; backend inferred unless <code>--backend</code>.</td></tr>
    <tr><td><code>adh export-latex</code></td><td>Convert a benchmark result file into a LaTeX table.</td></tr>
  </tbody>
</table>

---

## 12. Configuration and environment variables

Two YAML files carry configuration:

- `configs/registry/models.yaml` — the model manifest (source of truth for published models).
- A model configuration under `configs/models/` or `general/configs/models/` — executable
  parameters of one run; the manifest refers to it by filename and the runtime resolves it under
  `configs/models/` unless an explicit path is given.

Never commit a secret or a machine-specific absolute path; use an environment variable.

<table align="center">
  <thead>
    <tr>
      <th>Environment variable</th>
      <th>Default value</th>
      <th>Effect</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>AD_DATASETS_ROOT</code></td><td><code>&lt;repository&gt;/datasets</code></td><td>Root of the dataset tree.</td></tr>
    <tr><td><code>AD_COMPONENTS_ROOT</code></td><td><code>&lt;repository&gt;/components</code></td><td>Root of the component checkouts.</td></tr>
    <tr><td><code>AD_TEXTILE_ROOT</code></td><td><code>&lt;repository&gt;/textile</code></td><td>Root of the textile assets and artifacts.</td></tr>
    <tr><td><code>AD_GENERAL_ROOT</code></td><td><code>&lt;repository&gt;/general</code></td><td>Root of the general assets and artifacts.</td></tr>
    <tr><td><code>AD_ARTIFACTS_ROOT</code></td><td><code>&lt;repository&gt;/textile/artifacts</code></td><td>Textile artifact root; the weight manifest and trained artifacts go under the repository-level <code>artifacts/models/</code> (<a href="#131-output-tree">Section 13.1</a>).</td></tr>
    <tr><td><code>AD_CONFIGS_ROOT</code></td><td><code>&lt;repository&gt;/configs</code></td><td>Root of the configuration tree.</td></tr>
    <tr><td><code>AD_RESULTS_ROOT</code></td><td><code>&lt;repository&gt;/results</code></td><td>Root of the results tree.</td></tr>
    <tr><td><code>GRADIO_SERVER_PORT</code></td><td><code>6008</code></td><td>Web interface port; also listens on all interfaces.</td></tr>
    <tr><td><code>FDH_PROGRESS</code></td><td><code>1</code></td><td><code>0</code>, <code>false</code>, <code>no</code>, or <code>off</code> disables progress lines.</td></tr>
    <tr><td><code>FDH_PROGRESS_INTERVAL</code></td><td><code>5.0</code></td><td>Seconds between progress lines; invalid values restore the default.</td></tr>
    <tr><td><code>FDH_MODEL_CACHE</code></td><td><code>artifacts/models</code></td><td>Directory searched by the cloud preflight tool for a missing weight.</td></tr>
    <tr><td><code>FDH_MAMBAAD_SCAN_BUDGET</code></td><td><code>64000000</code></td><td>Element budget of one chunk of the portable MambaAD selective scan.</td></tr>
    <tr><td><code>FDH_BATCH_RUN_ID</code></td><td>Unset</td><td>Batch run identifier, set by <code>adh train-all</code> for each child run; recorded in the weight manifest.</td></tr>
    <tr><td><code>FDH_BATCH_MODEL_KEY</code></td><td>Unset</td><td>Model identifier of a <code>train-all</code> child run; recorded in the weight manifest.</td></tr>
  </tbody>
</table>

`tools/run_full_benchmark.sh` reads a separate set of `FDH_*` variables (not runtime
configuration): `FDH_POWER_MODE` (required), `FDH_DATASET`, `FDH_DATASET_ROOT`, `FDH_MODELS`,
`FDH_PATTERN`, `FDH_HELD_OUT_PATTERNS`, `FDH_NUM_SAMPLES`, `FDH_MEASURED_RUNS`, `FDH_WARMUP_RUNS`,
`FDH_DEVICE`, `FDH_OUTPUT`, `FDH_ANOMALY_MAP_DIR`, `FDH_PYTHON`, `FDH_RUN_ID`. See the script
header.

The web interface also reads a per-dataset root override variable, falling back to the declared
default root; the CLI ignores these and uses the declared root or `--dataset-root`.

<table align="center">
  <thead>
    <tr>
      <th>Dataset label in the web interface</th>
      <th>Dataset identifier</th>
      <th>Root override variable</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>ZJU-Leaper</td><td><code>zju-leaper</code></td><td><code>ZJU_LEAPER_ROOT</code></td></tr>
    <tr><td>RAW-FABRID</td><td><code>raw-fabric</code></td><td><code>RAW_FABRIC_ROOT</code></td></tr>
    <tr><td>MVTec AD</td><td><code>mvtec-ad</code></td><td><code>MVTEC_AD_ROOT</code></td></tr>
    <tr><td>MVTec LOCO</td><td><code>mvtec-loco</code></td><td><code>MVTEC_LOCO_ROOT</code></td></tr>
    <tr><td>VisA</td><td><code>visa</code></td><td><code>VISA_ROOT</code></td></tr>
    <tr><td>TILDA-400</td><td><code>tilda-400</code></td><td><code>TILDA_400_ROOT</code></td></tr>
    <tr><td>Fabric Defects</td><td><code>fabric-defects</code></td><td><code>FABRIC_DEFECTS_ROOT</code></td></tr>
    <tr><td>Tianchi</td><td><code>tianchi</code></td><td><code>TIANCHI_ROOT</code></td></tr>
  </tbody>
</table>

---

## 13. Outputs and reproducibility

### 13.1 Output tree

```text
artifacts/
├── models/
│   ├── <trained artifact>                       # the run-specific weight
│   ├── weight_manifest.jsonl                    # append-only provenance records
│   └── records/
│       └── <record identifier>.config.json      # the resolved configuration of one run
├── benchmarks/
│   └── <benchmark output directory>/
│       ├── leaderboard.csv                      # when report_path is configured
│       └── <experiment identifier>/
│           ├── result.json                      # ExperimentResult
│           └── predictions.json                 # Prediction list
└── runtime/
    └── anomaly_maps/
        ├── <sample identifier>.npy              # adh predict --output-dir
        └── benchmark/<experiment identifier>/   # web interface benchmark

<application domain>/artifacts/models/published/
└── <model identifier><extension>                # the published slot
    └── <model identifier><extension>.metadata.json   # optional provenance sidecar

runs/
└── leaderboard_log.jsonl                        # append-only evaluation and benchmark log

results/
└── <user-specified output>.json                 # adh predict --output
```

### 13.2 Reproducibility checklist

A provenance record states timestamp, Git revision, working-tree state, host, platform, Python
environment, package versions, and component revisions; the same block is attached to the weight
manifest and the run log.

Before reporting a result, confirm:

1. Model identifier, model configuration, dataset identifier, split, shot mode, and sample count
   stated exactly.
2. Published-slot state was `file` or `symlink` at run start.
3. The provenance record was retained.
4. Anomaly maps retained if any pixel-level metric is reported.
5. Metric keys quoted exactly as the evaluator emits them, headline metric named.
6. Uncomputable metrics reported as unavailable, not zero.

Do not commit generated datasets, trained artifacts, anomaly maps, caches, or results. Distribute
a durable weight as in [Section 9.2](#92-downloading-the-distributed-weights).

---

## 14. Troubleshooting

<table align="center">
  <thead>
    <tr>
      <th>Symptom</th>
      <th>Treatment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>A dataset is reported as unavailable</strong></td>
      <td><code>adh doctor</code>; check the directory at the declared root is complete; <code>--dataset-root</code> selects another location for one command.</td>
    </tr>
    <tr>
      <td><strong>A component checkout is missing</strong></td>
      <td><code>git submodule update --init --recursive</code>, then <code>git submodule status</code>, then <code>adh doctor</code>.</td>
    </tr>
    <tr>
      <td><strong>Importing the web interface raises a SOCKS proxy error</strong></td>
      <td><code>python -m pip install -r requirements.txt</code> (provides <code>httpx[socks]</code>).</td>
    </tr>
    <tr>
      <td><strong>A weight is missing</strong></td>
      <td><code>adh inventory</code> → <code>weight</code>, <code>weight_status</code>; place the weight at the reported path. Never rename a weight of a different architecture.</td>
    </tr>
    <tr>
      <td><strong>The web interface displays an obsolete label</strong></td>
      <td>The manifest is read at process start: stop and restart <code>adh-ui</code>.</td>
    </tr>
    <tr>
      <td><strong>No anomaly heat map is displayed</strong></td>
      <td>Check the capability declaration; GANomaly is image-level only. For an <code>anomaly_map</code> model, supply an output directory and a mask-carrying dataset for pixel metrics.</td>
    </tr>
    <tr>
      <td><strong>Training runs out of memory</strong></td>
      <td><code>--mode test</code>, a smaller batch size via <code>--set</code>, a lower input resolution, or a smaller model variant. A CUDA accelerator must match the installed PyTorch and CUDA toolchain.</td>
    </tr>
    <tr>
      <td><strong>A general training run fails because the dataset is not a one-class training source</strong></td>
      <td>Check that the dataset declares the <code>anomaly_train</code> role (<code>core/dataset_capabilities.py</code>, reported by <code>adh doctor</code>).</td>
    </tr>
  </tbody>
</table>

---

## 15. Tools reference

<table align="center">
  <thead>
    <tr>
      <th>Tool</th>
      <th>Function</th>
      <th>Interface</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>tools/download_datasets.py</code></td><td>Download a registered dataset.</td><td><code>mvtec-ad</code>, <code>visa</code>, or <code>zju-leaper</code>; required <code>--root</code>; optional <code>--category</code> and <code>--repo-id</code></td></tr>
    <tr><td><code>tools/download_weights.py</code></td><td>Download a weight from Hugging Face Hub.</td><td>positional <code>repo_id</code> and <code>filename</code>; required <code>--output</code>; optional <code>--revision</code></td></tr>
    <tr><td><code>tools/export_model.py</code></td><td>Export a trained artifact, optionally building a TensorRT engine or applying post-training ONNX quantization.</td><td>required <code>--backend</code>, <code>--model</code>, <code>--artifact</code>, and <code>--target</code>; optional <code>--precision</code>, <code>--calibration-dir</code>, <code>--quantize-level</code>, and <code>--input-size</code></td></tr>
    <tr><td><code>tools/visualize_predictions.py</code></td><td>Render boxes, masks, and anomaly maps onto images.</td><td>positional <code>samples</code>, <code>predictions</code>, and <code>output_dir</code></td></tr>
    <tr><td><code>tools/run_metric_sweep.py</code></td><td>Measure every implemented metric over every model that has a weight.</td><td><code>--dataset</code>, <code>--groups</code>, <code>--models</code>, and <code>--output</code>; use <code>--list</code> to print the plan without measuring</td></tr>
    <tr><td><code>tools/run_full_benchmark.sh</code></td><td>Run the full post-training measurement, including the cross-domain sweep, from environment variables.</td><td>environment variables prefixed <code>FDH_</code>; notably <code>FDH_POWER_MODE</code>, which is required</td></tr>
    <tr><td><code>tools/collect_cloud_artifacts.py</code></td><td>Collect and optionally archive the artifacts of a batch run.</td><td>required <code>--run-id</code>; optional <code>--verify-only</code> and <code>--archive</code></td></tr>
    <tr><td><code>tools/preflight_cloud_models.py</code></td><td>Report, before a cloud run, which model identifiers are blocked by a missing path.</td><td>no arguments; prints a report</td></tr>
    <tr><td><code>tools/smoke_test_all_backends.py</code></td><td>Train every model identifier for one step, as a wiring check.</td><td>no arguments; starts training immediately</td></tr>
    <tr><td><code>tools/convert_annotations.py</code></td><td>Convert COCO detection annotations into a <code>Sample</code> JSON file.</td><td>positional <code>annotations</code> and <code>output</code>; optional <code>--image-root</code></td></tr>
    <tr><td><code>tools/plot_training_curves.py</code></td><td>Render SVG training curves from the YOLO <code>results.csv</code> file and the torchvision <code>history.csv</code> file.</td><td>positional <code>paths</code>, being history CSV files or directories searched recursively; optional <code>--output-dir</code></td></tr>
    <tr><td><code>tools/prune_artifacts.py</code></td><td>Inspect, and optionally apply, the storage cleanup of <code>artifacts/models/</code>.</td><td>optional <code>--dedupe-published</code>, <code>--prune-checkpoints</code>, <code>--keep</code>, <code>--run-root</code>, <code>--project-root</code>, and <code>--apply</code>. Without <code>--apply</code>, the tool prints the plan and writes nothing.</td></tr>
    <tr><td><code>tools/export_benchmark_csv.mjs</code></td><td>Convert the JSONL output of the metric sweep into two CSV files.</td><td>no arguments; reads <code>artifacts/runtime/full_sweep.jsonl</code> and writes <code>artifacts/runtime/benchmark_results_long.csv</code> and <code>artifacts/runtime/benchmark_model_summary.csv</code>. Requires the Node.js package <code>@oai/artifact-tool</code>.</td></tr>
  </tbody>
</table>

`preflight_cloud_models.py` and `smoke_test_all_backends.py` accept **no arguments** and act as
soon as they are invoked; **`--help` does not print usage**.

---

## 16. Terminology

`<application domain>` is a placeholder for either `textile` or `general`.

<table align="center">
  <thead>
    <tr>
      <th>Term</th>
      <th>Definition</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>application domain</strong></td><td>A data family and its assets: <code>textile</code>, <code>general</code>.</td></tr>
    <tr><td><strong>backend</strong></td><td>The execution framework: <code>ultralytics</code>, <code>torchvision</code>, <code>anomalib</code>, <code>dinomaly</code>, <code>moeclip</code>, <code>mambaad</code>.</td></tr>
    <tr><td><strong>model variant</strong></td><td>The architecture within a backend, e.g. <code>yolov8n</code>, <code>PatchCore</code>.</td></tr>
    <tr><td><strong>model identifier</strong></td><td>The stable id of one manifest entry, e.g. <code>PatchCore</code>, <code>PatchCore_general</code>.</td></tr>
    <tr><td><strong>model manifest</strong></td><td>The catalogue of published models: <code>configs/registry/models.yaml</code>.</td></tr>
    <tr><td><strong>model configuration</strong></td><td>Executable parameters of one run, e.g. <code>configs/models/ultralytics_example.yaml</code>.</td></tr>
    <tr><td><strong>optimization recipe</strong></td><td>A paper-anchored hyperparameter profile (<code>src/fabric_defect_hub/recipes/</code>); not a model configuration.</td></tr>
    <tr><td><strong>dataset adapter</strong></td><td>A <code>DatasetAdapter</code> subclass converting one source dataset into <code>Sample</code> objects.</td></tr>
    <tr><td><strong>sample</strong></td><td>One image with its annotations (<code>core.types.Sample</code>).</td></tr>
    <tr><td><strong>prediction</strong></td><td>One model output (<code>core.types.Prediction</code>).</td></tr>
    <tr><td><strong>task</strong></td><td>One of <code>detection</code>, <code>segmentation</code>, <code>instance_segmentation</code>, <code>anomaly</code>, <code>industrial</code>.</td></tr>
    <tr><td><strong>annotation</strong></td><td>A ground-truth field: <code>boxes</code>, <code>masks</code>, <code>labels</code>, <code>is_anomalous</code>, <code>anomaly_mask</code>.</td></tr>
    <tr><td><strong>capability declaration</strong></td><td>A <code>ModelCapabilities</code> or <code>DatasetCapabilities</code> value: which tasks and output fields are supported, without running the model or reading the dataset.</td></tr>
    <tr><td><strong>shot mode</strong></td><td>A training sample budget: <code>full</code>, <code>medium</code>, <code>few</code>, <code>test</code>.</td></tr>
    <tr><td><strong>split</strong></td><td><code>train</code> or <code>test</code>.</td></tr>
    <tr><td><strong>trained artifact</strong></td><td>The run-specific weight under <code>artifacts/models/</code> (<code>registered_artifact</code>).</td></tr>
    <tr><td><strong>published slot</strong></td><td>The fixed path <code>&lt;application domain&gt;/artifacts/models/published/&lt;model identifier&gt;&lt;extension&gt;</code>; a regular file or symlink.</td></tr>
    <tr><td><strong>weight</strong></td><td>A model parameter file.</td></tr>
    <tr><td><strong>weight manifest</strong></td><td>The append-only provenance log <code>artifacts/models/weight_manifest.jsonl</code>.</td></tr>
    <tr><td><strong>provenance record</strong></td><td>One line of the weight manifest.</td></tr>
    <tr><td><strong>run log</strong></td><td>The append-only evaluation log <code>runs/leaderboard_log.jsonl</code>.</td></tr>
    <tr><td><strong>anomaly map</strong></td><td>A per-pixel anomaly score array at <code>&lt;output directory&gt;/&lt;sample identifier&gt;.npy</code>.</td></tr>
    <tr><td><strong>Minimal Working Example</strong></td><td>The shortest path to a visible result (<a href="#5-minimal-working-example">Section 5</a>).</td></tr>
  </tbody>
</table>

1. `model configuration` never means `optimization recipe`, and `published slot` never means
   `weight`.
2. Commands are written exactly as typed, including the `adh` program name.

---

## 17. Additional documentation

<table align="center">
  <thead>
    <tr>
      <th>Document</th>
      <th>Contents</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><a href="docs/model-integration.md">docs/model-integration.md</a></td><td>Model integration types; publishing checklist.</td></tr>
    <tr><td><a href="docs/interface.md">docs/interface.md</a></td><td>The unified model interface.</td></tr>
    <tr><td><a href="docs/config.md">docs/config.md</a></td><td>The configuration model.</td></tr>
    <tr><td><a href="docs/deployment.md">docs/deployment.md</a></td><td>Deployment procedure and release checks.</td></tr>
    <tr><td><a href="docs/weights.md">docs/weights.md</a></td><td>Weight storage, distribution, provenance.</td></tr>
    <tr><td><a href="docs/cloud_training_checklist.md">docs/cloud_training_checklist.md</a></td><td>Cloud training checklist.</td></tr>
    <tr><td><a href="docs/report.md">docs/report.md</a></td><td>Benchmark report format and metric taxonomy.</td></tr>
    <tr><td><a href="docs/add_new.md">docs/add_new.md</a></td><td>Adding a dataset, application domain, or backend.</td></tr>
    <tr><td><a href="docs/open-items.md">docs/open-items.md</a></td><td>Information this document does not yet provide.</td></tr>
    <tr><td><a href="docs/adr/0001-unified-registry-and-adapters.md">docs/adr/0001-unified-registry-and-adapters.md</a></td><td>ADR: unified registry and adapters.</td></tr>
    <tr><td><a href="general/README.md">general/README.md</a></td><td>The <code>general</code> application domain.</td></tr>
    <tr><td><a href="textile/README.md">textile/README.md</a></td><td>The <code>textile</code> application domain.</td></tr>
    <tr><td><a href="datasets/README.md">datasets/README.md</a></td><td>Dataset storage conventions.</td></tr>
    <tr><td><a href="examples/README.md">examples/README.md</a></td><td>The Minimal Working Example scripts.</td></tr>
    <tr><td><a href="CONTRIBUTING.md">CONTRIBUTING.md</a></td><td>Contribution procedure.</td></tr>
    <tr><td><a href="LICENSE">LICENSE</a></td><td>License.</td></tr>
  </tbody>
</table>
