# AnomalyDetection

An extensible, configuration-driven platform for supervised defect detection, defect segmentation, and industrial anomaly detection.

## Outline
- [AnomalyDetection](#anomalydetection)
  - [Outline](#outline)
  - [1. Project scope](#1-project-scope)
  - [2. Overview](#2-overview)
  - [3. Supported models](#3-supported-models)
  - [4. Architecture](#4-architecture)
  - [5. Requirements](#5-requirements)
    - [4.1 Software](#41-software)
    - [4.2 Hardware profiles](#42-hardware-profiles)
  - [6. Installation](#6-installation)
    - [6.1 Clone source and initialize components](#61-clone-source-and-initialize-components)
    - [6.2 Create and activate the Conda environment](#62-create-and-activate-the-conda-environment)
    - [6.3 Install requirements](#63-install-requirements)
  - [7. Data preparation](#7-data-preparation)
  - [8. Weight preparation](#8-weight-preparation)
  - [9. Web UI](#9-web-ui)
  - [10. Minimal Working Example](#10-minimal-working-example)
    - [Step 1: Select the task type](#step-1-select-the-task-type)
    - [Step 2: Select the application domain](#step-2-select-the-application-domain)
    - [Step 3: Select the model and inspect its metadata](#step-3-select-the-model-and-inspect-its-metadata)
    - [Step 4: Load the model](#step-4-load-the-model)
    - [Step 5: Select the dataset and split](#step-5-select-the-dataset-and-split)
    - [Step 6: Run detection](#step-6-run-detection)
  - [11. Training](#11-training)
  - [12. Inference](#12-inference)
  - [13. Evaluation and benchmarking](#13-evaluation-and-benchmarking)
  - [14. Adding a model or domain](#14-adding-a-model-or-domain)
    - [Add a model](#add-a-model)
    - [Add a domain](#add-a-domain)
  - [15. Outputs and reproducibility](#15-outputs-and-reproducibility)
  - [16. Troubleshooting](#16-troubleshooting)
    - [Dataset unavailable](#dataset-unavailable)
    - [Component checkout not found](#component-checkout-not-found)
    - [SOCKS proxy error when importing Gradio](#socks-proxy-error-when-importing-gradio)
    - [Model weight missing](#model-weight-missing)
    - [UI still shows an old label](#ui-still-shows-an-old-label)
    - [No anomaly heatmap](#no-anomaly-heatmap)
    - [Out-of-memory during training](#out-of-memory-during-training)
  - [17. Testing](#17-testing)
  - [Additional documentation](#additional-documentation)

## 1. Project scope

This repository currently delivers:

1. A general modular framework organized by application domain.
2. A runnable textile domain demonstration.
3. Deployment, architecture, Minimal Working Example, inference, evaluation, and output documentation.

Other application domains, datasets, and model weights are extension points. They are not claimed to be complete in the current release.

## 2. Overview

The project solves three related task families through one interface:

| Family | Output | Typical metrics |
| --- | --- | --- |
| Supervised defect detection | boxes, labels, confidence | mAP, precision, recall |
| Supervised defect segmentation | semantic or instance masks | mIoU, Dice, pixel F1 |
| Anomaly detection | image score and, when supported, anomaly map | image AUROC, pixel AUROC, AUPRO, IAP |

The textile weights included on the development machine were trained or adapted for the datasets declared in [`configs/registry/models.yaml`](configs/registry/models.yaml). They must not be treated as universal weights for metal, wood, electronics, food, or other materials.

## 3. Supported models

The manifest is authoritative; run `adh inventory` for the machine-readable live view.

| Category | Method | Integration | Training data/protocol |
| --- | --- | --- | --- |
| Detection | YOLOv8n, YOLOv8s, YOLO11n | Ultralytics | ZJU-Leaper |
| Detection | Faster R-CNN, Cascade R-CNN, DETR | torchvision/native factory | ZJU-Leaper |
| Segmentation | Mask R-CNN, UNet++, DeepLabV3+ | torchvision/native factory | ZJU-Leaper |
| Embedding anomaly | PatchCore, PaDiM | Anomalib YAML | ZJU-Leaper normal-only split |
| Teacher-student anomaly | RD4AD, EfficientAD, STFPM | Anomalib YAML | ZJU-Leaper normal-only split |
| Synthetic/generative anomaly | SuperSimpleNet, GANomaly | Anomalib YAML | ZJU-Leaper normal-only split |
| Vision-language anomaly | WinCLIP | Anomalib YAML | external CLIP pretraining; zero-shot |
| Foundation-model anomaly | Dinomaly | component repository | ZJU-Leaper normal-only split |
| Vision-language adapter | MoECLIP | component repository | MVTec AD auxiliary training; evaluated on ZJU-Leaper |
| State-space anomaly | MambaAD | native adapter | ZJU-Leaper normal-only split; local published weight currently missing |

GANomaly produces an image-level anomaly score but no pixel anomaly map. The UI therefore correctly shows no heatmap for GANomaly. Capabilities are queried from the backend rather than guessed by the frontend.

## 4. Architecture

```text
AnomalyDetection/
├── src/fabric_defect_hub/       # reusable SDK; legacy import name
│   ├── application/             # business API used by CLI and UI
│   ├── core/                    # registries and common contracts
│   ├── datasets/                # DatasetAdapter implementations
│   ├── models/                  # ModelAdapter implementations
│   ├── evaluation/              # backend-independent metrics
│   ├── profiling/               # runtime and resource measurement
│   └── web/                     # presentation and event binding only
├── components/                  # isolated upstream research checkouts
├── configs/
│   ├── registry/models.yaml     # published-model source of truth
│   └── models/                  # executable backend recipes
├── datasets/
│   ├── textile/                 # textile datasets
│   └── general/                 # MVTec, VisA, DTD, and auxiliary data
├── textile/                     # textile overlays and artifacts
│   └── artifacts/models/published/
├── examples/                    # Minimal Working Examples
├── tools/                       # conversion, export, and benchmark tools
├── tests/                       # runtime and architecture contracts
└── docs/                        # focused guides and design records
```

There are two registries:

1. The backend registry maps a backend name to a `ModelAdapter` implementation.
2. The model manifest describes each concrete published model: method, task, integration, domain, training dataset/split, evaluation datasets, config, and weight source.

The UI consumes `application/` services and the generated inventory. It does not define training, inference, evaluation, model paths, or model capabilities.

## 5. Requirements

### 4.1 Software

| Item | Requirement |
| --- | --- |
| Python | 3.10 or newer; the current local verification used Python 3.14.6 |
| Environment | Conda recommended; local environment name: `anomalib_env` |
| Operating system | macOS or Linux for the current workflows |
| GPU | optional for Minimal Working Example inference; strongly recommended for training |
| CUDA | required only for NVIDIA acceleration and CUDA-specific optional packages |

PyTorch emits deprecation warnings for TorchScript on Python 3.14. For a conservative training deployment, Python 3.11–3.13 is recommended until the upstream TorchScript transition is complete.

### 4.2 Hardware profiles

| Workflow | CPU/RAM | GPU | Disk |
| --- | --- | --- | --- |
| Catalog, tests, UI layout | 4+ cores, 16 GB RAM | not required | 10 GB plus weights |
| Minimal Working Example inference | 8+ cores, 16 GB RAM | optional | weights plus selected dataset |
| Full textile training | 8+ cores, 32 GB RAM | NVIDIA GPU recommended; capacity depends on model/batch | dataset and run artifacts, typically tens of GB |
| Full benchmark suite | 16+ cores, 32–64 GB RAM | recommended | all datasets, weights, maps, exports, and logs |

These are operational recommendations, not guaranteed minimums. Use smaller batches and `--mode test` on constrained machines.

## 6. Installation

### 6.1 Clone source and initialize components

```bash
git clone --recurse-submodules https://github.com/LINC-BIT/AnomalyDetection.git
cd AnomalyDetection
```

If the repository was cloned without `--recurse-submodules`, initialize all pinned components with:

```bash
git submodule update --init --recursive
```

Do not clone component repositories manually. The parent repository pins the tested commit of AnomalyDiffusion, Dinomaly, and MoECLIP through `.gitmodules` and Git links.

### 6.2 Create and activate the Conda environment

Create the environment first:

```bash
conda create -n anomalib_env python=3.12 -y
conda activate anomalib_env
```

### 6.3 Install requirements

Install the complete local stack:

```bash
python -m pip install -r requirements-full.txt
python -m pip install --no-deps --no-build-isolation -e .
```

For UI and lightweight inference only:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

## 7. Data preparation

Datasets are local runtime assets and are excluded from Git.

```text
datasets/textile/ZJU-Leaper/
datasets/textile/RAW_FABRID/
datasets/textile/Fabric Defects Dataset/
datasets/textile/TILDA_400/
datasets/textile/tianchi/
datasets/general/MVTec AD/
datasets/general/MVTec LOCO/
datasets/general/VisA/
```

Inspect registered defaults and availability:

```bash
adh inventory
adh doctor
```

An explicit root always overrides the default:

```bash
adh train patchcore --dataset zju-leaper --dataset-root /absolute/path/to/ZJU-Leaper --mode test
```

Supported environment overrides include `ZJU_LEAPER_ROOT`, `RAW_FABRIC_ROOT`, `MVTEC_AD_ROOT`, `MVTEC_LOCO_ROOT`, and `VISA_ROOT`.

Download MVTec AD or VisA through Anomalib's maintained downloaders:

```bash
python tools/download_datasets.py mvtec-ad --root datasets/general/MVTecAD --category bottle
python tools/download_datasets.py visa --root datasets/general/VisA --category capsules
```

Place ZJU-Leaper at `datasets/textile/ZJU-Leaper/` or set `ZJU_LEAPER_ROOT`.

### Public dataset links

| Dataset | Download |
|---|---|
| MVTec AD | [Official page](https://www.mvtec.com/company/research/datasets/mvtec-ad) · [Hugging Face search](https://huggingface.co/datasets?search=MVTec%20AD) |
| MVTec LOCO | [Official page](https://www.mvtec.com/company/research/datasets/mvtec-loco) · [Hugging Face search](https://huggingface.co/datasets?search=MVTec%20LOCO) |
| VisA | [Official repository](https://github.com/amazon-science/spot-diff) · [Hugging Face search](https://huggingface.co/datasets?search=VisA) |
| ZJU-Leaper | [Hugging Face search](https://huggingface.co/datasets?search=ZJU-Leaper) |

## 8. Weight preparation

Canonical local weights resolve under:

```text
textile/artifacts/models/published/<model-id>.<extension>
```

Weights are not committed to Git and the UI never downloads them implicitly. `adh inventory` reports `file`, `symlink`, `broken_link`, or `missing` for each slot.

The release policy is:

1. store public weights in a dedicated Hugging Face Hub repository;
2. pin an immutable revision;
3. record filename, size, and SHA-256 in the manifest;
4. download into the project cache and verify before publication.

The existing MoECLIP Google Drive URL is retained only as legacy provenance. See [docs/weights.md](docs/weights.md).

Small YOLO weights may be published with the project. WinCLIP and similar training-free models download their pretrained weights through the backend. Other weights will be distributed from the project Hugging Face repository.

When the project weight repository is published, use the generic downloader:

```bash
python tools/download_weights.py ORG/REPO PatchCore.ckpt \
  --revision REVISION \
  --output textile/artifacts/models/published/PatchCore.ckpt
adh inventory
```

## 9. Web UI

Start the UI before using the command-line workflows:

```bash
conda activate anomalib_env
adh-ui
```

Open the URL printed by Gradio. Select a task type, application domain, and model. The UI reads model, dataset, weight, and capability metadata from the backend registry; it does not define inference or training logic.

### 9.1 Model session

Use the selectors in this order:

1. **Task type**: Defect detection or Anomaly detection.
2. **Application domain**: textile, general, or another registered domain.
3. **Model**: models compatible with the selected task and domain.

The model panel displays the method, training dataset, training split, weight status, and supported outputs. Click **Load model** before running inference.

### 9.2 Benchmark

Open the **Benchmark** tab to evaluate one or more registered models on a selected dataset. Choose the dataset, category or pattern, shot mode, models, and optional profiling or resolution sweep, then click **Run benchmark**.

Benchmark results use two top-level groups:

- **Technical**
  - **Image level**: image AUROC, image F1, precision, and recall.
  - **Pixel level**: pixel AUROC, AUPRO, and IAP when masks and anomaly maps are available.
  - **Instance level**: detection boxes and instance metrics for models that provide them.
  - **Cross-domain**: scores on a held-out dataset or pattern.
- **Overhead**
  - **Compute overhead**: wall time, FPS, latency, and FLOPs.
  - **Memory overhead**: peak memory and allocator measurements when available.
  - **Communication overhead**: exported model transfer-size proxy.

LMEI is reported under **Compute overhead** when the selected profiler provides it.

Unavailable metrics are shown as unavailable; the UI does not infer values from unsupported model outputs. Benchmark reports are written to the configured run directory.

### 9.3 Run history

Open the **Run history** tab to read saved benchmark and evaluation results. Enter the JSON or JSONL report path, click **Refresh**, and optionally select a metric to filter the table. The table shows the run timestamp, model, dataset, metric values, and report path. History reads existing files only; it does not rerun experiments.

### 9.4 Video demonstration

Watch [Web UI demonstration](docs/videos/detection.mp4) for model selection, dataset selection, image loading, and detection output.

Watch [Benchmark demonstration](docs/videos/benchmark.mp4) for benchmark execution and history reading.

## 10. Minimal Working Example

This Minimal Working Example uses the web UI and an existing weight.

Start the UI as described in [Web UI](#9-web-ui), then complete the following steps:

### Step 1: Select the task type

Choose **Anomaly detection** in **Task type**:

![Minimal Working Example step 1 placeholder: select Anomaly detection](docs/images/img1.png)

### Step 2: Select the application domain

Choose **textile** for demonstration:

![Minimal Working Example step 2 placeholder: select application domain](docs/images/img2.png)

### Step 3: Select the model and inspect its metadata

Select `WinCLIP · LAION-400M zero-shot`. Confirm the displayed method, domain, training dataset, training split, and weight filename.

![Minimal Working Example step 3 placeholder: select model and inspect metadata](docs/images/img3.png)

### Step 4: Load the model

Click **Load model** and wait until the runtime status shows the selected model as loaded.

![Minimal Working Example step 4 placeholder: load model](docs/images/img4.png)

### Step 5: Select the dataset and split

Choose `ZJU-Leaper`, select a pattern or **All textures**, and select the `test` split. Set **Full-shot** to use the complete configured sample regime.

Then, click **Load random images**.

![Minimal Working Example step 5 placeholder: select dataset and split](docs/images/img5.png)

### Step 6: Run detection 

Click **Run detection**. Wait for the process to complete and observe the results.

![Minimal Working Example step 6 placeholder: run detection](docs/images/img6.png)

## 11. Training

List resolvable recipes:

```bash
adh train --list
adh models
```

Test training run:

```bash
adh train patchcore --dataset zju-leaper --mode test --no-publish
```

Full supervised detection:

```bash
adh train yolov8n --dataset zju-leaper --mode full
```

Override one nested parameter without copying a recipe:

```bash
adh train patchcore \
  --dataset zju-leaper \
  --mode medium \
  --set train.model_kwargs.coreset_sampling_ratio=0.05
```

MoECLIP uses separate training and evaluation datasets:

```bash
adh train moeclip \
  --dataset mvtec-ad \
  --test-dataset zju-leaper \
  --mode test \
  --no-publish
```

Use `--no-publish` for Minimal Working Example runs and experiments. Without it, a successful canonical run updates the published slot used by the UI.

## 12. Inference

Single image:

```bash
adh predict patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --image /absolute/path/to/image.jpg \
  --output-dir artifacts/runtime/anomaly_maps \
  --output results/patchcore-prediction.json
```

Dataset sampling:

```bash
adh predict patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --dataset zju-leaper \
  --split test \
  --pattern pattern1 \
  --num-samples 8 \
  --output-dir artifacts/runtime/anomaly_maps
```

`--output-dir` is required when a map-capable anomaly model should persist `.npy` anomaly maps. It cannot make an image-only model such as GANomaly produce a heatmap.

## 13. Evaluation and benchmarking

Evaluate an anomaly checkpoint:

```bash
adh evaluate patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --dataset zju-leaper \
  --split test \
  --num-samples 32 \
  --output-dir artifacts/runtime/anomaly_maps
```

Run a configuration-defined benchmark:

```bash
adh benchmark configs/archive/benchmark_example.yaml
```

Cross-pattern robustness:

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

Image-level metrics are available when scores and labels exist. Pixel metrics require ground-truth masks and a model-produced anomaly map. Detection and segmentation metrics require their corresponding annotations.

## 14. Adding a model or domain

### Add a model

1. Reuse a backend or implement `ModelAdapter`.
2. Implement `train`, `predict`, `export`, `load_trained_model`, and truthful `capabilities`.
3. Add a recipe under `configs/models/`.
4. Add one entry to `configs/registry/models.yaml`.
5. Supply a canonical weight or leave its status explicitly missing.
6. Run runtime and architecture tests.

### Add a domain

1. create `datasets/<domain>/` and its adapter;
2. declare dataset tasks, roles, and default root;
3. create `<domain>/domain.yaml`, overlays, and examples;
4. train or calibrate domain-specific weights;
5. add separate manifest entries with correct `domain` and `trained_on` values.

Do not copy reusable algorithms into a domain folder and do not relabel textile weights as general-purpose weights.

## 15. Outputs and reproducibility

Training produces registered artifacts, provenance, optional exports, and a canonical published path. Evaluation produces structured metrics and may persist anomaly maps. Benchmark runs use JSON/JSONL reports suitable for later tables.

Provenance records include timestamp, Git commit/dirty state, host/platform, Python environment, package versions, and component commits. Independent component checkouts and Git submodules are both recorded.

Do not commit generated datasets, checkpoints, maps, caches, or results. Publish durable checkpoints through the documented model distribution channel.

## 16. Troubleshooting

### Dataset unavailable

Run `adh doctor`, verify the categorized directory, or pass `--dataset-root`. The project does not require an SSD fallback.

### Component checkout not found

Run `git submodule update --init --recursive` and rerun `adh doctor`. The expected root-level layout is mandatory.

### SOCKS proxy error when importing Gradio

Install `httpx[socks]` through `requirements.txt`. The UI also removes unusable SOCKS proxy variables for its own process when `socksio` is unavailable.

### Model weight missing

Run `adh inventory` and place the correct file at the reported canonical path. Do not rename a different architecture's checkpoint to satisfy the slot.

### UI still shows an old label

Stop the existing Gradio process and launch `adh-ui` again. The manifest is loaded when the Python process starts.

### No anomaly heatmap

Check the model's `capabilities`. GANomaly is image-level only. For a map-capable model, provide an output directory and ensure the selected dataset has pixel masks if pixel metrics are needed.

### Out-of-memory during training

Use `--mode test`, lower the batch size with `--set`, reduce image size where supported, or select a smaller model. CUDA-only accelerators are optional and must match the installed PyTorch/CUDA toolchain.

## 17. Testing

Runtime and integration contracts:

```bash
pytest -q
```

Architecture boundary audit:

```bash
pytest -q -m architecture
```

Repository hygiene:

```bash
git diff --check
```

The architecture suite verifies, among other rules, that the UI does not import backend business logic and that every published model declares training provenance.

## Additional documentation

- [Model integration](docs/model-integration.md)
- [Unified interface](docs/interface.md)
- [Configuration](docs/config.md)
- [Deployment](docs/deployment.md)
- [Weight storage and distribution](docs/weights.md)
- [Cloud training checklist](docs/cloud_training_checklist.md)
- [Benchmark report](docs/report.md)
- [Architecture decision](docs/adr/0001-unified-registry-and-adapters.md)
- [Textile domain](textile/README.md)
