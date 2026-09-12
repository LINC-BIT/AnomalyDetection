# AnomalyDetection

An extensible, configuration-driven platform for supervised defect detection, defect segmentation, and industrial anomaly detection.

## Table of contents

- [AnomalyDetection](#anomalydetection)
  - [Table of contents](#table-of-contents)
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

The downloader may require acceptance of the dataset license or network access. ZJU-Leaper is a project-specific dataset; its downloader is TODO. Until released, place it at the registered path or set `ZJU_LEAPER_ROOT`. Do not commit dataset bytes to Git.

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

Weight download automation, checksum verification, and Hugging Face release packages are TODO. Upstream model weights are downloaded by their backend when supported; project weights must currently be copied into the canonical slots and verified with `adh inventory`. No training is needed for the MWE: use an existing published checkpoint such as WinCLIP or PatchCore.

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

## 10. Minimal Working Example

```bash
conda activate anomalib_env
adh inventory
adh doctor
python examples/01_inspect_catalog.py
```

Expected inventory shape:

```text
models: 20
datasets: 9
```

On the current local checkout, 19 canonical weights are present and MambaAD is missing. `doctor` should report every installed backend and explain its selected training dataset or missing prerequisite.

Run a real local inference:

```bash
adh predict yolov8n \
  --weights textile/artifacts/models/published/yolov8n.pt \
  --image datasets/textile/ZJU-Leaper/Images/052628.jpg \
  --output results/yolov8n-mwe.json
```

Expected result: JSON containing the model identity and one prediction with boxes, labels, and confidence scores. Exact detections depend on the checkpoint and input.

The page title and brand are **AnomalyDetection**. The model selector displays the actual training dataset, for example `PatchCore · ZJU-Leaper (normal only)` or `MoECLIP · MVTec AD adapter`.

Single-image workflow:

1. select a model and inspect its method/domain/training metadata;
2. load the model;
3. select a dataset, split, category/pattern, and sampling regime;
4. load sample images;
5. run inference;
6. inspect boxes, masks, anomaly score, or heatmap according to declared capabilities.

Restart `adh-ui` after changing the model manifest because a running process keeps its imported inventory in memory.

## 11. Training

List resolvable recipes:

```bash
adh train --list
adh models
```

Minimal Working Example training run:

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

No UI edit is permitted or required. See [docs/model-integration.md](docs/model-integration.md).

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
