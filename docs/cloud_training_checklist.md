# Cloud Completion Checklist

This document is retained as a historical cloud-run checklist. The published
project weights have now been uploaded; do not rerun the missing-model batch
merely to recreate them.

## Scope

| Model | Why it is missing locally | Cloud data | Base weight behavior |
| --- | --- | --- | --- |
| EfficientAD | Needs a separate natural-image regularization set | ZJU-Leaper + Imagenette | Anomalib/timm downloads or reuses its pretrained teacher |
| WinCLIP | Upstream zero-shot checkpoint; no learned ZJU checkpoint | ZJU-Leaper test set | OpenCLIP/LAION-400M pretrained backbone; no MVTec or textile fitting |
| MoECLIP | Needs an auxiliary zero-shot training corpus | MVTec AD + ZJU-Leaper | **Manual** `ViT-L-14-336px.pt` placement required |
| MambaAD | High-resource model; no accepted published baseline yet | ZJU-Leaper | timm downloads or reuses its pretrained ResNet teacher |

`MambaAD` remains a required attempted run. A CUDA out-of-memory or an
implementation/runtime failure is recorded as a failed run with its log; it
is not silently omitted from the batch.

## 1. What to upload

Upload or clone only the source repository. The repository's `.gitignore`
already excludes `data/`, `artifacts/`, `runs/`, `results/`, and all common
weight/export formats (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.engine`). Do
not upload the 16 local published weights or your local `artifacts/` tree.

On the cloud machine, manually stage these paths (copies, mounts, or symlinks
are all valid):

```text
data/ZJU-Leaper/                         # ZJU images + annotations + masks
data/MVTec AD/                           # MVTec AD train/test/ground_truth
data/imagenette/                         # Imagenette train/val directories
components/moeclip/model/ViT-L-14-336px.pt
```

The MoECLIP checkpoint is the only required manually placed model weight.
WinCLIP and MambaAD may download their pretrained backbones at first run, so
give the cloud machine network access or pre-populate its model cache.

## 2. Prepare and validate prerequisites

Run these commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-full.txt

export ZJU_LEAPER_ROOT="$PWD/data/ZJU-Leaper"
export MVTEC_AD_ROOT="$PWD/data/MVTec AD"
export IMAGENET_DIR="$PWD/data/imagenette"

python tools/preflight_cloud_models.py
```

The preflight command must report four `ready` models before a real run. It
checks the selected configs, required datasets, installed backend packages,
and MoECLIP's manually supplied backbone. It does not download weights or
start training.

## 3. Run every missing model

Use separate commands so every model has a clear log and an independent
success/failure record. The ZJU selections in the configs are pattern1-4,
with 1,200 train and 400 test samples total (300/100 per pattern requested).

```bash
mkdir -p artifacts/training_runs/cloud_missing_models/logs
set -o pipefail

adh train configs/models/anomalib_efficientad.yaml \
  --dataset-root "$ZJU_LEAPER_ROOT" \
  --set train.model_kwargs.imagenet_dir="$IMAGENET_DIR" \
  2>&1 | tee artifacts/training_runs/cloud_missing_models/logs/EfficientAD.log

adh train configs/models/anomalib_winclip.yaml \
  --dataset-root "$ZJU_LEAPER_ROOT" \
  2>&1 | tee artifacts/training_runs/cloud_missing_models/logs/WinCLIP.log

adh train configs/models/moeclip_example.yaml \
  --dataset-root "$MVTEC_AD_ROOT" \
  --test-dataset-root "$ZJU_LEAPER_ROOT" \
  2>&1 | tee artifacts/training_runs/cloud_missing_models/logs/MoECLIP.log

adh train configs/models/mambaad_example.yaml \
  --dataset-root "$ZJU_LEAPER_ROOT" \
  --num-samples 1200 --val-num-samples 400 --use-defect --defect-ratio 0.5 \
  2>&1 | tee artifacts/training_runs/cloud_missing_models/logs/MambaAD.log
```

The commands intentionally do not use `adh train-all`: that command would
re-run all 20 catalog entries, including the 16 models already completed
locally. Run MambaAD even if it fails; preserve its log for the project record.

## 4. Run the complete benchmark on cloud

Run the accuracy sweep with anomaly-map persistence enabled. The sweep now
uses `artifacts/runtime/anomaly_maps` automatically when the flag is omitted,
but keeping the option explicit makes the cloud bundle self-documenting:

```bash
python tools/run_metric_sweep.py \
  --dataset zju-leaper \
  --pattern 1,2,3,4 \
  --num-samples 400 \
  --groups accuracy,cross_domain,runtime,scaling,concurrency \
  --cross-domain-patterns 5,6,7,8 \
  --anomaly-map-dir artifacts/runtime/anomaly_maps \
  --output artifacts/runtime/full_sweep.jsonl \
  --device cuda:0
```

Detection models produce box/instance metrics only. Segmentation and anomaly
models that expose masks or anomaly maps produce both image-level and
pixel-level metrics. GANomaly is image-level only because its model output has
no spatial anomaly map. Every unsupported export or profiling measurement is
written as a `skipped` row with its reason; it must not be interpreted as a
zero score.

The `cross_domain` group is not enabled unless `--cross-domain-patterns` is
provided. It reports degradation of the task headline metric on held-out
patterns. Runtime, scaling, and concurrency are the overhead groups; they are
independent of accuracy and may be skipped when a backend cannot export or
profile its model.

## 5. What is created and what to download

Each successful `adh train` writes a registered weight under
`artifacts/models/`, publishes the four missing canonical slots when applicable,
writes `artifacts/models/weight_manifest.jsonl`, and writes model-specific run
files, metrics, and anomaly maps under `runs/` / `artifacts/anomaly_maps/`.
The `tee` commands also guarantee one raw cloud log per model.

After all four attempts, package only the newly generated cloud artifacts:

```bash
python tools/collect_cloud_artifacts.py \
  --run-id cloud_missing_models --archive
```

Download the single output file:

```text
artifacts/cloud_runs/cloud_missing_models.tar.gz
```

It contains project-generated weights, manifests, configs, logs, metrics,
curves, anomaly maps, environment information, and SHA-256 checksums. It does
not include your datasets or the manually staged MoECLIP backbone, by design.
