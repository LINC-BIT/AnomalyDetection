#!/usr/bin/env bash
set -Eeuo pipefail

# Full post-training measurement. Accuracy writes anomaly maps by default, so
# anomaly models receive both image-level and pixel-level metrics. Detection
# models remain box/instance-level only. Profiling failures are preserved as
# skipped rows in the JSONL rather than converted to zeroes.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${FDH_DATASET:-zju-leaper}"
DATASET_ROOT="${FDH_DATASET_ROOT:-}"
PATTERN="${FDH_PATTERN:-1,2,3,4}"
HELD_OUT="${FDH_HELD_OUT_PATTERNS:-5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"
SAMPLES="${FDH_NUM_SAMPLES:-400}"
RUNS="${FDH_MEASURED_RUNS:-50}"
WARMUP="${FDH_WARMUP_RUNS:-5}"
RUN_ID="${FDH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT="${FDH_OUTPUT:-artifacts/runtime/runs/${RUN_ID}/full_sweep.jsonl}"
MAP_DIR="${FDH_ANOMALY_MAP_DIR:-artifacts/runtime/runs/${RUN_ID}/anomaly_maps}"
PYTHON_BIN="${FDH_PYTHON:-python3}"
POWER_MODE="${FDH_POWER_MODE:?Set FDH_POWER_MODE to required or disabled}"

DEVICE="${FDH_DEVICE:-auto}"
MODELS="${FDH_MODELS:-}"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$OUTPUT")" "$MAP_DIR"

args=(
  --dataset "$DATASET"
  --pattern "$PATTERN"
  --num-samples "$SAMPLES"
  --groups accuracy,cross_domain,runtime,scaling,concurrency,communication
  --cross-domain-patterns "$HELD_OUT"
  --anomaly-map-dir "$MAP_DIR"
  --output "$OUTPUT"
  --device "$DEVICE"
  --measured-runs "$RUNS"
  --warmup-runs "$WARMUP"
  --power-mode "$POWER_MODE"
)
if [[ -n "$DATASET_ROOT" ]]; then
  args+=(--dataset-root "$DATASET_ROOT")
fi
if [[ -n "$MODELS" ]]; then
  args+=(--models "$MODELS")
fi

exec "$PYTHON_BIN" tools/run_metric_sweep.py "${args[@]}"
