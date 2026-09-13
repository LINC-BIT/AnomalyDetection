#!/usr/bin/env bash
set -euo pipefail

model="${1:-PatchCore}"
dataset="${2:-mvtec-ad}"
category="${3:-bottle}"

adh train general/configs/models/anomalib.yaml \
  --variant "$model" \
  --dataset "$dataset" \
  --category "$category"
