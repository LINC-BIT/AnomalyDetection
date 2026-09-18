#!/usr/bin/env bash
# Train every catalogued model in one command, `FDH_JOBS` of them at a time.
#
# Each model is its own `adh train` process on its own device — one CUDA GPU
# apiece, and a single process on an MPS or CPU host (a second worker there
# would only compete for the same memory). That isolation is the point: a
# backend that dies, whether from an out-of-memory kill or an illegal CUDA
# access, costs its own model instead of the whole batch. The batch is
# resumable: re-run it with the same run id and every model it already finished
# is skipped rather than retrained.
#
#   ./scripts/train_all.sh                              # one model per GPU
#   FDH_JOBS=2 ./scripts/train_all.sh                   # half the devices
#   FDH_ONLY="PatchCore PaDiM" ./scripts/train_all.sh   # just these
#   FDH_TRAIN_MODE=test FDH_NO_PUBLISH=1 ./scripts/train_all.sh   # 8-image wiring check
#   FDH_RESUME=1 ./scripts/train_all.sh                 # continue the last batch
#   FDH_DRY_RUN=1 ./scripts/train_all.sh                # print the plan only
#
# Environment: FDH_CONDA_ENV (default anomalib_env), FDH_RUN_ROOT, FDH_RUN_ID.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${FDH_CONDA_ENV:-anomalib_env}"
RUN_ROOT="${FDH_RUN_ROOT:-artifacts/training_runs}"
JOBS="${FDH_JOBS:-}"
MODE="${FDH_TRAIN_MODE:-}"
ONLY="${FDH_ONLY:-}"
RESUME="${FDH_RESUME:-0}"
DRY_RUN="${FDH_DRY_RUN:-0}"
NO_PUBLISH="${FDH_NO_PUBLISH:-0}"

cd "$PROJECT_ROOT"

args=(train-all --run-root "$RUN_ROOT")
[[ -n "$JOBS" ]] && args+=(--jobs "$JOBS")
[[ -n "$MODE" ]] && args+=(--mode "$MODE")
[[ -n "$ONLY" ]] && args+=(--only $ONLY)
[[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)
# Smoke runs (`FDH_TRAIN_MODE=test`) must not overwrite a published weight with
# an 8-image checkpoint: FDH_NO_PUBLISH=1 keeps the run to its own artifact.
[[ "$NO_PUBLISH" == "1" ]] && args+=(--no-publish)

if [[ "$RESUME" == "1" ]]; then
  # `--resume` needs the run id it continues; default to the newest batch so the
  # common case ("I interrupted it, carry on") is one variable, not two.
  run_id="${FDH_RUN_ID:-}"
  if [[ -z "$run_id" && -d "$RUN_ROOT" ]]; then
    run_id="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -1)"
  fi
  if [[ -z "$run_id" ]]; then
    echo "train_all: FDH_RESUME=1 needs FDH_RUN_ID, or an existing batch under $RUN_ROOT" >&2
    exit 2
  fi
  args+=(--run-id "$run_id" --resume)
  echo "train_all: resuming batch $run_id"
elif [[ -n "${FDH_RUN_ID:-}" ]]; then
  args+=(--run-id "$FDH_RUN_ID")
fi

echo "train_all: env=${CONDA_ENV} jobs=${JOBS:-auto (one per device)} mode=${MODE:-config default} root=${RUN_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || ! command -v conda >/dev/null 2>&1; then
  exec python -m fabric_defect_hub "${args[@]}"
fi

exec conda run --no-capture-output -n "$CONDA_ENV" python -m fabric_defect_hub "${args[@]}"
