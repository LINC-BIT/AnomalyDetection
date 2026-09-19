#!/usr/bin/env bash
# Retrain DETR on ZJU-Leaper pattern 1-4 with the corrected loss weighting.
#
# Context: the published `detr_resnet50.pt` collapsed — every one of its 100
# queries emitted the same box with score 0.005 for every image (mAP 0.005,
# mAP@50 0.010). Two recipe defects caused it and are fixed here:
#   1. `presets.SetCriterion` never applied DETR's `weight_dict`, so
#      `loss_bbox`/`loss_giou` trained at weight 1 instead of 5/2.
#   2. It ran 100 epochs at the COCO 800/1333 resize, on 512x512 source images.
# See the header of the config for the full write-up and the measured evidence.
#
#   ./scripts/retrain_detr.sh                  # 300 epochs on one GPU, in tmux
#   ./scripts/retrain_detr.sh --foreground     # same, but in this shell
#   FDH_EPOCHS=50 ./scripts/retrain_detr.sh    # short first pass; resume later
#   FDH_RESUME=1 ./scripts/retrain_detr.sh     # continue an interrupted run
#
# Any other arguments are forwarded to `adh train` verbatim, e.g.
#   ./scripts/retrain_detr.sh --mode test --no-publish   # 8-image wiring check
#
# The run overwrites the registered/published DETR slot on success. The
# collapsed bytes stay in runs/retrain_detr_100/detr_resnet50/best.pt.
#
# Environment: FDH_CONDA_ENV (default anomalib_env), FDH_TMUX_SESSION (default
# detr_fixed), FDH_DETR_CONFIG, FDH_RUN_DIR, FDH_ALLOW_CPU=1 to permit a CPU run.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${FDH_CONDA_ENV:-anomalib_env}"
SESSION="${FDH_TMUX_SESSION:-detr_fixed}"
CONFIG="${FDH_DETR_CONFIG:-configs/models/torchvision_detr.yaml}"
RUN_DIR="${FDH_RUN_DIR:-runs/retrain_detr_fixed}"
LOG="${FDH_DETR_LOG:-$RUN_DIR/session.log}"

FOREGROUND=0
extra=()
for arg in "$@"; do
  if [[ "$arg" == "--foreground" ]]; then
    FOREGROUND=1
  else
    extra+=("$arg")
  fi
done

cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR"

# `--set checkpoint.run_dir` keeps the config's run directory in step with
# FDH_RUN_DIR, so the log, the report and the checkpoints all agree.
args=(train "$CONFIG" --json-report "$RUN_DIR/report.json" --set "checkpoint.run_dir=$RUN_DIR")
[[ -n "${FDH_EPOCHS:-}" ]] && args+=(--set "train.epochs=${FDH_EPOCHS}")
[[ "${FDH_RESUME:-0}" == "1" ]] && args+=(--set "train.resume=true")
[[ ${#extra[@]} -gt 0 ]] && args+=("${extra[@]}")

# Same resolution as scripts/train_all.sh: use the active env, else `conda run`.
if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]] || ! command -v conda >/dev/null 2>&1; then
  py=(python)
else
  py=(conda run --no-capture-output -n "$CONDA_ENV" python)
fi
runner=("${py[@]}" -m fabric_defect_hub)

if [[ "${FDH_ALLOW_CPU:-0}" != "1" ]]; then
  # `conda run` prints its own ERROR banner when the child exits non-zero.
  if ! "${py[@]}" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "retrain_detr: no CUDA device visible to env '$CONDA_ENV' — this is a" >&2
    echo "  300-epoch job; set FDH_ALLOW_CPU=1 if you really mean to run it on CPU." >&2
    exit 3
  fi
fi

echo "retrain_detr: env=$CONDA_ENV config=$CONFIG run_dir=$RUN_DIR log=$LOG session=$SESSION"

if [[ "$FOREGROUND" == "1" ]]; then
  exec "${runner[@]}" "${args[@]}" 2>&1 | tee "$LOG"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "retrain_detr: tmux session '$SESSION' already exists — attach with: tmux attach -t $SESSION" >&2
  exit 2
fi

printf -v cmd '%q ' "${runner[@]}" "${args[@]}"
tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" "set -o pipefail; ${cmd}2>&1 | tee '${LOG}'"
echo "retrain_detr: started in tmux session '$SESSION'"
echo "  attach : tmux attach -t $SESSION"
echo "  tail   : tail -f $LOG"
echo "  history: $RUN_DIR/detr_resnet50/history.csv"
