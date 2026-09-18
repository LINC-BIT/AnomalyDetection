#!/usr/bin/env bash
# Sanity-check this checkout, activate the conda environment, start adh-ui.
#
#   ./scripts/start_web.sh                 # checks, then launch the UI
#   ./scripts/start_web.sh --check-only    # checks only (pre-flight, CI)
#   ./scripts/start_web.sh --skip-doctor   # skip the dataset/weight inventory
#   GRADIO_SERVER_PORT=7860 ./scripts/start_web.sh
#
# The checks are deliberately the cheap ones that catch the failures this
# project actually sees: a checkout that is not the project root, a conda
# environment that is not the one the training extras were installed into, an
# environment whose framework imports are broken, and a port already held by an
# earlier `adh-ui` (which `adh-ui` itself reports with the holder's PID, so this
# only warns).
#
# Environment: FDH_CONDA_ENV (default anomalib_env), GRADIO_SERVER_PORT.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${FDH_CONDA_ENV:-anomalib_env}"
PORT="${GRADIO_SERVER_PORT:-6008}"
CHECK_ONLY=0
SKIP_DOCTOR=0

for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
    --skip-doctor) SKIP_DOCTOR=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "start_web: unknown option: $arg" >&2; exit 2 ;;
  esac
done

ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mfail\033[0m %s\n' "$1"; failed=1; }
failed=0

cd "$PROJECT_ROOT"

echo "[1/4] checkout"
if [[ -d src/fabric_defect_hub && -f pyproject.toml ]]; then
  ok "project root: $PROJECT_ROOT"
else
  bad "src/fabric_defect_hub or pyproject.toml is missing — run this from the repository"
fi

echo "[2/4] conda environment '${CONDA_ENV}'"
use_conda_run=0
if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
  ok "already active"
elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  use_conda_run=1
  ok "found via conda (running through \`conda run\`)"
else
  warn "conda cannot see '${CONDA_ENV}'; using whatever \`python\` resolves to"
fi

run_python() {
  if [[ "$use_conda_run" == "1" ]]; then
    conda run --no-capture-output -n "$CONDA_ENV" python "$@"
  else
    python "$@"
  fi
}

echo "[3/4] imports and devices"
if run_python - <<'PY'
import sys

try:
    import fabric_defect_hub
    import gradio
    import torch
except Exception as exc:  # the message is the diagnosis; do not traceback it
    print(f"  import failed: {type(exc).__name__}: {exc}")
    print("  install with: python tools/install_deps.py ui && python tools/install_deps.py train")
    raise SystemExit(1)

from fabric_defect_hub.runtime_device import available_torch_devices

devices = available_torch_devices()
print(f"  python {sys.version.split()[0]} | gradio {gradio.__version__} | torch {torch.__version__}")
print(f"  training devices: {', '.join(devices)}"
      + (f" (parallel training will use {len(devices)} worker(s))" if len(devices) > 1 else ""))
PY
then
  ok "fabric_defect_hub, gradio and torch import"
else
  bad "imports failed (see above)"
fi

echo "[4/4] catalogue"
if [[ "$SKIP_DOCTOR" == "1" ]]; then
  warn "skipped (--skip-doctor)"
elif run_python -m fabric_defect_hub doctor >/dev/null 2>&1; then
  ok "\`adh doctor\` reports no missing backend or weight"
  echo "     run \`adh doctor\` yourself for the full report"
else
  warn "\`adh doctor\` reported problems (missing datasets/weights are often expected);"
  echo "     run \`adh doctor\` for details — the UI still starts"
fi

if [[ "$failed" != "0" ]]; then
  echo
  echo "start_web: sanity checks failed; not starting the UI" >&2
  exit 1
fi

echo
if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "start_web: checks passed (--check-only)"
  exit 0
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  warn "port ${PORT} is already served — \`adh-ui\` will name the holder and offer to reuse it"
fi

echo "start_web: starting adh-ui on http://127.0.0.1:${PORT}"
if [[ "$use_conda_run" == "1" ]]; then
  exec conda run --no-capture-output -n "$CONDA_ENV" adh-ui
fi
exec adh-ui
