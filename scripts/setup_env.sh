#!/usr/bin/env bash
set -euo pipefail

profile="${1:-core}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_dir}/.venv"

case "${profile}" in
  core|ui|train|eval|all) ;;
  *) echo "Usage: $0 [core|ui|train|eval|all]" >&2; exit 2 ;;
esac

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip

if [[ "${profile}" == "core" ]]; then
  # Project only: the CLI and the catalogue commands, no scenario extras.
  "${venv_dir}/bin/python" -m pip install --no-deps -e "${project_dir}"
else
  # install_deps.py installs the scenario extras and adds the `cuda` extra
  # when it detects an NVIDIA GPU on this machine.
  "${venv_dir}/bin/python" "${project_dir}/tools/install_deps.py" "${profile}" \
    --python "${venv_dir}/bin/python"
fi

PYTHONPATH="${project_dir}/src" "${venv_dir}/bin/python" "${project_dir}/examples/01_inspect_catalog.py"

echo
echo "Environment ready: ${venv_dir}"
echo "Activate with: source ${venv_dir}/bin/activate"
echo "Next check: adh doctor"
