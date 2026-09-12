#!/usr/bin/env bash
set -euo pipefail

profile="${1:-core}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_dir}/.venv"

case "${profile}" in
  core|ui|full) ;;
  *) echo "Usage: $0 [core|ui|full]" >&2; exit 2 ;;
esac

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip

if [[ "${profile}" == "ui" ]]; then
  "${venv_dir}/bin/python" -m pip install -r "${project_dir}/requirements.txt"
elif [[ "${profile}" == "full" ]]; then
  "${venv_dir}/bin/python" -m pip install -r "${project_dir}/requirements-full.txt"
fi

"${venv_dir}/bin/python" -m pip install -e "${project_dir}"
PYTHONPATH="${project_dir}/src" "${venv_dir}/bin/python" "${project_dir}/examples/01_inspect_catalog.py"

echo
echo "Environment ready: ${venv_dir}"
echo "Activate with: source ${venv_dir}/bin/activate"
echo "Next check: adh doctor"
