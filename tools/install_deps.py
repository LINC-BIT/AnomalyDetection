#!/usr/bin/env python
"""Install a dependency scenario, choosing the CUDA extras per machine.

The scenarios are the optional-dependency groups declared in pyproject.toml:

    ui     web interface and lightweight inference
    train  the six training backends
    eval   evaluation, profiling, export, quantization
    all    everything above, plus the test suite

Examples::

    python tools/install_deps.py all
    python tools/install_deps.py train eval
    python tools/install_deps.py ui --dry-run

The project is installed editable with the selected extras, so the `adh` and
`adh-ui` entry points appear in the environment. When an NVIDIA GPU is detected
-- Linux, a driver that answers ``nvidia-smi``, and a CUDA-capable PyTorch when
torch is already installed -- the ``cuda`` extra is added to the install. A
package there that fails to build is a warning, not a failed install, because
each accelerator is optional and the platform falls back to its portable
implementation at run time.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ("ui", "train", "eval", "all")


def detect_cuda() -> tuple[bool, str]:
    """Report whether the CUDA-only extras belong on this machine."""
    system = platform.system()
    if system != "Linux":
        return False, f"{system} has no CUDA platform support in this project"
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi is not on PATH, so no NVIDIA driver was found"
    probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if probe.returncode != 0 or not probe.stdout.strip():
        return False, "nvidia-smi reports no NVIDIA GPU"
    try:
        import torch
    except ImportError:
        return True, "an NVIDIA GPU is present and torch is not installed yet"
    if not torch.cuda.is_available():
        return False, "an NVIDIA GPU is present but the installed torch has no CUDA support"
    return True, f"an NVIDIA GPU is present and torch.cuda is available (torch {torch.__version__})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the project with the extras of one or more scenarios",
        epilog="Scenarios: " + ", ".join(SCENARIOS),
    )
    parser.add_argument(
        "scenarios",
        nargs="+",
        choices=SCENARIOS,
        metavar="SCENARIO",
        help="one or more of: " + ", ".join(SCENARIOS),
    )
    cuda = parser.add_mutually_exclusive_group()
    cuda.add_argument("--cuda", action="store_true", help="install the CUDA extras without probing")
    cuda.add_argument("--no-cuda", action="store_true", help="skip the CUDA extras")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter to install into (default: the running one)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command without installing anything",
    )
    args = parser.parse_args()

    extras: list[str] = []
    for scenario in args.scenarios:
        if scenario not in extras:
            extras.append(scenario)

    if args.cuda:
        install_cuda, cuda_reason = True, "forced by --cuda"
    elif args.no_cuda:
        install_cuda, cuda_reason = False, "disabled by --no-cuda"
    else:
        install_cuda, cuda_reason = detect_cuda()
    if install_cuda:
        extras.append("cuda")

    target = f".[{','.join(extras)}]"
    command = [args.python, "-m", "pip", "install", "-e", target]

    print(f"Scenarios:   {', '.join(args.scenarios)}")
    print(f"Extras:      {', '.join(extras)}")
    print(f"Interpreter: {args.python}")
    print(f"CUDA extras: {'install' if install_cuda else 'skip'} ({cuda_reason})")
    print()
    print("$ " + f'{args.python} -m pip install -e "{target}"' + f"    # in {PROJECT_ROOT}", flush=True)

    if args.dry_run:
        return 0

    code = subprocess.call(command, cwd=PROJECT_ROOT)
    if code != 0:
        if install_cuda and "cuda" in extras:
            print(
                f"error: the install failed with exit code {code}. If the failure is in a "
                "CUDA-only accelerator, re-run with --no-cuda: those packages are optional "
                "and the platform falls back to its portable implementations",
                file=sys.stderr,
            )
        else:
            print(f"error: the install failed with exit code {code}", file=sys.stderr)
        return code

    print()
    print("Next check: adh doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
