"""The operational shell scripts have to keep parsing and keep working.

`scripts/train_all.sh` and `scripts/start_web.sh` are what an operator actually
runs on a training host; a typo in either is invisible to the Python test suite
until the day someone needs it. `bash -n` catches the typo, and the help /
dry-run paths prove the argument plumbing still reaches the CLI.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win") or shutil.which("bash") is None,
    reason="the operational scripts are bash",
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["scripts/train_all.sh", "scripts/start_web.sh"])
def test_script_parses(script):
    path = ROOT / script

    assert path.is_file()
    assert path.stat().st_mode & 0o111, f"{script} is not executable"
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_train_all_dry_run_reaches_the_cli(tmp_path):
    """`FDH_DRY_RUN=1` has to print the plan, the job count included, without
    training anything — that is the smoke test for the whole wrapper."""

    import os

    # A PATH without `${CONDA_PREFIX}/bin` so `command -v conda` fails and the
    # script takes its plain-`python` branch: this test is about the argument
    # plumbing, not about conda being installed.
    path_without_conda = f"{Path(sys.executable).parent}:/usr/bin:/bin"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "train_all.sh")],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={**os.environ, "FDH_DRY_RUN": "1", "FDH_JOBS": "2", "PATH": path_without_conda},
    )

    assert result.returncode == 0, result.stderr
    assert '"jobs": 2' in result.stdout
    assert '"model_count"' in result.stdout
    assert "train_all: env=" in result.stdout


def test_start_web_help_lists_the_check_only_flag():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "start_web.sh"), "--help"],
        capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0
    assert "--check-only" in result.stdout
