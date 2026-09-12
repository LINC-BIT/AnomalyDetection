"""Provenance shared by `reporting.append_run_log` and `weight_registry.record_weight`."""

from __future__ import annotations

import platform
import subprocess
import sys
import os
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_commit(cwd: str | Path | None = None) -> str:
    """The current commit hash, or "unknown" off-repo / without git."""

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            cwd=cwd,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty(cwd: str | Path | None = None) -> bool | None:
    """Whether tracked or untracked files differ from HEAD; None off-repo."""

    try:
        output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True, text=True, timeout=10, check=True, cwd=cwd,
        ).stdout
        return bool(output.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str]:
    packages = (
        "fabric-defect-hub", "torch", "torchvision", "anomalib", "numpy",
        "scikit-learn", "scikit-image", "onnxruntime", "ultralytics",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def vendored_components(cwd: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Commit + dirty-state of component submodules or standalone checkouts."""

    try:
        output = subprocess.run(
            ["git", "submodule", "status"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            cwd=cwd,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}

    components: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        marker, rest = line[0], line[1:].split()
        if len(rest) < 2:
            continue
        commit, path = rest[0], rest[1]
        components[path] = {
            "commit": commit,
            "state": {" ": "clean", "+": "modified", "-": "uninitialized"}.get(marker, marker),
        }
    root = Path(cwd or Path.cwd())
    component_root = root / "components"
    if component_root.is_dir():
        for checkout in component_root.iterdir():
            key = f"components/{checkout.name}"
            if key in components or not checkout.is_dir():
                continue
            try:
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                    timeout=5, check=True, cwd=checkout,
                ).stdout.strip()
                dirty = subprocess.run(
                    ["git", "status", "--porcelain"], capture_output=True, text=True,
                    timeout=10, check=True, cwd=checkout,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                continue
            components[key] = {"commit": commit, "state": "modified" if dirty else "clean"}
    return components


def collect_provenance(cwd: str | Path | None = None) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(cwd),
        "git_dirty": git_dirty(cwd),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "package_versions": _package_versions(),
        "vendored_components": vendored_components(cwd),
    }


def describe_training(
    optimizer: Any = None,
    scheduler: Any = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Record for `Artifact.metadata["training"]`. Accepts a live optimizer/
    scheduler object (reads class name + lr/weight_decay off it) or a plain
    string when there's no object to introspect (e.g. Ultralytics' "auto").
    """

    record: dict[str, Any] = {"precision": precision}

    if optimizer is None:
        record["optimizer"] = None
    elif isinstance(optimizer, str):
        record["optimizer"] = optimizer
    else:
        record["optimizer"] = type(optimizer).__name__
        groups = getattr(optimizer, "param_groups", None)
        if groups:
            record["lr"] = groups[0].get("lr")
            if "weight_decay" in groups[0]:
                record["weight_decay"] = groups[0]["weight_decay"]

    if scheduler is None:
        record["scheduler"] = None
    elif isinstance(scheduler, str):
        record["scheduler"] = scheduler
    else:
        record["scheduler"] = type(scheduler).__name__

    return record
