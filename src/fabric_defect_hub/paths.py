"""Repository paths shared by CLI, services and adapters.

All paths may be overridden for deployments, while the source checkout uses
the repository-level multi-domain layout by default.
"""

from __future__ import annotations

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = Path(os.environ.get("AD_DATASETS_ROOT", REPOSITORY_ROOT / "datasets"))
COMPONENTS_ROOT = Path(os.environ.get("AD_COMPONENTS_ROOT", REPOSITORY_ROOT / "components"))
TEXTILE_ROOT = Path(os.environ.get("AD_TEXTILE_ROOT", REPOSITORY_ROOT / "textile"))
GENERAL_ROOT = Path(os.environ.get("AD_GENERAL_ROOT", REPOSITORY_ROOT / "general"))
ARTIFACTS_ROOT = Path(os.environ.get("AD_ARTIFACTS_ROOT", TEXTILE_ROOT / "artifacts"))
CONFIGS_ROOT = Path(os.environ.get("AD_CONFIGS_ROOT", REPOSITORY_ROOT / "configs"))
RESULTS_ROOT = Path(os.environ.get("AD_RESULTS_ROOT", REPOSITORY_ROOT / "results"))
