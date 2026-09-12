"""Minimal core-install inspection of the textile domain catalog."""

from __future__ import annotations

import fabric_defect_hub as adh
from fabric_defect_hub.catalog import CANONICAL_MODELS


def main() -> None:
    models = [model.key for model in CANONICAL_MODELS]
    datasets = adh.list_datasets()
    print(f"AnomalyDetection version: {adh.__version__}")
    print(f"Registered models: {len(models)}")
    print(f"Registered datasets: {len(datasets)}")
    print("Models:", ", ".join(models))
    print("Datasets:", ", ".join(map(str, datasets)))


if __name__ == "__main__":
    main()
