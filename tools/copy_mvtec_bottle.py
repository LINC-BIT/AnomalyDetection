#!/usr/bin/env python
"""Copy the bottle category from a staged MVTec AD tree into its own root."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy the bottle category from an existing MVTec AD tree"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/general/MVTec AD"),
        help="Existing MVTec AD root; this script never downloads data",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("datasets/general/Bottle"),
    )
    args = parser.parse_args()

    source = args.source_root / "bottle"
    if not (source / "train" / "good").is_dir() or not (source / "test").is_dir():
        raise RuntimeError(f"MVTec AD bottle category was not found under {args.source_root}")

    target = args.target_root / "bottle"
    if target.exists():
        shutil.rmtree(target)
    args.target_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    print(f"Copied {source} to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())