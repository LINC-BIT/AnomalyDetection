#!/usr/bin/env python
"""Download datasets through Anomalib's maintained dataset modules.

ZJU-Leaper is project-specific and is intentionally not downloaded here.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download an Anomalib dataset")
    parser.add_argument("dataset", choices=("mvtec-ad", "visa"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    from anomalib.data import MVTecAD, Visa

    cls = MVTecAD if args.dataset == "mvtec-ad" else Visa
    kwargs = {"root": args.root}
    if args.category:
        kwargs["category"] = args.category
    datamodule = cls(**kwargs)
    datamodule.prepare_data()
    print(f"Downloaded {args.dataset} to {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
