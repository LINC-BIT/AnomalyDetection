#!/usr/bin/env python
"""Download registered datasets into the project data directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a registered dataset")
    parser.add_argument("dataset", choices=("mvtec-ad", "visa", "zju-leaper"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--category", default=None)
    parser.add_argument(
        "--repo-id",
        default="AnupamaBandara/ZLU_Leaper",
        help="Hugging Face dataset repository for zju-leaper",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "mvtec-ad":
        from huggingface_hub import snapshot_download

        patterns = [f"{args.category}/**"] if args.category else None
        snapshot_download(
            repo_id="Voxel51/mvtec-ad",
            repo_type="dataset",
            local_dir=args.root,
            allow_patterns=patterns,
        )
        print(f"Downloaded Voxel51/mvtec-ad to {args.root}")
        return 0

    if args.dataset == "zju-leaper":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.root,
        )
        print(f"Downloaded {args.repo_id} to {args.root}")
        return 0

    from anomalib.data import MVTecAD, Visa

    cls = Visa
    kwargs = {"root": args.root}
    if args.category:
        kwargs["category"] = args.category
    datamodule = cls(**kwargs)
    datamodule.prepare_data()
    print(f"Downloaded {args.dataset} to {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
