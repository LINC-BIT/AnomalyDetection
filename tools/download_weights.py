#!/usr/bin/env python
"""Download published checkpoints from Hugging Face Hub into the project cache.

Every checkpoint::

    python tools/download_weights.py --all

One checkpoint, named by its path inside the repository::

    python tools/download_weights.py textile/artifacts/models/published/yolov8n.pt \
        --output textile/artifacts/models/published/yolov8n.pt

``--all`` writes every file of the published-weight repository at its runtime
path under the project root. A single download requires ``--output``: the
destination path is part of the runtime contract, so it must match the path the
manifest declares. The repository defaults to the published-weight repository;
pass ``--repo-id`` to read another one, optionally as a first positional
argument.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "AuroraLeeeeee/AnomalyDetection-textile-weights"


def download_all(repo_id: str, revision: str) -> int:
    from huggingface_hub import snapshot_download

    print(f"Downloading every checkpoint of {repo_id}@{revision} into {PROJECT_ROOT}")
    result = snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(PROJECT_ROOT))
    if isinstance(result, list):
        print(f"Downloaded {len(result)} files")
    print("Check the result with: adh inventory")
    return 0


def download_one(repo_id: str, filename: str, revision: str, output: Path) -> int:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(Path(cached).read_bytes())
    print(f"Downloaded {repo_id}/{filename}@{revision} to {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download published checkpoints from Hugging Face Hub",
        epilog=f"Default repository: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="the path of the file inside the repository, optionally preceded by the repository id",
    )
    parser.add_argument("--all", action="store_true", help="download every published checkpoint")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repository to read")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--output",
        type=Path,
        help="destination of a single download; part of the runtime contract",
    )
    args = parser.parse_args()

    if args.all:
        if args.paths or args.output:
            parser.error("--all takes neither a file path nor --output")
        return download_all(args.repo_id, args.revision)

    if len(args.paths) == 1:
        repo_id, filename = args.repo_id, args.paths[0]
    elif len(args.paths) == 2:
        repo_id, filename = args.paths
    else:
        parser.error("give the path of one file inside the repository, or --all")
    if args.output is None:
        parser.error("--output is required for a single download")
    return download_one(repo_id, filename, args.revision, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
