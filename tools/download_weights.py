#!/usr/bin/env python
"""Download one published weight from Hugging Face Hub into the project cache."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a model weight from Hugging Face Hub")
    parser.add_argument("repo_id", help="Hugging Face repository, for example org/anomalydetection-weights")
    parser.add_argument("filename")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=args.repo_id, filename=args.filename, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(Path(cached).read_bytes())
    print(f"Downloaded {args.repo_id}/{args.filename}@{args.revision} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
