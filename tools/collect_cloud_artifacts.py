"""Collect a self-contained, checksummed bundle from a cloud training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _copy_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".DS_Store"))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(run_id: str, verify_only: bool = False) -> Path:
    bundle = ROOT / "artifacts" / "cloud_runs" / run_id
    if not verify_only:
        bundle.mkdir(parents=True, exist_ok=True)
        _copy_tree(ROOT / "artifacts" / "models", bundle / "artifacts/models")
        _copy_tree(ROOT / "artifacts" / "runtime", bundle / "artifacts/runtime")
        _copy_tree(ROOT / "artifacts" / "anomaly_maps", bundle / "artifacts/anomaly_maps")
        _copy_tree(ROOT / "runs", bundle / "runs")
        _copy_tree(ROOT / "artifacts" / "training_runs", bundle / "artifacts/training_runs")
        _copy_tree(ROOT / "configs", bundle / "configs")
        _copy_if_exists(ROOT / "runs/leaderboard_log.jsonl", bundle / "runs/leaderboard_log.jsonl")
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except Exception:
            commit = "unknown"
        metadata = {
            "run_id": run_id,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "python": sys.version,
            "argv": sys.argv,
            "cwd": str(ROOT),
            "env": {k: v for k, v in os.environ.items() if k.endswith("_ROOT") or k.startswith("FDH_")},
        }
        (bundle / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        try:
            freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], cwd=ROOT, text=True)
            (bundle / "pip-freeze.txt").write_text(freeze)
        except Exception as exc:
            (bundle / "pip-freeze.error.txt").write_text(str(exc) + "\n")

    if not bundle.exists():
        raise SystemExit(f"bundle not found: {bundle}")
    checksums = bundle / "SHA256SUMS.txt"
    rows = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file() and p != checksums):
        rows.append(f"{_sha256(path)}  {path.relative_to(bundle)}")
    checksums.write_text("\n".join(rows) + ("\n" if rows else ""))
    required = [bundle / "metadata.json", bundle / "configs", bundle / "artifacts/models"]
    missing = [str(p.relative_to(bundle)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("incomplete cloud bundle; missing: " + ", ".join(missing))
    print(bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="also create artifacts/cloud_runs/<run-id>.tar.gz for one-file download",
    )
    args = parser.parse_args()
    bundle = collect(args.run_id, args.verify_only)
    if args.archive:
        archive = shutil.make_archive(str(bundle), "gztar", root_dir=bundle.parent, base_dir=bundle.name)
        print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
