#!/usr/bin/env python
"""Inspect and (optionally) apply `artifacts/models/` storage cleanup.

A thin front end over `fabric_defect_hub.retention` — the policy, the
protected set and the resume-safety reasoning all live in that module, so a
notebook, the Gradio app, or a cloud-side automation script can plan the same
cleanup without going through this file.

    # what would change, nothing touched (default)
    python tools/prune_artifacts.py

    # convert the pre-symlink published copies into symlinks
    python tools/prune_artifacts.py --dedupe-published --apply

    # drop registered checkpoints beyond the newest 3 per (backend, variant)
    python tools/prune_artifacts.py --prune-checkpoints --keep 3 --apply

Without `--apply` this prints the plan and exits 0 without writing anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabric_defect_hub.retention import (  # noqa: E402
    DEFAULT_KEEP_PER_VARIANT,
    apply_plan,
    plan_checkpoint_prune,
    plan_publish_dedupe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--dedupe-published", action="store_true",
        help="replace real files under published/ with symlinks into the model root",
    )
    parser.add_argument(
        "--prune-checkpoints", action="store_true",
        help="delete registered checkpoints beyond the newest --keep per (backend, variant)",
    )
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP_PER_VARIANT)
    parser.add_argument("--run-root", default=None, help="batch-run directory (default artifacts/training_runs)")
    parser.add_argument(
        "--apply", action="store_true",
        help="actually perform the plan; without this nothing is written",
    )
    args = parser.parse_args()

    if not (args.dedupe_published or args.prune_checkpoints):
        args.dedupe_published = args.prune_checkpoints = True

    total = 0
    for enabled, title, plan in (
        (args.dedupe_published, "published/ -> symlinks",
         plan_publish_dedupe(args.project_root) if args.dedupe_published else None),
        (args.prune_checkpoints, f"superseded checkpoints (keep {args.keep})",
         plan_checkpoint_prune(args.project_root, keep=args.keep, run_root=args.run_root)
         if args.prune_checkpoints else None),
    ):
        if not enabled or plan is None:
            continue
        print(f"\n== {title}")
        print(plan.render())
        for path, reason in sorted(plan.protected.items()):
            print(f"  [keep] {path}  ({reason})")
        total += plan.reclaimed_bytes
        if args.apply and plan.actions:
            applied = apply_plan(plan, confirm=True)
            print(f"  applied {len(applied)} action(s)")

    if not args.apply:
        print(f"\nDry run — nothing was written. Re-run with --apply to reclaim {total / 1024**3:.1f} GB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
