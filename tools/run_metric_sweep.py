#!/usr/bin/env python
"""Measure every implemented metric over every model that has weights.

A thin front end over `fabric_defect_hub.metric_sweep` — the discovery,
grouping and failure handling all live there, so the Gradio app or a
cloud-side automation script can run the same sweep without this file.

    # what would be swept, nothing measured
    python tools/run_metric_sweep.py --dataset zju-leaper --list

    # accuracy only, every trained model
    python tools/run_metric_sweep.py --dataset zju-leaper --groups accuracy

    # everything, including the cross-domain pattern sweep
    python tools/run_metric_sweep.py --dataset zju-leaper \\
        --pattern 1,2,3,4 --cross-domain-patterns 5,6,7,8 \\
        --anomaly-map-dir artifacts/runtime/sweep_maps

Results land in JSONL (one row per model x metric group) for secondary
analysis; nothing is aggregated into a report here on purpose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabric_defect_hub.metric_sweep import (  # noqa: E402
    METRIC_GROUPS,
    SweepRequest,
    SweepRow,
    broken_published_links,
    discover_trained_models,
    run_sweep,
    summarize,
)
from fabric_defect_hub.runtime_device import resolve_torch_device  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _warn_broken(broken: list[tuple[Path, str]]) -> None:
    """Dangling published links are not "model not trained" — say so loudly,
    because the fix (copy artifacts/models/ across) is nothing like the fix
    for a model that was genuinely never trained."""

    if not broken:
        return
    print(f"\n{len(broken)} published slot(s) point at missing files:", file=sys.stderr)
    for _, description in broken:
        print(f"  ! {description}", file=sys.stderr)
    print(file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="registered dataset name, e.g. zju-leaper")
    parser.add_argument("--dataset-root")
    parser.add_argument("--project-root", default=str(_PROJECT_ROOT))
    parser.add_argument(
        "--output", default=None,
        help="JSONL destination (default artifacts/runtime/metric_sweep.jsonl)",
    )
    parser.add_argument(
        "--groups", default=",".join(METRIC_GROUPS),
        help=f"comma-separated subset of {list(METRIC_GROUPS)}",
    )
    parser.add_argument("--models", help="comma-separated model keys/variants; default is all trained")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--pattern", help="ZJU-Leaper source pattern filter, e.g. '1,2,3,4'")
    parser.add_argument("--category", help="MVTec-AD category filter")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="auto",
        help="auto (CUDA > MPS > CPU), cpu, mps, cuda, or cuda:<index>",
    )
    parser.add_argument(
        "--anomaly-map-dir",
        help=(
            "persist anomaly maps here (default: artifacts/runtime/anomaly_maps); "
            "required for pixel_auroc / pixel_aupro / iap on anomaly models"
        ),
    )
    parser.add_argument("--cross-domain-patterns", help="held-out patterns, e.g. '5,6,7,8'")
    parser.add_argument("--cross-domain-k", type=int, default=3)
    parser.add_argument("--cross-domain-mode", default="worst", choices=("worst", "best"))
    parser.add_argument("--frame-budget-ms", type=float, default=33.0)
    parser.add_argument("--max-streams", type=int, default=8)
    parser.add_argument("--measured-runs", type=int, default=50)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument(
        "--power-mode", choices=("required", "disabled"),
        help="required for profiling groups: fail without a valid sensor, or explicitly disable power",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing JSONL file; by default an existing path is rejected",
    )
    parser.add_argument("--list", action="store_true", help="show what would be swept and exit")
    args = parser.parse_args()

    requested_groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())
    profiling_groups = {"runtime", "scaling", "concurrency"}
    if not args.list and profiling_groups.intersection(requested_groups) and args.power_mode is None:
        parser.error("profiling groups require explicit --power-mode required or --power-mode disabled")

    models = discover_trained_models(args.project_root)
    broken = broken_published_links(args.project_root)
    if args.list:
        print(f"{len(models)} model(s) with weights on disk:\n")
        for model in models:
            kind = "link" if model.weights.is_symlink() else "file"
            print(f"  {model.label:28s} {model.backend:12s} {model.variant:26s} [{kind}] {model.weights}")
        print(f"\ngroups available: {', '.join(METRIC_GROUPS)}")
        _warn_broken(broken)
        return 0
    if not models:
        print("No trained weights found — nothing to sweep.", file=sys.stderr)
        _warn_broken(broken)
        return 0
    _warn_broken(broken)

    try:
        device = resolve_torch_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    request = SweepRequest(
        project_root=Path(args.project_root),
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        output_path=Path(args.output) if args.output
        else Path(args.project_root) / "artifacts" / "runtime" / "metric_sweep.jsonl",
        groups=requested_groups,
        models=tuple(m.strip() for m in args.models.split(",")) if args.models else None,
        split=args.split,
        num_samples=args.num_samples,
        pattern=args.pattern,
        category=args.category,
        seed=args.seed,
        device=device,
        anomaly_map_dir=Path(args.anomaly_map_dir) if args.anomaly_map_dir else None,
        cross_domain_patterns=tuple(
            p.strip() for p in args.cross_domain_patterns.split(",")
        ) if args.cross_domain_patterns else (),
        cross_domain_k=args.cross_domain_k,
        cross_domain_mode=args.cross_domain_mode,
        frame_budget_ms=args.frame_budget_ms,
        max_streams_to_try=args.max_streams,
        measured_runs=args.measured_runs,
        warmup_runs=args.warmup_runs,
        power_mode=args.power_mode or "disabled",
        overwrite=args.overwrite,
    )

    def report(row: SweepRow) -> None:
        mark = {"ok": "OK  ", "skipped": "SKIP", "failed": "FAIL"}.get(row.status, "?   ")
        detail = "" if row.status == "ok" else f"  ({row.reason})"
        print(f"  [{mark}] {row.model:26s} {row.group:12s} {row.duration_s:6.1f}s{detail}", flush=True)

    print(
        f"Sweeping {len(models)} model(s) x {len(request.groups)} group(s) "
        f"on {device} -> {request.output_path}\n"
    )
    rows = run_sweep(request, on_row=report)
    print("\n" + json.dumps(summarize(row.to_json() for row in rows), indent=2))
    print(f"\nJSONL: {request.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
