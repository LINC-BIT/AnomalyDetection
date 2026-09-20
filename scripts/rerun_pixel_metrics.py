#!/usr/bin/env python3
"""Re-score the mask models so their pixel ranking metrics exist.

The 2026-09-20 fix made the torchvision adapter persist the continuous score
map that `pixel_auroc` / `pixel_aupro` / `iap` are computed from. Rows written
before it never had a map on disk, so those three columns cannot be recovered
by re-plotting: the models have to be scored again.

Two deliberate choices:

* profiling and the resolution sweep stay **off**. Those numbers were measured
  on a GPU and are not comparable with a CPU pass, so only the accuracy metrics
  are re-measured;
* the rows go to a **separate** log, so the existing snapshot is never edited in
  place.

What this does and does not measure, per model:

* **measured again** — the whole 350-image test split, so every accuracy metric
  is complete, including the `pixel_auroc` / `pixel_aupro` / `iap` the fix
  unlocked;
* **not measured** — `fps`, `latency_ms_*`, `peak_memory_mb`, `model_size_mb`,
  `flops_g`, `lmei`, `instantaneous_fps_*` (profiling) and
  `resolution_slope_*` (the sweep).

That is safe because the plot tooling folds rows **one metric at a time**
(`benchmark_plots.data._merge_rows`): a metric this run did not report keeps the
value the earlier run measured, and a metric it did report wins. Nothing is
zeroed out for not being re-measured.

Run in the environment the snapshot came from (`anomalib_env`):

    PYTHONPATH=src python scripts/rerun_pixel_metrics.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabric_defect_hub.application.benchmark import run_benchmark

# The mask models: their metrics come from `SegmentationEvaluator`, which used
# to have no map to sweep a threshold over.
DEFAULT_MODELS = ["UNet++ · ZJU-Leaper", "DeepLabV3+ · ZJU-Leaper", "Mask R-CNN · ZJU-Leaper"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-log", default="runs/benchmark_rows_pixel_rerun.jsonl")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dataset", default="ZJU-Leaper")
    parser.add_argument("--texture", default="Pattern 1-4 (Train patterns)")
    parser.add_argument("--shot-mode", default="Few-shot")
    args = parser.parse_args()

    for columns, rows, status, _ in run_benchmark(
        dataset_label=args.dataset,
        texture_label=args.texture,
        shot_mode=args.shot_mode,
        model_labels=args.models,
        include_profiling=False,
        include_resolution_sweep=False,
        calibrate_thresholds=True,
        row_log_path=args.row_log,
        run_log_path=None,
    ):
        print(f"status: {status}", flush=True)
        for row in rows:
            name = row[0] if row else "?"
            print(f"  scored: {name}", flush=True)

    print(f"rows -> {args.row_log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
