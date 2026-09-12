#!/usr/bin/env python3
"""Render SVG training curves from YOLO ``results.csv`` and Torchvision ``history.csv``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabric_defect_hub.reporting.training_curves import (
    discover_training_histories,
    load_training_curve,
    render_training_curve_svg,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="history CSVs or directories to search recursively")
    parser.add_argument("--output-dir", default="artifacts/training_curves", help="where SVG charts are written")
    args = parser.parse_args(argv)

    histories = discover_training_histories(args.paths)
    if not histories:
        parser.error("no history.csv or results.csv files found")
    output_dir = Path(args.output_dir)
    for index, history in enumerate(histories, start=1):
        curve = load_training_curve(history)
        output = render_training_curve_svg(curve, output_dir / f"{index:02d}_{history.parent.name}_{history.stem}.svg")
        print(f"{history} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
