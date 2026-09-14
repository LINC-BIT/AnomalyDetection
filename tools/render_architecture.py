#!/usr/bin/env python
"""Render the system architecture diagram as a dependency-free SVG.

Same approach as `reporting/training_curves.py`: the SVG is assembled from
plain strings (`<rect>`, `<line>`, `<text>`), with no plotting library, so it
renders anywhere the README is read and can be regenerated anywhere Python
runs.

    python tools/render_architecture.py
    python tools/render_architecture.py --output docs/images/architecture.svg
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "images" / "architecture.svg"

WIDTH, HEIGHT = 1240, 1000
FONT = 'font-family="Helvetica, Arial, sans-serif"'
INK = "#0f172a"
BODY = "#334155"
LINE = "#475569"


def box(x, y, w, h, title, lines=(), fill="#f8fafc", size=15) -> list[str]:
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
        f'stroke="{LINE}" stroke-width="1.5"/>',
        f'<text x="{x + 14}" y="{y + 24}" {FONT} font-size="{size}" font-weight="bold" '
        f'fill="{INK}">{escape(title)}</text>',
    ]
    for index, line in enumerate(lines):
        out.append(
            f'<text x="{x + 14}" y="{y + 47 + index * 18}" {FONT} font-size="12.5" '
            f'fill="{BODY}">{escape(line)}</text>'
        )
    return out


def band(x, y, w, h, title, fill="#eef2f7") -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
        f'stroke="{LINE}" stroke-width="1.5"/>',
        f'<text x="{x + 16}" y="{y + 26}" {FONT} font-size="15" font-weight="bold" '
        f'fill="{INK}">{escape(title)}</text>',
    ]


def arrow(x1, y1, x2, y2, label="", label_x=None) -> list[str]:
    out = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" '
        f'stroke-width="1.8" marker-end="url(#head)"/>'
    ]
    if label:
        anchor_x = label_x if label_x is not None else (x1 + x2) / 2 + 8
        out.append(
            f'<text x="{anchor_x}" y="{(y1 + y2) / 2 + 4}" {FONT} font-size="12" '
            f'fill="{BODY}">{escape(label)}</text>'
        )
    return out


def build() -> str:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">',
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        f'<text x="40" y="38" {FONT} font-size="22" font-weight="bold" fill="{INK}">'
        "AnomalyDetection - system architecture</text>",
    ]

    # -- frontend ----------------------------------------------------------
    out += band(40, 60, 1160, 140, "Frontend - Gradio web UI")
    out += box(60, 105, 350, 80, "Model session", ["task type - application domain - model",
                                                    "dataset, split, shot mode, random images"])
    out += box(445, 105, 350, 80, "Benchmark", ["dataset, shot mode, model identifiers",
                                                "optional profiling and resolution sweep"])
    out += box(830, 105, 350, 80, "Run history", ["read a saved JSON / JSONL report",
                                                  "filter by metric, list past runs"])
    out += arrow(620, 200, 620, 248, "event handlers")

    # -- application -------------------------------------------------------
    out += band(40, 250, 1160, 130, "Application layer - the only business boundary for the CLI and the web UI")
    out += box(60, 296, 560, 68, "Application services (`application/`)",
               ["workspace (model session) - benchmark - run records"])
    out += box(650, 296, 530, 68, "Command line (`adh`)",
               ["train - predict - evaluate - benchmark - train-all - doctor"])
    out += arrow(620, 380, 620, 418, "shared services")

    # -- core --------------------------------------------------------------
    out += band(40, 420, 1160, 120, "Core - the interfaces every layer shares")
    out += box(60, 462, 1120, 66, "Contracts and registries",
               ["ModelAdapter / DatasetAdapter / Evaluator contracts - capability declarations",
                "model, dataset and evaluator registries - Sample / Prediction - provenance - configuration"],
               fill="#f8fafc", size=14)

    # -- backends ----------------------------------------------------------
    out += band(40, 580, 1160, 230, "Backends - one interface, many implementations")
    out += box(60, 630, 270, 150, "Model backends",
               ["ultralytics", "torchvision", "anomalib", "dinomaly", "moeclip", "mambaad"])
    out += box(350, 630, 270, 150, "Dataset adapters",
               ["9 registered datasets", "textile: 6", "general: 3", "declared tasks and roles"])
    out += box(640, 630, 270, 150, "Evaluation",
               ["detection (boxes)", "segmentation (pixels)", "anomaly (image / pixel)", "industrial - cross-domain"])
    out += box(930, 630, 250, 150, "Profiling / quantization",
               ["pytorch - onnxruntime", "tensorrt - flops - power", "fp16 / INT8 ONNX export"])
    for x in (195, 485, 775, 1055):
        out += arrow(x, 540, x, 626)

    # -- outputs -----------------------------------------------------------
    for x in (195, 485, 775, 1055):
        out.append(f'<line x1="{x}" y1="780" x2="{x}" y2="815" stroke="{LINE}" stroke-width="1.8"/>')
    out.append(f'<line x1="195" y1="815" x2="1055" y2="815" stroke="{LINE}" stroke-width="1.8"/>')
    out += arrow(620, 815, 620, 856, "artifacts and logs")

    out += band(40, 860, 1160, 110, "Outputs")
    out += box(60, 896, 1120, 56, "Run products",
               ["published slots (`<domain>/artifacts/models/published/`) - weight_manifest.jsonl",
                "leaderboard_log.jsonl - results/ - anomaly maps - benchmark reports - exported models"],
               fill="#f8fafc", size=14)

    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the system architecture diagram")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
