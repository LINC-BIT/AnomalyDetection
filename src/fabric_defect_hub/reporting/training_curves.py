"""CSV training-history discovery and dependency-free SVG line charts."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class TrainingCurve:
    source: Path
    x_name: str
    x_values: list[float]
    series: dict[str, list[float]]


def load_training_curve(path: str | Path) -> TrainingCurve:
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0]:
        raise ValueError(f"training history {source} has no rows")

    headers = [header.strip() for header in rows[0] if header]
    x_name = next(
        (header for preferred in ("iteration", "step", "epoch") for header in headers if header.lower() == preferred),
        headers[0],
    )
    x_values = [_number(row.get(x_name), row_index) for row_index, row in enumerate(rows)]
    series: dict[str, list[float]] = {}
    for header in headers:
        if header == x_name:
            continue
        values = [_number(row.get(header), math.nan) for row in rows]
        if any(math.isfinite(value) for value in values):
            series[header] = values
    if not series:
        raise ValueError(f"training history {source} has no numeric metric columns")
    return TrainingCurve(source=source, x_name=x_name, x_values=x_values, series=series)


def discover_training_histories(paths: list[str | Path]) -> list[Path]:
    histories: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.name in {"history.csv", "results.csv"}:
            histories.add(path)
        elif path.is_dir():
            histories.update(candidate for name in ("history.csv", "results.csv") for candidate in path.rglob(name))
    return sorted(histories)


def render_training_curve_svg(curve: TrainingCurve, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height, margin = 1100, 650, 70
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    finite = [value for values in curve.series.values() for value in values if math.isfinite(value)]
    x_min, x_max = min(curve.x_values), max(curve.x_values)
    y_min, y_max = min(finite), max(finite)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        x = margin + (x_value - x_min) / (x_max - x_min) * plot_width
        y = height - margin - (y_value - y_min) / (y_max - y_min) * plot_height
        return x, y

    palette = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="32" font-family="sans-serif" font-size="20">{escape(curve.source.name)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#475569"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#475569"/>',
        f'<text x="{width - margin}" y="{height - 25}" text-anchor="end" font-family="sans-serif" font-size="14">{escape(curve.x_name)}</text>',
        f'<text x="{margin - 8}" y="{height - margin + 20}" text-anchor="end" font-family="sans-serif" font-size="12">{x_min:g}</text>',
        f'<text x="{width - margin}" y="{height - margin + 20}" text-anchor="end" font-family="sans-serif" font-size="12">{x_max:g}</text>',
        f'<text x="{margin - 8}" y="{height - margin}" text-anchor="end" font-family="sans-serif" font-size="12">{y_min:g}</text>',
        f'<text x="{margin - 8}" y="{margin + 5}" text-anchor="end" font-family="sans-serif" font-size="12">{y_max:g}</text>',
    ]
    for index, (name, values) in enumerate(curve.series.items()):
        color = palette[index % len(palette)]
        segments: list[str] = []
        for x_value, y_value in zip(curve.x_values, values):
            if math.isfinite(y_value):
                x, y = point(x_value, y_value)
                segments.append(f"{x:.2f},{y:.2f}")
        if len(segments) >= 2:
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(segments)}"/>')
        legend_y = 56 + 22 * index
        lines.append(f'<rect x="{width - 330}" y="{legend_y - 11}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{width - 312}" y="{legend_y}" font-family="sans-serif" font-size="13">{escape(name)}</text>')
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _number(value: object, fallback: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return fallback
