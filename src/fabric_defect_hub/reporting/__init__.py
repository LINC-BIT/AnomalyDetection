"""AnomalyDetection reporting, run logs, and LaTeX generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fabric_defect_hub.core.provenance import collect_provenance
from fabric_defect_hub.core.serialization import experiment_result_to_dict
from fabric_defect_hub.core.types import ExperimentResult
from fabric_defect_hub.reporting.latex_generator import generate_latex_table


def save_leaderboard(results: list[ExperimentResult], path: str | Path) -> Path:
    """Write a flat leaderboard as CSV or Markdown based on the suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _columns(results)
    rows = [_flatten(result, columns) for result in results]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    elif suffix in {".md", ".markdown"}:
        path.write_text(_markdown(columns, rows), encoding="utf-8")
    else:
        raise ValueError("leaderboard path must end in .csv, .md, or .markdown")
    return path


def _columns(results: list[ExperimentResult]) -> list[str]:
    metric_names = sorted({name for result in results for name in result.metrics})
    return [
        "experiment_id",
        "model",
        "backend",
        "task",
        "dataset",
        "split",
        "device",
        "engine",
        "precision",
        "input_size",
        *metric_names,
    ]


def _flatten(result: ExperimentResult, columns: list[str]) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": result.experiment_id,
        "model": result.model.name,
        "backend": result.model.backend,
        "task": result.model.task,
        "dataset": result.dataset.name,
        "split": result.dataset.split,
        "device": result.runtime.device,
        "engine": result.runtime.engine,
        "precision": result.runtime.precision,
        "input_size": "x".join(str(value) for value in result.runtime.input_size),
    }
    row.update(result.metrics)
    return {column: row.get(column, "") for column in columns}


def append_run_log(result: ExperimentResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**experiment_result_to_dict(result), "provenance": collect_provenance()}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def read_run_log(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_run_log_rows(rows: list[dict]) -> tuple[list[str], list[list[object]]]:
    metric_columns = sorted({key for row in rows for key in row.get("metrics", {})})
    columns = ["timestamp_utc", "model", "backend", "task", "dataset", "device", *metric_columns]
    table = []
    for row in rows:
        model = row.get("model", {})
        dataset = row.get("dataset", {})
        runtime = row.get("runtime", {})
        provenance = row.get("provenance", {})
        metrics = row.get("metrics", {})
        record = {
            "timestamp_utc": provenance.get("timestamp_utc", ""),
            "model": model.get("name", ""),
            "backend": model.get("backend", ""),
            "task": model.get("task", ""),
            "dataset": dataset.get("name", ""),
            "device": runtime.get("device", ""),
            **metrics,
        }
        table.append([record.get(column, "") for column in columns])
    return columns, table


def latest_run_per_model(rows: list[dict]) -> list[dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        name = row.get("model", {}).get("name", "")
        timestamp = row.get("provenance", {}).get("timestamp_utc", "")
        current_timestamp = latest.get(name, {}).get("provenance", {}).get("timestamp_utc", "")
        if name not in latest or timestamp > current_timestamp:
            latest[name] = row
    return list(latest.values())


def _markdown(columns: list[str], rows: list[dict[str, object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(cell(row[column]) for column in columns) + " |" for row in rows)
    return "\n".join(lines) + "\n"


__all__ = [
    "save_leaderboard",
    "append_run_log",
    "read_run_log",
    "flatten_run_log_rows",
    "latest_run_per_model",
    "generate_latex_table",
]
