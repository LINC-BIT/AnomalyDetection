"""Load and audit one reproducible local benchmark snapshot."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED_MODELS = 19
DEFAULT_LOG = Path("runs/benchmark_rows.jsonl")
# Dated JSON written by `application.benchmark.write_benchmark_snapshot`, one
# per benchmark run — the record the figures read.
SNAPSHOT_ROOT = Path("runs/benchmark_snapshots")


@dataclass(frozen=True)
class Snapshot:
    rows: list[dict[str, Any]]
    commit: str
    dataset: str
    samples: int
    timestamp: str
    source: str = ""


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _eligible(row: dict[str, Any]) -> bool:
    selection = row.get("selection", {})
    return (
        row.get("status") == "ok"
        and row.get("dataset") == "ZJU-Leaper"
        and row.get("samples") == 350
        and selection.get("shot_mode") == "Few-shot"
        and selection.get("include_profiling") is True
        and selection.get("calibrate_thresholds") is True
        and selection.get("texture") == ["Pattern 1-4 (Train patterns)"]
    )


def _merge_rows(base: dict[str, Any], newer: dict[str, Any]) -> dict[str, Any]:
    """`newer` folded onto `base`, **one metric at a time**.

    A re-run rarely measures everything the previous one did: re-scoring models
    after a metric fix walks the whole test split but skips profiling, so its
    row carries no `fps` / latency / memory / FLOPs, and a resolution sweep
    usually stays off too. Replacing the row wholesale blanks those columns for
    exactly the models that were re-scored, and three figures lose their bars
    without anything looking wrong.

    So: a metric the newer row does not report keeps the older value, and a
    metric it does report always wins. "The latest run is the truth" still holds
    for everything the latest run actually measured. Row-level fields
    (`provenance`, `selection`, `warnings`, `status`) come from the newer row,
    because they describe the run, not a metric.
    """

    merged = {**base, **newer}
    merged["metrics"] = dict(base.get("metrics") or {})
    for key, value in (newer.get("metrics") or {}).items():
        if value is not None:
            merged["metrics"][key] = value
    return merged


def _benchmark_rows(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only rows shaped like a benchmark row.

    `runs/benchmark_snapshots/` is a directory a driver scans, and a stray JSON
    in it must not take the figures down. A row is usable when it carries its
    `provenance` (for the timestamp) and a nested `metrics` mapping; anything
    else is skipped rather than half-read.
    """

    return [
        row for row in raw
        if isinstance(row, dict) and isinstance(row.get("provenance"), dict) and isinstance(row.get("metrics"), dict)
    ]


def _newest_per_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model: the latest measurement of each of its metrics.

    The benchmark log only ever appends. Re-scoring a model therefore leaves
    more than one row for it, and `load_snapshot` used to hand all of them to
    the figures, drawing the model twice. Folding oldest-to-newest with
    `_merge_rows` gives one row per model that keeps every metric the newest run
    did not re-measure.
    """

    newest: dict[str, dict[str, Any]] = {}
    ordered = sorted(rows, key=lambda row: _parse_time(row["provenance"]["timestamp_utc"]))
    for row in ordered:
        model = row["model"]
        current = newest.get(model)
        newest[model] = row if current is None else _merge_rows(current, row)
    return list(newest.values())


def load_snapshot(path: str | Path = DEFAULT_LOG, commit: str | None = None) -> Snapshot:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not _eligible(row):
                continue
            provenance = row.get("provenance", {})
            key = (provenance.get("git_commit", ""), row["dataset"], int(row["samples"]))
            groups.setdefault(key, []).append(row)

    if commit is not None:
        candidates = [rows for (row_commit, _, _), rows in groups.items() if row_commit == commit]
    else:
        complete = [rows for rows in groups.values() if len({row["model"] for row in rows}) == EXPECTED_MODELS]
        candidates = complete or list(groups.values())

    if not candidates:
        raise ValueError("No eligible local benchmark rows were found.")

    def group_time(rows: list[dict[str, Any]]) -> datetime:
        return max(_parse_time(row["provenance"]["timestamp_utc"]) for row in rows)

    rows = _newest_per_model(max(candidates, key=group_time))
    models = {row["model"] for row in rows}
    if len(models) != EXPECTED_MODELS:
        raise ValueError(f"Selected snapshot has {len(models)} models; expected {EXPECTED_MODELS}.")
    rows = sorted(rows, key=lambda row: row["model"])
    first = rows[0]
    return Snapshot(
        rows=rows,
        commit=first["provenance"]["git_commit"],
        dataset=first["dataset"],
        samples=int(first["samples"]),
        timestamp=max(row["provenance"]["timestamp_utc"] for row in rows),
        source=str(path),
    )


def _snapshot_files(root: str | Path = SNAPSHOT_ROOT) -> list[Path]:
    """Dated snapshot JSONs, oldest first. The name starts with the timestamp,
    so a filename sort is a time sort and no file has to be opened to sort."""

    return sorted(Path(root).glob("*.json"))


def load_dated_snapshot(path: str | Path) -> Snapshot:
    """One dated snapshot JSON, as written by a benchmark run."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = _newest_per_model(list(payload.get("rows") or []))
    if not rows:
        raise ValueError(f"{path} holds no rows.")
    first = rows[0]
    return Snapshot(
        rows=sorted(rows, key=lambda row: row["model"]),
        commit=str(first.get("provenance", {}).get("git_commit", "")),
        dataset=str(payload.get("dataset") or first.get("dataset", "")),
        samples=int(first.get("samples") or 0),
        timestamp=str(payload.get("created_utc") or max(row["provenance"]["timestamp_utc"] for row in rows)),
        source=str(path),
    )


def load_latest_snapshot(
    root: str | Path = SNAPSHOT_ROOT,
    log: str | Path = DEFAULT_LOG,
    commit: str | None = None,
) -> Snapshot:
    """The newest dated snapshots, merged per model, newest row winning.

    The figures read the benchmark's own JSON output — dated files holding every
    metric of a run, technical and overhead together — never a hand-kept
    document.

    Merging across files is what makes a partial re-run usable. Re-scoring three
    models writes a three-row snapshot; taking only the newest file would drop
    the other sixteen models from every figure. Instead the files are read
    newest first and each model keeps the first (newest) row seen, so a re-run
    contributes exactly the models it re-scored and everything else comes from
    the last complete run.

    Falls back to the append-only JSONL for a checkout that has not run a
    benchmark since the dated snapshots were introduced; it is re-grouped and
    de-duplicated the same way, so the figures see the same shape either way.
    """

    merged: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for path in reversed(_snapshot_files(root)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = _newest_per_model(_benchmark_rows(list(payload.get("rows") or [])))
        if not rows:
            continue
        if commit is not None and str(rows[0].get("provenance", {}).get("git_commit", "")) != commit:
            continue
        for row in rows:
            current = merged.get(row["model"])
            # Files are read newest first, so `current` is the newer measurement
            # and `row` is the older one being folded underneath it.
            merged[row["model"]] = row if current is None else _merge_rows(row, current)
        sources.append(str(path))
        if len(merged) >= EXPECTED_MODELS:
            break

    if len(merged) >= EXPECTED_MODELS:
        rows = sorted(merged.values(), key=lambda row: row["model"])
        return Snapshot(
            rows=rows,
            commit=str(rows[0].get("provenance", {}).get("git_commit", "")),
            dataset=str(rows[0].get("dataset", "")),
            samples=int(rows[0].get("samples") or 0),
            timestamp=max(row["provenance"]["timestamp_utc"] for row in rows),
            source=" + ".join(sources),
        )
    return load_snapshot(log, commit)


def task(row: dict[str, Any]) -> str:
    name = row["model"].lower()
    metrics = row.get("metrics", {})
    if "map_50" in metrics:
        return "detection"
    if "miou" in metrics:
        return "segmentation"
    return "anomaly"


def family(row: dict[str, Any]) -> str:
    """The family a row is grouped under in the figures and the audit table.

    Only two families are reported. A segmentation model predicts a binary
    pixel mask that is scored with pixel-level overlap metrics, which is
    pixel-level defect detection, so it is grouped with the anomaly models
    rather than given a third colour and a third legend entry.
    """

    return "detection" if task(row) == "detection" else "anomaly"


def metric_rows(snapshot: Snapshot, metrics: list[str]) -> list[dict[str, Any]]:
    output = []
    for row in snapshot.rows:
        values = row.get("metrics", {})
        available = [metric for metric in metrics if values.get(metric) is not None]
        for metric in metrics:
            output.append({
                "model": row["model"],
                "family": family(row),
                "metric": metric,
                "value": values.get(metric),
                "available": values.get(metric) is not None,
                "reason": "" if values.get(metric) is not None else "not reported by this task/backend",
                "commit": snapshot.commit,
                "timestamp_utc": row["provenance"]["timestamp_utc"],
                "memory_measurement_kind": values.get("memory_measurement_kind", ""),
            })
    return output


def write_audit(snapshot: Snapshot, output_path: str | Path, metric_groups: dict[str, list[str]]) -> None:
    rows: list[dict[str, Any]] = []
    for chart, metrics in metric_groups.items():
        for item in metric_rows(snapshot, metrics):
            item["chart"] = chart
            rows.append(item)
    fields = ["chart", "model", "family", "metric", "value", "available", "reason", "commit", "timestamp_utc", "memory_measurement_kind"]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_snapshot_metadata(snapshot: Snapshot, output_path: str | Path) -> None:
    payload = {
        "source": str(DEFAULT_LOG),
        "dataset": snapshot.dataset,
        "samples": snapshot.samples,
        "model_count": len(snapshot.rows),
        "git_commit": snapshot.commit,
        "latest_timestamp_utc": snapshot.timestamp,
        "models": [row["model"] for row in snapshot.rows],
    }
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
