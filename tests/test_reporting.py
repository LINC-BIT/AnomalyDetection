import csv
import json

import pytest

from fabric_defect_hub.core.types import DatasetInfo, ExperimentResult, ModelInfo, RuntimeInfo
from fabric_defect_hub.reporting import (
    append_run_log,
    flatten_run_log_rows,
    latest_run_per_model,
    read_run_log,
    save_leaderboard,
)


def _result():
    return ExperimentResult(
        experiment_id="exp-1",
        model=ModelInfo(name="model", backend="backend", task="detection"),
        dataset=DatasetInfo(name="dataset", split="test"),
        runtime=RuntimeInfo(device="cpu", engine="onnxruntime", precision="fp32", input_size=(640, 640)),
        metrics={"map50": 0.8, "fps": 100.0},
    )


def test_save_leaderboard_csv(tmp_path):
    path = save_leaderboard([_result()], tmp_path / "board.csv")
    with path.open() as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["experiment_id"] == "exp-1"
    assert rows[0]["map50"] == "0.8"


def test_save_leaderboard_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="must end"):
        save_leaderboard([_result()], tmp_path / "board.json")


def test_append_run_log_writes_one_json_line_with_provenance(tmp_path):
    path = append_run_log(_result(), tmp_path / "runs_log.jsonl")

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["experiment_id"] == "exp-1"
    assert row["metrics"] == {"map50": 0.8, "fps": 100.0}
    assert {
        "timestamp_utc",
        "git_commit",
        "hostname",
        "vendored_components",
    } <= set(row["provenance"])


def test_append_run_log_appends_across_different_metric_shapes(tmp_path):
    detection_result = _result()
    anomaly_result = ExperimentResult(
        experiment_id="exp-2",
        model=ModelInfo(name="patchcore", backend="anomalib", task="anomaly"),
        dataset=DatasetInfo(name="dataset", split="test"),
        runtime=RuntimeInfo(device="cpu", engine="pytorch", precision="fp32", input_size=(256, 256)),
        metrics={"image_auroc": 0.95, "pixel_aupro": 0.9, "iap": 0.7},
    )

    log_path = tmp_path / "runs_log.jsonl"
    append_run_log(detection_result, log_path)
    append_run_log(anomaly_result, log_path)

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["metrics"] == {"map50": 0.8, "fps": 100.0}
    assert rows[1]["metrics"] == {"image_auroc": 0.95, "pixel_aupro": 0.9, "iap": 0.7}


def test_read_run_log_missing_file_returns_empty_list(tmp_path):
    assert read_run_log(tmp_path / "does-not-exist.jsonl") == []


def test_read_run_log_round_trips_append_run_log(tmp_path):
    log_path = tmp_path / "runs_log.jsonl"
    append_run_log(_result(), log_path)
    append_run_log(_result(), log_path)

    rows = read_run_log(log_path)
    assert len(rows) == 2
    assert rows[0]["experiment_id"] == "exp-1"
    assert rows[0]["metrics"] == {"map50": 0.8, "fps": 100.0}


def test_flatten_run_log_rows_unions_metric_keys_across_tasks():
    detection_row = {
        "model": {"name": "yolo", "backend": "ultralytics", "task": "detection"},
        "dataset": {"name": "zju-leaper", "split": "test"},
        "runtime": {"device": "cpu"},
        "metrics": {"map50": 0.8},
        "provenance": {"timestamp_utc": "2026-01-01T00:00:00+00:00"},
    }
    anomaly_row = {
        "model": {"name": "patchcore", "backend": "anomalib", "task": "anomaly"},
        "dataset": {"name": "mvtec-ad", "split": "test"},
        "runtime": {"device": "cpu"},
        "metrics": {"image_auroc": 0.95},
        "provenance": {"timestamp_utc": "2026-01-02T00:00:00+00:00"},
    }

    columns, table = flatten_run_log_rows([detection_row, anomaly_row])

    assert columns == [
        "timestamp_utc", "model", "backend", "task", "dataset", "device", "image_auroc", "map50",
    ]
    row_by_model = {row[columns.index("model")]: row for row in table}
    assert row_by_model["yolo"][columns.index("map50")] == 0.8
    assert row_by_model["yolo"][columns.index("image_auroc")] == ""
    assert row_by_model["patchcore"][columns.index("image_auroc")] == 0.95


def test_flatten_run_log_rows_empty_input():
    columns, table = flatten_run_log_rows([])
    assert table == []
    assert columns == ["timestamp_utc", "model", "backend", "task", "dataset", "device"]


def test_latest_run_per_model_keeps_only_the_most_recent_row_per_model():
    older = {
        "model": {"name": "yolo"},
        "provenance": {"timestamp_utc": "2026-01-01T00:00:00+00:00"},
        "metrics": {"map50": 0.5},
    }
    newer = {
        "model": {"name": "yolo"},
        "provenance": {"timestamp_utc": "2026-01-02T00:00:00+00:00"},
        "metrics": {"map50": 0.9},
    }
    other_model = {
        "model": {"name": "patchcore"},
        "provenance": {"timestamp_utc": "2026-01-01T12:00:00+00:00"},
        "metrics": {"image_auroc": 0.8},
    }

    latest = latest_run_per_model([older, newer, other_model])

    assert len(latest) == 2
    by_name = {row["model"]["name"]: row for row in latest}
    assert by_name["yolo"]["metrics"]["map50"] == 0.9
    assert by_name["patchcore"]["metrics"]["image_auroc"] == 0.8


def test_append_run_log_survives_concurrent_appends(tmp_path):
    """Benchmark models are scored in separate processes, so rows can land at
    the same moment; the file has to stay one valid JSON object per line (a
    single interleaved line makes `read_run_log` raise for the whole file)."""

    import threading

    log = tmp_path / "leaderboard_log.jsonl"
    padded = _result()
    padded.metrics = {f"metric_{index:03d}": float(index) for index in range(200)}
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for _ in range(10):
                append_run_log(padded, log)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    rows = read_run_log(log)
    assert len(rows) == 80
    assert all(row["experiment_id"] == "exp-1" for row in rows)
