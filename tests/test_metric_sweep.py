"""The unified metric sweep: discovery, per-group isolation, and JSONL output.

The behaviour under test is mostly about *not* failing — a sweep that dies on
the first broken backend is useless after an overnight training batch, so
most of these assert that a failure became a row rather than an exception.
"""

from __future__ import annotations

import json

import pytest

from fabric_defect_hub.metric_sweep import (
    METRIC_GROUPS,
    SweepRequest,
    SweepRow,
    discover_trained_models,
    read_sweep,
    run_sweep,
    summarize,
)


def _tree(root):
    models = root / "artifacts" / "models"
    (models / "published").mkdir(parents=True, exist_ok=True)
    return models


def _weight(models, name, size=512):
    path = models / name
    path.write_bytes(b"\0" * size)
    return path


def _manifest(models, records):
    with (models / "weight_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _record(backend, variant, path, *, recorded_at="2026-07-30T00:00:00Z", kind="trained", key=None):
    return {
        "kind": kind, "backend": backend, "variant": variant, "model_key": key,
        "recorded_at": recorded_at, "config_source_path": None,
        "artifact": {"registered_path": str(path), "published_path": None},
    }


def _request(root, **kwargs):
    defaults = dict(
        project_root=root, dataset="zju-leaper",
        output_path=root / "out" / "sweep.jsonl", groups=("accuracy",),
    )
    return SweepRequest(**{**defaults, **kwargs})


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_discovers_models_from_the_manifest(tmp_path):
    models = _tree(tmp_path)
    yolo = _weight(models, "yolov8n_run1.pt")
    patchcore = _weight(models, "PatchCore_run1.ckpt")
    _manifest(models, [
        _record("ultralytics", "yolov8n", yolo),
        _record("anomalib", "PatchCore", patchcore),
    ])

    found = discover_trained_models(tmp_path)

    assert {(m.backend, m.variant) for m in found} == {
        ("ultralytics", "yolov8n"), ("anomalib", "PatchCore"),
    }


def test_newest_surviving_checkpoint_wins_per_variant(tmp_path):
    models = _tree(tmp_path)
    old = _weight(models, "yolov8n_old.pt")
    new = _weight(models, "yolov8n_new.pt")
    _manifest(models, [
        _record("ultralytics", "yolov8n", old, recorded_at="2026-07-01T00:00:00Z"),
        _record("ultralytics", "yolov8n", new, recorded_at="2026-07-09T00:00:00Z"),
    ])

    (found,) = discover_trained_models(tmp_path)

    assert found.weights == new


def test_a_pruned_checkpoint_falls_back_to_an_older_surviving_one(tmp_path):
    """A manifest row whose file was pruned must not shadow a live older one."""

    models = _tree(tmp_path)
    old = _weight(models, "yolov8n_old.pt")
    _manifest(models, [
        _record("ultralytics", "yolov8n", old, recorded_at="2026-07-01T00:00:00Z"),
        _record("ultralytics", "yolov8n", models / "deleted.pt", recorded_at="2026-07-09T00:00:00Z"),
    ])

    (found,) = discover_trained_models(tmp_path)

    assert found.weights == old


def test_quantized_records_are_not_swept_as_models(tmp_path):
    models = _tree(tmp_path)
    fp32 = _weight(models, "yolov8n.pt")
    quant = _weight(models, "yolov8n_int8.onnx")
    _manifest(models, [
        _record("ultralytics", "yolov8n", fp32),
        _record("ultralytics", "yolov8n-int8", quant, kind="quantized"),
    ])

    found = discover_trained_models(tmp_path)

    assert [m.variant for m in found] == ["yolov8n"]


def test_discovery_is_empty_rather_than_failing_on_a_bare_tree(tmp_path):
    assert discover_trained_models(tmp_path) == []


def test_the_same_model_under_two_spellings_is_one_entry(tmp_path):
    """The manifest records the config's spelling (`ganomaly`) while catalog
    uses the canonical one (`GANomaly`) — a case-sensitive key would sweep
    the same checkpoint twice."""

    models = _tree(tmp_path)
    lower = _weight(models, "Ganomaly.ckpt")
    _manifest(models, [_record("anomalib", "ganomaly", lower)])
    (models / "published" / "GANomaly.ckpt").symlink_to("../Ganomaly.ckpt")

    found = discover_trained_models(tmp_path)

    assert len(found) == 1, [m.variant for m in found]


# --------------------------------------------------------------------------- #
# Failure isolation — the whole point
# --------------------------------------------------------------------------- #
def test_a_failing_model_becomes_a_row_not_an_exception(tmp_path, monkeypatch):
    models = _tree(tmp_path)
    _manifest(models, [_record("ultralytics", "yolov8n", _weight(models, "yolov8n.pt"))])

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("fabric_defect_hub.metric_sweep._accuracy", boom)

    rows = run_sweep(_request(tmp_path))

    (row,) = rows
    assert row.status == "failed"
    assert "CUDA out of memory" in row.reason


def test_one_broken_model_does_not_cost_the_others_their_measurements(tmp_path, monkeypatch):
    models = _tree(tmp_path)
    _manifest(models, [
        _record("ultralytics", "yolov8n", _weight(models, "yolov8n.pt")),
        _record("anomalib", "PatchCore", _weight(models, "PatchCore.ckpt")),
    ])

    def selective(model, request):
        if model.backend == "ultralytics":
            raise RuntimeError("broken")
        return {"image_auroc": 0.91}

    monkeypatch.setattr("fabric_defect_hub.metric_sweep._accuracy", selective)

    rows = run_sweep(_request(tmp_path))

    by_backend = {row.backend: row for row in rows}
    assert by_backend["ultralytics"].status == "failed"
    assert by_backend["anomalib"].status == "ok"
    assert by_backend["anomalib"].metrics["image_auroc"] == 0.91


def test_a_missing_dependency_is_skipped_not_failed(tmp_path, monkeypatch):
    """An uninstalled backend is an environment fact, not a defect."""

    models = _tree(tmp_path)
    _manifest(models, [_record("dinomaly", "dinov2reg_vit_base_14", _weight(models, "d.pth"))])
    monkeypatch.setattr(
        "fabric_defect_hub.metric_sweep._accuracy",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("No module named 'timm'")),
    )

    (row,) = run_sweep(_request(tmp_path))

    assert row.status == "skipped"
    assert "timm" in row.reason


def test_an_unsupported_operation_is_skipped(tmp_path, monkeypatch):
    models = _tree(tmp_path)
    _manifest(models, [_record("moeclip", "ViT-L-14-336", _weight(models, "m.pth"))])
    monkeypatch.setattr(
        "fabric_defect_hub.metric_sweep._accuracy",
        lambda *a, **k: (_ for _ in ()).throw(NotImplementedError("export")),
    )

    (row,) = run_sweep(_request(tmp_path))

    assert row.status == "skipped"


@pytest.mark.parametrize("returned", [{}, {"sample_count": 4}])
def test_a_group_producing_no_real_metrics_is_skipped_not_reported_as_ok(
    tmp_path, monkeypatch, returned
):
    """`sample_count` says the run happened, not that anything was measured.
    Counting it as a metric is how an evaluator/task mismatch first showed up
    as a green row carrying nothing.
    """

    models = _tree(tmp_path)
    _manifest(models, [_record("ultralytics", "yolov8n", _weight(models, "yolov8n.pt"))])
    monkeypatch.setattr("fabric_defect_hub.metric_sweep._accuracy", lambda *a, **k: returned)

    (row,) = run_sweep(_request(tmp_path))

    assert row.status == "skipped"
    assert "no metrics produced" in row.reason


def test_cross_domain_is_skipped_when_no_patterns_are_given(tmp_path):
    models = _tree(tmp_path)
    _manifest(models, [_record("ultralytics", "yolov8n", _weight(models, "yolov8n.pt"))])

    (row,) = run_sweep(_request(tmp_path, groups=("cross_domain",)))

    assert row.status == "skipped" and "cross-domain-patterns" in row.reason


# --------------------------------------------------------------------------- #
# JSONL output
# --------------------------------------------------------------------------- #
def test_rows_are_written_incrementally_so_a_kill_keeps_partial_results(tmp_path, monkeypatch):
    models = _tree(tmp_path)
    _manifest(models, [
        _record("ultralytics", "yolov8n", _weight(models, "a.pt")),
        _record("anomalib", "PatchCore", _weight(models, "b.ckpt")),
    ])
    request = _request(tmp_path)

    # Models are swept in (backend, variant) order, so anomalib runs first
    # and ultralytics is the one interrupted.
    def record_then_die(model, req):
        if model.backend == "ultralytics":
            raise KeyboardInterrupt
        return {"map_50": 0.5}

    monkeypatch.setattr("fabric_defect_hub.metric_sweep._accuracy", record_then_die)

    with pytest.raises(KeyboardInterrupt):
        run_sweep(request)

    # A Ctrl-C (or a dropped SSH session) stops the sweep — but whatever
    # finished before it is already on disk rather than lost with the process.
    _, rows = read_sweep(request.output_path)
    assert [row["backend"] for row in rows] == ["anomalib"]
    assert rows[0]["status"] == "ok" and rows[0]["metrics"]["map_50"] == 0.5


def test_the_header_carries_provenance_for_secondary_analysis(tmp_path, monkeypatch):
    models = _tree(tmp_path)
    _manifest(models, [_record("ultralytics", "yolov8n", _weight(models, "a.pt"))])
    monkeypatch.setattr("fabric_defect_hub.metric_sweep._accuracy", lambda *a, **k: {"map_50": 0.4})
    request = _request(tmp_path)

    run_sweep(request)
    header, rows = read_sweep(request.output_path)

    assert header["dataset"] == "zju-leaper"
    assert header["model_count"] == 1
    assert "provenance" in header
    assert rows[0]["metrics"]["map_50"] == 0.4


def test_read_sweep_tolerates_a_truncated_final_line(tmp_path):
    path = tmp_path / "sweep.jsonl"
    path.write_text(
        json.dumps({"kind": "sweep_header", "dataset": "d"}) + "\n"
        + json.dumps({"model": "a", "status": "ok"}) + "\n"
        + '{"model": "b", "stat'
    )

    header, rows = read_sweep(path)

    assert header["dataset"] == "d"
    assert [row["model"] for row in rows] == ["a"]


def test_summarize_counts_by_status_and_group():
    rows = [
        {"status": "ok", "group": "accuracy"},
        {"status": "failed", "group": "accuracy"},
        {"status": "skipped", "group": "runtime"},
    ]

    summary = summarize(rows)

    assert summary["total"] == 3
    assert summary["by_status"] == {"ok": 1, "failed": 1, "skipped": 1}
    assert summary["by_group"]["accuracy"] == {"ok": 1, "failed": 1}


# --------------------------------------------------------------------------- #
# Request validation & selection
# --------------------------------------------------------------------------- #
def test_unknown_metric_group_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="unknown metric group"):
        _request(tmp_path, groups=("accuracy", "telepathy"))


def test_every_declared_group_is_dispatchable(tmp_path, monkeypatch):
    """Guards against a group name existing in the vocabulary with no branch."""

    models = _tree(tmp_path)
    _manifest(models, [_record("ultralytics", "yolov8n", _weight(models, "a.pt"))])
    monkeypatch.setattr(
        "fabric_defect_hub.metric_sweep._try_export", lambda *a, **k: (None, "no export in test")
    )
    monkeypatch.setattr("fabric_defect_hub.metric_sweep._accuracy", lambda *a, **k: {"map_50": 1.0})

    rows = run_sweep(_request(tmp_path, groups=METRIC_GROUPS))

    assert {row.group for row in rows} == set(METRIC_GROUPS)
    assert not any(row.reason and "unknown group" in row.reason for row in rows)


def test_models_filter_selects_by_variant(tmp_path, monkeypatch):
    models = _tree(tmp_path)
    _manifest(models, [
        _record("ultralytics", "yolov8n", _weight(models, "a.pt")),
        _record("anomalib", "PatchCore", _weight(models, "b.ckpt")),
    ])
    monkeypatch.setattr("fabric_defect_hub.metric_sweep._accuracy", lambda *a, **k: {"map_50": 1.0})

    rows = run_sweep(_request(tmp_path, models=("PatchCore",)))

    assert [row.variant for row in rows] == ["PatchCore"]


def test_config_hint_survives_a_path_from_another_machine(tmp_path):
    """The manifest stores the training box's absolute path; only the
    filename is portable back to this checkout."""

    from fabric_defect_hub.metric_sweep import TrainedModel

    model = TrainedModel(
        backend="anomalib", variant="ganomaly", weights=tmp_path / "w.ckpt",
        config="/root/autodl-tmp/FabricDefectHub/configs/models/anomalib_ganomaly.yaml",
    )

    assert model.config_hint == "anomalib_ganomaly"


def test_config_hint_falls_back_to_the_variant(tmp_path):
    from fabric_defect_hub.metric_sweep import TrainedModel

    model = TrainedModel(backend="ultralytics", variant="yolov8n", weights=tmp_path / "w.pt")

    assert model.config_hint == "yolov8n"


def test_sweep_row_serialises_with_a_timestamp():
    row = SweepRow(model="m", backend="b", variant="v", group="accuracy", status="ok")
    payload = row.to_json()
    assert payload["recorded_at"].endswith("Z")
    assert payload["status"] == "ok"
