"""Tests for `web/benchmark.py`'s Gradio-facing leaderboard engine: the
opt-in profiling pass, run-log persistence, and composite scoring added on
top of the plain accuracy leaderboard.

Uses its own uniquely-named fake dataset/model registration
("*-webbench" suffix), mirroring `test_benchmark.py`'s pattern for the same
`core.registry` duplicate-name reason explained there -- registered once at
module import time (not per test), since `register_model` raises on a
second registration of the same name.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from fabric_defect_hub.core.registry import register_dataset, register_model
from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.datasets.base import DatasetAdapter
from fabric_defect_hub.models.base import ExportedArtifact, ModelAdapter, ModelCapabilities
from fabric_defect_hub.application import benchmark as web_benchmark

MODEL_LABEL = "Fake Model"


@register_dataset("fake-fabric-webbench")
class _FakeWebBenchDataset(DatasetAdapter):
    name = "fake-fabric-webbench"

    def load_samples(self) -> list[Sample]:
        return [
            Sample(
                id=f"sample-{i:04d}", image_path=f"{self.root}/{i:04d}.jpg", task="anomaly",
                annotations=Annotations(is_anomalous=bool(i % 2)),
            )
            for i in range(4)
        ]


class _TinyModule(nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2, 3))


@register_model("fake-backend-webbench")
class _FakeWebBenchModel(ModelAdapter):
    name = "fake-backend-webbench-model"
    backend = "fake-backend-webbench"

    def capabilities(self):
        return ModelCapabilities(
            tasks=("anomaly",),
            prediction_fields=("anomaly_score",),
            export_targets=("torchscript",),
        )

    def train(self, config):
        return None

    def predict(self, samples, artifact=None, output_dir=None, config=None):
        return [Prediction(sample_id=s.id, anomaly_score=0.9) for s in samples]

    def raw_module(self):
        # A real Conv2d (unlike `_TinyModule`'s parameter-free `.mean()`)
        # so `profiling.flops.compute_model_flops`'s hook-based counter has
        # something nonzero to count -- see test_run_benchmark_with_profiling
        # _adds_flops_and_lmei.
        return nn.Conv2d(3, 4, 3, bias=False)

    def export(self, artifact, target, config=None):
        assert target == "torchscript"
        fd, path = tempfile.mkstemp(suffix=f".{target}")
        os.close(fd)
        torch.jit.save(torch.jit.script(_TinyModule()), path)
        return ExportedArtifact(path=path, target=target)


@register_dataset("fake-fabric-webbench-target")
class _FakeWebBenchTargetDataset(DatasetAdapter):
    """A second, differently-named domain for cross-domain-degradation
    tests -- same shape as `_FakeWebBenchDataset`, registered separately
    since `register_dataset` rejects a second registration of the same
    name (see module docstring)."""

    name = "fake-fabric-webbench-target"

    def load_samples(self) -> list[Sample]:
        return [
            Sample(
                id=f"target-{i:04d}", image_path=f"{self.root}/{i:04d}.jpg", task="anomaly",
                annotations=Annotations(is_anomalous=bool(i % 2)),
            )
            for i in range(4)
        ]


def _install_fake_catalog(monkeypatch, tmp_path):
    # The UI reads "which tasks can this dataset be scored on" from
    # `core.dataset_capabilities`, not from its own catalog, so a fake
    # dataset has to declare capabilities exactly like a real one does.
    # That is the point of the arrangement -- the fixture models the
    # production requirement instead of restating the answer.
    from fabric_defect_hub.core.dataset_capabilities import (
        all_capabilities,
        register_capabilities,
    )

    already = all_capabilities()
    for name in ("fake-fabric-webbench", "fake-fabric-webbench-target"):
        # `register_capabilities` refuses a re-registration (a real dataset
        # declaring itself twice is a bug); this fixture runs once per test.
        if name not in already:
            register_capabilities(name, default_root=str(tmp_path), roles=set(), tasks=("anomaly",))

    dataset_catalog = {
        "Fake Dataset": {
            "name": "fake-fabric-webbench",
            "slice_kwarg": None,
        },
        "Fake Target Dataset": {
            "name": "fake-fabric-webbench-target",
            "slice_kwarg": None,
        },
    }
    model_catalog = {
        MODEL_LABEL: {
            "backend": "fake-backend-webbench",
            "name": "fake-backend-webbench-model",
            "checkpoint": str(tmp_path / "fake.ckpt"),
            "task": "anomaly",
            "metadata": {},
        },
    }
    monkeypatch.setattr(web_benchmark, "DATASET_CATALOG", dataset_catalog)
    monkeypatch.setattr(web_benchmark, "MODEL_CATALOG", model_catalog)
    monkeypatch.setattr(web_benchmark, "default_dataset_root", lambda label: str(tmp_path))
    monkeypatch.setattr(web_benchmark, "_detect_device", lambda: "cpu")


def test_run_benchmark_basic_leaderboard_has_no_score_columns_without_metrics(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL], run_log_path=None,
    )

    assert "composite_score" in columns
    assert rows[0][columns.index("model")] == MODEL_LABEL
    # No profiling metrics were requested, so overhead_score has nothing to
    # average and composite falls back to the technical (accuracy) score.
    assert rows[0][columns.index("composite_score")] == rows[0][columns.index("technical_score")]
    # The chart-facing payload carries the same run as name-keyed dicts.
    assert scored[0]["model"] == MODEL_LABEL
    assert "composite_score" in scored[0]


def test_run_benchmark_with_profiling_adds_overhead_metrics_and_scores(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        include_profiling=True, run_log_path=None,
    )

    assert "fps" in columns
    assert "latency_ms_mean" in columns
    row = rows[0]
    assert row[columns.index("fps")] > 0
    assert row[columns.index("overhead_score")] != ""
    assert row[columns.index("composite_score")] != ""


def test_run_benchmark_keeps_accuracy_when_profiling_export_is_unsupported(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)
    monkeypatch.setattr(
        _FakeWebBenchModel,
        "capabilities",
        lambda self: ModelCapabilities(
            tasks=("anomaly",), prediction_fields=("anomaly_score",), export_targets=("onnx",),
        ),
    )

    *_, (columns, rows, status, _) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        include_profiling=True, run_log_path=None,
    )

    assert rows[0][columns.index("model")] == MODEL_LABEL
    assert "image_auroc" in columns
    assert "profiling skipped" in status


def test_run_benchmark_with_profiling_adds_flops_and_allocator_lmei(monkeypatch, tmp_path):
    pytest.importorskip("thop")
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        include_profiling=True, run_log_path=None,
    )

    assert "flops_g" in columns
    assert "params_m" in columns
    row = rows[0]
    assert row[columns.index("flops_g")] >= 0
    assert row[columns.index("params_m")] >= 0
    # LMEI is defined using accelerator allocator memory.  CPU profiling
    # reports process RSS, which is useful telemetry but is not a comparable
    # substitute for VRAM and therefore must not produce a misleading score.
    memory_kind = row[columns.index("memory_measurement_kind")]
    if memory_kind == "device_allocator":
        assert "lmei" in columns
        assert row[columns.index("lmei")] != ""
    else:
        assert "lmei" not in columns


def test_run_benchmark_with_resolution_sweep_adds_slope_columns(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        include_resolution_sweep=True, run_log_path=None,
    )

    assert "resolution_slope_beta" in columns
    assert "resolution_slope_alpha" in columns
    row = rows[0]
    assert isinstance(row[columns.index("resolution_slope_beta")], float)


def test_run_benchmark_with_cross_domain_dataset_adds_degradation_column(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        cross_domain_dataset_label="Fake Target Dataset", run_log_path=None,
    )

    assert "cross_domain_delta_acc_pct" in columns
    row = rows[0]
    assert isinstance(row[columns.index("cross_domain_delta_acc_pct")], float)


def test_run_benchmark_cross_domain_skips_column_for_incompatible_target(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        cross_domain_dataset_label="Nonexistent Dataset", run_log_path=None,
    )

    assert "cross_domain_delta_acc_pct" not in columns
    assert rows  # the row itself still succeeds, just without the extra column


def test_run_benchmark_appends_to_run_log(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)
    log_path = tmp_path / "log.jsonl"

    list(web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL], run_log_path=str(log_path),
    ))

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["model"]["backend"] == "fake-backend-webbench"


def test_run_benchmark_custom_preset_uses_custom_weight(monkeypatch, tmp_path):
    _install_fake_catalog(monkeypatch, tmp_path)

    *_, (columns, rows, status, scored) = web_benchmark.run_benchmark(
        "Fake Dataset", "All textures", "Full-shot", [MODEL_LABEL],
        score_preset="custom", custom_technical_weight=0.9, run_log_path=None,
    )

    assert rows[0][columns.index("composite_score")] != ""
