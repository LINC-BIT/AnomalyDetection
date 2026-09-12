"""Runs against this repo's real submodules, not mocks — update the expected
set in `test_vendored_components_lists_every_submodule` if `components/`
changes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fabric_defect_hub.core.provenance import (
    collect_provenance,
    describe_training,
    git_commit,
    vendored_components,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_git_commit_in_repo_is_a_hash():
    commit = git_commit(REPO_ROOT)
    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_git_commit_off_repo_degrades_to_unknown(tmp_path):
    assert git_commit(tmp_path) == "unknown"


def test_vendored_components_lists_every_submodule():
    components = vendored_components(REPO_ROOT)
    assert set(components) == {
        "components/dinomaly",
        "components/moeclip",
        "components/anomalydiffusion",
    }
    for record in components.values():
        assert re.fullmatch(r"[0-9a-f]{40}", record["commit"])
        assert record["state"] in {"clean", "modified", "uninitialized"}


def test_vendored_components_off_repo_degrades_to_empty(tmp_path):
    assert vendored_components(tmp_path) == {}


def test_collect_provenance_block_shape():
    block = collect_provenance(REPO_ROOT)
    assert {"timestamp_utc", "git_commit", "hostname", "vendored_components"} <= set(block)
    assert "components/moeclip" in block["vendored_components"]


def test_describe_training_accepts_strings_and_none():
    record = describe_training("auto", "linear", precision="amp-mixed")
    assert record == {"optimizer": "auto", "scheduler": "linear", "precision": "amp-mixed"}
    assert describe_training() == {"optimizer": None, "scheduler": None, "precision": "fp32"}


def test_describe_training_reads_live_torch_objects():
    torch = pytest.importorskip("torch")

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    record = describe_training(optimizer, scheduler)
    assert record["optimizer"] == "SGD"
    assert record["scheduler"] == "StepLR"
    assert record["lr"] == pytest.approx(0.01)
    assert record["weight_decay"] == pytest.approx(0.0005)
    assert record["precision"] == "fp32"
