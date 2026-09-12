from pathlib import Path

import pytest

from fabric_defect_hub.core.preflight import require_cached_weight, resolve_cached_weight


def test_resolve_cached_weight_from_explicit_path(tmp_path):
    weight = tmp_path / "model.pt"
    weight.write_bytes(b"weights")
    assert resolve_cached_weight(str(weight)) == weight.resolve()


def test_resolve_cached_weight_from_extra_cache(tmp_path):
    weight = tmp_path / "model.pt"
    weight.write_bytes(b"weights")
    assert resolve_cached_weight("https://example.invalid/model.pt", [tmp_path]) == weight.resolve()


def test_offline_missing_weight_error_is_actionable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FDH_MODEL_CACHE", str(tmp_path / "empty-cache"))
    with pytest.raises(FileNotFoundError, match="FDH_MODEL_CACHE"):
        require_cached_weight("missing.pt", "test-backend")
