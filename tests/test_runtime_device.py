from __future__ import annotations

import pytest

from fabric_defect_hub.runtime_device import resolve_torch_device


def test_auto_selects_a_real_local_backend():
    assert resolve_torch_device("auto") in {"cpu", "mps", "cuda:0"}


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match="device must be"):
        resolve_torch_device("banana")


def test_explicit_unavailable_accelerator_never_falls_back():
    try:
        selected = resolve_torch_device("mps")
    except RuntimeError as exc:
        assert "MPS was explicitly requested" in str(exc)
    else:
        assert selected == "mps"
