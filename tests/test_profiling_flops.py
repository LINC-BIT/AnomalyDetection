"""Tests for `profiling/flops.py`'s FLOPs counter -- the input
`evaluation.lmei_profiler.calculate_lmei`'s `flops_g` parameter needed
that nothing in this project computed before.
"""

from __future__ import annotations

import builtins

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("thop")

from fabric_defect_hub.profiling.flops import compute_model_flops


def test_compute_model_flops_matches_hand_computed_conv2d_macs():
    # Conv2d(3, 4, kernel=3, no bias) on an 8x8 input -> 6x6 output.
    # MACs = out_elements * (in_channels * kh * kw) = (4*6*6) * (3*3*3) = 3888.
    # FLOPs = 2 * MACs (one multiply + one add per MAC).
    model = torch.nn.Conv2d(3, 4, 3, bias=False)
    flops_g = compute_model_flops(model, input_size=(8, 8), batch_size=1, input_style="batched")
    assert flops_g == pytest.approx((3888 * 2) / 1e9)


def test_compute_model_flops_scales_with_batch_size():
    model = torch.nn.Conv2d(3, 4, 3, bias=False)
    single = compute_model_flops(model, input_size=(8, 8), batch_size=1, input_style="batched")
    doubled = compute_model_flops(model, input_size=(8, 8), batch_size=2, input_style="batched")
    assert doubled == pytest.approx(single * 2)


class _ListInputModel(torch.nn.Module):
    """Mirrors torchvision detection models' native calling convention:
    forward takes a `List[Tensor]` of unbatched `[3, H, W]` images (see
    `profiling.base.ProfileConfig.input_style`'s 'list' branch), not a
    single batched tensor."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, bias=False)

    def forward(self, images):
        return [self.conv(image.unsqueeze(0)) for image in images]


def test_compute_model_flops_supports_list_input_style():
    model = _ListInputModel()
    flops_g = compute_model_flops(model, input_size=(8, 8), batch_size=1, input_style="list")
    assert flops_g == pytest.approx((3888 * 2) / 1e9)


def test_compute_model_flops_rejects_unknown_input_style():
    model = torch.nn.Conv2d(3, 4, 3, bias=False)
    with pytest.raises(ValueError, match="input_style"):
        compute_model_flops(model, input_size=(8, 8), input_style="ragged")


def test_compute_model_flops_raises_actionable_error_without_thop(monkeypatch):
    model = torch.nn.Conv2d(3, 4, 3, bias=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "thop":
            raise ImportError("No module named 'thop'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="profiling-flops"):
        compute_model_flops(model, input_size=(8, 8))
