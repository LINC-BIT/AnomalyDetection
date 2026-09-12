"""FabricMamba clean-room modules and their Ultralytics integration.

Mirrors `test_mambaad.py`'s split: module-level shape/semantic checks that
run everywhere, then integration checks that exercise the real Ultralytics
`parse_model` path (still CPU-cheap — building the graph, not training it).
"""

from __future__ import annotations

import pytest
import torch

from fabric_defect_hub.models.fabricmamba.modules import (
    LSKA,
    MVSS,
    PLSKA,
    CrossScanSS2D,
    DropPath,
    DySample,
    register_with_ultralytics,
)


# --------------------------------------------------------------------- #
# Modules in isolation
# --------------------------------------------------------------------- #
def test_lska_gates_without_changing_shape():
    module = LSKA(16)
    x = torch.randn(2, 16, 13, 17)  # non-square, odd sizes on purpose
    assert module(x).shape == x.shape


def test_plska_preserves_channels_like_the_sppf_it_replaces():
    module = PLSKA(32)
    x = torch.randn(2, 32, 16, 16)
    out = module(x)
    assert out.shape == x.shape
    out.sum().backward()  # every branch participates in the graph


def test_cross_scan_ss2d_round_trips_the_scan_views():
    """The four-direction scan and its inverse must be exact inverses:
    decoding an encoded identity signal has to reproduce `4 * x` (one copy
    per direction), independent of feature-map size or squareness.
    """

    module = CrossScanSS2D(8, d_state=4)
    x = torch.randn(2, 16, 5, 7)  # (batch, d_inner, H, W) grid, non-square
    views = module._multi_directional_scan(x)
    assert views.shape == (2, 4, 16, 35)
    restored = module._undo_multi_directional_scan(views, 5, 7)
    assert torch.allclose(restored.view(2, 16, 5, 7), 4 * x, atol=1e-6)


def test_cross_scan_ss2d_forward_any_size():
    module = CrossScanSS2D(16, d_state=4)
    for h, w in ((8, 8), (5, 9)):
        x = torch.randn(2, h, w, 16)  # NHWC, matching SS2D.forward
        assert module(x).shape == (2, h, w, 16)


def test_cross_scan_ss2d_rejects_other_direction_counts():
    with pytest.raises(ValueError):
        CrossScanSS2D(16, num_direction=8)


def test_droppath_is_identity_in_eval_and_scales_in_train():
    dp = DropPath(0.5)
    x = torch.ones(64, 3)
    dp.eval()
    assert torch.equal(dp(x), x)
    dp.train()
    out = dp(x)
    kept = out[out.sum(dim=1) != 0]
    if kept.numel():  # kept rows are rescaled by 1/keep_prob
        assert torch.allclose(kept, torch.full_like(kept, 2.0))


def test_mvss_preserves_channels_and_uses_every_block():
    module = MVSS(32, n=2)
    x = torch.randn(2, 32, 8, 8)
    out = module(x)
    assert out.shape == x.shape
    out.sum().backward()
    for block in module.blocks:
        grads = [p.grad for p in block.parameters() if p.grad is not None]
        assert grads, "an MVSS block did not participate in the forward pass"


def test_dysample_upsamples_by_scale_and_starts_as_plain_bilinear():
    module = DySample(16, scale=2)
    x = torch.randn(2, 16, 8, 8)
    out = module(x)
    assert out.shape == (2, 16, 16, 16)
    # The offset conv is zero-initialised, so before training DySample must
    # reproduce a regular bilinear resize exactly (the paper's motivation:
    # start from the safe upsampler, learn to deviate).
    expected = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    assert torch.allclose(out, expected, atol=1e-5)


def test_dysample_rejects_indivisible_group_counts():
    with pytest.raises(ValueError):
        DySample(10, groups=4)


# --------------------------------------------------------------------- #
# Ultralytics integration
# --------------------------------------------------------------------- #
def test_registration_is_idempotent_and_visible_to_parse_model():
    register_with_ultralytics()
    register_with_ultralytics()
    import ultralytics.nn.tasks as tasks

    assert tasks.PLSKA is PLSKA
    assert tasks.MVSS is MVSS
    assert tasks.DySample is DySample


def test_variant_resolves_to_packaged_yaml():
    from fabric_defect_hub.models.ultralytics.presets import resolve_variant, variant_weights

    assert resolve_variant("FabricMamba") == "fabricmamba"
    assert resolve_variant("fabric-mamba") == "fabricmamba"
    for pretrained in (True, False):  # no pretrained ckpt exists: both = arch yaml
        path = variant_weights("fabricmamba", pretrained=pretrained)
        assert path.endswith("fabricmamba_n.yaml")


def test_model_builds_with_expected_size_and_detects():
    """Build the real DetectionModel from the yaml and check it against the
    paper's published budget (3.8M parameters for the n scale). Ours
    measures 3.6M — the difference is the paper's unstated SS2D/MLP
    hyperparameters — so the assertion brackets the target rather than
    pinning our own output.
    """

    from fabric_defect_hub.models.ultralytics.presets import variant_weights
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(variant_weights("fabricmamba"), ch=3, nc=1, verbose=False)
    n_params = sum(p.numel() for p in model.parameters())
    assert 3.0e6 < n_params < 4.6e6

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 320, 320))
    # Detect head inference output: (concatenated predictions, per-level maps).
    preds = out[0] if isinstance(out, tuple) else out
    assert preds.shape[0] == 1 and preds.shape[1] == 4 + 1  # xywh + 1 class


def test_checkpoint_round_trip_restores_custom_modules():
    import io

    from fabric_defect_hub.models.ultralytics.presets import variant_weights
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(variant_weights("fabricmamba"), ch=3, nc=1, verbose=False)
    buffer = io.BytesIO()
    torch.save({"model": model}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)["model"]
    assert any(isinstance(m, MVSS) for m in restored.modules())
    assert any(isinstance(m, PLSKA) for m in restored.modules())
    assert any(isinstance(m, DySample) for m in restored.modules())


def test_fabricmamba_config_resolves_for_cli_train():
    from fabric_defect_hub.training import resolve_model_config_and_variant

    path, variant = resolve_model_config_and_variant("fabricmamba")
    assert path.name == "fabricmamba_example.yaml"
    assert variant == "fabricmamba"
