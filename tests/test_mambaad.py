"""Fast, framework-free-where-possible tests for the MambaAD backend.

`scan.py` is pure NumPy (no torch) and gets the most scrutiny here: a
subtly wrong scan order is the kind of bug that still produces a valid
permutation (so shape/dtype checks pass) but a discontinuous curve, which
silently degrades the model rather than erroring -- see the adjacency
checks below, which is exactly the property a bug in `_hilbert_d2xy`
violated during development (fixed by using the sub-square size `s`, not
the full grid `n`, as the rotation bound -- verified against a scalar port
of the reference algorithm).

`ssm.py`/`decoder.py`/`network.py` need torch; the shape/gradient/freeze
checks here use tiny synthetic tensors, not a real training run.
"""

from __future__ import annotations

import numpy as np
import pytest

from fabric_defect_hub.models.mambaad.scan import SCAN_TYPES, scan_order

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------- #
# scan.py -- pure NumPy, no torch needed
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("scan_type", SCAN_TYPES)
@pytest.mark.parametrize("size", [4, 8, 16])
def test_scan_order_is_a_permutation(scan_type, size):
    order = scan_order(size, scan_type)
    assert sorted(order.tolist()) == list(range(size * size))


@pytest.mark.parametrize("scan_type", SCAN_TYPES)
def test_scan_order_is_invertible(scan_type):
    order = scan_order(8, scan_type)
    inverse = np.argsort(order)
    assert np.array_equal(order[inverse], np.arange(64))


@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_hilbert_scan_steps_stay_grid_adjacent(size):
    # The defining property of a Hilbert curve: consecutive scan steps are
    # always a neighboring grid cell (Manhattan distance 1) -- a wrong
    # rotation bound in _hilbert_d2xy produces a valid permutation that
    # silently fails exactly this property (see module docstring).
    order = scan_order(size, "hilbert")
    coords = [(int(cell) // size, int(cell) % size) for cell in order]
    distances = [
        abs(coords[i][0] - coords[i + 1][0]) + abs(coords[i][1] - coords[i + 1][1])
        for i in range(len(coords) - 1)
    ]
    assert max(distances) == 1


def test_hilbert_scan_requires_power_of_two_size():
    with pytest.raises(ValueError, match="power-of-two"):
        scan_order(6, "hilbert")


def test_unknown_scan_type_is_rejected():
    with pytest.raises(ValueError, match="unknown scan_type"):
        scan_order(8, "spiral")


# --------------------------------------------------------------------- #
# ssm.py -- HSCANS round-trip, selective_scan shape/grad, SS2D shape
# --------------------------------------------------------------------- #
def test_hscans_encode_decode_round_trips():
    from fabric_defect_hub.models.mambaad.ssm import HSCANS

    scans = HSCANS(size=8, scan_type="hilbert")
    x = torch.randn(2, 4, 64)
    assert torch.allclose(scans.decode(scans.encode(x)), x, atol=1e-6)


def test_selective_scan_shape_and_grad():
    from fabric_defect_hub.models.mambaad.ssm import selective_scan

    batch, d_inner, d_state, length = 2, 6, 4, 10
    u = torch.randn(batch, d_inner, length, requires_grad=True)
    delta = torch.rand(batch, d_inner, length)
    A = -torch.rand(d_inner, d_state)
    B = torch.randn(batch, d_state, length)
    C = torch.randn(batch, d_state, length)
    D = torch.randn(d_inner)

    y = selective_scan(u, delta, A, B, C, D, delta_softplus=True)
    assert y.shape == (batch, d_inner, length)
    y.sum().backward()
    assert u.grad is not None


# The chunked scan trades arithmetic for far fewer sequential steps; it is
# only a valid substitute if it is *numerically* the same recurrence, so it
# is checked against the sequential oracle rather than merely for shape.
@pytest.mark.parametrize(
    ("batch", "d_inner", "d_state", "length", "groups"),
    [(2, 8, 4, 16, 1), (1, 16, 4, 64, 2), (2, 32, 8, 100, 4), (1, 64, 16, 257, 8)],
)
def test_chunked_scan_matches_sequential_oracle(batch, d_inner, d_state, length, groups):
    from fabric_defect_hub.models.mambaad.ssm import selective_scan_chunked, selective_scan_reference

    torch.manual_seed(0)
    u = torch.randn(batch, d_inner, length)
    delta = torch.rand(batch, d_inner, length)
    A = -torch.rand(d_inner, d_state).exp()
    B = torch.randn(batch, groups, d_state, length)
    C = torch.randn(batch, groups, d_state, length)
    D = torch.randn(d_inner)
    bias = torch.randn(d_inner)

    reference = selective_scan_reference(u, delta, A, B, C, D, delta_bias=bias, delta_softplus=True)
    for chunk_size in (1, 7, 32, None):
        chunked = selective_scan_chunked(
            u, delta, A, B, C, D, delta_bias=bias, delta_softplus=True, chunk_size=chunk_size
        )
        assert torch.allclose(reference, chunked, atol=2e-4), f"chunk_size={chunk_size}"


def test_chunked_scan_matches_sequential_gradients():
    from fabric_defect_hub.models.mambaad.ssm import selective_scan_chunked, selective_scan_reference

    torch.manual_seed(0)
    u = torch.randn(1, 8, 50)
    delta = torch.rand(1, 8, 50)
    A = -torch.rand(8, 4).exp()
    B, C = torch.randn(1, 4, 50), torch.randn(1, 4, 50)

    grads = []
    for scan in (selective_scan_reference, selective_scan_chunked):
        leaf = u.clone().requires_grad_(True)
        scan(leaf, delta, A, B, C, delta_softplus=True).sum().backward()
        grads.append(leaf.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-4)


def test_chunked_scan_does_not_retain_every_chunks_decay_matrix():
    """The chunked scan budgets its `(B, G, P, N, Lc, Lc)` decay matrix per
    chunk, but without recomputation autograd keeps *every* chunk's copy
    alive until backward — so the retained working set grows with the chunk
    length, and a detection-sized feature map OOMs a 32GB card on the first
    iteration (observed on FabricMamba/MambaAD, 30.7GB allocated at step 0).
    Checkpointing each chunk makes the retained set independent of the chunk
    length instead.

    Measured by tallying the elements autograd actually saves: with the
    quadratic per-chunk terms retained this grows roughly linearly in
    `chunk_size` (12.7K -> 69.1K elements across the sizes below); with
    recomputation it stays flat.
    """

    from fabric_defect_hub.models.mambaad.ssm import selective_scan_chunked

    def saved_elements(chunk_size: int, length: int = 64) -> int:
        total = 0

        def pack(tensor):
            nonlocal total
            total += tensor.numel()
            return tensor

        torch.manual_seed(0)
        u = torch.randn(1, 8, length, requires_grad=True)
        delta, A = torch.rand(1, 8, length), -torch.rand(8, 4).exp()
        B, C = torch.randn(1, 2, 4, length), torch.randn(1, 2, 4, length)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            selective_scan_chunked(u, delta, A, B, C, chunk_size=chunk_size).sum().backward()
        return total

    smallest, largest = saved_elements(4), saved_elements(32)
    # An 8x chunk-length increase must not inflate the retained set; allow a
    # little slack for the per-chunk *inputs*, which do vary slightly.
    assert largest <= smallest * 1.5, (
        f"retained working set scales with chunk length ({smallest} -> {largest}); "
        "per-chunk recomputation is not in effect"
    )


def test_grouped_and_ungrouped_bc_agree():
    # SS2D folds all scan directions into one grouped call; a single group
    # must therefore behave identically to the plain ungrouped form.
    from fabric_defect_hub.models.mambaad.ssm import selective_scan_reference

    torch.manual_seed(0)
    u, delta = torch.randn(1, 8, 32), torch.rand(1, 8, 32)
    A = -torch.rand(8, 4)
    B, C = torch.randn(1, 4, 32), torch.randn(1, 4, 32)
    ungrouped = selective_scan_reference(u, delta, A, B, C, delta_softplus=True)
    grouped = selective_scan_reference(u, delta, A, B.unsqueeze(1), C.unsqueeze(1), delta_softplus=True)
    assert torch.equal(ungrouped, grouped)


@pytest.mark.parametrize("num_direction", [2, 4, 8])
@pytest.mark.parametrize("scan_type", SCAN_TYPES)
def test_ss2d_forward_preserves_shape(num_direction, scan_type):
    from fabric_defect_hub.models.mambaad.ssm import SS2D

    module = SS2D(d_model=32, d_state=8, size=8, scan_type=scan_type, num_direction=num_direction)
    x = torch.randn(2, 8, 8, 32)
    out = module(x)
    assert out.shape == x.shape


def test_ss2d_rejects_invalid_num_direction():
    from fabric_defect_hub.models.mambaad.ssm import SS2D

    with pytest.raises(ValueError, match="num_direction"):
        SS2D(d_model=32, num_direction=3)


# --------------------------------------------------------------------- #
# decoder.py / network.py -- shape and gradient/freeze checks
# --------------------------------------------------------------------- #
def test_mamba_up_net_forward_shapes():
    from fabric_defect_hub.models.mambaad.decoder import MambaUPNet

    net = MambaUPNet(dims_decoder=[64, 32, 16, 8], depths_decoder=[3, 4, 6, 3], base_size=4,
                     scan_type="hilbert", num_direction=8)
    x = torch.randn(1, 64, 4, 4)
    outputs = net(x)
    expected_shapes = [(1, 8, 32, 32), (1, 16, 16, 16), (1, 32, 8, 8)]
    assert [tuple(o.shape) for o in outputs] == expected_shapes
    sum(o.sum() for o in outputs).backward()


def test_mambaad_net_teacher_stays_frozen():
    timm = pytest.importorskip("timm")
    from fabric_defect_hub.models.mambaad.network import MambaADNet

    teacher = timm.create_model("resnet34", pretrained=False, features_only=True, out_indices=[1, 2, 3])
    net = MambaADNet(
        teacher=teacher, teacher_channels=teacher.feature_info.channels(),
        dims_decoder=[512, 256, 128, 64], depths_decoder=[3, 4, 6, 3],
        base_size=4, scan_type="hilbert", num_direction=8,
    )
    for param in net.teacher.parameters():
        param.requires_grad = False
    net.train()

    images = torch.randn(1, 3, 128, 128)
    teacher_features, student_features = net(images)
    for t, s in zip(teacher_features, student_features):
        assert t.shape == s.shape
        assert not t.requires_grad  # detached -- never a backprop target itself

    loss = sum((1 - torch.nn.functional.cosine_similarity(t.flatten(2), s.flatten(2), dim=1)).mean()
              for t, s in zip(teacher_features, student_features))
    loss.backward()
    assert all(p.grad is None for p in net.teacher.parameters())
    assert any(p.grad is not None for p in net.fusion.parameters())


def test_mambaad_net_train_mode_keeps_teacher_in_eval():
    timm = pytest.importorskip("timm")
    from fabric_defect_hub.models.mambaad.network import MambaADNet

    teacher = timm.create_model("resnet34", pretrained=False, features_only=True, out_indices=[1, 2, 3])
    net = MambaADNet(
        teacher=teacher, teacher_channels=teacher.feature_info.channels(),
        dims_decoder=[512, 256, 128, 64], depths_decoder=[3, 4, 6, 3], base_size=4,
    )
    net.train()  # a plain train() call on the whole model...
    assert not net.teacher.training  # ...must not un-freeze the teacher's batch-norm/dropout


# --------------------------------------------------------------------- #
# adapter.py -- fidelity of the loss and anomaly-map math against the
# upstream definitions they were ported from (ADer's loss/base_loss.py and
# util/metric.py, as called by MambaAD's own trainer).
# --------------------------------------------------------------------- #
def test_reconstruction_loss_is_mse_not_cosine():
    # Upstream's config selects L2Loss (nn.MSELoss) with lam=5.0, NOT the
    # cosine objective the rest of the reverse-distillation family uses.
    from fabric_defect_hub.models.mambaad.adapter import MambaADAdapter

    torch.manual_seed(0)
    teacher = [torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4)]
    student = [torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4)]
    expected = sum(torch.nn.functional.mse_loss(s, t) * 5.0 for t, s in zip(teacher, student))
    assert torch.allclose(MambaADAdapter._reconstruction_loss(teacher, student, lam=5.0), expected)


@pytest.mark.parametrize("sigma", [1.0, 2.0, 4.0])
def test_gaussian_blur_matches_scipy(sigma):
    # Upstream blurs the anomaly map with scipy.ndimage.gaussian_filter
    # (sigma=4). Both the 4*sigma kernel radius and scipy's symmetric
    # 'reflect' boundary have to match, or pixel metrics shift for reasons
    # unrelated to the model.
    ndimage = pytest.importorskip("scipy.ndimage")
    from fabric_defect_hub.models.mambaad.adapter import MambaADAdapter

    rng = np.random.default_rng(0)
    array = rng.random((2, 40, 55), dtype=np.float32)
    expected = np.stack([ndimage.gaussian_filter(a, sigma=sigma) for a in array])
    actual = MambaADAdapter._gaussian_blur(torch.from_numpy(array).unsqueeze(1), sigma)
    assert np.abs(expected - actual.squeeze(1).numpy()).max() < 1e-5


def test_anomaly_map_matches_upstream_construction():
    # ADer's cal_anomaly_map(uni_am=False, use_cos=True, amap_mode='add'):
    # per-level (1 - cosine), upsampled, summed, divided by
    # len(levels) * sum(weights), then blurred.
    from fabric_defect_hub.models.mambaad.adapter import MambaADAdapter

    torch.manual_seed(0)
    size = 32
    teacher = [torch.randn(1, 4, 16, 16), torch.randn(1, 8, 8, 8)]
    student = [torch.randn(1, 4, 16, 16), torch.randn(1, 8, 8, 8)]

    expected = None
    for t, s in zip(teacher, student):
        level = (1 - torch.nn.functional.cosine_similarity(t, s, dim=1)).unsqueeze(1)
        level = torch.nn.functional.interpolate(level, size=size, mode="bilinear", align_corners=True)
        expected = level if expected is None else expected + level
    expected = expected / (len(teacher) * len(teacher))

    actual = MambaADAdapter._anomaly_map(teacher, student, size, gaussian_sigma=0.0)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert actual.shape == (1, 1, size, size)


# --------------------------------------------------------------------- #
# config.py / presets.py -- framework-free
# --------------------------------------------------------------------- #
def test_config_applies_upstream_defaults():
    from fabric_defect_hub.models.mambaad.config import MambaADConfig

    config = MambaADConfig.from_dict(
        {"model": {"name": "resnet34"},
         "data": {"dataset": "zju-leaper", "dataset_root": "data/ZJU-Leaper"}}
    )
    assert config.train.total_iters == 5000
    assert config.train.image_size == 256


def test_config_unknown_backbone_is_rejected():
    from fabric_defect_hub.models.mambaad.config import MambaADConfig

    with pytest.raises(KeyError, match="unknown MambaAD encoder"):
        MambaADConfig.from_dict(
            {"model": {"name": "resnet999"},
             "data": {"dataset": "zju-leaper", "dataset_root": "data/ZJU-Leaper"}}
        )


def test_config_requires_dataset_root():
    from fabric_defect_hub.models.mambaad.config import MambaADConfig

    with pytest.raises(ValueError, match="requires 'dataset_root'"):
        MambaADConfig.from_dict({"data": {"dataset": "zju-leaper"}})


def test_shipped_example_config_parses():
    from fabric_defect_hub.models.mambaad.config import MambaADConfig
    from fabric_defect_hub.training import apply_default_dataset_root, load_raw_config

    raw = apply_default_dataset_root(load_raw_config("configs/models/mambaad_example.yaml"))
    config = MambaADConfig.from_dict(raw)
    assert config.data.train_selection["use_defect"] is False  # one-class training
    assert config.model.name == "resnet34"


# --------------------------------------------------------------------- #
# adapter.py -- one-class enforcement, no real training run
# --------------------------------------------------------------------- #
def test_train_rejects_defective_samples():
    from fabric_defect_hub.core.types import Annotations, Sample
    from fabric_defect_hub.models.mambaad.adapter import MambaADAdapter

    adapter = MambaADAdapter.__new__(MambaADAdapter)
    defective = Sample(id="d1", image_path="d1.png", task="anomaly", annotations=Annotations(is_anomalous=True))
    with pytest.raises(ValueError, match="must be all-normal"):
        adapter.train({"train_samples": [defective], "total_iters": 1})


def test_train_requires_samples():
    from fabric_defect_hub.models.mambaad.adapter import MambaADAdapter

    adapter = MambaADAdapter.__new__(MambaADAdapter)
    with pytest.raises(ValueError, match="requires config\\['train_samples'\\]"):
        adapter.train({})
