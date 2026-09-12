"""The data-side half of the interface ratchet: every `Sample` -> batch
conversion in the project implements the same `DataAdapter` contract.

The conversions themselves are deliberately *not* shared — each backend's
native preprocessing genuinely differs, and an earlier attempt at one common
DataLoader had to be reverted. What these tests pin down is that the seam
around them is uniform: same construction, same map-style protocol, a declared
`batch_spec()`, and a collate the caller never has to guess at.
"""

from __future__ import annotations

import pytest

from fabric_defect_hub.core.data_adapter import (
    IMAGE_LAYOUTS,
    ITEM_KINDS,
    MASK_SEMANTICS,
    BatchSpec,
    DataAdapter,
    Normalization,
)
from fabric_defect_hub.core.types import Annotations, Sample


def _samples(n: int = 2) -> list[Sample]:
    return [
        Sample(
            id=f"s-{i}",
            image_path=f"/nonexistent/{i}.jpg",
            task="anomaly",
            annotations=Annotations(is_anomalous=False),
        )
        for i in range(n)
    ]


def _implementations():
    """Every `DataAdapter` in the project, constructed but never indexed —
    `__getitem__` would need real image files on disk, which these tests
    deliberately avoid; the contract under test is the seam, not the decoding.
    """

    from fabric_defect_hub.models.mambaad.data import ImageOnlyDataset
    from fabric_defect_hub.models.moeclip.data import SampleDataset
    from fabric_defect_hub.models.torchvision.dataset import (
        SampleDetectionDataset,
        SampleSegmentationDataset,
    )

    samples = _samples()
    return [
        pytest.param(ImageOnlyDataset(samples, image_size=256), id="mambaad"),
        pytest.param(
            SampleDataset(samples, img_size=336, class_name_fn=lambda s: "fabric", train=False),
            id="moeclip",
        ),
        pytest.param(SampleDetectionDataset(samples, class_map={"defect": 1}), id="tv-detection"),
        pytest.param(
            SampleDetectionDataset(samples, class_map={"defect": 1}, with_masks=True),
            id="tv-instance-seg",
        ),
        pytest.param(SampleSegmentationDataset(samples), id="tv-segmentation"),
    ]


@pytest.mark.parametrize("adapter", _implementations())
def test_implements_the_dataadapter_contract(adapter):
    assert isinstance(adapter, DataAdapter)
    assert len(adapter) == 2                     # inherited __len__ over `samples`
    assert adapter.samples[0].id == "s-0"        # `samples` is always the first argument
    assert adapter.collate_fn is None or callable(adapter.collate_fn)


@pytest.mark.parametrize("adapter", _implementations())
def test_declares_a_wellformed_batch_spec(adapter):
    spec = adapter.batch_spec()

    assert isinstance(spec, BatchSpec)
    assert spec.item_kind in ITEM_KINDS
    assert spec.image_layout in IMAGE_LAYOUTS
    assert spec.mask_semantics is None or spec.mask_semantics in MASK_SEMANTICS
    if spec.normalization is not None:
        assert len(spec.normalization.mean) == len(spec.normalization.std)


@pytest.mark.parametrize("adapter", _implementations())
def test_batch_spec_is_json_safe_for_the_run_log(adapter):
    import json

    metadata = adapter.batch_spec().as_run_metadata()
    json.dumps(metadata)  # must not raise: it goes into the run log verbatim
    assert set(metadata) == {
        "item_kind",
        "image_layout",
        "image_dtype",
        "image_size",
        "mask_semantics",
        "normalization_mean",
        "normalization_std",
    }


def test_backends_declare_their_own_normalization_not_a_shared_assumption():
    """The concrete reason `batch_spec()` exists: MambaAD normalizes with
    ImageNet statistics and MoECLIP with CLIP's. Both are correct for their
    backend, and nothing else in the codebase records which was used.
    """

    from fabric_defect_hub.models.mambaad.data import IMAGENET_MEAN, ImageOnlyDataset
    from fabric_defect_hub.models.moeclip.data import CLIP_MEAN, SampleDataset

    mamba = ImageOnlyDataset(_samples(), image_size=256).batch_spec()
    moe = SampleDataset(
        _samples(), img_size=336, class_name_fn=lambda s: "fabric", train=False
    ).batch_spec()

    assert mamba.normalization.mean == IMAGENET_MEAN
    assert moe.normalization.mean == CLIP_MEAN
    assert mamba.normalization.mean != moe.normalization.mean


def test_detection_declares_a_collate_because_targets_cannot_be_stacked():
    from fabric_defect_hub.models.torchvision.dataset import (
        SampleDetectionDataset,
        detection_collate_fn,
    )

    adapter = SampleDetectionDataset(_samples(), class_map={"defect": 1})
    assert adapter.collate_fn is detection_collate_fn


def test_batch_spec_rejects_typos():
    with pytest.raises(ValueError, match="item_kind"):
        BatchSpec(item_kind="images")
    with pytest.raises(ValueError, match="mask_semantics"):
        BatchSpec(item_kind="image", mask_semantics="binary")
    with pytest.raises(ValueError, match="image_layout"):
        BatchSpec(item_kind="image", image_layout="NCHW")


def test_normalization_rejects_degenerate_statistics():
    with pytest.raises(ValueError, match="mean/std length mismatch"):
        Normalization(mean=(0.5, 0.5, 0.5), std=(0.5,))
    with pytest.raises(ValueError, match="non-zero"):
        Normalization(mean=(0.5,), std=(0.0,))


def test_incomplete_implementation_cannot_be_instantiated():
    """The ratchet itself: a new backend that forgets `batch_spec()` fails at
    construction, not at the first training step.
    """

    class _MissingBatchSpec(DataAdapter):
        def __getitem__(self, index):
            return None

    with pytest.raises(TypeError, match="batch_spec"):
        _MissingBatchSpec(_samples())
