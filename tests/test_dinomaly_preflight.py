from pathlib import Path

import pytest

from fabric_defect_hub.core.types import Annotations, Sample
from fabric_defect_hub.datasets.anomalib_folder import anomalib_folder_staging_dir
from fabric_defect_hub.models.dinomaly.adapter import DinomalyAdapter


def _sample(sample_id: str, anomalous: bool, mask_path: str | None = None) -> Sample:
    return Sample(
        id=sample_id,
        image_path="image.png",
        task="anomaly",
        annotations=Annotations(is_anomalous=anomalous, anomaly_mask=mask_path),
    )


def test_dinomaly_preflight_rejects_defects_without_masks():
    with pytest.raises(ValueError, match="missing masks for 1 sample.*'missing'"):
        DinomalyAdapter._validate_test_masks([_sample("missing", anomalous=True)])


def test_dinomaly_preflight_accepts_existing_masks(tmp_path: Path):
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask")
    DinomalyAdapter._validate_test_masks([
        _sample("normal", anomalous=False),
        _sample("defect", anomalous=True, mask_path=str(mask)),
    ])


def test_dinomaly_staging_normalizes_mask_suffix(tmp_path: Path):
    image = tmp_path / "image.jpg"
    mask = tmp_path / "mask.jpg"
    image.write_bytes(b"image")
    mask.write_bytes(b"mask")
    train = _sample("train", anomalous=False)
    train.image_path = str(image)
    defect = _sample("defect", anomalous=True, mask_path=str(mask))
    defect.image_path = str(image)

    with anomalib_folder_staging_dir(
        [train], [defect], mask_suffix=".png", image_suffix=".png"
    ) as layout:
        staged_mask = layout.root / "ground_truth" / "defect" / "defect.png"
        staged_train = layout.root / "train" / "good" / "train.png"
        staged_image = layout.root / "test" / "defect" / "defect.png"
        assert staged_mask.is_symlink()
        assert staged_train.is_symlink()
        assert staged_image.is_symlink()
        assert staged_mask.resolve() == mask
