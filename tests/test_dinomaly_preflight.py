import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from fabric_defect_hub.core.types import Annotations, Sample
from fabric_defect_hub.datasets.anomalib_folder import anomalib_folder_staging_dir
from fabric_defect_hub.models.dinomaly import presets
from fabric_defect_hub.models.dinomaly.adapter import (
    DinomalyAdapter,
    _bounded_network_wait,
    _load_pretrained_encoder,
    backbone_weights_url,
)

_VENDORED_ENCODER_SOURCE = (
    Path(__file__).resolve().parents[1] / "components" / "dinomaly" / "models" / "vit_encoder.py"
)


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


def test_backbone_weights_url_covers_every_shipped_encoder():
    assert backbone_weights_url("dinov2reg_vit_base_14") == (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
    )
    assert backbone_weights_url("dinov2reg_vit_small_14").endswith(
        "dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
    )
    assert backbone_weights_url("dinov2reg_vit_large_14").endswith(
        "dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
    )
    for encoder_name in presets.ENCODER_PRESETS:
        assert backbone_weights_url(encoder_name) is not None


def test_backbone_weights_url_refuses_spellings_this_adapter_never_ships():
    # The vendored loader also understands dinov1/beit/mae/moco/... names whose
    # checkpoints live elsewhere. Naming a file for those would be a guess, and
    # the guess is what the user would be told to download.
    assert backbone_weights_url("dinov1_vit_base_16") is None
    assert backbone_weights_url("dinov2reg_vit_huge_14") is None
    assert backbone_weights_url("") is None


def test_backbone_weights_url_still_matches_the_vendored_downloader():
    """`backbone_weights_url` is a copy of a convention that lives in a
    vendored checkout, so it can drift without anything failing until a user is
    told to stage a filename the loader never asks for. Compare against the
    loader's own f-string templates (``{patchsize}`` unformatted, as written).
    """

    source = _VENDORED_ENCODER_SOURCE.read_text()
    initials = {"small": "s", "base": "b", "large": "l"}
    for encoder_name in presets.ENCODER_PRESETS:
        initial = initials[encoder_name.split("_")[2]]
        stem = f"dinov2_vit{initial}{{patchsize}}"
        assert f"{stem}/{stem}_reg4_pretrain.pth" in source, encoder_name


def test_bounded_network_wait_sets_then_restores_the_default_timeout():
    previous = socket.getdefaulttimeout()
    with _bounded_network_wait(5):
        assert socket.getdefaulttimeout() == 5
    assert socket.getdefaulttimeout() == previous


def test_first_encoder_load_announces_the_download(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DINOMALY_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setenv("FDH_PROGRESS", "1")

    encoder = SimpleNamespace(load=lambda name: f"built:{name}")
    assert _load_pretrained_encoder("dinov2reg_vit_base_14", encoder) == "built:dinov2reg_vit_base_14"

    out = capsys.readouterr().out
    assert "dinov2_vitb14_reg4_pretrain.pth" in out
    assert "downloading" in out.lower()


def test_cached_encoder_is_not_announced(tmp_path, monkeypatch, capsys):
    (tmp_path / "dinov2_vitb14_reg4_pretrain.pth").write_bytes(b"cached")
    monkeypatch.setenv("DINOMALY_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setenv("FDH_PROGRESS", "1")

    _load_pretrained_encoder("dinov2reg_vit_base_14", SimpleNamespace(load=lambda name: name))
    assert capsys.readouterr().out == ""


def test_encoder_load_failure_names_the_file_and_where_to_put_it(tmp_path, monkeypatch):
    # An unreachable CDN used to block forever with no output: the vendored
    # downloader has no timeout and logs through `logging.info`. It must now
    # fail with the checkpoint name, its directory, and the original cause.
    monkeypatch.setenv("DINOMALY_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setenv("FDH_PROGRESS", "1")

    class _Unreachable:
        def load(self, name):
            raise OSError("connection timed out")

    with pytest.raises(RuntimeError) as caught:
        _load_pretrained_encoder("dinov2reg_vit_base_14", _Unreachable())

    message = str(caught.value)
    assert "dinov2_vitb14_reg4_pretrain.pth" in message
    assert str(tmp_path) in message
    assert "connection timed out" in message


def test_dinomaly_declares_a_probe_size_its_patch_size_accepts():
    """ViTill asserts the input is a whole number of 14-pixel patches. The
    FLOPs prober used to feed it a 640x640 default, so every Dinomaly
    benchmark row lost its FLOPs/LMEI columns to an assertion about the probe
    rather than about the model.
    """

    caps = DinomalyAdapter(name="dinov2reg_vit_base_14").capabilities()
    assert caps.probe_input_size == (448, 448)
    assert caps.probe_input_size[0] % 14 == 0
