from pathlib import Path

import pytest

from fabric_defect_hub.core.serialization import sample_to_dict
from fabric_defect_hub.core.types import Annotations, Prediction, Sample
from fabric_defect_hub.models.base import ModelCapabilities
from fabric_defect_hub.application import workspace
from fabric_defect_hub.application.workspace import (
    ALL_TEXTURES,
    DATASET_CATALOG,
    DEFECT_ONLY,
    MODEL_CATALOG,
    NORMAL_ONLY,
    SHOT_FEW,
    SHOT_FULL,
    build_gallery_state,
    current_image,
    default_dataset_root,
    detect_loaded_model,
    detect_current,
    empty_gallery_state,
    load_selected_model,
    move_image,
    render_prediction_tags,
    texture_choices,
)


def _samples(tmp_path):
    paths = []
    for name in ("first.jpg", "second.jpg", "third.jpg"):
        path = tmp_path / name
        path.write_bytes(b"not-an-image-but-present")
        paths.append(path)
    return [
        Sample(path.stem, str(path), "detection", Annotations(is_anomalous=index == 0))
        for index, path in enumerate(paths)
    ]


def test_gallery_state_is_seeded_and_bounded(tmp_path):
    state = build_gallery_state(_samples(tmp_path), count=10, seed=7, dataset_label="ZJU-Leaper")
    assert len(state["samples"]) == 3
    assert state["index"] == 0
    assert state["dataset"] == "ZJU-Leaper"


def test_navigation_wraps_and_preserves_caption(tmp_path):
    state = build_gallery_state(_samples(tmp_path), count=3, seed=0, dataset_label="ZJU-Leaper")
    moved, path, caption = move_image(state, -1)
    assert Path(path).name in {"first.jpg", "second.jpg", "third.jpg"}
    assert moved["index"] == 2
    assert "3 / 3" in caption


def test_empty_state_has_clear_browsing_message():
    path, caption = current_image(empty_gallery_state())
    assert path is None
    assert "No image" in caption


def test_dataset_root_prefers_the_configured_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ZJU_LEAPER_ROOT", str(tmp_path))
    assert default_dataset_root() == str(tmp_path.resolve())


def test_texture_choices_are_safe_when_the_dataset_is_not_connected(monkeypatch, tmp_path):
    monkeypatch.setenv("ZJU_LEAPER_ROOT", str(tmp_path / "missing"))
    assert ALL_TEXTURES in texture_choices("ZJU-Leaper")


def test_raw_fabrid_is_registered_alongside_zju_leaper():
    assert set(DATASET_CATALOG) == {
        "ZJU-Leaper", "RAW-FABRID", "MVTec AD", "MVTec LOCO", "VisA",
        "TILDA-400", "Fabric Defects", "Tianchi",
    }
    assert DATASET_CATALOG["RAW-FABRID"]["name"] == "raw-fabric"
    assert DATASET_CATALOG["MVTec AD"]["name"] == "mvtec-ad"
    assert DATASET_CATALOG["VisA"]["name"] == "visa"
    assert DATASET_CATALOG["MVTec LOCO"]["name"] == "mvtec-loco"


def test_raw_fabrid_has_no_texture_subdivision(monkeypatch, tmp_path):
    monkeypatch.setenv("RAW_FABRIC_ROOT", str(tmp_path / "missing"))
    assert texture_choices("RAW-FABRID") == [ALL_TEXTURES]


def test_raw_fabrid_root_prefers_the_configured_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RAW_FABRIC_ROOT", str(tmp_path))
    assert default_dataset_root("RAW-FABRID") == str(tmp_path.resolve())


def test_mvtec_ad_root_prefers_the_configured_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MVTEC_AD_ROOT", str(tmp_path))
    assert default_dataset_root("MVTec AD") == str(tmp_path.resolve())


def test_mvtec_ad_texture_choices_list_categories_found_on_disk(monkeypatch, tmp_path):
    for category in ("bottle", "cable"):
        (tmp_path / category / "train" / "good").mkdir(parents=True)
    monkeypatch.setenv("MVTEC_AD_ROOT", str(tmp_path))

    assert texture_choices("MVTec AD") == [ALL_TEXTURES, "bottle", "cable"]


def test_mvtec_ad_texture_choices_are_safe_when_the_dataset_is_not_connected(monkeypatch):
    monkeypatch.setattr(workspace, "default_dataset_root", lambda dataset_label: "")
    assert texture_choices("MVTec AD") == [ALL_TEXTURES]


def test_dataset_root_follows_symlinks_to_the_real_storage_location(monkeypatch, tmp_path):
    real_root = tmp_path / "real_storage"
    real_root.mkdir()
    link = tmp_path / "data_link"
    link.symlink_to(real_root)
    monkeypatch.setenv("ZJU_LEAPER_ROOT", str(link))

    resolved = default_dataset_root("ZJU-Leaper")

    assert resolved == str(real_root.resolve())
    assert "data_link" not in resolved


def test_anomalib_selection_dispatches_a_trusted_artifact_and_map_directory(monkeypatch, tmp_path):
    image = tmp_path / "fabric.jpg"
    image.write_bytes(b"placeholder")
    sample = Sample("fabric", str(image), "detection", Annotations())
    state = {"samples": [sample_to_dict(sample)], "index": 0, "dataset": "ZJU-Leaper"}
    captured = {}

    class FakeAnomalibModel:
        def predict(self, samples, artifact, output_dir=None):
            captured["artifact"] = artifact
            captured["output_dir"] = output_dir
            return [Prediction(sample_id=samples[0].id, anomaly_score=0.8)]

        def capabilities(self):
            return ModelCapabilities(
                tasks=("anomaly",), prediction_fields=("anomaly_score", "anomaly_map")
            )

    monkeypatch.setattr(workspace, "load_model", lambda backend, name: FakeAnomalibModel())
    monkeypatch.setattr(workspace, "render_prediction", lambda image_path, prediction: "rendered")
    monkeypatch.setattr(workspace, "model_status", lambda label, lang="en": "🟢 **Ready**")

    image, summary, status = detect_current(state, "PatchCore · ZJU-Leaper (normal only)")

    assert image == "rendered"
    assert summary["task"] == "anomaly"
    assert status.startswith("🟢")
    assert captured["artifact"].metadata["trusted"] is True
    assert captured["artifact"].metadata["model_class"] == "Patchcore"
    assert "anomaly_maps" in captured["output_dir"]


def test_loaded_detection_uses_the_session_manager_without_creating_an_adapter(monkeypatch, tmp_path):
    image = tmp_path / "fabric.jpg"
    image.write_bytes(b"placeholder")
    sample = Sample("fabric", str(image), "detection", Annotations())
    state = {"samples": [sample_to_dict(sample)], "index": 0, "dataset": "ZJU-Leaper"}
    captured = {}

    class FakeSessionManager:
        def predict(self, model_id, samples, **kwargs):
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return [Prediction(sample_id=samples[0].id, boxes=[])]

        def capabilities(self, model_id):
            return ModelCapabilities(
                tasks=("detection",), prediction_fields=("boxes", "labels", "scores")
            )

    monkeypatch.setattr(workspace, "render_prediction", lambda image_path, prediction: "rendered")

    image, summary, status = detect_loaded_model(FakeSessionManager(), state, "YOLOv8n · ZJU-Leaper")

    assert image == "rendered"
    assert summary["task"] == "detection"
    assert status.startswith("🟢")
    assert captured == {"model_id": "YOLOv8n · ZJU-Leaper", "kwargs": {}}


def test_load_selected_model_refuses_a_missing_checkpoint_without_calling_session_manager(monkeypatch, tmp_path):
    # Regression: Ultralytics' YOLO(path) loader treats a missing path whose
    # filename matches a known official release asset (e.g. "yolo11n.pt") as
    # a request to auto-download that asset — confirmed live, it silently
    # pulled generic COCO-pretrained weights into this catalog's published
    # slot for an unfinished model. The UI-facing loader must fail before
    # ever reaching the session manager/adapter for a missing checkpoint.
    missing = tmp_path / "does-not-exist.pt"
    monkeypatch.setitem(MODEL_CATALOG, "Fake Model", {**MODEL_CATALOG["YOLOv8n · ZJU-Leaper"], "checkpoint": missing})

    class ExplodingSessionManager:
        def load(self, model_id, spec, artifact):
            raise AssertionError("session_manager.load must not be called for a missing checkpoint")

    with pytest.raises(FileNotFoundError, match="Fake Model"):
        load_selected_model(ExplodingSessionManager(), "Fake Model")


def test_load_selected_model_delegates_to_session_manager_when_checkpoint_exists(monkeypatch, tmp_path):
    present = tmp_path / "weights.pt"
    present.write_bytes(b"placeholder")
    monkeypatch.setitem(MODEL_CATALOG, "Fake Model", {**MODEL_CATALOG["YOLOv8n · ZJU-Leaper"], "checkpoint": present})
    captured = {}

    class FakeSessionManager:
        def load(self, model_id, spec, artifact, samples=None):
            captured["model_id"] = model_id
            captured["artifact_path"] = artifact.path
            captured["samples"] = samples
            return {"active_model": model_id}

    result = load_selected_model(FakeSessionManager(), "Fake Model")

    assert result == {"active_model": "Fake Model"}
    assert captured == {"model_id": "Fake Model", "artifact_path": str(present), "samples": []}


def test_image_scope_filters_defect_and_normal_samples(monkeypatch, tmp_path):
    normal_path = tmp_path / "normal.jpg"
    defect_path = tmp_path / "defect.jpg"
    normal_path.write_bytes(b"normal")
    defect_path.write_bytes(b"defect")
    samples = [
        Sample("normal", str(normal_path), "detection", Annotations(is_anomalous=False)),
        Sample("defect", str(defect_path), "detection", Annotations(is_anomalous=True)),
    ]

    class FakeDataset:
        name = "zju-leaper"

        def load_samples(self):
            return samples

    monkeypatch.setattr(workspace, "default_dataset_root", lambda dataset_label="ZJU-Leaper": str(tmp_path))
    monkeypatch.setattr(workspace, "load_dataset", lambda *args, **kwargs: FakeDataset())

    defect_state, *_ = workspace.load_random_samples("ZJU-Leaper", "test", 8, seed=3, image_scope=DEFECT_ONLY)
    normal_state, *_ = workspace.load_random_samples("ZJU-Leaper", "train", 8, seed=3, image_scope=NORMAL_ONLY)

    assert defect_state["samples"][0]["annotations"]["is_anomalous"] is True
    assert normal_state["samples"][0]["annotations"]["is_anomalous"] is False


def test_shot_mode_controls_the_dataset_adapters_sample_count(monkeypatch, tmp_path):
    normal_path = tmp_path / "normal.jpg"
    normal_path.write_bytes(b"normal")
    samples = [Sample("normal", str(normal_path), "detection", Annotations(is_anomalous=False))]
    captured = {}

    class FakeDataset:
        name = "zju-leaper"

        def load_samples(self):
            return samples

    def fake_load_dataset(name, **kwargs):
        captured.update(kwargs)
        return FakeDataset()

    monkeypatch.setattr(workspace, "default_dataset_root", lambda dataset_label="ZJU-Leaper": str(tmp_path))
    monkeypatch.setattr(workspace, "load_dataset", fake_load_dataset)

    workspace.load_random_samples("ZJU-Leaper", "test", 8, seed=1, shot_mode=SHOT_FULL)
    assert captured["num_samples"] is None

    workspace.load_random_samples("ZJU-Leaper", "test", 8, seed=1, shot_mode=SHOT_FEW)
    assert captured["num_samples"] == workspace.FEW_SHOT_SAMPLE_COUNT
    assert captured["defect_ratio"] == workspace.FEW_SHOT_DEFECT_RATIO


def test_mvtec_ad_texture_selection_forwards_the_category_kwarg(monkeypatch, tmp_path):
    normal_path = tmp_path / "normal.jpg"
    normal_path.write_bytes(b"normal")
    samples = [Sample("bottle/good/normal", str(normal_path), "anomaly", Annotations(is_anomalous=False))]
    captured = {}

    class FakeDataset:
        name = "mvtec-ad"

        def load_samples(self):
            return samples

    def fake_load_dataset(name, **kwargs):
        captured.update(kwargs)
        return FakeDataset()

    monkeypatch.setattr(workspace, "default_dataset_root", lambda dataset_label="ZJU-Leaper": str(tmp_path))
    monkeypatch.setattr(workspace, "load_dataset", fake_load_dataset)

    workspace.load_random_samples("MVTec AD", "test", 8, seed=1, texture_label="bottle")
    assert captured["category"] == "bottle"

    workspace.load_random_samples("MVTec AD", "test", 8, seed=1, texture_label=ALL_TEXTURES)
    assert captured["category"] is None


def test_prediction_tags_render_defect_label_and_confidence_for_detection_and_anomaly():
    detection = render_prediction_tags(
        {"task": "detection", "detections": 1, "labels": ["Defect"], "scores": [0.2567], "anomaly_score": None, "has_anomaly_map": False}
    )
    anomaly = render_prediction_tags(
        {"task": "anomaly", "detections": 0, "labels": ["anomaly"], "scores": [], "anomaly_score": 0.7014, "has_anomaly_map": True}
    )

    assert "Defect" in detection
    assert "25.7%" in detection
    assert "Anomalous" in anomaly
    assert "70.1%" in anomaly
    assert "Heatmap available" in anomaly


def test_prediction_tags_separate_a_missing_heatmap_from_a_model_that_has_none():
    """An image-level-only model (GANomaly: the anomaly score is the distance
    between two latent vectors, so there is no spatial output) must not be
    reported the same way as a model whose map merely didn't arrive — the
    first is permanent, the second is fixable by passing an output_dir.
    """

    base = {"task": "anomaly", "detections": 0, "labels": [], "scores": [], "anomaly_score": 0.6}

    unsupported = render_prediction_tags({**base, "has_anomaly_map": False, "pixel_map_supported": False})
    not_produced = render_prediction_tags({**base, "has_anomaly_map": False, "pixel_map_supported": True})
    unknown = render_prediction_tags({**base, "has_anomaly_map": False})

    assert "Image-level score only" in unsupported
    assert "Heatmap not available" in not_produced
    # No capabilities passed -> the older, non-committal wording, so a caller
    # that doesn't know cannot accidentally assert the stronger claim.
    assert "Heatmap not available" in unknown


def test_prediction_summary_reports_pixel_map_support_from_capabilities():
    from fabric_defect_hub.web.single_image import prediction_summary

    prediction = Prediction(sample_id="x", anomaly_score=0.5)
    image_level_only = ModelCapabilities(tasks=("anomaly",), prediction_fields=("anomaly_score",))
    pixel_level = ModelCapabilities(
        tasks=("anomaly",), prediction_fields=("anomaly_score", "anomaly_map")
    )

    assert prediction_summary(prediction, image_level_only)["pixel_map_supported"] is False
    assert prediction_summary(prediction, pixel_level)["pixel_map_supported"] is True
    assert prediction_summary(prediction)["pixel_map_supported"] is None
