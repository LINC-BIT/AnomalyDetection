import json

from fabric_defect_hub.models.base import Artifact
from fabric_defect_hub.weight_registry import MANIFEST_FILENAME, read_weight_manifest, record_weight


def test_record_weight_captures_both_storage_locations_and_config(tmp_path):
    registered = tmp_path / "artifacts" / "models" / "run-specific.pt"
    registered.parent.mkdir(parents=True)
    registered.write_bytes(b"weights")
    published = registered.parent / "published" / "yolov8n.pt"
    published.parent.mkdir()
    published.write_bytes(b"weights")
    config_source = tmp_path / "configs" / "model.yaml"
    config_source.parent.mkdir()
    config_source.write_text("model: {}\n")

    manifest = record_weight(
        project_root=tmp_path,
        backend="ultralytics",
        variant="yolov8n",
        model_key="yolov8n",
        run_id="zju-medium",
        config_path=config_source,
        resolved_config={"train": {"batch": 32}, "data": {"num_samples": 600}},
        registered_artifact=Artifact(
            path=str(registered),
            backend="ultralytics",
            metadata={"parameter_count": 3_200_000, "trainable_parameter_count": 3_200_000},
        ),
        published_path=published,
        metrics={"map50": 0.7},
    )

    assert manifest == tmp_path / "artifacts" / "models" / MANIFEST_FILENAME
    record = read_weight_manifest(tmp_path)[0]
    assert record["batch_run_id"] == "zju-medium"
    assert record["artifact"]["registered_path"] == str(registered)
    assert record["artifact"]["published_path"] == str(published)
    assert record["artifact"]["primary_location"] == "published"
    assert record["parameters"] == {
        "total": 3_200_000,
        "trainable": 3_200_000,
        "status": "measured_during_training",
    }
    snapshot = json.loads((tmp_path / "artifacts" / "models" / "records" / f"{record['record_id']}.config.json").read_text())
    assert snapshot["train"]["batch"] == 32
    # The manifest carries the same provenance block as the run log, so a
    # weight and a result row can be matched to one code state.
    assert {
        "timestamp_utc",
        "git_commit",
        "hostname",
        "vendored_components",
    } <= set(record["provenance"])


def test_record_weight_marks_unavailable_parameter_counts(tmp_path):
    artifact = tmp_path / "artifacts" / "models" / "zero-shot.ckpt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metadata-only")
    config_source = tmp_path / "config.yaml"
    config_source.write_text("model: {}\n")

    record_weight(
        project_root=tmp_path,
        backend="anomalib",
        variant="WinClip",
        config_path=config_source,
        resolved_config={},
        registered_artifact=Artifact(path=str(artifact), backend="anomalib"),
        published_path=None,
    )

    record = read_weight_manifest(tmp_path)[0]
    assert record["artifact"]["primary_location"] == "registered"
    assert record["parameters"] == {"total": None, "trainable": None, "status": "unavailable"}
