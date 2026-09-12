import json

from fabric_defect_hub.training_runs import BatchRunTracker


def test_batch_tracker_skips_completed_models_when_resuming(tmp_path):
    models = [{"key": "yolo", "backend": "ultralytics", "variant": "yolov8n", "config": "x.yaml"}]
    tracker = BatchRunTracker(tmp_path, "run-1", models)
    tracker.begin("yolo")
    tracker.finish("yolo", succeeded=True, detail="published/yolo.pt")

    resumed = BatchRunTracker(tmp_path, "run-1", models, resume=True)
    assert resumed.should_run("yolo") is False
    assert json.loads((tmp_path / "run-1" / "state.json").read_text())["models"]["yolo"]["status"] == "succeeded"


def test_batch_tracker_retries_interrupted_model(tmp_path):
    models = [{"key": "faster", "backend": "torchvision", "variant": "fasterrcnn", "config": "x.yaml"}]
    tracker = BatchRunTracker(tmp_path, "run-2", models)
    tracker.begin("faster")
    tracker.interrupt("faster")

    resumed = BatchRunTracker(tmp_path, "run-2", models, resume=True)
    assert resumed.should_run("faster") is True
