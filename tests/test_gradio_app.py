import pytest


gradio = pytest.importorskip("gradio")

from fabric_defect_hub.web import app as app_module
from fabric_defect_hub.web.app import CSS, create_app


def test_gradio_app_builds_with_single_image_and_benchmark_tabs():
    app = create_app()
    config = app.get_config_file()
    labels = {component.get("props", {}).get("label") for component in config["components"]}
    values = [component.get("props", {}).get("value") for component in config["components"]]
    assert "Selected dataset image" in labels
    assert "Inference result" in labels
    assert "Local trained model" in labels
    assert "Inspect checkpoint" in values
    assert "Load model" in values
    assert "Unload model" in values
    assert "Texture / pattern" in labels
    assert "Image selection" in labels
    assert "Checkpoint path" not in labels
    assert "Dataset root" not in labels
    run_index = next(
        index for index, component in enumerate(config["components"])
        if component.get("props", {}).get("value") == "Run detection"
    )
    source_index = next(
        index for index, component in enumerate(config["components"])
        if component.get("props", {}).get("label") == "Selected dataset image"
    )
    assert run_index < source_index


def test_styles_define_complete_light_and_dark_theme_rules():
    assert "body.dark .gradio-container" in CSS
    assert "body:not(.dark) .gradio-container" in CSS
    assert ":root { color-scheme: dark; }" not in CSS


def test_launch_injects_the_application_stylesheet(monkeypatch):
    captured_kwargs = {}

    class FakeApp:
        def launch(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "launched"

    monkeypatch.setattr(app_module, "create_app", lambda: FakeApp())
    monkeypatch.setattr(
        app_module,
        "default_dataset_root",
        lambda dataset_label: f"/external/{dataset_label}",
    )

    assert app_module.launch(server_port=7860) == "launched"
    assert captured_kwargs["css"] == CSS
    assert captured_kwargs["allowed_paths"] == [
        f"/external/{label}" for label in app_module.DATASET_CATALOG
    ]
