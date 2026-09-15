import io
import sys

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
    assert "Detection result" in labels
    assert "Model" in labels
    assert "Task type" in labels
    assert "Application domain" in labels
    assert "Inspect checkpoint" in values
    assert "Load model" in values
    assert "Unload model" in values
    assert "Texture / pattern" in labels
    assert "Image selection" in labels
    # The old, second name for the Model dropdown: README 4.1.2 calls it
    # "Model", and the page must not label the same control twice.
    assert "Local trained model" not in labels
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


def test_model_selectors_only_offer_models_with_a_staged_weight():
    """A registered slot with no file is a model that was never trained or
    downloaded here. It must not reach either selector — offering it and then
    answering "Checkpoint missing" is a wasted click."""

    from pathlib import Path

    from fabric_defect_hub.application.workspace import MODEL_CATALOG, available_model_labels

    labels = available_model_labels()
    assert labels  # the fallback below keeps this list non-empty
    config = create_app().get_config_file()
    choices = {
        component["props"]["label"]: [value for _, value in component["props"]["choices"]]
        for component in config["components"]
        if component["props"].get("label") in {"Model", "Models to benchmark"}
    }
    assert choices["Model"] == labels
    assert set(choices["Models to benchmark"]) <= set(labels)

    staged = {
        label for label in MODEL_CATALOG
        if Path(MODEL_CATALOG[label]["checkpoint"]).is_file()
    }
    if staged:
        # Weights are staged here, so the unfiltered fallback must not be in
        # play: the selectors offer exactly the staged slots, nothing else.
        assert set(labels) == staged


def test_styles_define_complete_light_and_dark_theme_rules():
    assert "body.dark .gradio-container" in CSS
    assert "body:not(.dark) .gradio-container" in CSS
    assert ":root { color-scheme: dark; }" not in CSS


def test_launch_prints_a_loopback_url_but_still_binds_all_interfaces(monkeypatch):
    """README 4.1.1 documents the app at a loopback address *and* as listening
    on all interfaces, so the banner Gradio prints must name the loopback
    host while `server_name` stays `0.0.0.0`."""

    captured_kwargs = {}
    installed_while_launching = {}

    class FakeApp:
        def launch(self, **kwargs):
            captured_kwargs.update(kwargs)
            # Gradio prints the banner part-way through `launch`, so the
            # filter has to be installed then, not merely around the call.
            installed_while_launching["stdout"] = sys.stdout
            return "launched"

    monkeypatch.setattr(app_module, "create_app", lambda: FakeApp())

    assert app_module.launch(server_port=7860) == "launched"
    assert isinstance(installed_while_launching["stdout"], app_module._LoopbackBanner)
    assert captured_kwargs["server_name"] == "0.0.0.0"


def test_loopback_banner_rewrites_only_the_wildcard_url():
    stream = io.StringIO()
    banner = app_module._LoopbackBanner(stream)
    banner.write("* Running on local URL:  http://0.0.0.0:6008")
    banner.write("\n")
    banner.write("* To create a public link, set `share=True` in `launch()`.\n")

    assert stream.getvalue() == (
        "* Running on local URL:  http://localhost:6008\n"
        "* To create a public link, set `share=True` in `launch()`.\n"
    )


def test_launch_restores_stdout_when_the_server_stops(monkeypatch):
    class FakeApp:
        def launch(self, **kwargs):
            return "launched"

    monkeypatch.setattr(app_module, "create_app", lambda: FakeApp())
    original = sys.stdout
    app_module.launch(server_port=7860)
    assert sys.stdout is original


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
