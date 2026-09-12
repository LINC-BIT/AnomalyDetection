import pytest

from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.inference.session import InferenceSessionManager, ModelNotLoadedError
from fabric_defect_hub.models.base import Artifact


class FakeAdapter:
    def __init__(self):
        self.loaded = None
        self.unloaded = False

    def load_trained_model(self, artifact):
        self.loaded = artifact
        return artifact

    def predict(self, samples, artifact, **kwargs):
        assert artifact is self.loaded
        return [Prediction(sample_id=sample.id) for sample in samples]

    def unload(self):
        self.unloaded = True


def test_session_requires_explicit_load_and_evicts_previous_model():
    adapters = []

    def loader(backend, name):
        adapter = FakeAdapter()
        adapters.append(adapter)
        return adapter

    manager = InferenceSessionManager(model_loader=loader)
    spec = {"backend": "fake", "name": "first", "task": "detection"}
    artifact = Artifact(path="first.pt", backend="fake")

    with pytest.raises(ModelNotLoadedError):
        manager.predict("first", [])

    status = manager.load("first", spec, artifact)
    assert status["loaded"] is True
    assert manager.predict("first", [Sample("one", "one.jpg", "detection", None)])[0].sample_id == "one"

    manager.load("second", {**spec, "name": "second"}, Artifact(path="second.pt", backend="fake"))
    assert adapters[0].unloaded is True
    assert manager.status()["model_id"] == "second"

    manager.unload()
    assert adapters[1].unloaded is True
    assert manager.status()["loaded"] is False
