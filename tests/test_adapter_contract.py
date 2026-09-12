"""The interface ratchet: every registered model backend must satisfy the
`ModelAdapter` contract, checked the same way for all of them.

Without this file the contract is a docstring. Signatures drifted apart once
already — the anomaly backends grew `predict(..., output_dir=...)` while the
detection backends grew `predict(..., config=...)`, and `loader.py` ended up
calling `inspect.signature(model.export)` at runtime to find out which kind of
adapter it was holding before it dared call it. These tests fail on the *next*
divergence instead of letting it be discovered by a caller.

Nothing here loads weights or runs inference: adapters are inspected, not
executed, so the whole file runs without a GPU, a dataset, or any optional ML
framework beyond what is importable on the machine.
"""

from __future__ import annotations

import inspect

import pytest

from fabric_defect_hub.core.registry import get_model_cls, list_models
from fabric_defect_hub.core.train_config import CANONICAL_KEYS, TrainConfig
from fabric_defect_hub.loader import import_all_model_backends, list_model_backends
from fabric_defect_hub.models.base import (
    ANNOTATION_FIELDS,
    PREDICTION_FIELDS,
    TASKS,
    ModelAdapter,
    ModelCapabilities,
)

# Import every optional backend that is installed here, so the parametrisation
# below covers what this machine can actually run (a machine without anomalib
# simply tests fewer backends, rather than erroring).
import_all_model_backends()

INSTALLED_BACKENDS = sorted(set(list_models()) & set(list_model_backends()))

# A representative variant per backend: `capabilities()` is an instance method
# because a backend's answer can depend on the variant (torchvision), so the
# adapter has to be constructed. Construction loads no weights.
_VARIANTS = {
    "ultralytics": "yolov8n",
    "torchvision": "fasterrcnn_resnet50_fpn",
    "anomalib": "PatchCore",
    "dinomaly": "dinov2reg_vit_base_14",
    "moeclip": "ViT-L-14-336",
    "mambaad": "resnet34",
}


def _adapter(backend: str) -> ModelAdapter:
    cls = get_model_cls(backend)
    variant = _VARIANTS.get(backend)
    return cls(name=variant) if variant else cls()


def test_at_least_one_backend_is_installed():
    # Guards against this whole file silently passing as zero parametrised
    # cases if backend registration ever breaks.
    assert INSTALLED_BACKENDS, "no model backend importable; the contract went unchecked"


# --------------------------------------------------------------------------- #
# Signatures — identical across every backend, including ignored parameters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
@pytest.mark.parametrize("method", ["train", "predict", "export", "capabilities"])
def test_method_signature_matches_the_base_contract(backend, method):
    base_params = list(inspect.signature(getattr(ModelAdapter, method)).parameters)
    actual_params = list(inspect.signature(getattr(get_model_cls(backend), method)).parameters)

    # A backend may add trailing optional parameters of its own, but the
    # contract's parameters must come first, in order, under the same names —
    # so any caller can call any backend positionally or by keyword.
    assert actual_params[: len(base_params)] == base_params, (
        f"{backend}.{method}{tuple(actual_params)} diverges from "
        f"ModelAdapter.{method}{tuple(base_params)}"
    )


@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
def test_export_is_callable_without_signature_introspection(backend):
    # The exact call `loader.run_experiment` makes. Binding it proves the call
    # is valid for every backend without inspecting anything first.
    signature = inspect.signature(get_model_cls(backend).export)
    signature.bind(object(), object(), target="onnx", config={})


# --------------------------------------------------------------------------- #
# Capabilities — declared, well-formed, and consistent with the rest
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
def test_backend_declares_wellformed_capabilities(backend):
    caps = _adapter(backend).capabilities()

    assert isinstance(caps, ModelCapabilities)
    assert set(caps.tasks) <= set(TASKS)
    assert set(caps.prediction_fields) <= set(PREDICTION_FIELDS)
    assert set(caps.required_annotations) <= set(ANNOTATION_FIELDS)
    assert caps.tasks and caps.prediction_fields


@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
def test_anomaly_backends_declare_the_fields_pixel_metrics_need(backend):
    caps = _adapter(backend).capabilities()
    if "anomaly" not in caps.tasks:
        pytest.skip(f"{backend} is not an anomaly backend")

    # An anomaly backend that filled neither an image score nor a map would
    # leave `evaluation/anomaly.py` nothing to score.
    assert caps.fills("anomaly_score") or caps.fills("anomaly_map")


@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
def test_detection_backends_fill_boxes(backend):
    caps = _adapter(backend).capabilities()
    if "detection" not in caps.tasks:
        pytest.skip(f"{backend} is not a detection backend")

    assert caps.fills("boxes") and caps.fills("scores")


@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
def test_every_backend_declares_a_known_export_input_style(backend):
    """The shape an exported module wants (`batched` vs `list`) has to be
    declared, because whoever profiles it must synthesize an input for it and
    cannot introspect a TorchScript signature. It used to be decided in
    `web/benchmark.py` by an `if backend == "torchvision" and ...`, which put
    a fact about torchvision's forward signature in the UI.
    """

    from fabric_defect_hub.models.base import EXPORT_INPUT_STYLES

    assert _adapter(backend).capabilities().export_input_style in EXPORT_INPUT_STYLES


def test_export_input_style_is_vocabulary_checked():
    with pytest.raises(ValueError, match="export_input_style"):
        ModelCapabilities(
            tasks=("detection",), prediction_fields=("boxes",), export_input_style="tensor"
        )


@pytest.mark.architecture
@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
def test_declared_export_targets_are_honest(backend):
    """A backend whose `export()` unconditionally raises `NotImplementedError`
    must declare no export targets — otherwise a caller that checks
    `can_export()` first still gets an exception.
    """

    adapter = _adapter(backend)
    caps = adapter.capabilities()
    source = inspect.getsource(type(adapter).export)
    unconditionally_raises = source.count("raise NotImplementedError") == 1 and "if " not in source

    if unconditionally_raises:
        assert caps.export_targets == (), (
            f"{backend}.export() always raises NotImplementedError but declares "
            f"export_targets={caps.export_targets}"
        )


# --------------------------------------------------------------------------- #
# Portable training settings — every backend translates the same vocabulary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
def test_backend_declares_its_train_config_key_map(backend):
    key_map = get_model_cls(backend).TRAIN_CONFIG_KEYS

    assert key_map, f"{backend} declares no TRAIN_CONFIG_KEYS; TrainConfig cannot reach it"
    unknown = sorted(set(key_map) - set(CANONICAL_KEYS))
    assert not unknown, f"{backend}.TRAIN_CONFIG_KEYS has non-canonical names {unknown}"


@pytest.mark.parametrize("backend", INSTALLED_BACKENDS)
def test_train_config_renders_into_each_backends_own_names(backend):
    key_map = get_model_cls(backend).TRAIN_CONFIG_KEYS
    config = TrainConfig(
        epochs=3, lr=0.01, batch_size=8, image_size=256, device="cpu", num_workers=2
    )

    rendered = config.to_backend_dict(key_map)

    # Only names this backend actually accepts come out, and nothing canonical
    # leaks through under a name the backend would reject.
    assert set(rendered) <= set(key_map.values())
    if "lr" in key_map:
        assert rendered[key_map["lr"]] == 0.01


def test_the_same_train_config_reaches_two_backends_under_different_names():
    """The concrete problem TrainConfig solves: one learning rate, two names."""

    ultralytics = pytest.importorskip("fabric_defect_hub.models.ultralytics.adapter")
    dinomaly = pytest.importorskip("fabric_defect_hub.models.dinomaly.adapter")

    config = TrainConfig(lr=0.005, batch_size=4)
    yolo_kwargs = config.to_backend_dict(ultralytics.UltralyticsAdapter.TRAIN_CONFIG_KEYS)
    dino_kwargs = config.to_backend_dict(dinomaly.DinomalyAdapter.TRAIN_CONFIG_KEYS)

    assert yolo_kwargs["lr0"] == 0.005 and yolo_kwargs["batch"] == 4
    assert dino_kwargs["lr"] == 0.005 and dino_kwargs["batch_size"] == 4


def test_capabilities_rejects_a_typo():
    # The point of the fixed vocabularies: a misspelt field is a hard error at
    # declaration time, not a metric that silently never fires.
    with pytest.raises(ValueError, match="prediction_fields"):
        ModelCapabilities(tasks=("anomaly",), prediction_fields=("anomaly_maps",))
    with pytest.raises(ValueError, match="tasks"):
        ModelCapabilities(tasks=("detect",), prediction_fields=("boxes",))
