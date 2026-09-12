"""`TrainConfig`: validation and translation, independent of any backend.

The behaviour under test is what `train(config: dict[str, Any])` could not
provide — a typo is rejected instead of silently ignored, and one set of names
renders into six different vocabularies without any of them seeing a key it
would reject.
"""

from __future__ import annotations

import pytest

from fabric_defect_hub.core.train_config import (
    CANONICAL_KEYS,
    TrainConfig,
    resolve_train_config,
)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_rejects_unknown_precision():
    with pytest.raises(ValueError, match="precision"):
        TrainConfig(precision="fp8")


def test_rejects_setting_both_run_length_knobs():
    # Backends read one or the other; setting both means one is silently lost.
    with pytest.raises(ValueError, match="not both"):
        TrainConfig(epochs=10, max_iters=5000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": -1},
        {"image_size": 0},
        {"lr": 0.0},
        {"max_iters": -100},
    ],
)
def test_rejects_nonpositive_values(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_num_workers_zero_is_allowed():
    # 0 means "load in the main process", a real and common setting.
    assert TrainConfig(num_workers=0).num_workers == 0


def test_rejects_backend_specific_shadowing_a_canonical_key():
    with pytest.raises(ValueError, match="must not repeat canonical keys"):
        TrainConfig(lr=0.01, backend_specific={"lr": 0.02})


# --------------------------------------------------------------------------- #
# Construction from a flat dict
# --------------------------------------------------------------------------- #
def test_from_dict_splits_canonical_from_backend_specific():
    config = TrainConfig.from_dict(
        {"epochs": 5, "lr": 0.01, "hm_percent_final": 0.9, "grad_clip_max_norm": 0.1}
    )

    assert config.epochs == 5
    assert config.lr == 0.01
    # A backend's own knobs are kept, not dropped — they just aren't portable.
    assert config.backend_specific == {"hm_percent_final": 0.9, "grad_clip_max_norm": 0.1}


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #
def test_to_backend_dict_uses_the_backends_own_names():
    config = TrainConfig(epochs=10, lr=0.01, batch_size=16)

    rendered = config.to_backend_dict({"epochs": "epochs", "lr": "lr0", "batch_size": "batch"})

    assert rendered == {"epochs": 10, "lr0": 0.01, "batch": 16}


def test_unset_fields_do_not_appear():
    # An unset field must not arrive as None and override a backend default.
    rendered = TrainConfig(lr=0.01).to_backend_dict({"lr": "lr0", "epochs": "epochs"})

    assert rendered == {"lr0": 0.01}


def test_a_field_the_backend_cannot_express_is_dropped_not_mistranslated():
    # Ultralytics has no iteration budget; forwarding `max_iters` under some
    # other name would crash its trainer.
    rendered = TrainConfig(max_iters=5000).to_backend_dict({"epochs": "epochs", "lr": "lr0"})

    assert rendered == {}


def test_backend_specific_passes_through_untouched():
    config = TrainConfig(lr=0.01, backend_specific={"hm_factor": 0.1})

    rendered = config.to_backend_dict({"lr": "lr"})

    assert rendered == {"lr": 0.01, "hm_factor": 0.1}


def test_to_backend_dict_rejects_a_bad_key_map():
    with pytest.raises(ValueError, match="non-canonical keys"):
        TrainConfig(lr=0.01).to_backend_dict({"learning_rate": "lr0"})


# --------------------------------------------------------------------------- #
# Back-compatibility: a plain dict still works
# --------------------------------------------------------------------------- #
def test_resolve_passes_a_plain_dict_through_unchanged():
    raw = {"total_iters": 100, "train_samples": ["<sample>"]}

    assert resolve_train_config(raw, {"max_iters": "total_iters"}) is raw


def test_resolve_translates_a_train_config():
    resolved = resolve_train_config(TrainConfig(max_iters=100), {"max_iters": "total_iters"})

    assert resolved == {"total_iters": 100}


# --------------------------------------------------------------------------- #
# Run provenance
# --------------------------------------------------------------------------- #
def test_run_metadata_is_json_safe_and_omits_unset_fields():
    import json

    metadata = TrainConfig(epochs=5, lr=0.01).as_run_metadata()
    json.dumps(metadata)

    assert metadata == {"epochs": 5, "lr": 0.01, "precision": "fp32"}
    assert set(metadata) <= {*CANONICAL_KEYS, "precision"}


# --------------------------------------------------------------------------- #
# Run length, per backend
# --------------------------------------------------------------------------- #
def test_run_length_keys_match_test_speed_overrides():
    """`training.RUN_LENGTH_KEYS` tells `api.load_config` where a backend's
    run length lives; `_apply_test_speed_overrides` writes to that same place
    when `--mode test` caps a smoke run. They are two statements of one fact,
    so this pins them together -- otherwise a backend could get its run-length
    key changed in one and not the other, and `epochs=` would silently write
    to a key nobody reads.
    """

    from fabric_defect_hub.training import RUN_LENGTH_KEYS, _apply_test_speed_overrides

    for backend, (dotted_key, unit) in RUN_LENGTH_KEYS.items():
        assert unit in {"epochs", "iterations"}, f"{backend}: unknown run-length unit {unit!r}"

        capped = _apply_test_speed_overrides({}, backend)
        section: object = capped
        for part in dotted_key.split("."):
            assert isinstance(section, dict) and part in section, (
                f"{backend}: RUN_LENGTH_KEYS says the run length is at {dotted_key!r}, but "
                f"--mode test does not write there"
            )
            section = section[part]
        assert isinstance(section, int) and section > 0
