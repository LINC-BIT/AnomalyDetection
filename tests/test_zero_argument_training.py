"""The published contract: `fdh train <model>` works, from the project
root, with no environment-specific flags.

Every model this project ships must be reachable by its bare name, and no
shipped config may pin a device — otherwise a release forces its users to
discover and pass `--set` overrides that only exist to undo a checked-in
setting, which is exactly what these tests exist to prevent regressing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fabric_defect_hub.catalog import CANONICAL_MODELS
from fabric_defect_hub.training import (
    DEFAULT_MODEL_CONFIG_DIR,
    find_model_configs,
    resolve_model_config_and_variant,
)

# Models shipped but not (yet) in the catalog — no trained weights, so no
# canonical entry, but `fdh train <name>` must still reach them.
_UNCATALOGUED = ("fabricmamba", "draem")


def _all_shipped_names() -> list[str]:
    return [model.key for model in CANONICAL_MODELS] + list(_UNCATALOGUED)


@pytest.mark.parametrize("name", _all_shipped_names())
def test_every_shipped_model_resolves_from_its_bare_name(name):
    path, variant = resolve_model_config_and_variant(name)
    assert path.is_file()
    assert variant is not None, f"{name} resolved no variant; the run would use the config's default"


@pytest.mark.parametrize("name", _all_shipped_names())
def test_bare_names_are_case_insensitive(name):
    """`fdh train patchcore` and `fdh train PatchCore` must agree — users
    type the spelling they remember, not the catalog's."""

    lowered = resolve_model_config_and_variant(name.lower())
    exact = resolve_model_config_and_variant(name)
    assert lowered == exact


@pytest.mark.parametrize("config", find_model_configs(DEFAULT_MODEL_CONFIG_DIR), ids=lambda p: p.stem)
def test_no_shipped_config_pins_a_training_device(config: Path):
    """A checked-in device makes a GPU host train on CPU silently (anomalib's
    `engine_kwargs.accelerator` used to do exactly this) and forces every
    caller to pass `--set` to undo it. Leave it unset: anomalib inherits
    Lightning's "auto", and the other backends run their own
    cuda > mps > cpu detection.
    """

    raw = yaml.safe_load(config.read_text()) or {}
    train = raw.get("train") or {}

    accelerator = (train.get("engine_kwargs") or {}).get("accelerator")
    assert accelerator in (None, "auto"), (
        f"{config.name} pins engine_kwargs.accelerator={accelerator!r}; "
        "leave it unset so the host decides"
    )

    device = train.get("device")
    assert device in (None, "auto"), (
        f"{config.name} pins train.device={device!r}; leave it unset so the host decides"
    )


def test_missing_config_dir_blames_the_directory_not_the_model_name(tmp_path, monkeypatch):
    """Running `fdh` from outside the project root is the single most common
    way every model name fails at once; the error has to say so rather than
    claim the name is unknown.
    """

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_model_config_and_variant("fabricmamba")
