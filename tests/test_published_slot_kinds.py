"""`published/` must work whether a slot holds a real file or a symlink.

`publish_artifact` writes symlinks, but a tree copied down from the training
box arrives as regular files, and both have to be loadable without a
migration step. The third state — a symlink whose target did not come along
— must be reported as its own problem rather than as "never trained".
"""

from __future__ import annotations

import pytest

from fabric_defect_hub.catalog import (
    PUBLISHED_STATES,
    describe_published,
    published_is_usable,
    published_status,
)
from fabric_defect_hub.metric_sweep import broken_published_links, discover_trained_models


def _published(root):
    path = root / "artifacts" / "models" / "published"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _real_file(published, name="PatchCore.ckpt"):
    path = published / name
    path.write_bytes(b"\0" * 32)
    return path


def _valid_link(root, published, name="PatchCore.ckpt", target_name="PatchCore_run1.ckpt"):
    target = root / "artifacts" / "models" / target_name
    target.write_bytes(b"\0" * 32)
    link = published / name
    link.symlink_to(f"../{target_name}")
    return link, target


def _broken_link(published, name="PatchCore.ckpt"):
    link = published / name
    link.symlink_to("../never_copied.ckpt")
    return link


# --------------------------------------------------------------------------- #
# The three states
# --------------------------------------------------------------------------- #
def test_a_regular_file_is_usable(tmp_path):
    path = _real_file(_published(tmp_path))

    assert published_status(path) == "file"
    assert published_is_usable(path)


def test_a_resolving_symlink_is_usable(tmp_path):
    link, _ = _valid_link(tmp_path, _published(tmp_path))

    assert published_status(link) == "symlink"
    assert published_is_usable(link)


def test_a_dangling_symlink_is_its_own_state(tmp_path):
    """`is_file()` is False here too, which is why every reader used to call
    this "not published" and send the user off to retrain."""

    link = _broken_link(_published(tmp_path))

    assert published_status(link) == "broken_link"
    assert not published_is_usable(link)


def test_an_absent_slot_is_missing(tmp_path):
    assert published_status(_published(tmp_path) / "nothing.ckpt") == "missing"


def test_every_state_is_in_the_declared_vocabulary(tmp_path):
    published = _published(tmp_path)
    states = {
        published_status(_real_file(published, "a.ckpt")),
        published_status(_valid_link(tmp_path, published, "b.ckpt", "b_run.ckpt")[0]),
        published_status(_broken_link(published, "c.ckpt")),
        published_status(published / "d.ckpt"),
    }
    assert states == set(PUBLISHED_STATES)


# --------------------------------------------------------------------------- #
# Discovery treats file and link alike
# --------------------------------------------------------------------------- #
def test_discovery_finds_a_model_published_as_a_regular_file(tmp_path):
    """The copied-from-the-cloud-box case."""

    _real_file(_published(tmp_path))

    found = discover_trained_models(tmp_path)

    assert [(m.backend, m.variant) for m in found] == [("anomalib", "PatchCore")]
    assert not found[0].weights.is_symlink()


def test_discovery_finds_a_model_published_as_a_symlink(tmp_path):
    _, target = _valid_link(tmp_path, _published(tmp_path))

    found = discover_trained_models(tmp_path)

    assert [(m.backend, m.variant) for m in found] == [("anomalib", "PatchCore")]
    assert found[0].weights == target.resolve()


def test_discovery_skips_a_dangling_link_rather_than_offering_unloadable_weights(tmp_path):
    _broken_link(_published(tmp_path))

    assert discover_trained_models(tmp_path) == []


# --------------------------------------------------------------------------- #
# Dangling links are reported
# --------------------------------------------------------------------------- #
def test_broken_links_are_reported_with_a_fix(tmp_path):
    _broken_link(_published(tmp_path))

    (path, description), = broken_published_links(tmp_path)

    assert path.name == "PatchCore.ckpt"
    assert "artifacts/models/" in description  # the directory to copy across


def test_healthy_slots_report_nothing(tmp_path):
    published = _published(tmp_path)
    _real_file(published, "a.ckpt")
    _valid_link(tmp_path, published, "b.ckpt", "b_run.ckpt")

    assert broken_published_links(tmp_path) == []


def test_reporting_survives_a_project_with_no_published_dir(tmp_path):
    assert broken_published_links(tmp_path) == []


# --------------------------------------------------------------------------- #
# The facade distinguishes the two failures
# --------------------------------------------------------------------------- #
def test_from_pretrained_names_the_dangling_link_case(tmp_path, monkeypatch):
    from fabric_defect_hub import api, catalog

    published = _published(tmp_path)
    monkeypatch.setattr(catalog, "PUBLISHED_MODEL_ROOT", published)
    _broken_link(published)

    with pytest.raises(FileNotFoundError, match="unreachable"):
        api.from_pretrained("PatchCore")


def test_from_pretrained_still_says_untrained_when_nothing_is_published(tmp_path, monkeypatch):
    from fabric_defect_hub import api, catalog

    monkeypatch.setattr(catalog, "PUBLISHED_MODEL_ROOT", _published(tmp_path))

    with pytest.raises(FileNotFoundError, match="no published weights"):
        api.from_pretrained("PatchCore")


def test_from_pretrained_accepts_a_regular_file(tmp_path, monkeypatch):
    from fabric_defect_hub import api, catalog

    published = _published(tmp_path)
    monkeypatch.setattr(catalog, "PUBLISHED_MODEL_ROOT", published)
    _real_file(published)

    assert api.from_pretrained("PatchCore").backend == "anomalib"


def test_list_pretrained_counts_a_regular_file_as_available(tmp_path, monkeypatch):
    from fabric_defect_hub import api, catalog

    published = _published(tmp_path)
    monkeypatch.setattr(catalog, "PUBLISHED_MODEL_ROOT", published)
    _real_file(published)

    assert "PatchCore" in api.list_pretrained(available_only=True)


def test_list_pretrained_excludes_a_dangling_link(tmp_path, monkeypatch):
    from fabric_defect_hub import api, catalog

    published = _published(tmp_path)
    monkeypatch.setattr(catalog, "PUBLISHED_MODEL_ROOT", published)
    _broken_link(published)

    assert "PatchCore" not in api.list_pretrained(available_only=True)


def test_describe_reads_out_the_link_target(tmp_path):
    link, _ = _valid_link(tmp_path, _published(tmp_path))

    assert "PatchCore_run1.ckpt" in describe_published(link)
