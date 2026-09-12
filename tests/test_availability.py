"""Fast, framework-free tests for `core.availability`: is a dataset actually
staged on *this* machine right now, independent of whether the project knows
how to use it (that's `core.dataset_capabilities`/`core.registry`'s job).
"""

from fabric_defect_hub.core.availability import (
    dataset_is_staged,
    root_is_staged,
    staged_datasets,
)


def test_root_is_staged_true_for_a_real_non_empty_directory(tmp_path):
    (tmp_path / "sample.jpg").write_text("x")
    assert root_is_staged(str(tmp_path)) is True


def test_root_is_staged_false_for_an_empty_directory(tmp_path):
    # An existing-but-empty symlink target is functionally "not staged".
    assert root_is_staged(str(tmp_path)) is False


def test_root_is_staged_false_for_missing_none_and_placeholder():
    assert root_is_staged(None) is False
    assert root_is_staged("") is False
    assert root_is_staged("/definitely/does/not/exist/anywhere") is False
    assert root_is_staged("${SOME_ENV_VAR}") is False


def test_dataset_is_staged_uses_the_given_root_map(tmp_path):
    (tmp_path / "f.jpg").write_text("x")
    root_map = {"zju-leaper": str(tmp_path), "raw-fabric": "/nope"}

    assert dataset_is_staged("zju-leaper", root_map=root_map) is True
    assert dataset_is_staged("raw-fabric", root_map=root_map) is False
    assert dataset_is_staged("unregistered-name", root_map=root_map) is False


def test_staged_datasets_filters_to_only_whats_present_on_disk(tmp_path):
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "f.jpg").write_text("x")
    root_map = {"a": str(staged_dir), "b": "/nope", "c": None}

    assert staged_datasets({"a", "b", "c"}, root_map=root_map) == {"a"}


def test_staged_datasets_empty_when_nothing_matches(tmp_path):
    root_map = {"a": "/nope"}
    assert staged_datasets({"a"}, root_map=root_map) == set()
