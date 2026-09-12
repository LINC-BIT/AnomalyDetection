"""Fast, framework-free tests for `core.decision`: the dataset-selection
decision tree that lets a backend train on whatever's actually staged, rather
than hard-failing the moment a config names a dataset this machine doesn't
have. See the module docstring for the exact policy this pins down.
"""

from fabric_defect_hub.core.decision import decide_dataset


def test_requested_dataset_staged_is_used_without_substitution(tmp_path):
    (tmp_path / "f.jpg").write_text("x")
    root_map = {"zju-leaper": str(tmp_path), "raw-fabric": "/nope"}

    decision = decide_dataset("zju-leaper", {"zju-leaper", "raw-fabric"}, root_map=root_map)

    assert decision.runnable is True
    assert decision.dataset == "zju-leaper"
    assert decision.substituted is False
    assert decision.requested == "zju-leaper"


def test_requested_dataset_not_staged_substitutes_a_staged_alternative(tmp_path):
    (tmp_path / "f.jpg").write_text("x")
    root_map = {"zju-leaper": str(tmp_path), "raw-fabric": "/nope"}

    decision = decide_dataset("raw-fabric", {"zju-leaper", "raw-fabric"}, root_map=root_map)

    assert decision.runnable is True
    assert decision.dataset == "zju-leaper"
    assert decision.substituted is True
    assert decision.requested == "raw-fabric"
    assert "substituted" in decision.reason


def test_nothing_staged_is_not_runnable_with_actionable_reason():
    root_map = {"zju-leaper": "/nope1", "raw-fabric": "/nope2"}

    decision = decide_dataset("zju-leaper", {"zju-leaper", "raw-fabric"}, root_map=root_map)

    assert decision.runnable is False
    assert decision.dataset is None
    assert "zju-leaper" in decision.reason and "raw-fabric" in decision.reason


def test_no_dataset_requested_picks_deterministically_by_name(tmp_path):
    (tmp_path / "f.jpg").write_text("x")
    root_map = {"b-set": str(tmp_path), "a-set": str(tmp_path)}

    decision = decide_dataset(None, {"a-set", "b-set"}, root_map=root_map)

    # Alphabetically-first staged dataset, not a hand-tuned "best" ranking
    # (see module docstring for why this project doesn't pretend to rank
    # dataset quality without benchmark evidence).
    assert decision.dataset == "a-set"
    assert decision.runnable is True


def test_substitution_reason_lists_other_staged_alternatives(tmp_path):
    (tmp_path / "f.jpg").write_text("x")
    root_map = {"a": str(tmp_path), "b": str(tmp_path), "c": "/nope"}

    decision = decide_dataset("c", {"a", "b", "c"}, root_map=root_map)

    assert decision.dataset == "a"
    assert "b" in decision.reason
