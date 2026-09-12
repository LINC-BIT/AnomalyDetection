"""Storage hygiene: symlinked publishing, dedupe/adopt, and prune safety.

The tests that matter here are the ones about what is *not* deleted — a
prune that frees space is easy, a prune that never eats a checkpoint someone
still needs is the actual requirement.
"""

from __future__ import annotations

import json

import pytest

from fabric_defect_hub.retention import (
    Plan,
    apply_plan,
    plan_checkpoint_prune,
    plan_pretrain_cleanup,
    plan_publish_dedupe,
    protected_paths,
)


def _models_root(root):
    path = root / "artifacts" / "models"
    (path / "published").mkdir(parents=True, exist_ok=True)
    return path


def _weight(models_root, name: str, size: int = 1024):
    path = models_root / name
    path.write_bytes(b"\0" * size)
    return path


def _manifest(models_root, records):
    with (models_root / "weight_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _record(backend, variant, path, *, recorded_at, run_id=None):
    return {
        "kind": "trained", "backend": backend, "variant": variant,
        "recorded_at": recorded_at, "batch_run_id": run_id,
        "artifact": {"registered_path": str(path), "published_path": None},
    }


# --------------------------------------------------------------------------- #
# Publishing writes a relative symlink, not a copy
# --------------------------------------------------------------------------- #
def test_publish_creates_a_relative_symlink(tmp_path, monkeypatch):
    from fabric_defect_hub import catalog

    models_root = _models_root(tmp_path)
    source = _weight(models_root, "Patchcore_run7_best.ckpt", size=2048)
    monkeypatch.setattr(catalog, "PUBLISHED_MODEL_ROOT", models_root / "published")

    destination = catalog.publish_artifact("anomalib", "PatchCore", str(source))

    assert destination is not None and destination.is_symlink()
    # The published name drops the run suffix; the link resolves to the file
    # that actually holds the bytes.
    assert destination.name == "PatchCore.ckpt"
    assert destination.resolve() == source.resolve()
    # Relative, so the whole artifacts/ tree stays movable between machines.
    import os
    assert not os.path.isabs(os.readlink(destination))


def test_republishing_replaces_an_existing_copy(tmp_path, monkeypatch):
    from fabric_defect_hub import catalog

    models_root = _models_root(tmp_path)
    monkeypatch.setattr(catalog, "PUBLISHED_MODEL_ROOT", models_root / "published")
    stale = models_root / "published" / "PatchCore.ckpt"
    stale.write_bytes(b"\1" * 4096)  # a leftover copy from before symlinks

    source = _weight(models_root, "Patchcore_run8_best.ckpt")
    destination = catalog.publish_artifact("anomalib", "PatchCore", str(source))

    assert destination.is_symlink()
    assert destination.resolve() == source.resolve()


def test_publishing_missing_weights_fails_loudly(tmp_path, monkeypatch):
    from fabric_defect_hub import catalog

    models_root = _models_root(tmp_path)
    monkeypatch.setattr(catalog, "PUBLISHED_MODEL_ROOT", models_root / "published")
    with pytest.raises(FileNotFoundError):
        catalog.publish_artifact("anomalib", "PatchCore", str(models_root / "nope.ckpt"))


# --------------------------------------------------------------------------- #
# Dedupe / adopt
# --------------------------------------------------------------------------- #
def test_duplicate_published_copy_becomes_a_symlink(tmp_path):
    models_root = _models_root(tmp_path)
    registered = _weight(models_root, "Ganomaly.ckpt", size=4096)
    duplicate = models_root / "published" / "Ganomaly.ckpt"
    duplicate.write_bytes(b"\0" * 4096)

    plan = plan_publish_dedupe(tmp_path)

    assert [action.kind for action in plan.actions] == ["dedupe"]
    assert plan.reclaimed_bytes == 4096
    apply_plan(plan, confirm=True)
    assert duplicate.is_symlink() and duplicate.resolve() == registered.resolve()


def test_published_file_with_no_counterpart_is_adopted_not_deleted(tmp_path):
    """The only copy of a model's weights must survive the cleanup."""

    models_root = _models_root(tmp_path)
    orphan = models_root / "published" / "detr_resnet50.pt"
    orphan.write_bytes(b"\2" * 2048)

    plan = plan_publish_dedupe(tmp_path)

    assert [action.kind for action in plan.actions] == ["adopt"]
    apply_plan(plan, confirm=True)
    adopted = models_root / "detr_resnet50.pt"
    assert adopted.is_file() and adopted.read_bytes() == b"\2" * 2048
    assert orphan.is_symlink() and orphan.resolve() == adopted.resolve()


def test_adopt_does_not_clobber_a_differently_cased_registered_file(tmp_path):
    """macOS's default filesystem is case-insensitive, so `GANomaly.ckpt` and
    `Ganomaly.ckpt` are one path — but on this project's tree they are two
    different checkpoints. Adopting must not move one onto the other.
    """

    models_root = _models_root(tmp_path)
    registered = _weight(models_root, "Ganomaly.ckpt", size=4096)
    published = models_root / "published" / "GANomaly.ckpt"
    published.write_bytes(b"\3" * 8192)  # a different size => a different run

    plan = plan_publish_dedupe(tmp_path)

    (action,) = plan.actions
    assert action.kind == "adopt"
    apply_plan(plan, confirm=True)

    assert registered.read_bytes() == b"\0" * 4096, "the existing checkpoint survived untouched"
    assert published.is_symlink()
    assert published.resolve().read_bytes() == b"\3" * 8192, "the adopted weights survived too"


def test_already_symlinked_published_entries_are_left_alone(tmp_path):
    models_root = _models_root(tmp_path)
    registered = _weight(models_root, "Stfpm.ckpt")
    link = models_root / "published" / "STFPM.ckpt"
    link.symlink_to("../Stfpm.ckpt")

    assert plan_publish_dedupe(tmp_path).actions == []
    assert link.resolve() == registered.resolve()


# --------------------------------------------------------------------------- #
# Prune: the keep window
# --------------------------------------------------------------------------- #
def test_prune_keeps_the_newest_n_per_variant(tmp_path):
    models_root = _models_root(tmp_path)
    paths = [_weight(models_root, f"yolov8n_run{i}.pt") for i in range(5)]
    _manifest(models_root, [
        _record("ultralytics", "yolov8n", path, recorded_at=f"2026-07-{20 + i}T00:00:00Z")
        for i, path in enumerate(paths)
    ])

    plan = plan_checkpoint_prune(tmp_path, keep=2)

    doomed = {action.path for action in plan.actions}
    assert doomed == set(paths[:3])  # the three oldest
    assert paths[3] not in doomed and paths[4] not in doomed


def test_variants_get_independent_keep_windows(tmp_path):
    models_root = _models_root(tmp_path)
    old_yolo = _weight(models_root, "yolov8n_old.pt")
    new_yolo = _weight(models_root, "yolov8n_new.pt")
    lone_rcnn = _weight(models_root, "fasterrcnn_only.pt")
    _manifest(models_root, [
        _record("ultralytics", "yolov8n", old_yolo, recorded_at="2026-07-01T00:00:00Z"),
        _record("ultralytics", "yolov8n", new_yolo, recorded_at="2026-07-02T00:00:00Z"),
        _record("torchvision", "fasterrcnn_resnet50_fpn", lone_rcnn, recorded_at="2026-06-01T00:00:00Z"),
    ])

    plan = plan_checkpoint_prune(tmp_path, keep=1)

    # The oldest torchvision weight is also the only one, so it stays.
    assert {action.path for action in plan.actions} == {old_yolo}


def test_keep_must_be_at_least_one(tmp_path):
    _models_root(tmp_path)
    with pytest.raises(ValueError, match="keep must be at least 1"):
        plan_checkpoint_prune(tmp_path, keep=0)


# --------------------------------------------------------------------------- #
# Prune: what must never be deleted
# --------------------------------------------------------------------------- #
def test_a_published_symlink_target_is_never_pruned(tmp_path):
    """The hazard the copy->symlink switch introduces: deleting a registered
    checkpoint would leave a dangling published slot."""

    models_root = _models_root(tmp_path)
    old = _weight(models_root, "yolov8n_old.pt")
    new = _weight(models_root, "yolov8n_new.pt")
    (models_root / "published" / "yolov8n.pt").symlink_to("../yolov8n_old.pt")
    _manifest(models_root, [
        _record("ultralytics", "yolov8n", old, recorded_at="2026-07-01T00:00:00Z"),
        _record("ultralytics", "yolov8n", new, recorded_at="2026-07-02T00:00:00Z"),
    ])

    plan = plan_checkpoint_prune(tmp_path, keep=1)

    assert plan.actions == []
    assert old.resolve() in plan.protected
    assert "published" in plan.protected[old.resolve()]


def test_an_unfinished_batch_run_protects_its_finished_models(tmp_path):
    """`--resume` skips models already recorded `succeeded` and reads their
    results back, so their weights have to outlive the prune."""

    models_root = _models_root(tmp_path)
    done = _weight(models_root, "yolov8n_batch.pt")
    newer = _weight(models_root, "yolov8n_later.pt")
    _manifest(models_root, [
        _record("ultralytics", "yolov8n", done, recorded_at="2026-07-01T00:00:00Z", run_id="R1"),
        _record("ultralytics", "yolov8n", newer, recorded_at="2026-07-09T00:00:00Z"),
    ])
    run_dir = tmp_path / "artifacts" / "training_runs" / "R1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps({
        "run_id": "R1",
        "models": {"yolov8n": {"status": "succeeded"}, "yolo11n": {"status": "pending"}},
    }))

    plan = plan_checkpoint_prune(tmp_path, keep=1)

    assert plan.actions == []
    assert "still to finish" in plan.protected[done]


def test_a_completed_batch_run_stops_protecting_its_models(tmp_path):
    models_root = _models_root(tmp_path)
    done = _weight(models_root, "yolov8n_batch.pt")
    newer = _weight(models_root, "yolov8n_later.pt")
    _manifest(models_root, [
        _record("ultralytics", "yolov8n", done, recorded_at="2026-07-01T00:00:00Z", run_id="R1"),
        _record("ultralytics", "yolov8n", newer, recorded_at="2026-07-09T00:00:00Z"),
    ])
    run_dir = tmp_path / "artifacts" / "training_runs" / "R1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps({
        "run_id": "R1", "models": {"yolov8n": {"status": "succeeded"}},
    }))

    plan = plan_checkpoint_prune(tmp_path, keep=1)

    assert {action.path for action in plan.actions} == {done}


def test_an_unreadable_batch_state_protects_rather_than_assumes_finished(tmp_path):
    models_root = _models_root(tmp_path)
    weight = _weight(models_root, "yolov8n_batch.pt")
    newer = _weight(models_root, "yolov8n_later.pt")
    _manifest(models_root, [
        _record("ultralytics", "yolov8n", weight, recorded_at="2026-07-01T00:00:00Z", run_id="R1"),
        _record("ultralytics", "yolov8n", newer, recorded_at="2026-07-09T00:00:00Z"),
    ])
    run_dir = tmp_path / "artifacts" / "training_runs" / "R1"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{ truncated")

    assert plan_checkpoint_prune(tmp_path, keep=1).actions == []


def test_weights_with_no_manifest_entry_are_reported_not_deleted(tmp_path):
    """A hand-placed pretrained backbone has no provenance for this module to
    judge, so it is surfaced and left alone."""

    models_root = _models_root(tmp_path)
    stranger = _weight(models_root, "resnet50_imagenet.pth")
    _manifest(models_root, [])

    plan = plan_checkpoint_prune(tmp_path)

    assert plan.actions == []
    assert stranger in plan.unknown


def test_a_manifest_pointing_at_a_deleted_file_is_not_an_error(tmp_path):
    models_root = _models_root(tmp_path)
    _manifest(models_root, [
        _record("ultralytics", "yolov8n", models_root / "gone_a.pt", recorded_at="2026-07-01T00:00:00Z"),
        _record("ultralytics", "yolov8n", models_root / "gone_b.pt", recorded_at="2026-07-02T00:00:00Z"),
    ])

    assert plan_checkpoint_prune(tmp_path, keep=1).actions == []


# --------------------------------------------------------------------------- #
# Applying is opt-in
# --------------------------------------------------------------------------- #
def test_apply_refuses_without_confirmation(tmp_path):
    models_root = _models_root(tmp_path)
    doomed = _weight(models_root, "yolov8n_old.pt")
    keep = _weight(models_root, "yolov8n_new.pt")
    _manifest(models_root, [
        _record("ultralytics", "yolov8n", doomed, recorded_at="2026-07-01T00:00:00Z"),
        _record("ultralytics", "yolov8n", keep, recorded_at="2026-07-02T00:00:00Z"),
    ])
    plan = plan_checkpoint_prune(tmp_path, keep=1)

    with pytest.raises(ValueError, match="confirm=True"):
        apply_plan(plan)
    assert doomed.is_file(), "planning alone must never remove a checkpoint"


def test_render_summarises_without_touching_anything(tmp_path):
    models_root = _models_root(tmp_path)
    doomed = _weight(models_root, "yolov8n_old.pt", size=2048)
    keep = _weight(models_root, "yolov8n_new.pt")
    _manifest(models_root, [
        _record("ultralytics", "yolov8n", doomed, recorded_at="2026-07-01T00:00:00Z"),
        _record("ultralytics", "yolov8n", keep, recorded_at="2026-07-02T00:00:00Z"),
    ])

    text = plan_checkpoint_prune(tmp_path, keep=1).render()

    assert "delete" in text and "reclaimable" in text
    assert doomed.is_file()


def test_empty_plan_renders_and_reclaims_nothing():
    plan = Plan()
    assert plan.reclaimed_bytes == 0
    assert "nothing to do" in plan.render()


def test_protected_paths_survives_a_project_with_no_artifacts(tmp_path):
    assert protected_paths(tmp_path) == {}


# --------------------------------------------------------------------------- #
# Pre-training offer
# --------------------------------------------------------------------------- #
def test_pretrain_cleanup_is_scoped_to_the_model_about_to_train(tmp_path):
    models_root = _models_root(tmp_path)
    old_pc = _weight(models_root, "PatchCore_run1.ckpt")
    new_pc = _weight(models_root, "PatchCore_run2.ckpt")
    old_yolo = _weight(models_root, "yolov8n_run1.pt")
    new_yolo = _weight(models_root, "yolov8n_run2.pt")
    _manifest(models_root, [
        _record("anomalib", "PatchCore", old_pc, recorded_at="2026-07-01T00:00:00Z"),
        _record("anomalib", "PatchCore", new_pc, recorded_at="2026-07-02T00:00:00Z"),
        _record("ultralytics", "yolov8n", old_yolo, recorded_at="2026-07-01T00:00:00Z"),
        _record("ultralytics", "yolov8n", new_yolo, recorded_at="2026-07-02T00:00:00Z"),
    ])

    plan = plan_pretrain_cleanup(tmp_path, "anomalib", "PatchCore")

    # Starting a PatchCore run must never propose touching the YOLO weights.
    assert {action.path for action in plan.actions} == {old_pc}


def test_pretrain_cleanup_keeps_one_predecessor_by_default(tmp_path):
    models_root = _models_root(tmp_path)
    paths = [_weight(models_root, f"PatchCore_run{i}.ckpt") for i in range(3)]
    _manifest(models_root, [
        _record("anomalib", "PatchCore", path, recorded_at=f"2026-07-0{i + 1}T00:00:00Z")
        for i, path in enumerate(paths)
    ])

    plan = plan_pretrain_cleanup(tmp_path, "anomalib", "PatchCore")

    assert {action.path for action in plan.actions} == {paths[0], paths[1]}
    assert paths[2].is_file(), "the newest predecessor stays as a fallback"


def test_pretrain_cleanup_still_respects_published_links(tmp_path):
    models_root = _models_root(tmp_path)
    old = _weight(models_root, "PatchCore_run1.ckpt")
    new = _weight(models_root, "PatchCore_run2.ckpt")
    (models_root / "published" / "PatchCore.ckpt").symlink_to("../PatchCore_run1.ckpt")
    _manifest(models_root, [
        _record("anomalib", "PatchCore", old, recorded_at="2026-07-01T00:00:00Z"),
        _record("anomalib", "PatchCore", new, recorded_at="2026-07-02T00:00:00Z"),
    ])

    plan = plan_pretrain_cleanup(tmp_path, "anomalib", "PatchCore")

    assert plan.actions == []
    assert old.resolve() in plan.protected


def test_pretrain_cleanup_never_prompts(tmp_path, monkeypatch):
    """The library must be usable from a headless sweep: no stdin, no input()."""

    models_root = _models_root(tmp_path)
    _manifest(models_root, [])
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("retention must not prompt"))

    assert plan_pretrain_cleanup(tmp_path, "anomalib", "PatchCore").actions == []
