"""Disk hygiene for `artifacts/models/`: deduplicating published copies and
pruning superseded checkpoints, as a plan you inspect before anything is
touched.

Two operations live here because they share one hard-won piece of knowledge —
*which files must never be deleted* — and splitting them would mean writing
that set twice:

`plan_publish_dedupe`
    Converts the pre-symlink world (every published model existed twice: once
    under `artifacts/models/`, once as a byte-identical copy in
    `published/`) into the symlink layout `catalog.publish_artifact` now
    writes. A published file with no counterpart under `artifacts/models/` is
    *adopted* — moved down into the model root and replaced by a link —
    rather than deleted, since it is the only copy of those weights.

`plan_checkpoint_prune`
    Proposes deleting superseded registered checkpoints, keeping the newest
    `keep` per (backend, variant) plus everything in the protected set.

Nothing here deletes on its own. Both return a `Plan`; `apply_plan` is a
separate call that requires `confirm=True`. That split is the whole point —
these weights are the output of GPU-hours that cannot be re-run cheaply, so
"show me what you would do" has to be the default and the destructive step
has to be typed out explicitly.

Resume safety
-------------
Pruning `artifacts/models/` cannot break either kind of resume, and the
reason is a layout fact worth stating rather than rediscovering:

* **Within-model resume** (`torchvision`'s `train.resume`) reads `last.pt`
  from `CheckpointSpec.run_dir`, which defaults to `runs/fabric_defect_hub_tv`
  — a transient *training* directory outside `artifacts/`. This module never
  looks there.
* **Batch resume** (`fdh train-all --resume`) skips models already recorded
  `succeeded` in a run's `state.json`, and the final report reads their
  metrics back. Deleting a succeeded model's weights would leave a run that
  claims a result it can no longer produce, so every artifact belonging to a
  batch run with unfinished work is protected until that run completes.

The genuinely new hazard is not resume at all: now that `published/` holds
symlinks instead of copies, deleting a registered checkpoint breaks the
published slot that points at it. Symlink targets are therefore protected
unconditionally, ahead of any keep-window arithmetic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fabric_defect_hub.weight_registry import read_weight_manifest

ActionKind = Literal["dedupe", "adopt", "delete"]

DEFAULT_KEEP_PER_VARIANT = 3


@dataclass(frozen=True)
class Action:
    """One proposed change, with the bytes it frees and why it is proposed."""

    kind: ActionKind
    path: Path
    reclaimed_bytes: int
    reason: str
    # `dedupe`/`adopt` only: what `path` will point at once applied.
    links_to: Path | None = None


@dataclass
class Plan:
    """Proposed actions plus the files deliberately left alone.

    `protected` and `unknown` are part of the output rather than an internal
    detail: the question a reader actually has when handed a deletion plan is
    "what did it decide *not* to touch, and why", and answering that is what
    makes the plan reviewable.
    """

    actions: list[Action] = field(default_factory=list)
    protected: dict[Path, str] = field(default_factory=dict)
    unknown: list[Path] = field(default_factory=list)

    @property
    def reclaimed_bytes(self) -> int:
        return sum(action.reclaimed_bytes for action in self.actions)

    def render(self) -> str:
        """Human-readable summary — what `tools/prune_artifacts.py` prints."""

        lines: list[str] = []
        for action in self.actions:
            size = _human_bytes(action.reclaimed_bytes)
            arrow = f" -> {action.links_to}" if action.links_to else ""
            lines.append(f"  [{action.kind}] {action.path}{arrow}  ({size})  {action.reason}")
        if not lines:
            lines.append("  (nothing to do)")
        lines.append("")
        lines.append(f"  reclaimable: {_human_bytes(self.reclaimed_bytes)}")
        if self.protected:
            lines.append(f"  protected:   {len(self.protected)} file(s)")
        if self.unknown:
            lines.append(
                f"  unrecorded:  {len(self.unknown)} file(s) with no manifest entry, left untouched"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The protected set — computed once, shared by both plans
# --------------------------------------------------------------------------- #
def published_link_targets(project_root: str | Path) -> dict[Path, str]:
    """Every file a `published/` symlink resolves to.

    Deleting one of these would leave a dangling published slot, which is the
    one regression the copy->symlink switch could introduce.
    """

    published = Path(project_root) / "artifacts" / "models" / "published"
    targets: dict[Path, str] = {}
    if not published.is_dir():
        return targets
    for entry in sorted(published.iterdir()):
        if entry.is_symlink():
            resolved = entry.resolve()
            targets[resolved] = f"published as {entry.name}"
    return targets


def active_batch_artifacts(project_root: str | Path, run_root: str | Path | None = None) -> dict[Path, str]:
    """Artifacts belonging to batch runs that still have unfinished work.

    A run with any model not yet `succeeded` is resumable, and `--resume`
    reads the succeeded models' results back rather than retraining them, so
    their weights have to outlive the prune.
    """

    root = Path(run_root) if run_root else Path(project_root) / "artifacts" / "training_runs"
    protected: dict[Path, str] = {}
    if not root.is_dir():
        return protected

    records = read_weight_manifest(project_root)
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        state_path = run_dir / "state.json"
        if not state_path.is_file():
            continue
        try:
            state = _read_json(state_path)
        except ValueError:
            # A half-written state file is a reason to protect the run, not
            # to assume it finished.
            protected.update(_artifacts_for_run(records, run_dir.name, "unreadable batch state"))
            continue
        models = state.get("models", {})
        if not isinstance(models, dict):
            continue
        unfinished = [
            key for key, entry in models.items()
            if isinstance(entry, dict) and entry.get("status") != "succeeded"
        ]
        if unfinished:
            reason = f"batch run {run_dir.name} has {len(unfinished)} model(s) still to finish"
            protected.update(_artifacts_for_run(records, state.get("run_id", run_dir.name), reason))
    return protected


def _artifacts_for_run(records: list[dict[str, Any]], run_id: str, reason: str) -> dict[Path, str]:
    found: dict[Path, str] = {}
    for record in records:
        if record.get("batch_run_id") != run_id:
            continue
        registered = (record.get("artifact") or {}).get("registered_path")
        if registered:
            found[Path(registered)] = reason
    return found


def protected_paths(
    project_root: str | Path, run_root: str | Path | None = None
) -> dict[Path, str]:
    """Union of every "must not delete" rule, path -> reason."""

    protected = dict(published_link_targets(project_root))
    for path, reason in active_batch_artifacts(project_root, run_root).items():
        protected.setdefault(path, reason)
    return protected


# --------------------------------------------------------------------------- #
# Plan 1: published copies -> symlinks
# --------------------------------------------------------------------------- #
def plan_publish_dedupe(project_root: str | Path) -> Plan:
    """Turn every real file under `published/` into a symlink.

    Two cases, distinguished by whether the bytes exist elsewhere:

    * a counterpart of the same size already sits under `artifacts/models/`
      -> `dedupe`: drop the published copy, link to the counterpart.
    * no counterpart -> `adopt`: move the file down into
      `artifacts/models/` (keeping its published name as the archival name)
      and link to it, so the model root really does hold every trained file.
    """

    root = Path(project_root)
    models_root = root / "artifacts" / "models"
    published = models_root / "published"
    plan = Plan()
    if not published.is_dir():
        return plan

    candidates = _registered_files(models_root)
    for entry in sorted(published.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            continue
        size = entry.stat().st_size
        twin = next(
            (path for path in candidates if path.stat().st_size == size and path.name == entry.name),
            None,
        ) or next((path for path in candidates if path.stat().st_size == size), None)

        if twin is not None:
            plan.actions.append(Action(
                kind="dedupe", path=entry, reclaimed_bytes=size, links_to=twin,
                reason=f"identical {_human_bytes(size)} copy already registered",
            ))
        else:
            destination = _archival_destination(models_root, entry)
            note = "" if destination.name == entry.name else f" as {destination.name}"
            plan.actions.append(Action(
                kind="adopt", path=entry, reclaimed_bytes=0, links_to=destination,
                reason=f"only copy of these weights; move into the model root{note} and link",
            ))
    return plan


def _archival_destination(models_root: Path, published_file: Path) -> Path:
    """A free name under `artifacts/models/` for an adopted published file.

    Naively reusing the published name loses data on a case-insensitive
    filesystem, which is the default on macOS: `published/GANomaly.ckpt` and
    the already-registered `Ganomaly.ckpt` are the *same path* to APFS, yet
    on this project's own tree they are two different checkpoints (their
    sizes differ by 3968 bytes). Moving one onto the other would silently
    destroy a trained model, so a collision earns a disambiguated name here
    rather than an error at apply time.
    """

    candidate = models_root / published_file.name
    if not candidate.exists():
        return candidate
    stem, suffix = published_file.stem, published_file.suffix
    for attempt in ("published", *(f"published-{n}" for n in range(2, 100))):
        candidate = models_root / f"{stem}.{attempt}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no free archival name for {published_file}")


# --------------------------------------------------------------------------- #
# Plan 2: prune superseded checkpoints
# --------------------------------------------------------------------------- #
def plan_checkpoint_prune(
    project_root: str | Path,
    *,
    keep: int = DEFAULT_KEEP_PER_VARIANT,
    run_root: str | Path | None = None,
) -> Plan:
    """Propose deleting registered checkpoints beyond the newest `keep` for
    each (backend, variant).

    Only files the manifest knows about are ever proposed for deletion. A
    weight file with no manifest entry is reported under `Plan.unknown` and
    left alone — it may be a hand-placed pretrained backbone, and this module
    has no provenance with which to judge it.
    """

    if keep < 1:
        raise ValueError(f"keep must be at least 1, got {keep}")

    root = Path(project_root)
    models_root = root / "artifacts" / "models"
    plan = Plan(protected=protected_paths(root, run_root))
    records = read_weight_manifest(root)

    recorded: dict[Path, list[dict[str, Any]]] = {}
    by_variant: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for record in records:
        registered = (record.get("artifact") or {}).get("registered_path")
        if not registered:
            continue
        path = Path(registered)
        recorded.setdefault(path, []).append(record)
        key = (record.get("backend") or "?", record.get("variant") or "?")
        by_variant.setdefault(key, []).append((record.get("recorded_at") or "", path))

    superseded: dict[Path, str] = {}
    for (backend, variant), entries in by_variant.items():
        # Newest first; the same path can appear more than once (a re-run that
        # overwrote its own registered name), so collapse before counting.
        ordered: list[Path] = []
        for _, path in sorted(entries, key=lambda item: item[0], reverse=True):
            if path not in ordered:
                ordered.append(path)
        for path in ordered[keep:]:
            superseded[path] = f"superseded: not among the newest {keep} for {backend}/{variant}"

    for path, reason in sorted(superseded.items()):
        if path in plan.protected:
            continue
        if not path.is_file() or path.is_symlink():
            continue
        plan.actions.append(Action(
            kind="delete", path=path, reclaimed_bytes=path.stat().st_size, reason=reason,
        ))

    for path in _registered_files(models_root):
        if path not in recorded and path not in plan.protected:
            plan.unknown.append(path)
    return plan


def plan_pretrain_cleanup(
    project_root: str | Path,
    backend: str,
    variant: str | None,
    *,
    keep: int = 1,
    run_root: str | Path | None = None,
) -> Plan:
    """What could be freed for `(backend, variant)` before starting a new run.

    Meant to be called *before* training so the answer can be offered to
    whoever started it — "you have 4 previous PatchCore checkpoints (6.1 GB);
    clear them?" — with the same protection rules a manual prune obeys.

    This module deliberately does not prompt. Asking is the caller's job,
    because the sensible way to ask differs per front end (a terminal
    question from the CLI, a checkbox in the Gradio app, a config flag in an
    automated sweep) and a library that reads stdin cannot be used by two of
    those three. The contract is: plan here, ask there, then `apply_plan`.

        plan = plan_pretrain_cleanup(root, "anomalib", "PatchCore")
        if plan.actions and ui_confirms(plan.render()):
            apply_plan(plan, confirm=True)

    `keep` defaults to 1 rather than 3: the caller is about to produce a
    fresh checkpoint, so keeping the single most recent predecessor is enough
    to fall back on if the new run turns out worse.
    """

    full = plan_checkpoint_prune(project_root, keep=keep, run_root=run_root)
    records = read_weight_manifest(project_root)
    mine = {
        Path((record.get("artifact") or {}).get("registered_path") or "")
        for record in records
        if record.get("backend") == backend
        and (variant is None or record.get("variant") == variant)
    }
    return Plan(
        actions=[action for action in full.actions if action.path in mine],
        protected={path: why for path, why in full.protected.items() if path in mine},
        unknown=[],
    )


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #
def apply_plan(plan: Plan, *, confirm: bool = False) -> list[Action]:
    """Execute `plan`. Requires `confirm=True`; returns the actions applied.

    The keyword is not ceremony: every caller of this module defaults to
    planning, so the only way bytes are ever removed is a caller that spelled
    out `confirm=True` at the call site.
    """

    if not confirm:
        raise ValueError(
            "apply_plan requires confirm=True; call it only after showing Plan.render() "
            "to whoever owns these weights"
        )

    applied: list[Action] = []
    for action in plan.actions:
        if action.kind == "delete":
            action.path.unlink()
        elif action.kind == "dedupe":
            assert action.links_to is not None
            action.path.unlink()
            _symlink_relative(action.links_to, action.path)
        elif action.kind == "adopt":
            assert action.links_to is not None
            destination = action.links_to
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"cannot adopt {action.path}: {destination} already exists")
            action.path.replace(destination)
            _symlink_relative(destination, action.path)
        applied.append(action)
    return applied


def _symlink_relative(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source, destination.parent))


def _registered_files(models_root: Path) -> list[Path]:
    """Weight files directly under `artifacts/models/` (not `published/`,
    `records/`, or `quantized/`, which have their own lifecycles)."""

    if not models_root.is_dir():
        return []
    suffixes = {".pt", ".pth", ".ckpt", ".onnx", ".engine"}
    return sorted(
        path for path in models_root.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in suffixes
    )


def _read_json(path: Path) -> dict[str, Any]:
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc


def _human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"
