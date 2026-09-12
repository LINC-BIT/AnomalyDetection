"""The dataset-selection decision tree.

Given a backend's allowed set of training datasets (already role-filtered by
the caller -- e.g. `training.ANOMALY_TRAINABLE_DATASETS`) and what's actually
staged on this machine right now (`core.availability`), decide which
dataset to train on instead of hard-failing the moment a config names a
dataset this machine doesn't happen to have staged.

The policy is deliberately simple and the same for every backend, so "why
did it pick that one" is always answerable without special-casing:

  1. the dataset the config asked for, if it is staged here -> use it.
  2. otherwise, the alphabetically-first *staged* dataset in the same
     allowed set -> use it, and say so loudly (this is a substitution, not
     silent). Alphabetical, not a hand-tuned "best dataset" ranking -- this
     project doesn't have benchmark evidence that one in-domain fabric
     source trains a better model than another, so it does not pretend to.
  3. otherwise -> not runnable; the reason names exactly what to stage.

See `docs/EXTENDING.md` for how a new dataset or backend plugs into this.
"""

from __future__ import annotations

from dataclasses import dataclass

from fabric_defect_hub.core.availability import staged_datasets


@dataclass(frozen=True)
class DatasetDecision:
    """The outcome of `decide_dataset`, always carrying a human-readable
    `reason` so a caller (or `fdh doctor`) can explain itself without
    re-deriving the logic."""

    runnable: bool
    dataset: str | None  # the dataset to actually use; None if not runnable
    requested: str | None  # what was originally asked for (may be None)
    substituted: bool  # True iff `dataset` differs from `requested`
    reason: str


def decide_dataset(
    requested: str | None,
    allowed_datasets: set[str] | frozenset[str],
    root_map: dict[str, str] | None = None,
) -> DatasetDecision:
    """Decide which of `allowed_datasets` to train on.

    `requested`, if given, is assumed to already be a *valid* member of
    `allowed_datasets` (role-legality is the caller's concern, e.g.
    `training._enforce_trainable_dataset` running first) -- this function
    only adds the "...and is it actually staged here" axis.
    """

    available = staged_datasets(allowed_datasets, root_map=root_map)

    if requested and requested in available:
        return DatasetDecision(
            runnable=True,
            dataset=requested,
            requested=requested,
            substituted=False,
            reason=f"{requested!r} is staged on this machine and matches what was requested.",
        )

    if available:
        chosen = sorted(available)[0]
        others = ", ".join(sorted(available - {chosen})) or "none"
        if requested:
            reason = (
                f"{requested!r} was requested but is not staged on this machine; "
                f"substituted {chosen!r} (staged, same allowed dataset set). "
                f"Other staged alternatives: {others}."
            )
        else:
            reason = f"No dataset was requested; picked {chosen!r} (staged). Other staged alternatives: {others}."
        return DatasetDecision(runnable=True, dataset=chosen, requested=requested, substituted=True, reason=reason)

    return DatasetDecision(
        runnable=False,
        dataset=None,
        requested=requested,
        substituted=False,
        reason=(
            f"None of the allowed datasets ({', '.join(sorted(allowed_datasets)) or 'none declared'}) "
            "are staged on this machine. Stage one under data/<Dataset> (see "
            "training.DEFAULT_DATASET_ROOTS) to make this trainable here."
        ),
    )
