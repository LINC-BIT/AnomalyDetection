"""Cross-domain accuracy degradation metrics and bootstrap confidence intervals.

Two granularities, deliberately separate:

* **Dataset level** — `cross_domain_degradation` compares one source
  accuracy against one target accuracy. `web/benchmark.py` drives this with
  a whole second dataset (e.g. trained on ZJU-Leaper, evaluated on
  MVTec-AD).
* **Pattern level** — `pattern_sweep_degradation` holds the dataset fixed
  and varies the *texture pattern*: train on a few patterns, evaluate on
  every remaining one, then aggregate the per-pattern degradations. This is
  the protocol a textile benchmark actually needs, because "a fabric the
  model never saw" is a different question from "a different dataset
  entirely", and ZJU-Leaper ships 19 patterns precisely so it can be asked.

The sweep takes the per-pattern evaluation as a callback rather than
loading datasets itself — same injection style as
`profiling.scaling.max_concurrent_streams` — so this module stays free of
dataset and model imports and the aggregation stays unit-testable without
staged data on disk.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import statistics

from fabric_defect_hub.stats import bootstrap_ci, bootstrap_group_ci

# How `top_k` picks which held-out patterns represent the result.
#
# "worst" (the default) takes the k *largest* degradations: a robustness
# claim, i.e. "even on the fabrics it handles least well, the model loses at
# most this much". "best" takes the k smallest, which characterises the
# favourable end instead. The distinction is not cosmetic -- the two report
# opposite things about the same run -- so it is an explicit argument with
# no silent default beyond the conservative one, and the chosen mode is
# echoed back in the result dict so a table can state which was used.
TOPK_MODES = ("worst", "best")


def cross_domain_degradation(acc_src: float, acc_tgt: float) -> float:
    """Compute relative accuracy degradation percentage: (acc_src - acc_tgt) / acc_src * 100."""

    if acc_src == 0:
        raise ValueError("acc_src must be nonzero to compute a relative degradation.")
    return (acc_src - acc_tgt) / acc_src * 100.0


def cross_domain_degradation_ci(
    src_correct: Sequence[bool],
    tgt_correct: Sequence[bool],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for cross-domain accuracy degradation."""

    def _combine(acc_src: float, acc_tgt: float) -> float:
        return cross_domain_degradation(acc_src, acc_tgt)

    result = bootstrap_group_ci(
        list(src_correct), list(tgt_correct), statistics.fmean, _combine, n_resamples, confidence, seed
    )
    return {"delta_acc_pct": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]}


def top_k_mean(degradations: Mapping[str, float], k: int, mode: str = "worst") -> dict[str, object]:
    """Mean of the `k` most (or least) degraded patterns, plus which ones
    were picked.

    Reporting the mean over *all* held-out patterns hides the shape of the
    failure: a model that is fine on fifteen fabrics and collapses on one
    averages out to "acceptable", which is exactly the case a textile line
    cares about. Taking the k worst keeps that visible while still not
    resting the whole claim on a single pattern.

    `k` is clamped to the number of patterns available rather than raising,
    so a sweep that could only stage four patterns still reports a number
    (with `k_effective` saying what was actually used).
    """

    if mode not in TOPK_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {TOPK_MODES}.")
    if not degradations:
        raise ValueError("top_k_mean needs at least one pattern degradation.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    ordered = sorted(degradations.items(), key=lambda kv: kv[1], reverse=(mode == "worst"))
    k_effective = min(k, len(ordered))
    picked = ordered[:k_effective]
    return {
        "top_k_mean_degradation_pct": statistics.fmean(value for _, value in picked),
        "selected_patterns": [name for name, _ in picked],
        "k_effective": k_effective,
        "mode": mode,
    }


def pattern_sweep_degradation(
    acc_src: float,
    target_patterns: Sequence[str],
    evaluate_pattern: Callable[[str], float | None],
    k: int = 3,
    mode: str = "worst",
    with_ci: bool = True,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, object]:
    """Evaluate one trained model across every held-out pattern and reduce
    the result to a single reportable degradation.

    `acc_src` is the model's accuracy on the patterns it was trained on.
    `evaluate_pattern(pattern)` returns that model's accuracy on one
    held-out pattern, or `None` when the pattern cannot be scored on this
    machine (not staged, no defective samples, wrong task) -- skipped
    patterns are listed in the result rather than counted as a degradation
    of zero, which would silently flatter the model.

    The bootstrap resamples the *patterns*, not the images: the question a
    reader has is "would this number hold up on a different set of
    fabrics", and the pattern list is the sample from that population. It
    needs at least two scored patterns, so `ci_low`/`ci_high` are absent
    when fewer survive.
    """

    if acc_src == 0:
        raise ValueError("acc_src must be nonzero to compute a relative degradation.")

    per_pattern: dict[str, float] = {}
    skipped: list[str] = []
    for pattern in target_patterns:
        accuracy = evaluate_pattern(pattern)
        if accuracy is None:
            skipped.append(pattern)
            continue
        per_pattern[pattern] = cross_domain_degradation(acc_src, accuracy)

    if not per_pattern:
        return {
            "acc_src": acc_src,
            "per_pattern_degradation_pct": {},
            "skipped_patterns": skipped,
            "scored_pattern_count": 0,
        }

    summary = top_k_mean(per_pattern, k, mode)
    result: dict[str, object] = {
        "acc_src": acc_src,
        "per_pattern_degradation_pct": per_pattern,
        "skipped_patterns": skipped,
        "scored_pattern_count": len(per_pattern),
        "mean_degradation_pct": statistics.fmean(per_pattern.values()),
        **summary,
    }

    if with_ci and len(per_pattern) >= 2:
        values = list(per_pattern.values())

        def _top_k_of(sample: Sequence[float]) -> float:
            ordered = sorted(sample, reverse=(mode == "worst"))
            return statistics.fmean(ordered[: min(k, len(ordered))])

        interval = bootstrap_ci(values, _top_k_of, n_resamples, confidence, seed)
        result["ci_low"] = interval["ci_low"]
        result["ci_high"] = interval["ci_high"]

    return result


# Which metric represents "accuracy" for a task, most specific first. A
# degradation is only meaningful between two runs scored on the *same* key,
# so the choice is made once from the source run and reused for every
# held-out pattern rather than re-picked per run (where a pattern that
# happened to produce no pixel maps would silently switch the comparison to
# image level mid-sweep).
#
# Detection lists both this project's normalised key (`map50`, the
# Ultralytics adapter) and torchmetrics' raw keys (`map_50`/`map`,
# `evaluation/detection.py`), since either dict may arrive here. Anomaly
# falls through from pixel to image level on its own: a model that fills no
# anomaly map simply has no `pixel_auroc` in its metrics.
HEADLINE_PRIORITY: dict[str, tuple[str, ...]] = {
    "detection": ("map50", "map_50", "map"),
    "instance_segmentation": ("map50", "map_50", "map", "miou"),
    "segmentation": ("miou", "dice"),
    "anomaly": ("pixel_auroc", "image_auroc"),
}


def resolve_headline_metric(
    task: str, metrics: Mapping[str, float], preferred: str | None = None
) -> str:
    """The metric key a cross-domain comparison should be computed over.

    `preferred` wins when given and present, so a caller reporting a
    non-default number (e.g. `image_auroc` for a model that does produce
    pixel maps) can say so explicitly instead of being overridden by the
    priority table.
    """

    if preferred:
        if preferred not in metrics:
            raise KeyError(f"requested metric {preferred!r} not in {sorted(metrics)}")
        return preferred
    priority = HEADLINE_PRIORITY.get(task)
    if priority is None:
        raise ValueError(f"no headline metric defined for task {task!r}")
    for key in priority:
        if key in metrics:
            return key
    raise KeyError(
        f"none of {priority} present in metrics {sorted(metrics)} — "
        "was the evaluation run with the matching task evaluator?"
    )
