"""Model-upgrade comparison: how much a checkpoint swap moves the decision
threshold, and how many samples flip their pass/fail call as a result --
neither number is visible from each version's own metrics in isolation,
since two decent-looking threshold/accuracy pairs can still disagree on
which specific samples they flag (e.g. one recovers a few previously-missed
small defects but starts missing others). Deliberately plain functions
over already-computed thresholds/decisions rather than a new `Evaluator`:
"old" and "new" here are two runs of an *existing* evaluator (e.g.
`IndustrialEvaluator`'s recall-first threshold, or `AnomalyEvaluator`'s
F1-optimal one), compared after the fact.
"""

from __future__ import annotations

from typing import Sequence


def threshold_drift(tau_old: float, tau_new: float) -> float:
    """Delta_tau = |tau_new - tau_old|."""

    return abs(tau_new - tau_old)


def upgrade_inconsistency_rate(decisions_old: Sequence[bool], decisions_new: Sequence[bool]) -> float:
    """Fraction of samples whose binary decision (e.g. flagged-as-defective
    at each version's own chosen threshold) flips between the old and new
    model version. Both sequences must be aligned to the same sample order.
    """

    if len(decisions_old) != len(decisions_new):
        raise ValueError("decisions_old and decisions_new must be the same length.")
    if not decisions_old:
        return 0.0
    mismatches = sum(1 for old, new in zip(decisions_old, decisions_new) if bool(old) != bool(new))
    return mismatches / len(decisions_old)
