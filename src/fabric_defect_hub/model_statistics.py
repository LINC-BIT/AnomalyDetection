"""Lightweight model-size helpers shared by training backends."""

from __future__ import annotations

from typing import Any


def parameter_counts(model: Any) -> dict[str, int]:
    """Return total and trainable parameter counts without importing torch.

    Backends call this while their native model is still materialized after
    training. Keeping it duck-typed makes the bookkeeping usable by every
    optional backend and avoids reopening checkpoint files just to count
    tensors.
    """

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return {}
    try:
        total = 0
        trainable = 0
        for parameter in parameters():
            count = int(parameter.numel())
            total += count
            if bool(getattr(parameter, "requires_grad", False)):
                trainable += count
    except (AttributeError, RuntimeError, TypeError):
        return {}
    return {"parameter_count": total, "trainable_parameter_count": trainable}
