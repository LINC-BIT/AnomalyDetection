"""Serializable inventory shared verbatim by SDK, CLI and web UI."""

from __future__ import annotations

from typing import Any

from fabric_defect_hub.catalog import CANONICAL_MODELS, published_path, published_status
from fabric_defect_hub.core.dataset_capabilities import all_capabilities


def model_inventory(*, published_only: bool = False) -> list[dict[str, Any]]:
    rows = []
    for model in CANONICAL_MODELS:
        weight = published_path(model)
        status = published_status(weight)
        if published_only and status not in {"file", "symlink"}:
            continue
        rows.append({
            "id": model.key,
            "label": model.label,
            "backend": model.backend,
            "variant": model.variant,
            "task": model.task,
            "category": model.category,
            "subtype": model.subtype,
            "integration": model.integration,
            "training_mode": model.training_mode,
            "method": model.method,
            "trained_on": list(model.trained_on),
            "training_split": model.training_split,
            "evaluated_on": list(model.evaluated_on),
            "domain": model.domain,
            "config": model.config,
            "weight": str(weight),
            "weight_status": status,
            "source": model.source,
            "legacy_weight_url": model.legacy_weight_url,
        })
    return rows


def dataset_inventory() -> list[dict[str, Any]]:
    return [
        {
            "id": name,
            "default_root": caps.default_root,
            "roles": sorted(caps.roles),
            "tasks": list(caps.tasks),
        }
        for name, caps in sorted(all_capabilities().items())
    ]


def platform_inventory() -> dict[str, Any]:
    return {"models": model_inventory(), "datasets": dataset_inventory()}
