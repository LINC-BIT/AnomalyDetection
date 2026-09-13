"""Single source of truth for what each registered dataset may be used for.

Before this module, "what can dataset X train/evaluate?" was answered by
several independently-maintained structures scattered across the project --
`training.ANOMALY_TRAINABLE_DATASETS`, `training.ZERO_SHOT_TRAINABLE_DATASETS`,
`training.DEFAULT_DATASET_ROOTS`, `datasets/fabric_train.py`'s hardcoded
`_MEMBERS` tuple, and `web/single_image.py`'s `DATASET_CATALOG`. Adding a
dataset meant remembering to touch all of them (and nothing enforced that
you had); getting one wrong silently mis-scoped what a model could train on.

This module replaces the *decision* structures (not the on-disk parsing
metadata `DATASET_CATALOG` still owns, like env var names and UI slice
dropdowns) with one declaration per dataset: `register_capabilities(name,
default_root=..., roles={...}, tasks=(...))`. `training.py` and
`datasets/fabric_train.py` now *derive* their sets from this registry
instead of hand-maintaining parallel copies, so declaring a new dataset's
capabilities here is enough to make it a training-eligible source, a
`fabric-train` member, or both.

Roles (a dataset may hold more than one):
- "anomaly_train"       -- normal-only training source for one-class anomaly backends
                           (Anomalib/Dinomaly/MambaAD) and MoECLIP's
                           evaluation target.
- "zero_shot_train"     -- labelled auxiliary corpus on which MoECLIP may
                           train. A general benchmark can hold both roles.
- "detection_train"      -- bbox-labelled, a legitimate source for the
                           Ultralytics/torchvision detection backends.
- "fabric_train_member"  -- contributes samples to the `fabric-train`
                           composite (datasets/fabric_train.py); implies the
                           dataset exposes normal/defect images the same
                           way every other fabric adapter does.

A dataset with none of these roles (e.g. a composite like `fabric-train`
itself, which doesn't nest into its own union) is eval-only / non-training
by omission -- there is no separate "eval_only" tag, it's just the absence
of "anomaly_train"/"zero_shot_train"/"detection_train".
"""

from __future__ import annotations

from dataclasses import dataclass, field

Role = str  # kept loose (not a Literal) so new roles don't require a type-checker update here.


@dataclass(frozen=True)
class DatasetCapabilities:
    """What a registered dataset may be used for, and where it lives by default."""

    default_root: str | None = None
    roles: frozenset[Role] = field(default_factory=frozenset)
    tasks: tuple[str, ...] = ()
    domain: str = "general"

    def supports(self, role: Role) -> bool:
        return role in self.roles


_CAPABILITIES: dict[str, DatasetCapabilities] = {}


def register_capabilities(
    name: str,
    *,
    default_root: str | None = None,
    roles: frozenset[Role] | set[Role] = frozenset(),
    tasks: tuple[str, ...] = (),
    domain: str = "general",
) -> None:
    if name in _CAPABILITIES:
        raise ValueError(f"capabilities for dataset '{name}' are already registered")
    _CAPABILITIES[name] = DatasetCapabilities(
        default_root=default_root, roles=frozenset(roles), tasks=tasks, domain=domain
    )


def capabilities_for(name: str) -> DatasetCapabilities:
    """Capabilities for `name`, or an empty (no roles, no default root)
    placeholder for a dataset that hasn't declared any -- e.g. a
    hand-staged YOLO folder like SDUST-FDD, which is used via `data_yaml`
    rather than a registered `DatasetAdapter` and so has nothing to declare
    here."""

    return _CAPABILITIES.get(name, DatasetCapabilities())


def names_with_role(role: Role) -> set[str]:
    return {name for name, caps in _CAPABILITIES.items() if role in caps.roles}


def default_dataset_roots() -> dict[str, str]:
    return {name: caps.default_root for name, caps in _CAPABILITIES.items() if caps.default_root}


def all_capabilities() -> dict[str, DatasetCapabilities]:
    return dict(_CAPABILITIES)


# --------------------------------------------------------------------------
# Declarations. Keep in step with `datasets/__init__.py`'s registered
# adapters -- a dataset with no entry here still loads fine (`load_dataset`
# only needs `@register_dataset`), it just won't be selectable as a training
# source, a `fabric-train` member, or get a default root fallback.
# --------------------------------------------------------------------------

register_capabilities(
    "zju-leaper",
    default_root="datasets/textile/ZJU-Leaper",
    roles={"anomaly_train", "detection_train", "fabric_train_member"},
    tasks=("detection", "segmentation", "anomaly"),
    domain="textile",
)
register_capabilities(
    "raw-fabric",
    default_root="datasets/textile/RAW_FABRID",
    roles={"anomaly_train", "fabric_train_member"},
    tasks=("anomaly", "segmentation"),
    domain="textile",
)
register_capabilities(
    "tilda-400",
    default_root="datasets/textile/TILDA_400",
    roles={"anomaly_train", "fabric_train_member"},
    tasks=("anomaly",),
    domain="textile",
)
register_capabilities(
    "fabric-defects",
    default_root="datasets/textile/Fabric Defects Dataset",
    roles={"anomaly_train", "fabric_train_member"},
    tasks=("anomaly", "segmentation"),
    domain="textile",
)
register_capabilities(
    # Tianchi Guangdong fabric defect challenge: native bbox annotations
    # (feeds detection training directly) *and* a real normal_Images pool
    # per part, so it both stands alone as an anomaly training source and
    # contributes its good samples to the `fabric-train` composite -- see
    # `datasets/tianchi.py`'s module docstring.
    "tianchi",
    default_root="datasets/textile/tianchi",
    roles={"anomaly_train", "detection_train", "fabric_train_member"},
    tasks=("detection", "anomaly"),
    domain="textile",
)
register_capabilities(
    # The composite itself: not a member of its own union, but otherwise a
    # normal anomaly-training corpus (see datasets/fabric_train.py).
    "fabric-train",
    default_root="datasets/textile",
    roles={"anomaly_train"},
    tasks=("anomaly", "segmentation", "detection"),
    domain="textile",
)
register_capabilities(
    "mvtec-ad",
    default_root="datasets/general/MVTec AD",
    roles={"anomaly_train", "zero_shot_train"},
    tasks=("anomaly", "segmentation"),
)
register_capabilities(
    "mvtec-loco",
    default_root="datasets/general/MVTec LOCO",
    roles={"anomaly_train", "zero_shot_train"},
    tasks=("anomaly", "segmentation"),
)
register_capabilities(
    "visa",
    default_root="datasets/general/VisA",
    roles={"anomaly_train", "zero_shot_train"},
    tasks=("anomaly", "segmentation"),
)
