"""TrainConfig: one validated vocabulary for the settings every backend has.

`ModelAdapter.train(config: dict[str, Any])` accepts anything. That is not a
contract — it is six private vocabularies wearing the same type annotation:
the learning rate is `lr0` to Ultralytics, `lr` to torchvision, Dinomaly and
MambaAD; the run length is `epochs` here and `total_iters` there; a typo in
any of them is silently ignored and the run quietly uses a default.

`TrainConfig` fixes the handful of knobs that genuinely mean the same thing
everywhere, validates them once, and translates them into whatever the target
backend really calls them. Everything else — a backend's own knobs, which are
not portable and should not pretend to be — travels in `backend_specific` and
is passed through untouched.

Translation lives with the vocabulary it describes: each adapter declares its
own `TRAIN_CONFIG_KEYS` mapping (canonical name -> that backend's real
argument name), the same way `UltralyticsAdapter` already owns
`_RECIPE_TRAINER_ARGS`. `core` therefore knows the canonical names and
nothing about any specific framework.

Backwards compatible on purpose: `train()` still accepts a plain dict, which
is passed straight through. `TrainConfig` is the validated front door for
callers that want one, not a wall in front of the backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

PRECISIONS = ("fp32", "fp16", "bf16")

# The canonical names. A backend maps the subset it understands.
CANONICAL_KEYS = (
    "epochs",
    "max_iters",
    "lr",
    "batch_size",
    "image_size",
    "device",
    "seed",
    "num_workers",
    "work_dir",
)


@dataclass(frozen=True)
class TrainConfig:
    """Portable training settings, in names that mean the same thing for every backend."""

    epochs: int | None = None
    max_iters: int | None = None
    lr: float | None = None
    batch_size: int | None = None
    image_size: int | None = None
    device: str | None = None
    seed: int | None = None
    precision: str = "fp32"
    num_workers: int | None = None
    work_dir: str | None = None
    backend_specific: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"unknown precision {self.precision!r}; expected one of {PRECISIONS}")
        if self.epochs is not None and self.max_iters is not None:
            raise ValueError(
                "set either epochs or max_iters, not both: they are two ways to say the same "
                "thing and backends read only one of them, so the other would be silently ignored"
            )
        for name, value in (
            ("epochs", self.epochs),
            ("max_iters", self.max_iters),
            ("batch_size", self.batch_size),
            ("image_size", self.image_size),
            ("num_workers", self.num_workers),
        ):
            if value is not None and value <= 0 and not (name == "num_workers" and value == 0):
                raise ValueError(f"{name} must be positive, got {value}")
        if self.lr is not None and self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

        # Reject a backend-specific key that shadows a canonical one: it would
        # be set twice with no defined winner.
        shadowed = sorted(set(self.backend_specific) & set(CANONICAL_KEYS))
        if shadowed:
            raise ValueError(
                f"backend_specific must not repeat canonical keys {shadowed}; "
                "set them as TrainConfig fields instead"
            )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        """Build from a flat dict: recognised keys become fields, everything
        else lands in `backend_specific` rather than being dropped.
        """

        known = {f.name for f in fields(cls)} - {"backend_specific"}
        canonical = {k: v for k, v in data.items() if k in known}
        rest = {k: v for k, v in data.items() if k not in known}
        return cls(**canonical, backend_specific=rest)

    # ------------------------------------------------------------------ #
    # Translation
    # ------------------------------------------------------------------ #
    def to_backend_dict(self, key_map: dict[str, str]) -> dict[str, Any]:
        """Render into one backend's real argument names.

        `key_map` maps canonical name -> that backend's name (see an adapter's
        `TRAIN_CONFIG_KEYS`). A canonical field the backend has no name for is
        dropped — deliberately: forwarding `total_iters` to a trainer that
        takes `epochs` is exactly the crash this class exists to prevent.
        """

        unknown = sorted(set(key_map) - set(CANONICAL_KEYS))
        if unknown:
            raise ValueError(
                f"key_map has non-canonical keys {unknown}; expected names from {CANONICAL_KEYS}"
            )

        rendered: dict[str, Any] = {}
        for canonical_name, backend_name in key_map.items():
            value = getattr(self, canonical_name)
            if value is not None:
                rendered[backend_name] = value
        rendered.update(self.backend_specific)
        return rendered

    def as_run_metadata(self) -> dict[str, Any]:
        """JSON-safe record of the portable settings, for the run log."""

        return {
            name: getattr(self, name)
            for name in (*CANONICAL_KEYS, "precision")
            if getattr(self, name) is not None
        }


def resolve_train_config(config: Any, key_map: dict[str, str]) -> dict[str, Any]:
    """Normalise whatever `train()` was handed into the dict its backend reads.

    A `TrainConfig` is translated through `key_map`; a plain dict (still the
    common case, and what every existing caller passes) is returned unchanged.
    """

    if isinstance(config, TrainConfig):
        return config.to_backend_dict(key_map)
    return config
