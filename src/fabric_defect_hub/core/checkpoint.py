"""Non-executing provenance diagnostics for a trained checkpoint.

Every backend in this project stores its weights as a torch archive, and a
torch archive can carry arbitrary Python objects. This module establishes
what a checkpoint *is* -- size, SHA-256, and the globals its pickle declares
-- by reading the archive's metadata, never by deserialising model state, so
an operator can decide whether to trust a file before anything loads it.

Lives in `core/` rather than under one backend because nothing here is
backend-specific: it hashes bytes and asks `torch.serialization` what the
archive declares. It sat in `models/anomalib/` for a while, and the UI
consequently offered this diagnostic for anomalib checkpoints only, telling
everyone else "native Ultralytics artifact" -- wrong for four of the six
backends, and an unnecessary blind spot for the `.pt`/`.pth` files, which are
pickles too (verified: `get_unsafe_globals_in_checkpoint` reads a YOLO `.pt`
perfectly well).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointDiagnostic:
    path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    unsafe_globals: tuple[str, ...]


def inspect_checkpoint(path: str | Path) -> CheckpointDiagnostic:
    """Inspect a checkpoint without executing its pickle payload."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        return CheckpointDiagnostic(str(checkpoint), False, None, None, ())

    import torch

    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    unsafe = tuple(sorted(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)))
    return CheckpointDiagnostic(
        path=str(checkpoint),
        exists=True,
        size_bytes=checkpoint.stat().st_size,
        sha256=digest.hexdigest(),
        unsafe_globals=unsafe,
    )
