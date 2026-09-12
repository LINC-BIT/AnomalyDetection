"""Pre-flight checks that fail before entering framework download code."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def resolve_cached_weight(reference: str, extra_dirs: list[str | Path] | None = None) -> Path | None:
    """Resolve a local weight reference across project and common cache locations."""

    parsed = urlparse(reference)
    filename = Path(parsed.path).name
    direct = Path(os.path.expanduser(os.path.expandvars(reference)))
    candidates = [direct]
    if filename:
        candidates.extend(
            [
                Path.cwd() / filename,
                Path(os.getenv("FDH_MODEL_CACHE", "artifacts/models")).expanduser() / filename,
                Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename,
                Path.home() / ".cache" / "ultralytics" / filename,
            ]
        )
        candidates.extend(Path(directory).expanduser() / filename for directory in (extra_dirs or []))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    return None


def require_cached_weight(
    reference: str, backend: str, extra_dirs: list[str | Path] | None = None
) -> Path:
    """Return a cached weight path or raise an actionable offline error."""

    cached = resolve_cached_weight(reference, extra_dirs=extra_dirs)
    if cached is not None:
        return cached
    filename = Path(urlparse(reference).path).name or reference
    raise FileNotFoundError(
        f"Offline {backend} pre-flight failed: pretrained weight {filename!r} was not found. "
        "Download/cache it while online, set FDH_MODEL_CACHE, pass an explicit weights path, "
        "or disable offline/pretrained mode."
    )
