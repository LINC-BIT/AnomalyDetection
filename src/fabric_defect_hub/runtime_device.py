"""Torch device selection shared by command-line benchmark entry points."""

from __future__ import annotations


def resolve_torch_device(requested: str | None = "auto") -> str:
    """Resolve an explicit device or select the best usable local backend.

    Explicit accelerator requests must never silently fall back to CPU: doing
    so invalidates latency and memory measurements while leaving a long sweep
    apparently healthy.
    """

    import torch

    device = (requested or "auto").strip().lower()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cpu":
        return "cpu"
    if device == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        built = torch.backends.mps.is_built()
        detail = "compiled into PyTorch but unavailable at runtime" if built else "not compiled into PyTorch"
        raise RuntimeError(
            "MPS was explicitly requested but is " + detail + ". "
            "Use a compatible Apple-Silicon PyTorch environment, then verify "
            "`python -c 'import torch; print(torch.backends.mps.is_available())'` returns True."
        )
    if device == "cuda" or device.startswith("cuda:"):
        if torch.cuda.is_available():
            return "cuda:0" if device == "cuda" else device
        raise RuntimeError(
            f"{device} was explicitly requested but torch.cuda.is_available() is False. "
            "Use --device auto, --device cpu, or run on a CUDA-enabled host."
        )
    raise ValueError("device must be one of auto, cpu, mps, cuda, or cuda:<index>")
