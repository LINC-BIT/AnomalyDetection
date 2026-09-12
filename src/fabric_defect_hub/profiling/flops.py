"""FLOPs (floating-point operations) counting for a `torch.nn.Module` --
the one input `evaluation.lmei_profiler.calculate_lmei`'s `flops_g`
parameter needed that nothing in this project computed (see that module's
docstring: it took FLOPs as an external input, with no calculator behind
it).

Uses `thop` (already a common transitive dependency via Ultralytics; the
`profiling-flops` extra pins it directly so it isn't an undeclared
accident) since FLOPs are a pure architecture property counted once from a
dummy forward pass -- unlike FPS/latency/memory, they need no engine or
hardware, so they don't belong in `profiling/base.py`'s per-run
`BackendProfiler` hierarchy.
"""

from __future__ import annotations

from typing import Any


def compute_model_flops(
    model: Any,
    input_size: tuple[int, int],
    batch_size: int = 1,
    input_style: str = "batched",
    device: str | Any = "cpu",
) -> float:
    """FLOPs (in GFLOPs, i.e. 10^9 FLOPs) for one forward pass at
    `input_size`, via `thop.profile`.

    `input_style` mirrors `profiling.base.ProfileConfig.input_style`:
    'batched' builds a single `[batch, 3, H, W]` tensor (CNN-style models);
    'list' builds a `List[[3, H, W]]` of `batch_size` tensors (torchvision
    detection models' native calling convention). `model` can be a plain
    `torch.nn.Module` or an already-exported `torch.jit.ScriptModule` --
    both expose the `named_modules()` hook targets `thop` counts. `device`
    is taken as an explicit argument (not inferred from `model.parameters()`)
    since a purely functional module (no learnable weights) has none to
    read a device off of.

    `thop.profile` reports MACs (multiply-accumulate operations); FLOPs
    are conventionally 2x that (one multiply + one add per MAC) -- the
    same convention fvcore/ptflops and most CNN papers use.

    Requires the `profiling-flops` extra (`pip install thop`); raises
    `ImportError` with that instruction rather than silently returning 0,
    matching this project's "no fabric-appropriate default" philosophy for
    a genuinely-missing optional dependency (see `AnomalibAdapter`'s
    EfficientAD `imagenet_dir` error for the same pattern).
    """

    try:
        import thop
    except ImportError as exc:
        raise ImportError(
            "FLOPs counting needs the 'profiling-flops' extra: pip install thop "
            "(or `pip install -e '.[profiling-flops]'`)."
        ) from exc
    import torch

    height, width = input_size
    # The model's own device wins over the `device` argument whenever it has
    # one. A caller passing the *session's* device (e.g. "mps") for a module
    # still resident on CPU produced `Input type (MPSFloatType) and weight
    # type (torch.FloatTensor) should be the same` — a mismatch the caller
    # cannot see, since it never chose where the adapter put its weights.
    # `device` remains the fallback for a purely functional module, which is
    # the case the parameter was introduced for.
    target = next((parameter.device for parameter in model.parameters()), None) or device

    if input_style == "list":
        dummy: Any = [torch.rand(3, height, width, device=target) for _ in range(batch_size)]
    elif input_style == "batched":
        dummy = torch.rand(batch_size, 3, height, width, device=target)
    else:
        raise ValueError(f"unknown input_style {input_style!r}; expected 'batched' or 'list'.")

    macs, _params = thop.profile(model, inputs=(dummy,), verbose=False)
    return (macs * 2) / 1e9
