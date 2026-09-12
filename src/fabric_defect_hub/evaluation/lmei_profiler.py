"""Latency-Memory Efficiency Index (LMEI) Profiler for Edge Deployment.

Calculates a normalized trade-off score considering throughput (FPS), peak VRAM memory,
and model FLOPs parameter count for real-world industrial deployment benchmarking.
"""

from __future__ import annotations

import math


def calculate_lmei(fps: float, vram_mb: float, flops_g: float, params_m: float) -> float:
    """Calculates Latency-Memory Efficiency Index (LMEI).

    Formulation:
        LMEI = (FPS / 100.0) / (log10(FLOPs_G + 1.0) * log10(VRAM_MB / 100.0 + 1.0) + 1e-5)

    Higher score indicates superior real-time edge hardware deployment trade-off.
    """
    if fps <= 0 or vram_mb <= 0 or flops_g < 0:
        return 0.0

    denom = math.log10(flops_g + 1.0) * math.log10(max(1.0, vram_mb / 100.0) + 1.0) + 1e-5
    score = (fps / 100.0) / denom
    return round(float(score), 4)
