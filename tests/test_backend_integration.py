"""Opt-in real-backend integration tests for CUDA/cloud execution.

Set ``FDH_RUN_BACKEND_INTEGRATION=1`` and one or more backend-specific
config paths to exercise the actual config-driven lifecycle on a machine
where the data, framework extras, and model weights are available.
"""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    ("backend", "environment_key"),
    [
        ("ultralytics", "FDH_ULTRALYTICS_INTEGRATION_CONFIG"),
        ("torchvision", "FDH_TORCHVISION_INTEGRATION_CONFIG"),
        ("anomalib", "FDH_ANOMALIB_INTEGRATION_CONFIG"),
    ],
)
def test_configured_backend_lifecycle(backend, environment_key):
    if os.getenv("FDH_RUN_BACKEND_INTEGRATION") != "1":
        pytest.skip("set FDH_RUN_BACKEND_INTEGRATION=1 on a prepared backend host")
    config_path = os.getenv(environment_key)
    if not config_path:
        pytest.skip(f"set {environment_key} to a real backend config")

    if backend == "ultralytics":
        from fabric_defect_hub.models.ultralytics.pipeline import run_from_yaml
    elif backend == "torchvision":
        from fabric_defect_hub.models.torchvision.pipeline import run_from_yaml
    else:
        from fabric_defect_hub.models.anomalib.pipeline import run_from_yaml

    result = run_from_yaml(config_path)
    assert result.trained_artifact is not None or result.registered_artifact is not None


def test_configured_real_backend_benchmark():
    if os.getenv("FDH_RUN_BACKEND_INTEGRATION") != "1":
        pytest.skip("set FDH_RUN_BACKEND_INTEGRATION=1 on a prepared backend host")
    config_path = os.getenv("FDH_BENCHMARK_INTEGRATION_CONFIG")
    if not config_path:
        pytest.skip("set FDH_BENCHMARK_INTEGRATION_CONFIG to a real benchmark config")

    from fabric_defect_hub.benchmark import BenchmarkConfig

    results = BenchmarkConfig.from_yaml(config_path).run()
    assert results
    assert all(result.artifacts.get("predictions") for result in results)
