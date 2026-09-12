import pytest

from fabric_defect_hub.core.registry import (
    get_evaluator_cls,
    get_profiler_cls,
    list_evaluators,
    list_profilers,
    register_evaluator,
    register_profiler,
)


class _FakeEvaluator:
    task = "fake-registry-task"


class _FakeProfiler:
    engine = "fake-registry-engine"


def test_register_evaluator_reads_task_off_the_class():
    register_evaluator(_FakeEvaluator)

    assert get_evaluator_cls("fake-registry-task") is _FakeEvaluator
    assert "fake-registry-task" in list_evaluators()


def test_register_profiler_reads_engine_off_the_class():
    register_profiler(_FakeProfiler)

    assert get_profiler_cls("fake-registry-engine") is _FakeProfiler
    assert "fake-registry-engine" in list_profilers()


def test_register_evaluator_rejects_duplicate_task():
    class _DuplicateTask:
        task = "fake-registry-task"

    with pytest.raises(ValueError, match="already registered"):
        register_evaluator(_DuplicateTask)


def test_get_evaluator_cls_unknown_task_lists_known_ones():
    with pytest.raises(KeyError, match="fake-registry-task"):
        get_evaluator_cls("does-not-exist")


def test_real_evaluators_and_profilers_are_registered_via_import():
    import fabric_defect_hub.evaluation  # noqa: F401
    import fabric_defect_hub.profiling  # noqa: F401

    assert {"anomaly", "detection", "industrial", "segmentation"} <= set(list_evaluators())
    assert {"onnxruntime", "pytorch", "tensorrt"} <= set(list_profilers())
