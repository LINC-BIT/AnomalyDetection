"""Evaluates one benchmark model — in its own process.

`application/benchmark.py` used to run every model in a worker *thread* of the
Gradio process. Two library-level hazards made that cost rows that had nothing
to do with the failure:

* `torch.export` patches module dispatch process-wide while it traces, so a
  worker that was merely predicting got captured by a sibling worker's export
  (`AssertionError: Unexpected key relu@1, expected relu`, Mask R-CNN profiling
  vs DeepLabV3+ predicting, 2026-09-17 — see `core/execution.py`, which still
  guards whatever shares a process);
* an illegal memory access leaves a process's CUDA context permanently broken
  ("Sticky error detected"), after which *every* later model in that process
  fails with the same message — one bad kernel cost ten rows.

One process per model makes both local: a model's tracing cannot see another
model's forward, and a crash takes down exactly the model that crashed. The
parent keeps the scheduling, the rows and the status line, and never loads a
model itself, so it holds no CUDA context to poison.

Run as `python -m fabric_defect_hub.application.benchmark_worker`: reads one
JSON payload on stdin, writes its result JSON to `payload["results_path"]`.
stdout/stderr are inherited on purpose, so per-image progress and framework
warnings keep reaching the terminal the UI was started from.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from fabric_defect_hub.application import benchmark as benchmark_module
from fabric_defect_hub.application import load_dataset, load_model, run_experiment
from fabric_defect_hub.application.benchmark import (
    BENCHMARK_ANOMALY_MAP_ROOT,
    THRESHOLD_CALIBRATION_DEFECT_RATIO,
    THRESHOLD_CALIBRATION_SAMPLE_COUNT,
    MetricNotApplicable,
    _PRIMARY_ACCURACY_METRIC,
    _activate_device,
    _cross_domain_probe,
    _flops_and_lmei,
    _profile_model,
    _resolution_sweep,
    _slug,
)
from fabric_defect_hub.application.workspace import dataset_tasks, task_text
from fabric_defect_hub.core.execution import model_execution
from fabric_defect_hub.core.types import ModelInfo, RuntimeInfo
from fabric_defect_hub.evaluation import evaluator_for_task, ground_truth_task
from fabric_defect_hub.evaluation.anomaly import AnomalyEvaluator, calibrate_thresholds
from fabric_defect_hub.core.provenance import collect_provenance
from fabric_defect_hub.i18n import DEFAULT_LANGUAGE, tr
from fabric_defect_hub.reporting import append_jsonl
from fabric_defect_hub.inference.session import clear_accelerator_cache


def _calibrate_thresholds(
    *,
    dataset_label: str,
    texture: str | None,
    spec: dict[str, Any],
    dataset_task: str,
    model: Any,
    model_spec: dict[str, Any],
    artifact: Any,
    device: str,
    input_size: int = 640,
) -> tuple[float | None, float | None, int]:
    """Fit the image and pixel decision thresholds on the dataset's train split.

    The train split is the only split besides the one being reported on, and
    the dataset contract keeps the two disjoint (`ImageSets/<pattern>.json`
    lists them separately), so the thresholds are fitted on samples the test
    pass never scores. That is the whole point: `AnomalyEvaluator` will not
    pick its own threshold on the split it reports on, and a threshold fitted
    on the test scores would be exactly the same-set optimum it refuses.

    Anomaly maps are written to a throwaway directory rather than the
    benchmark's own artifact tree: they exist only long enough to fit the
    pixel threshold, and the scored pass writes the maps that matter.

    Returns `(image_threshold, pixel_threshold, scored_samples)`; a threshold
    is `None` when its data cannot separate the two classes — the normal case
    for the normal-only train splits (MVTec AD, flat-folder datasets) and for
    any model that does not persist an anomaly map. The caller reports that as
    a note rather than failing the row.
    """

    calibration_kwargs: dict[str, Any] = dict(
        root=benchmark_module.default_dataset_root(dataset_label),
        split="train",
        use_defect=True,
        num_samples=THRESHOLD_CALIBRATION_SAMPLE_COUNT,
        defect_ratio=THRESHOLD_CALIBRATION_DEFECT_RATIO,
    )
    if spec["slice_kwarg"] is not None:
        calibration_kwargs[spec["slice_kwarg"]] = benchmark_module.slice_value(
            dataset_label, texture
        )
    calibration_dataset = load_dataset(spec["name"], task=dataset_task, **calibration_kwargs)
    calibration_samples = calibration_dataset.load_samples()

    # Same predict contract `run_experiment` uses for the scored split (see
    # `loader.run_experiment`): a model that can persist an anomaly map gets an
    # `output_dir`, because the pixel threshold can only come from the maps.
    with tempfile.TemporaryDirectory(prefix="fdh-threshold-calibration-") as map_dir:
        predict_config: dict[str, Any] = {"device": device, "raw_anomaly": True}
        if model_spec["backend"] == "ultralytics":
            predict_config["imgsz"] = input_size
        predict_kwargs: dict[str, Any] = {"config": predict_config}
        if model.capabilities().fills("anomaly_map"):
            predict_kwargs["output_dir"] = map_dir
        with model_execution():
            predictions = model.predict(calibration_samples, artifact, **predict_kwargs)

        return calibrate_thresholds(calibration_samples, predictions)


def _calibrated_threshold_detail(
    image_threshold: float | None, pixel_threshold: float | None,
) -> str:
    """The fitted thresholds, as the status-line note spells them out.

    Only the ones that were actually fitted are named: a model that persists
    no anomaly map has no pixel threshold, and naming "pixel None" would read
    as a failure instead of "not measurable for this model".
    """

    parts = []
    if image_threshold is not None:
        parts.append(f"image {image_threshold:.4f}")
    if pixel_threshold is not None:
        parts.append(f"pixel {pixel_threshold:.4f}")
    return ", ".join(parts)


def evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    """Produce one model's benchmark row.

    Returns `{"row", "warnings", "count", "error", "traceback"}` — the report the
    parent turns into a table row, a status line and a failure-log entry.
    """

    model_label = payload["model_label"]
    dataset_label = payload["dataset_label"]
    shot_mode = payload["shot_mode"]
    device = payload["device"]
    lang = payload.get("lang", DEFAULT_LANGUAGE)

    # Read the catalogs through the `benchmark` module rather than importing the
    # objects: the UI-facing tests replace them with fakes, and an in-process
    # worker has to see the same replacement the parent does.
    model_spec = benchmark_module.MODEL_CATALOG[model_label]
    spec = benchmark_module.DATASET_CATALOG[dataset_label]
    dataset_task = ground_truth_task(model_spec["task"])
    if dataset_task not in dataset_tasks(spec["name"]):
        return _report(warnings=[
            tr(
                lang, "bench_task_mismatch",
                model=model_label, dataset=dataset_label,
                task=task_text(lang, model_spec["task"]),
            )
        ])

    num_samples, defect_ratio = benchmark_module.shot_regime_kwargs(shot_mode)
    dataset_kwargs: dict[str, Any] = dict(
        root=benchmark_module.default_dataset_root(dataset_label),
        split="test",
        use_defect=True,
        num_samples=num_samples,
        defect_ratio=defect_ratio,
    )
    if spec["slice_kwarg"] is not None:
        dataset_kwargs[spec["slice_kwarg"]] = benchmark_module.slice_value(
            dataset_label, payload["texture"]
        )

    include_profiling = bool(payload.get("include_profiling"))
    include_resolution_sweep = bool(payload.get("include_resolution_sweep"))
    cross_domain_dataset_label = payload.get("cross_domain_dataset_label")
    calibrate_thresholds = bool(payload.get("calibrate_thresholds"))
    run_log_path = payload.get("run_log_path")
    row_log_path = payload.get("row_log_path")
    artifact_for_model = benchmark_module.artifact_for_model

    model = None
    warnings: list[str] = []
    notes: list[str] = []
    try:
        # Pin this process to the device the parent assigned it: the process now
        # owns its whole CUDA context, and the adapters that resolve "cuda"
        # implicitly must land on the intended GPU.
        _activate_device(device)
        dataset = load_dataset(spec["name"], task=dataset_task, **dataset_kwargs)
        model = load_model(model_spec["backend"], model_spec["name"])
        evaluator = evaluator_for_task(dataset_task)
        # Timing starts before the calibration pass: that pass is work this row
        # asked for, so its wall clock belongs to `runtime_s` rather than
        # vanishing from it.
        started = time.perf_counter()
        if calibrate_thresholds and dataset_task == "anomaly":
            # Best-effort, like every other opt-in pass here: a dataset whose
            # train split is unreadable (or single-class) must cost this model
            # its thresholded columns, not its whole row.
            try:
                image_threshold, pixel_threshold, calibrated_samples = _calibrate_thresholds(
                    dataset_label=dataset_label,
                    texture=payload.get("texture"),
                    spec=spec,
                    dataset_task=dataset_task,
                    model=model,
                    model_spec=model_spec,
                    artifact=artifact_for_model(model_spec),
                    device=device,
                )
            except Exception as exc:
                warnings.append(
                    f"{model_label}: threshold calibration skipped ({type(exc).__name__}: {exc})"
                )
            else:
                if image_threshold is None and pixel_threshold is None:
                    notes.append(tr(
                        lang, "bench_threshold_uncalibratable",
                        model=model_label, samples=calibrated_samples,
                    ))
                else:
                    evaluator = AnomalyEvaluator(
                        image_threshold=image_threshold, pixel_threshold=pixel_threshold,
                    )
                    notes.append(tr(
                        lang, "bench_threshold_calibrated",
                        model=model_label, samples=calibrated_samples,
                        detail=_calibrated_threshold_detail(image_threshold, pixel_threshold),
                    ))
        result = run_experiment(
            experiment_id=f"benchmark-{_slug(model_label)}",
            dataset=dataset,
            model=model,
            model_info=ModelInfo(
                name=model_spec["name"], backend=model_spec["backend"], task=model_spec["task"]
            ),
            runtime=RuntimeInfo(device=device, engine="python", precision="fp32", input_size=(640, 640)),
            evaluator=evaluator,
            artifact=artifact_for_model(model_spec),
            output_dir=str(Path(BENCHMARK_ANOMALY_MAP_ROOT) / _slug(model_label)),
            run_log_path=run_log_path,
            # Only the anomaly backends have a stored normalization to bypass;
            # sending the flag to a detector put an unknown keyword in front of
            # ultralytics' strict argument validation.
            raw_anomaly_scores=dataset_task == "anomaly",
        )
        constant_scores = result.metrics.get("constant_anomaly_scores")
        if constant_scores:
            warnings.append(
                f"{model_label}: all {int(constant_scores)} predictions carry the same anomaly "
                "score, so this row does not discriminate between images — its AUROC/AUPRO are "
                "an artefact of a saturated model, not a measurement (see the failure log)"
            )
            benchmark_module._append_benchmark_failure(
                model_label,
                f"degenerate predictions: all {int(constant_scores)} samples share one anomaly "
                "score, so the ranking carries no information and this row's AUROC/AUPRO are "
                "artefacts of a saturated output rather than a measurement. Check the "
                "checkpoint, the post-processor's stored normalization and the input scale "
                "before trusting any metric from this row.\n",
            )
        count = len(dataset.load_samples())
        row: dict[str, Any] = {
            "model": model_label,
            "runtime_s": round(time.perf_counter() - started, 1),
            **result.metrics,
        }
        if include_profiling:
            try:
                profile_started = time.perf_counter()
                row.update(
                    _profile_model(model, artifact_for_model(model_spec), device, dataset.load_samples())
                )
                row["runtime_s"] = round(row["runtime_s"] + time.perf_counter() - profile_started, 1)
            except Exception as exc:
                warnings.append(f"{model_label}: profiling skipped ({type(exc).__name__}: {exc})")
        # Every opt-in addition below is best-effort: a failure in one (e.g.
        # thop missing for FLOPs, a target dataset erroring mid-probe) only
        # forfeits that addition's columns, never the base row.
        if include_resolution_sweep:
            try:
                row.update(_resolution_sweep(model, artifact_for_model(model_spec), device))
            except MetricNotApplicable as exc:
                # A property of the model, not a failure: say so without the
                # warning triangle the panel reserves for things going wrong.
                notes.append(f"{model_label}: no resolution slope — {exc}")
            except Exception as exc:
                warnings.append(f"{model_label}: resolution sweep skipped ({type(exc).__name__}: {exc})")
        if include_profiling:
            try:
                row.update(
                    _flops_and_lmei(
                        model, model_spec, device, fps=row.get("fps"),
                        vram_mb=row.get("peak_memory_mb"),
                        memory_kind=row.get("memory_measurement_kind"),
                    )
                )
            except Exception as exc:
                warnings.append(f"{model_label}: FLOPs/LMEI skipped ({type(exc).__name__}: {exc})")
        if cross_domain_dataset_label:
            try:
                metric_key = _PRIMARY_ACCURACY_METRIC.get(dataset_task)
                acc_src = result.metrics.get(metric_key) if metric_key else None
                if acc_src is not None:
                    target_metrics = _cross_domain_probe(
                        model, artifact_for_model(model_spec), dataset_task,
                        cross_domain_dataset_label, num_samples, defect_ratio,
                    )
                    acc_tgt = target_metrics.get(metric_key) if target_metrics else None
                    if acc_tgt is not None and acc_src != 0:
                        from fabric_defect_hub.evaluation.cross_domain import cross_domain_degradation

                        row["cross_domain_delta_acc_pct"] = cross_domain_degradation(acc_src, acc_tgt)
            except Exception as exc:
                warnings.append(f"{model_label}: cross-domain probe skipped ({type(exc).__name__}: {exc})")
        if row_log_path:
            # The whole row, not just `result.metrics`: the overhead columns
            # (`fps`, latency percentiles, peak memory, FLOPs, LMEI) are added
            # after `run_experiment` has logged the accuracy result, so this is
            # the only record that can back a compute table. `warnings` travels
            # with it so a blank cell has its reason next to it.
            append_jsonl(
                {
                    "experiment_id": f"benchmark-{_slug(model_label)}",
                    "model": model_label,
                    "dataset": dataset_label,
                    "selection": {
                        "texture": payload.get("texture"),
                        "shot_mode": shot_mode,
                        "device": device,
                        "include_profiling": include_profiling,
                        "include_resolution_sweep": include_resolution_sweep,
                        "calibrate_thresholds": calibrate_thresholds,
                        "cross_domain_dataset_label": cross_domain_dataset_label,
                    },
                    "status": "ok",
                    "samples": count,
                    "metrics": row,
                    "warnings": list(warnings),
                    "notes": list(notes),
                    "provenance": collect_provenance(),
                },
                row_log_path,
            )
        return _report(row=row, warnings=warnings, notes=notes, count=count)
    except Exception as exc:
        return _report(
            warnings=warnings,
            notes=notes,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    finally:
        if model is not None:
            unload = getattr(model, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:
                    pass
        clear_accelerator_cache()


def _report(
    *,
    row: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    notes: list[str] | None = None,
    count: int | None = None,
    error: str | None = None,
    traceback: str | None = None,
) -> dict[str, Any]:
    """`warnings` are failures that cost optional columns; `notes` are metrics
    this model cannot produce at all (see `MetricNotApplicable`)."""

    return {
        "row": row,
        "warnings": list(warnings or []),
        "notes": list(notes or []),
        "count": count,
        "error": error,
        "traceback": traceback,
    }


def main(argv: list[str] | None = None) -> int:
    payload = json.loads(sys.stdin.read())
    result = evaluate(payload)
    results_path = payload.get("results_path")
    if results_path:
        Path(results_path).write_text(json.dumps(result), encoding="utf-8")
    else:  # pragma: no cover - only reachable when run by hand
        print(json.dumps(result))
    return 0 if result["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
