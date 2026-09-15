"""Command-line entry point for config-driven model and benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any


def _shot_modes() -> tuple[str, ...]:
    """The `--mode` vocabulary, read off `training.ShotMode` rather than
    retyped here. The two used to be separate literals that had to agree.
    """

    import typing

    from fabric_defect_hub.training import ShotMode

    return typing.get_args(ShotMode)


def _model_backend_choices() -> tuple[str, ...]:
    from fabric_defect_hub.loader import list_model_backends

    return tuple(list_model_backends())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adh", description="AnomalyDetection runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a model or benchmark YAML config")
    run_parser.add_argument("config", help="path to YAML config")
    run_parser.add_argument(
        "--backend", choices=_model_backend_choices(),
        help="model backend; inferred from the config when omitted",
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="run a benchmark YAML config")
    benchmark_parser.add_argument("config", help="path to benchmark YAML config")

    subparsers.add_parser(
        "list",
        help="list every registered dataset/model-backend/evaluator/profiler "
        "(self-describing platform catalog, not hardcoded documentation)",
    )
    subparsers.add_parser(
        "inventory",
        help="machine-readable unified model and dataset registry used by the UI",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="unified training entry point: pick a model config, optionally override its dataset + shot mode",
    )
    train_parser.add_argument(
        "model", nargs="?",
        help=(
            "a model config: a path (configs/models/ultralytics_example.yaml), a filename stem "
            "under --config-dir (ultralytics_example), or a model keyword matched against every "
            "config's model.variant/model.name there (yolov8n, patchcore, ...). "
            "Omit with --list to see what's available."
        ),
    )
    train_parser.add_argument(
        "--config-dir", default="configs/models",
        help="directory searched when 'model' is a filename stem or keyword (default: configs/models)",
    )
    train_parser.add_argument(
        "--list", action="store_true", help="list resolvable model configs under --config-dir and exit"
    )
    train_parser.add_argument(
        "--backend", choices=_model_backend_choices(),
        help="override backend keyword detection (model.name -> anomalib, model.variant -> ultralytics/torchvision)",
    )
    train_parser.add_argument(
        "--variant",
        help=(
            "override which model in the backend's family gets trained: written to "
            "model.variant for ultralytics/torchvision (e.g. yolov8n, yolov8s, yolo11n, "
            "fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn) or model.name for anomalib/dinomaly/moeclip/mambaad "
            "(e.g. PatchCore, PaDiM, RD4AD, EfficientAD, SuperSimpleNet, dinov2reg_vit_base_14); "
            "lets one config file train any model its backend supports instead of needing one "
            "YAML per model"
        ),
    )
    train_parser.add_argument(
        "--dataset", help="registered dataset name (e.g. zju-leaper, raw-fabric, mvtec-ad); overrides data.dataset"
    )
    train_parser.add_argument("--dataset-root", help="dataset root path; overrides data.dataset_root")
    train_parser.add_argument(
        "--test-dataset",
        help="zero-shot backends (moeclip) only: the dataset to *evaluate* on, when it differs "
        "from the training corpus (overrides data.test_dataset)",
    )
    train_parser.add_argument(
        "--test-dataset-root", help="root path for --test-dataset; falls back to data/<Dataset>"
    )
    train_parser.add_argument(
        "--mode", choices=_shot_modes(), default=None,
        help=(
            "shot mode: full=use every sample, every ZJU-Leaper pattern (num_samples=null); "
            "medium=every ZJU-Leaper pattern but capped per-pattern (150 train / 50 val each); "
            "few=leave the config's own declared few-shot count and pattern subset untouched; "
            "test=quick 8-image smoke run of the whole pipeline"
        ),
    )
    train_parser.add_argument(
        "--num-samples", type=int,
        help="explicit sample-count override for the train split (and val/test split unless --val-num-samples is given)",
    )
    train_parser.add_argument(
        "--val-num-samples", type=int, help="explicit sample-count override for the val/test split only"
    )
    defect_group = train_parser.add_mutually_exclusive_group()
    defect_group.add_argument("--use-defect", dest="use_defect", action="store_true", default=None)
    defect_group.add_argument("--no-use-defect", dest="use_defect", action="store_false")
    train_parser.add_argument("--defect-ratio", type=float, help="fraction of the loaded split that is defective")
    train_parser.add_argument(
        "--pattern",
        help="ZJU-Leaper pattern/group filter override: one of patternN, N, groupN, or a texture name",
    )
    train_parser.add_argument("--category", help="MVTec-AD category filter override")
    train_parser.add_argument("--seed", type=int, help="subsampling RNG seed override")
    train_parser.add_argument(
        "--set", dest="set_overrides", action="append", default=[], metavar="path.to.key=value",
        help=(
            "tuning window: override any config value by dotted path, repeatable "
            "(e.g. --set train.model_kwargs.coreset_sampling_ratio=0.05 --set train.model_kwargs.lr=0.0005). "
            "Value is YAML-parsed (numbers/booleans/lists work unquoted); wins over the config file, "
            "the recipe/profile defaults, and every other --* override"
        ),
    )
    train_parser.add_argument("--enable-tiling", action="store_true", help="YOLO only: train on coordinate-aware image tiles")
    train_parser.add_argument("--tile-size", type=int, help="YOLO only: square tile size when --enable-tiling is set")
    train_parser.add_argument("--tile-overlap", type=float, help="YOLO only: tile overlap in [0, 1)")
    train_parser.add_argument("--profile", default="configs/training_profile.yaml", help="shared ZJU training policy YAML")
    train_parser.add_argument("--no-profile", action="store_true", help="ignore the shared training profile")
    train_parser.add_argument(
        "--no-publish",
        action="store_true",
        help=(
            "keep the result out of artifacts/models/published/. Training a model that is in "
            "catalog.CANONICAL_MODELS otherwise overwrites the checkpoint the web UI serves for "
            "it — use this for smoke runs and experiments"
        ),
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help=(
            "unified inference entry point: pick a model config (same resolution as 'train'), "
            "load a previously trained artifact, and run it over images or a dataset selection"
        ),
    )
    predict_parser.add_argument(
        "model",
        help=(
            "a model config: a path, a filename stem under --config-dir, or a model keyword "
            "(yolov8n, patchcore, ...) — resolved exactly like 'adh train MODEL'"
        ),
    )
    predict_parser.add_argument(
        "--weights", required=True,
        help=(
            "path to a trained/registered artifact to load, e.g. 'adh train's "
            "registered_artifact.path output (artifacts/models/<name>.pt or .ckpt)"
        ),
    )
    predict_parser.add_argument(
        "--config-dir", default="configs/models",
        help="directory searched when 'model' is a filename stem or keyword (default: configs/models)",
    )
    predict_parser.add_argument(
        "--backend", choices=_model_backend_choices(),
        help="override backend keyword detection (model.name -> anomalib, model.variant -> ultralytics/torchvision)",
    )
    predict_parser.add_argument(
        "--variant",
        help="override which model in the backend's family runs inference (see 'train --variant'); "
        "must match what --weights was actually trained as",
    )
    predict_parser.add_argument(
        "--image", action="append", dest="images", default=[],
        help="path to an image to run inference on; repeatable. Mutually exclusive with --dataset",
    )
    predict_parser.add_argument(
        "--dataset", help="registered dataset name (e.g. zju-leaper, raw-fabric, mvtec-ad) to draw samples from"
    )
    predict_parser.add_argument("--dataset-root", help="dataset root path; falls back to data/<Dataset> if omitted")
    predict_parser.add_argument("--enable-tiling", action="store_true", help="YOLO only: sliding-window prediction with global NMS")
    predict_parser.add_argument("--enable-tta", action="store_true", help="YOLO only: enable native flip/multiscale TTA")
    predict_parser.add_argument("--tile-size", type=int, help="YOLO only: square tile size when --enable-tiling is set")
    predict_parser.add_argument("--tile-overlap", type=float, help="YOLO only: tile overlap in [0, 1)")
    subparsers.add_parser(
        "recipes",
        help="list every model config profile",
    )

    models_parser = subparsers.add_parser(
        "models",
        help=(
            "list every model this project can run, grouped by backend — the CLI view of "
            "`fabric_defect_hub.list_models()`. Not the same list as `adh recipes` (profiles) "
            "or the web UI's dropdown (`catalog.CANONICAL_MODELS`, i.e. what gets published)"
        ),
    )
    models_parser.add_argument(
        "--backend",
        help="only this backend (ultralytics | torchvision | anomalib | dinomaly | moeclip | mambaad)",
    )

    subparsers.add_parser(
        "doctor",
        help=(
            "the availability decision tree: for every model backend, report whether it's "
            "trainable right now on this machine (framework installed + a matching dataset "
            "actually staged under data/<Dataset>), which dataset would be picked, and why — "
            "runnable backends first"
        ),
    )

    export_latex_parser = subparsers.add_parser(
        "export-latex",
        help="export benchmark results to IEEE/CVPR paper-grade LaTeX table code",
    )
    export_latex_parser.add_argument("results_json", help="path to benchmark results JSON file")
    export_latex_parser.add_argument("--output", help="optional output .tex file path")

    predict_parser.add_argument("--split", default="test", choices=("train", "test"), help="dataset split to draw from")
    predict_parser.add_argument("--num-samples", type=int, help="how many dataset samples to run inference on")
    predict_parser.add_argument(
        "--pattern",
        help="ZJU-Leaper pattern/group filter: one of patternN, N, groupN, or a texture name",
    )
    predict_parser.add_argument("--category", help="MVTec-AD category filter")
    predict_parser.add_argument("--seed", type=int, default=0, help="subsampling RNG seed")
    predict_parser.add_argument(
        "--output", help="write predictions as JSON to this path (in addition to stdout)"
    )
    predict_parser.add_argument(
        "--output-dir",
        help="anomalib/dinomaly/moeclip/mambaad only: also persist each sample's pixel-level anomaly map (.npy) under this directory",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help=(
            "score a trained artifact against a registered dataset's ground truth (e.g. tilda-400, "
            "fabric-defects, mvtec-ad) — the CLI-scriptable equivalent of the web Benchmark tab, "
            "for validating a model without a browser"
        ),
    )
    evaluate_parser.add_argument(
        "model",
        help="a model config: a path, a filename stem under --config-dir, or a model keyword — resolved like 'adh train MODEL'",
    )
    evaluate_parser.add_argument(
        "--weights", required=True,
        help="path to a trained/registered artifact to load, e.g. 'adh train's registered_artifact.path output",
    )
    evaluate_parser.add_argument(
        "--config-dir", default="configs/models",
        help="directory searched when 'model' is a filename stem or keyword (default: configs/models)",
    )
    evaluate_parser.add_argument(
        "--backend", choices=_model_backend_choices(),
        help="override backend keyword detection (model.name -> anomalib, model.variant -> ultralytics/torchvision)",
    )
    evaluate_parser.add_argument(
        "--variant",
        help="override which model in the backend's family gets scored; must match what --weights was trained as",
    )
    evaluate_parser.add_argument(
        "--dataset", required=True,
        help="registered dataset name to draw ground-truth samples from (e.g. tilda-400, fabric-defects, zju-leaper)",
    )
    evaluate_parser.add_argument("--dataset-root", help="dataset root path; falls back to data/<Dataset> if omitted")
    evaluate_parser.add_argument("--split", default="test", choices=("train", "test"), help="dataset split to draw from")
    evaluate_parser.add_argument("--num-samples", type=int, help="how many dataset samples to evaluate on")
    evaluate_parser.add_argument(
        "--pattern",
        help="ZJU-Leaper pattern/group filter: one of patternN, N, groupN, or a texture name",
    )
    evaluate_parser.add_argument("--category", help="MVTec-AD category filter")
    evaluate_parser.add_argument("--seed", type=int, default=0, help="subsampling RNG seed")
    evaluate_parser.add_argument(
        "--task", choices=("anomaly", "detection", "segmentation"),
        help="force a specific evaluator instead of the dataset samples' own task",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        help=(
            "anomalib/dinomaly/moeclip/mambaad only: persist each sample's pixel-level anomaly "
            "map (.npy) here, which is what makes pixel_auroc/pixel_aupro/iap computable — "
            "without it an anomaly model is scored on image-level metrics alone"
        ),
    )
    evaluate_parser.add_argument(
        "--cross-domain-patterns",
        help=(
            "ZJU-Leaper pattern-level cross-domain sweep: comma-separated held-out patterns to "
            "evaluate the same weights against (e.g. '5,6,7,8' or 'pattern5,pattern6'), after "
            "scoring the source pattern given by --pattern. Reports each pattern's relative "
            "accuracy drop plus a top-k mean with a bootstrap CI over the patterns. A pattern "
            "that cannot be scored here is reported as skipped, not as a zero drop"
        ),
    )
    evaluate_parser.add_argument(
        "--cross-domain-k", type=int, default=3,
        help="how many held-out patterns the reported mean degradation averages over (default 3)",
    )
    evaluate_parser.add_argument(
        "--cross-domain-mode", choices=("worst", "best"), default="worst",
        help=(
            "which k patterns to average: 'worst' (default) takes the largest drops — a "
            "robustness claim; 'best' takes the smallest. The two report opposite things, so "
            "the chosen mode is echoed back in the output"
        ),
    )
    evaluate_parser.add_argument(
        "--cross-domain-metric",
        help="metric key the degradation is computed over; defaults to the task's headline metric",
    )
    evaluate_parser.add_argument("--enable-tiling", action="store_true", help="YOLO only: sliding-window prediction with global NMS")
    evaluate_parser.add_argument("--enable-tta", action="store_true", help="YOLO only: enable native flip/multiscale TTA")
    evaluate_parser.add_argument("--tile-size", type=int, help="YOLO only: square tile size when --enable-tiling is set")
    evaluate_parser.add_argument("--tile-overlap", type=float, help="YOLO only: tile overlap in [0, 1)")

    train_all_parser = subparsers.add_parser(
        "train-all",
        help=(
            "train every catalogued model in one batch, resumably — the same per-model run "
            "`adh train` performs, looped, with per-model logs and restartable state"
        ),
    )
    train_all_parser.add_argument(
        "--only", nargs="+", metavar="MODEL",
        help="train only these models (catalog keys, e.g. yolov8n PatchCore), instead of all of them",
    )
    train_all_parser.add_argument(
        "--mode", choices=("full", "medium", "few", "test"),
        help=(
            "shot mode for every model. Omit to use each config's own sample counts. "
            "'test' is an 8-image wiring check that does NOT produce usable weights"
        ),
    )
    train_all_parser.add_argument(
        "--dry-run", action="store_true", help="print the plan without training anything",
    )
    train_all_parser.add_argument(
        "--run-id", help="persistent batch identifier (default: a UTC timestamp)",
    )
    train_all_parser.add_argument(
        "--run-root", default="artifacts/training_runs", help="directory for batch state and per-model logs",
    )
    train_all_parser.add_argument(
        "--resume", action="store_true",
        help="continue --run-id, skipping models already recorded as succeeded",
    )
    train_all_parser.add_argument(
        "--config-dir", default="configs/models", help="directory model names resolve against",
    )
    train_all_parser.add_argument(
        "--no-publish", action="store_true",
        help="keep results out of artifacts/models/published/ (use for --mode test runs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "benchmark":
            payload = _run_benchmark(args.config)
        elif args.command == "list":
            payload = _run_list()
        elif args.command == "inventory":
            from fabric_defect_hub.application import platform_inventory
            payload = platform_inventory()
        elif args.command == "recipes":
            payload = _run_recipes()
        elif args.command == "models":
            payload = _run_models(args.backend)
        elif args.command == "doctor":
            payload = _run_doctor()
        elif args.command == "export-latex":
            payload = _run_export_latex(args.results_json, args.output)
        elif args.command == "train":
            payload = _run_train(args)
        elif args.command == "train-all":
            payload = _run_train_all(args)
        elif args.command == "predict":
            payload = _run_predict(args)
        elif args.command == "evaluate":
            payload = _run_evaluate(args)
        else:
            payload = _run_config(args.config, args.backend)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"adh: {exc}") from exc
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_models(backend: str | None) -> Any:
    from fabric_defect_hub.api import list_models

    return list_models(backend)


def _run_recipes() -> dict[str, Any]:
    from fabric_defect_hub.core.registry import list_recipes, get_recipe
    import fabric_defect_hub.recipes  # Ensure registration

    summary = {}
    for recipe_id in list_recipes():
        recipe = get_recipe(recipe_id)
        summary[recipe_id] = recipe.get_recipe_summary()
    return summary


def _run_doctor() -> dict[str, Any]:
    """The availability decision tree, surfaced: for every known model
    backend, whether it's trainable right now on *this* machine, which
    dataset would actually be used, and why — runnable backends first, so
    "what can I train given what's staged here" is one command instead of
    reading tracebacks from a training run that got partway through.
    """

    import importlib

    from fabric_defect_hub.core.availability import backend_is_importable
    from fabric_defect_hub.core.decision import decide_dataset
    from fabric_defect_hub.loader import list_model_backends
    from fabric_defect_hub.training import DEFAULT_DATASET_ROOTS, _BACKEND_TRAINABLE_DATASETS

    importlib.import_module("fabric_defect_hub.datasets")  # populate dataset_capabilities-derived roles

    report: dict[str, dict[str, Any]] = {}
    for backend in list_model_backends():
        framework_installed = backend_is_importable(backend)
        entry: dict[str, Any] = {"framework_installed": framework_installed}

        if backend in _BACKEND_TRAINABLE_DATASETS:
            allowed_set, kind = _BACKEND_TRAINABLE_DATASETS[backend]
            decision = decide_dataset(None, allowed_set, root_map=DEFAULT_DATASET_ROOTS)
            entry["dataset_kind"] = kind
            entry["dataset"] = decision.dataset
            entry["trainable_now"] = framework_installed and decision.runnable
            entry["reason"] = decision.reason if framework_installed else "framework not installed"
        else:
            # Detection backends (ultralytics/torchvision) train on
            # whatever detection dataset the caller's config names; there
            # is no fixed allowed-set to pick a default from here.
            entry["trainable_now"] = framework_installed
            entry["reason"] = "framework installed" if framework_installed else "framework not installed"

        report[backend] = entry

    ordered = sorted(report, key=lambda b: (not report[b]["trainable_now"], b))
    return {"backends": {name: report[name] for name in ordered}}


def _run_export_latex(json_path: str, output_path: str | None = None) -> str:
    from fabric_defect_hub.reporting.latex_generator import generate_latex_table

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results_list = data if isinstance(data, list) else data.get("leaderboard", [])
    latex_code = generate_latex_table(results_list)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_code)
    return latex_code



def _run_config(path: str, backend: str | None) -> Any:
    import yaml

    with open(path) as file:
        raw = yaml.safe_load(file) or {}
    if isinstance(raw, dict) and "runs" in raw:
        return _run_benchmark(path)
    selected = backend or _infer_backend(raw)
    if selected == "ultralytics":
        from fabric_defect_hub.models.ultralytics.pipeline import run_from_yaml
    elif selected == "torchvision":
        from fabric_defect_hub.models.torchvision.pipeline import run_from_yaml
    elif selected == "dinomaly":
        from fabric_defect_hub.models.dinomaly.pipeline import run_from_yaml
    elif selected == "moeclip":
        from fabric_defect_hub.models.moeclip.pipeline import run_from_yaml
    elif selected == "mambaad":
        from fabric_defect_hub.models.mambaad.pipeline import run_from_yaml
    else:
        from fabric_defect_hub.models.anomalib.pipeline import run_from_yaml
    result = run_from_yaml(path)
    return {
        "backend": selected,
        "metrics": result.metrics,
        "trained_artifact": _artifact_dict(result.trained_artifact),
        "registered_artifact": _artifact_dict(result.registered_artifact),
        "exports": [asdict(artifact) for artifact in result.exports],
    }


def _run_benchmark(path: str) -> list[dict[str, Any]]:
    from fabric_defect_hub.benchmark import BenchmarkConfig
    from fabric_defect_hub.core.serialization import experiment_result_to_dict

    config = BenchmarkConfig.from_yaml(path)
    return [experiment_result_to_dict(result) for result in config.run()]


def _run_list() -> Any:
    import importlib

    from fabric_defect_hub.core import registry
    from fabric_defect_hub.loader import import_all_model_backends, list_model_backends

    importlib.import_module("fabric_defect_hub.datasets")
    importlib.import_module("fabric_defect_hub.evaluation")
    importlib.import_module("fabric_defect_hub.profiling")
    import_all_model_backends()

    return {
        "datasets": registry.list_datasets(),
        "model_backends": {"known": list_model_backends(), "available": registry.list_models()},
        "evaluators": registry.list_evaluators(),
        "profilers": registry.list_profilers(),
    }


def _parse_set_overrides(raw_items: list[str]) -> dict[str, Any]:
    """Parse repeated `--set path.to.key=value` CLI args into the dotted-path
    dict `training.apply_raw_overrides` expects. `value` is YAML-parsed so
    `--set train.epochs=50` and `--set train.model_kwargs.pre_trained=false`
    both come through as the right Python type, not the literal string.
    """

    import yaml

    overrides: dict[str, Any] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"--set expects path.to.key=value, got {item!r}")
        path, _, value = item.partition("=")
        path = path.strip()
        if not path:
            raise ValueError(f"--set expects a non-empty dotted path, got {item!r}")
        overrides[path] = yaml.safe_load(value)
    return overrides


def _run_train(args: argparse.Namespace) -> Any:
    # '5' and 'pattern5' both name pattern 5, on every command that takes a
    # pattern — see `datasets/zju_leaper.coerce_pattern`.
    from fabric_defect_hub.datasets.zju_leaper import coerce_pattern
    from fabric_defect_hub.training import DatasetOverrides, find_model_configs, run_train

    if args.list:
        return {
            "config_dir": args.config_dir,
            "configs": [str(path) for path in find_model_configs(args.config_dir)],
        }
    if not args.model:
        raise ValueError("'model' is required unless --list is given")

    overrides = DatasetOverrides(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        test_dataset=args.test_dataset,
        test_dataset_root=args.test_dataset_root,
        mode=args.mode,
        num_samples=args.num_samples,
        val_num_samples=args.val_num_samples,
        use_defect=args.use_defect,
        defect_ratio=args.defect_ratio,
        pattern=coerce_pattern(args.pattern),
        category=args.category,
        seed=args.seed,
    )
    set_overrides = _parse_set_overrides(args.set_overrides)
    if args.enable_tiling:
        set_overrides["data.tiling"] = True
    if args.tile_size is not None:
        set_overrides["data.tile_size"] = [args.tile_size, args.tile_size]
    if args.tile_overlap is not None:
        set_overrides["data.overlap"] = args.tile_overlap
    run = run_train(
        args.model,
        backend=args.backend,
        overrides=overrides,
        config_dir=args.config_dir,
        variant=args.variant,
        set_overrides=set_overrides,
        profile=None if args.no_profile else args.profile,
        publish=not args.no_publish,
    )
    result = run.result
    return {
        "backend": run.backend,
        "resolved_config": run.config_path,
        "resolved_variant": run.variant,
        "metrics": result.metrics,
        "trained_artifact": _artifact_dict(result.trained_artifact),
        "registered_artifact": _artifact_dict(result.registered_artifact),
        "published_path": run.published_path,
        "weight_manifest_path": run.weight_manifest_path,
        "exports": [asdict(artifact) for artifact in result.exports],
    }


def _run_train_all(args: argparse.Namespace) -> Any:
    """Batch counterpart of `_run_train`: the same `run_train` call per
    model, in-process, with resumable state and one log file each.

    In-process rather than by spawning `adh train` subprocesses (which is
    what the removed `tools/train_all_models.py` did) so a batch cannot drift from a
    single run: identical resolution, identical defaults, one code path.
    """

    import contextlib
    import io
    import time

    from fabric_defect_hub.catalog import CANONICAL_MODELS
    from fabric_defect_hub.training import DatasetOverrides, run_train
    from fabric_defect_hub.training_runs import BatchRunTracker, default_run_id

    selected = list(CANONICAL_MODELS)
    if args.only:
        wanted = {name.strip().lower() for name in args.only}
        selected = [model for model in selected if model.key.lower() in wanted]
        missing = wanted - {model.key.lower() for model in selected}
        if missing:
            raise ValueError(
                f"unknown model(s): {', '.join(sorted(missing))}. See `adh models`."
            )

    plan = [
        {"key": model.key, "backend": model.backend, "variant": model.variant, "config": model.config}
        for model in selected
    ]
    if args.dry_run:
        return {"plan": plan, "model_count": len(plan)}

    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id")

    tracker = BatchRunTracker(
        args.run_root, args.run_id or default_run_id(), plan, resume=args.resume
    )
    print(f"Batch state: {tracker.directory}", flush=True)

    results = []
    for model in selected:
        if not tracker.should_run(model.key):
            print(f"SKIP {model.key}: already succeeded in this batch", flush=True)
            results.append({"model": model.key, "status": "skipped"})
            continue

        print(f"\n{'=' * 70}\n>>> {model.key} ({model.backend} / {model.variant})\n{'=' * 70}", flush=True)
        tracker.begin(model.key)
        started = time.monotonic()
        captured = io.StringIO()
        try:
            # Tee rather than swallow: the operator watches the terminal,
            # the log file is what gets attached to a bug report later.
            with contextlib.redirect_stdout(_Tee(sys.stdout, captured)):
                run = run_train(
                    model.key,
                    config_dir=args.config_dir,
                    overrides=DatasetOverrides(mode=args.mode) if args.mode else None,
                    publish=not args.no_publish,
                )
        except Exception as exc:  # one model failing must not abort the batch
            elapsed = time.monotonic() - started
            detail = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {model.key} after {elapsed:.0f}s: {detail}", flush=True)
            tracker.log_path(model.key).write_text(captured.getvalue() + f"\n{detail}\n")
            tracker.finish(model.key, succeeded=False, detail=detail)
            results.append({"model": model.key, "status": "failed", "detail": detail})
            continue

        elapsed = time.monotonic() - started
        detail = run.published_path or (run.result.registered_artifact.path if run.result.registered_artifact else "")
        print(f"OK   {model.key} in {elapsed:.0f}s -> {detail}", flush=True)
        tracker.log_path(model.key).write_text(captured.getvalue())
        tracker.finish(model.key, succeeded=True, detail=str(detail))
        results.append({
            "model": model.key,
            "status": "succeeded",
            "published_path": run.published_path,
            "metrics": run.result.metrics,
        })

    succeeded = sum(1 for r in results if r["status"] in ("succeeded", "skipped"))
    return {
        "batch_state": str(tracker.directory),
        "succeeded": succeeded,
        "total": len(results),
        "results": results,
    }


class _Tee:
    """Write to two streams at once (terminal + in-memory log buffer)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _run_predict(args: argparse.Namespace) -> Any:
    # '5' and 'pattern5' both name pattern 5 — see
    # `datasets/zju_leaper.coerce_pattern`.
    from fabric_defect_hub.datasets.zju_leaper import coerce_pattern
    from fabric_defect_hub.inference.runner import PredictInput, run_predict

    source = PredictInput(
        images=args.images,
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        split=args.split,
        num_samples=args.num_samples,
        pattern=coerce_pattern(args.pattern),
        category=args.category,
        seed=args.seed,
    )
    run = run_predict(
        args.model,
        weights=args.weights,
        source=source,
        backend=args.backend,
        variant=args.variant,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        enable_tiling=args.enable_tiling,
        enable_tta=args.enable_tta,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )
    predictions = [asdict(prediction) for prediction in run.predictions]
    if args.output:
        from fabric_defect_hub.core.serialization import save_predictions

        save_predictions(run.predictions, args.output)
    return {
        "backend": run.backend,
        "resolved_config": run.config_path,
        "variant": run.variant,
        "num_predictions": len(predictions),
        "predictions": predictions,
    }


def _run_evaluate(args: argparse.Namespace) -> Any:
    # '5' and 'pattern5' both name pattern 5 here too, so the flag behaves the
    # same with and without a cross-domain sweep on the same command.
    from fabric_defect_hub.datasets.zju_leaper import coerce_pattern
    from fabric_defect_hub.inference.runner import PredictInput, run_evaluate

    source = PredictInput(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        split=args.split,
        num_samples=args.num_samples,
        pattern=coerce_pattern(args.pattern),
        category=args.category,
        seed=args.seed,
    )
    run = run_evaluate(
        args.model,
        weights=args.weights,
        source=source,
        backend=args.backend,
        variant=args.variant,
        config_dir=args.config_dir,
        task=args.task,
        output_dir=args.output_dir,
        enable_tiling=args.enable_tiling,
        enable_tta=args.enable_tta,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )
    payload: dict[str, Any] = {
        "backend": run.backend,
        "resolved_config": run.config_path,
        "variant": run.variant,
        "sample_count": run.sample_count,
        "metrics": run.metrics,
    }
    if getattr(args, "cross_domain_patterns", None):
        payload["cross_domain"] = _run_cross_domain_sweep(args, run, source)
    return payload


def _run_cross_domain_sweep(args: argparse.Namespace, source_run: Any, source: Any) -> dict[str, Any]:
    """Re-evaluate the same weights on each held-out pattern and reduce the
    per-pattern drops to one reportable number.

    The metric key is resolved once from the source run and reused for every
    pattern: a degradation between two different metrics is not a
    degradation. Patterns that cannot be scored on this machine come back as
    `None` and are listed as skipped rather than counted as a 0% drop.
    """

    from dataclasses import replace

    from fabric_defect_hub.datasets.zju_leaper import coerce_pattern
    from fabric_defect_hub.evaluation.cross_domain import (
        pattern_sweep_degradation,
        resolve_headline_metric,
    )
    from fabric_defect_hub.inference.runner import run_evaluate

    # '5' and 'pattern5' both name pattern 5. The source selection needs the
    # same treatment as the held-out ones: `--pattern 5` reaching the adapter
    # as the string "5" is looked up as a texture name and rejected. Only a
    # string is converted, so a `None` selection stays "the whole benchmark".
    patterns = [coerce_pattern(p.strip()) for p in args.cross_domain_patterns.split(",") if p.strip()]
    if isinstance(source.pattern, str):
        source = replace(source, pattern=coerce_pattern(source.pattern))
    task = args.task or source_run.metrics.get("task") or _sweep_task(source_run.metrics)
    metric_key = resolve_headline_metric(task, source_run.metrics, args.cross_domain_metric)

    def _evaluate_pattern(pattern: str) -> float | None:
        try:
            run = run_evaluate(
                args.model,
                weights=args.weights,
                source=replace(source, pattern=pattern),
                backend=args.backend,
                variant=args.variant,
                config_dir=args.config_dir,
                task=args.task,
                output_dir=args.output_dir,
                enable_tiling=args.enable_tiling,
                enable_tta=args.enable_tta,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
            )
        except (FileNotFoundError, ValueError):
            # Pattern not staged on this machine, or it yielded no scorable
            # samples. Skipping keeps a partially-staged benchmark honest.
            return None
        value = run.metrics.get(metric_key)
        return None if value is None else float(value)

    result = pattern_sweep_degradation(
        acc_src=float(source_run.metrics[metric_key]),
        target_patterns=patterns,
        evaluate_pattern=_evaluate_pattern,
        k=args.cross_domain_k,
        mode=args.cross_domain_mode,
    )
    result["metric"] = metric_key
    result["source_patterns"] = args.pattern
    return result


def _sweep_task(metrics: dict[str, float]) -> str:
    """Infer the task from which metric keys the source run produced, for
    the case where `--task` was left to the dataset samples themselves.
    """

    if "map50" in metrics or "map_50" in metrics or "map" in metrics:
        return "detection"
    if "pixel_auroc" in metrics or "image_auroc" in metrics:
        return "anomaly"
    if "miou" in metrics or "dice" in metrics:
        return "segmentation"
    raise ValueError(
        f"cannot infer the task from metrics {sorted(metrics)}; pass --task explicitly"
    )


def _infer_backend(raw: object) -> str:
    from fabric_defect_hub.training import infer_backend

    return infer_backend(raw)  # type: ignore[arg-type]


def _artifact_dict(artifact) -> dict[str, Any] | None:
    return asdict(artifact) if artifact is not None else None


if __name__ == "__main__":
    raise SystemExit(main())
