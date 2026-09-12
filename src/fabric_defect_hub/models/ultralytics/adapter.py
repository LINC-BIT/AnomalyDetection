"""`ModelAdapter` implementation backed by the `ultralytics` package,
covering the full training lifecycle for YOLOv8n / YOLOv8s / YOLO11n:

    pretrained-weight loading  ->  train  ->  validate (metrics)
        ->  predict  ->  register/save the trained model  ->  load it back
        ->  export (ONNX / TensorRT / ...).

Detection-only: predictions fill `boxes`, `labels`, `scores`.

Design notes
------------
* Config-driven. Every method accepts either a plain kwargs dict (low-level,
  passed through to Ultralytics) or is driven by an `UltralyticsConfig`
  (see `config.py`) through `models/ultralytics/pipeline.py`. Nothing here
  depends on argv.
* Data ingestion. `train`/`validate` accept an existing `data.yaml` *or* a
  `Sample` selection straight out of a `DatasetAdapter`, which is converted
  to YOLO format and symlinked into a temporary staging directory for the
  duration of the call only (see `datasets.yolo_bbox`). No converted copy of
  the dataset is persisted.
* Model registry. Ultralytics writes `best.pt` inside a transient run dir;
  `register_trained_model()` copies it to a stable, named location so a
  trained model can be reloaded later independent of that run dir.

Requires the `ultralytics` extra: `pip install -e ".[ultralytics]"`.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.provenance import describe_training
from fabric_defect_hub.core.registry import register_model
from fabric_defect_hub.core.train_config import TrainConfig, resolve_train_config
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.datasets.yolo_bbox import yolo_staging_dir
from fabric_defect_hub.model_statistics import parameter_counts
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter, ModelCapabilities
from fabric_defect_hub.models.ultralytics.presets import resolve_variant, variant_weights

# Ultralytics DetMetrics.results_dict keys (ultralytics 8.4.x), normalised to
# our flat metric names in `validate()`.
_METRIC_KEY_MAP = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
    "fitness": "fitness",
}


@register_model("ultralytics")
class UltralyticsAdapter(ModelAdapter):
    """Full-lifecycle wrapper around `ultralytics.YOLO`.

    `name` is a variant ('yolov8n', 'yolov8s', 'yolo11n') or any checkpoint
    path Ultralytics can open. The underlying model is loaded lazily and
    cached; call `load_pretrained()` / `load_weights()` to control exactly
    what is in memory.
    """

    backend = "ultralytics"

    # Ultralytics `YOLO.train` argument names that a recipe is allowed to set.
    # A recipe's raw hyperparameter dict may also carry model-constructor knobs
    # and differently-named loss gains; forwarding those to `train()` would
    # raise, so only these curated keys are injected.
    _RECIPE_TRAINER_ARGS = frozenset(
        {
            "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "epochs", "batch", "optimizer",
            "box", "cls", "dfl",  # Ultralytics loss gains
        }
    )

    def __init__(self, name: str = "yolov8n", **kwargs):
        super().__init__(name=name, **kwargs)
        self._model = None
        self._loaded_from: str | None = None

    # ------------------------------------------------------------------ #
    # Model / weight loading
    # ------------------------------------------------------------------ #
    @property
    def model(self):
        """The live `YOLO` object, lazily initialised from `self.name`."""

        if self._model is None:
            self.load_weights(self._default_weights())
        return self._model

    def raw_module(self):
        """`self._model` is a `YOLO` wrapper, not a raw `torch.nn.Module` --
        `.model` (nested) is the actual network `ModelAdapter.raw_module`'s
        callers (e.g. FLOPs counting) need. See that base method's
        docstring."""

        return self._model.model if self._model is not None else None

    def _default_weights(self) -> str:
        """Resolve `self.name` to a loadable weights/arch reference."""

        try:
            return variant_weights(self.name, pretrained=True)
        except KeyError:
            # Not a known variant name — assume it's already a path Ultralytics
            # can open (e.g. a previously trained best.pt).
            return self.name

    def load_weights(self, weights: str):
        """Load an explicit checkpoint / architecture file into memory."""

        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._loaded_from = str(weights)
        return self._model

    def load_pretrained(self, variant: str | None = None, offline: bool = False):
        """Load the COCO-pretrained checkpoint for a variant (transfer
        learning starting point). Defaults to this adapter's variant.
        """

        variant = variant or self.name
        weights = variant_weights(variant, pretrained=True)
        if offline:
            from fabric_defect_hub.core.preflight import require_cached_weight

            weights = str(require_cached_weight(weights, self.backend))
        return self.load_weights(weights)

    def load_scratch(self, variant: str | None = None):
        """Load the architecture spec with random init (train from scratch)."""

        variant = variant or self.name
        return self.load_weights(variant_weights(variant, pretrained=False))

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    # Canonical `TrainConfig` field -> Ultralytics' real `YOLO.train` argument
    # name. Ultralytics has no separate iteration budget, so `max_iters` has no
    # entry and is dropped rather than mistranslated.
    TRAIN_CONFIG_KEYS = {
        "epochs": "epochs",
        "lr": "lr0",
        "batch_size": "batch",
        "image_size": "imgsz",
        "device": "device",
        "seed": "seed",
        "num_workers": "workers",
        "work_dir": "project",
    }

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tasks=("detection",),
            prediction_fields=("boxes", "labels", "scores"),
            required_annotations=("boxes", "labels"),
            # `YOLO.export(format=...)` values verified against ultralytics 8.4.x.
            export_targets=("onnx", "engine", "torchscript", "openvino", "coreml", "tflite", "paddle", "ncnn"),
            # Ultralytics' trainer takes a real `amp` argument (default True).
            supports_amp=True,
        )

    def train(self, config: dict[str, Any] | TrainConfig) -> Artifact:
        """Run a full training job and return an `Artifact` pointing at the
        resulting `best.pt`.

        Data source (mutually exclusive):
          * `config['data']`      — path to an existing YOLO `data.yaml`.
          * `config['samples']`   — a `Sample` list, or `{'train': [...],
                                     'val': [...]}`, from a `DatasetAdapter`.
                                     Converted+staged on the fly.

        Weight initialisation (optional, else uses the variant's pretrained
        checkpoint):
          * `config['weights']`   — explicit checkpoint to fine-tune from.
          * `config['pretrained']` (bool) — False to train from scratch.

        Every remaining key is passed straight to `YOLO.train(**kwargs)`
        (e.g. epochs, imgsz, batch, lr0, project, name, resume). On success
        the trained model is left loaded in this adapter.
        """

        config = resolve_train_config(config, self.TRAIN_CONFIG_KEYS)

        cfg = dict(config)
        samples = cfg.pop("samples", None)
        class_names = cfg.pop("class_names", None)
        sample_summary = cfg.pop("sample_summary", None)
        tiling = cfg.pop("tiling", False)
        tile_size = cfg.pop("tile_size", (256, 256))
        tile_overlap = cfg.pop("tile_overlap", 0.25)
        data_yaml = cfg.pop("data", None)
        weights = cfg.pop("weights", None)
        pretrained = cfg.pop("pretrained", True)
        offline = cfg.pop("offline", False)

        if samples is None and data_yaml is None:
            raise ValueError(
                "UltralyticsAdapter.train: provide either config['data'] "
                "(a data.yaml path) or config['samples'] (a DatasetAdapter selection)."
            )

        # Fold in a recipe's trainer hyperparameters as defaults (anything the
        # caller set explicitly in `config` wins). `_recipe_hparams` is attached
        # by `recipes.apply.apply_recipe_to_training` when a recipe is active.
        recipe_hparams = getattr(self, "_recipe_hparams", None)
        if recipe_hparams:
            from fabric_defect_hub.recipes.apply import recipe_trainer_overrides

            for key, value in recipe_trainer_overrides(recipe_hparams, self._RECIPE_TRAINER_ARGS).items():
                cfg.setdefault(key, value)

        # Initialise weights before training.
        if weights is not None:
            self.load_weights(weights)
        elif pretrained:
            self.load_pretrained(offline=offline)
        else:
            self.load_scratch()

        if samples is not None:
            with yolo_staging_dir(samples, class_names=class_names, tiling=tiling, tile_size=tuple(tile_size), overlap=tile_overlap) as staged_yaml:
                results = self.model.train(data=str(staged_yaml), **cfg)
        else:
            results = self.model.train(data=data_yaml, **cfg)

        save_dir = Path(results.save_dir)
        best = save_dir / "weights" / "best.pt"
        last = save_dir / "weights" / "last.pt"
        best_path = best if best.exists() else last

        # Re-load the trained best weights so subsequent predict/val/export use them.
        if best_path.exists():
            self.load_weights(str(best_path))

        return Artifact(
            path=str(best_path),
            backend=self.backend,
            metadata={
                "variant": self._safe_variant(),
                "run_dir": str(save_dir),
                "last_weights": str(last) if last.exists() else None,
                "train_kwargs": cfg,
                "training": self._training_facilities(cfg),
                "results_csv": str(save_dir / "results.csv"),
                "sample_summary": sample_summary,
                **parameter_counts(self.model.model),
            },
        )

    def resume(self, last_weights: str, config: dict[str, Any] | None = None) -> Artifact:
        """Resume an interrupted run from a `last.pt` checkpoint.

        Ultralytics restores optimizer state, epoch counter, and the original
        training args from the checkpoint itself, so `config` only needs
        overrides you want to change (usually none).
        """

        self.load_weights(last_weights)
        cfg = dict(config or {})
        cfg["resume"] = True
        results = self.model.train(**cfg)
        save_dir = Path(results.save_dir)
        best = save_dir / "weights" / "best.pt"
        if best.exists():
            self.load_weights(str(best))
        return Artifact(
            path=str(best),
            backend=self.backend,
            metadata={
                "variant": self._safe_variant(),
                "run_dir": str(save_dir),
                "resumed_from": last_weights,
                "training": self._training_facilities(cfg),
            },
        )

    def _training_facilities(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Ultralytics' `optimizer="auto"` picks the real class at runtime,
        so read it off the finished trainer; fall back to the requested
        config if the trainer doesn't expose it.
        """

        trainer = getattr(self.model, "trainer", None)
        optimizer = getattr(trainer, "optimizer", None) or cfg.get("optimizer", "auto")
        scheduler = getattr(trainer, "scheduler", None) or (
            "cosine" if cfg.get("cos_lr") else "linear"
        )
        amp = getattr(trainer, "amp", cfg.get("amp", True))
        return describe_training(
            optimizer, scheduler, precision="amp-mixed" if amp else "fp32"
        )

    # ------------------------------------------------------------------ #
    # Validation / metrics
    # ------------------------------------------------------------------ #
    def validate(self, artifact: Artifact | None = None, config: dict[str, Any] | None = None) -> dict[str, float]:
        """Run Ultralytics' native detection validation and return a flat
        metrics dict (`precision`, `recall`, `map50`, `map50_95`, `fitness`).

        `config` may carry `data` (a data.yaml) or `samples` (a
        `DatasetAdapter` selection to stage); everything else is passed to
        `YOLO.val(**kwargs)`.
        """

        if artifact is not None:
            self.load_weights(artifact.path)

        cfg = dict(config or {})
        samples = cfg.pop("samples", None)
        class_names = cfg.pop("class_names", None)
        cfg.pop("sample_summary", None)
        cfg.pop("tiling", None)
        cfg.pop("tile_size", None)
        cfg.pop("tile_overlap", None)

        if samples is not None:
            with yolo_staging_dir(samples, class_names=class_names) as staged_yaml:
                metrics = self.model.val(data=str(staged_yaml), **cfg)
        else:
            metrics = self.model.val(**cfg)

        return self._normalise_metrics(metrics)

    @staticmethod
    def _normalise_metrics(metrics) -> dict[str, float]:
        results = getattr(metrics, "results_dict", None) or {}
        flat = {
            _METRIC_KEY_MAP[k]: float(v)
            for k, v in results.items()
            if k in _METRIC_KEY_MAP
        }
        # Per-class mAP50-95 if available (single-class fabric -> length 1).
        maps = getattr(metrics, "maps", None)
        if maps is not None:
            try:
                flat["map50_95_mean"] = float(sum(maps) / len(maps))
            except (TypeError, ZeroDivisionError):
                pass
        if not flat:
            available = sorted(results) if isinstance(results, dict) else []
            raise RuntimeError(
                "Ultralytics validation returned no recognized metrics; "
                f"available results_dict keys: {available or '<none>'}."
            )
        return flat

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(
        self,
        samples: list[Sample],
        artifact: Artifact | None = None,
        output_dir: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Run inference over `samples`, returning one unified `Prediction`
        each (`boxes` as pixel xyxy, `labels`, `scores`).

        `output_dir` is part of the uniform `ModelAdapter.predict` signature
        and unused here: a detector's output is boxes, not per-sample image
        files. `config` overrides inference defaults (conf, iou, imgsz,
        max_det, device, augment, ...). If `artifact` is given its weights are loaded
        first; otherwise whatever is currently loaded (or the variant's
        pretrained checkpoint) is used.
        """

        if artifact is not None and self._loaded_from != artifact.path:
            self.load_weights(artifact.path)

        cfg = dict(config or {})
        if cfg.pop("tta_mode", None) == "flip_multiscale":
            cfg["augment"] = True
        tiling = cfg.pop("tiling", False)
        tile_size = tuple(cfg.pop("tile_size", (256, 256)))
        tile_overlap = float(cfg.pop("tile_overlap", 0.25))
        cfg.setdefault("verbose", False)
        if tiling:
            return self._predict_tiled(samples, cfg, tile_size=tile_size, overlap=tile_overlap)
        image_paths = [s.image_path for s in samples]
        results = self.model.predict(source=image_paths, **cfg)

        predictions: list[Prediction] = []
        for sample, result in zip(samples, results):
            predictions.append(self._result_to_prediction(sample, result))
        return predictions

    def _predict_tiled(self, samples: list[Sample], cfg: dict[str, Any], *, tile_size: tuple[int, int], overlap: float) -> list[Prediction]:
        """Predict overlapping tiles, restore their coordinates, then globally NMS."""
        from PIL import Image
        from fabric_defect_hub.datasets.yolo_bbox import _tile_starts

        tile_w, tile_h = tile_size
        if tile_w <= 0 or tile_h <= 0 or not 0 <= overlap < 1:
            raise ValueError("tile_size must be positive and tile_overlap must be in [0, 1)")
        step_x, step_y = max(1, round(tile_w * (1 - overlap))), max(1, round(tile_h * (1 - overlap)))
        grouped: dict[str, list[tuple[list[float], str, float]]] = {sample.id: [] for sample in samples}
        with tempfile.TemporaryDirectory(prefix="fdh_yolo_predict_tiles_") as directory:
            tile_paths, origins = [], []
            for sample in samples:
                with Image.open(sample.image_path) as image:
                    for top in _tile_starts(image.height, tile_h, step_y):
                        for left in _tile_starts(image.width, tile_w, step_x):
                            path = Path(directory) / f"{sample.id}_{left}_{top}.jpg"
                            image.crop((left, top, min(left + tile_w, image.width), min(top + tile_h, image.height))).save(path)
                            tile_paths.append(str(path))
                            origins.append((sample.id, left, top))
            results = self.model.predict(source=tile_paths, **cfg)
            for result, (sample_id, left, top) in zip(results, origins):
                boxes_obj = result.boxes
                if boxes_obj is None:
                    continue
                for box, score, class_id in zip(boxes_obj.xyxy.tolist(), boxes_obj.conf.tolist(), boxes_obj.cls.tolist()):
                    grouped[sample_id].append(([box[0] + left, box[1] + top, box[2] + left, box[3] + top], result.names[int(class_id)], score))
        threshold = float(cfg.get("iou", 0.7))
        return [self._merged_tile_prediction(sample, grouped[sample.id], threshold) for sample in samples]

    @staticmethod
    def _merged_tile_prediction(sample: Sample, candidates: list[tuple[list[float], str, float]], iou_threshold: float) -> Prediction:
        kept = []
        for candidate in sorted(candidates, key=lambda item: item[2], reverse=True):
            if all(candidate[1] != prior[1] or _box_iou(candidate[0], prior[0]) < iou_threshold for prior in kept):
                kept.append(candidate)
        return Prediction(sample_id=sample.id, boxes=[item[0] for item in kept], labels=[item[1] for item in kept], scores=[item[2] for item in kept])

    @staticmethod
    def _result_to_prediction(sample: Sample, result) -> Prediction:
        names = result.names
        boxes_obj = result.boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return Prediction(sample_id=sample.id, boxes=[], labels=[], scores=[])
        boxes = boxes_obj.xyxy.tolist()
        scores = boxes_obj.conf.tolist()
        class_ids = boxes_obj.cls.tolist()
        labels = [names[int(c)] for c in class_ids]
        return Prediction(sample_id=sample.id, boxes=boxes, labels=labels, scores=scores)

    # ------------------------------------------------------------------ #
    # Model registry: persist / reload trained models
    # ------------------------------------------------------------------ #
    def register_trained_model(
        self, artifact: Artifact, registry_dir: str, model_name: str | None = None
    ) -> Artifact:
        """Copy a trained `best.pt` out of its transient run directory into a
        stable, named location so it can be reloaded later.

        Returns a new `Artifact` pointing at the registered copy.
        """

        src = Path(artifact.path)
        if not src.exists():
            raise FileNotFoundError(f"cannot register missing weights: {src}")

        registry = Path(registry_dir)
        registry.mkdir(parents=True, exist_ok=True)
        variant = artifact.metadata.get("variant") or self._safe_variant()
        filename = model_name or f"{variant}_{src.parent.parent.name}.pt"
        dst = registry / filename
        shutil.copy2(src, dst)

        metadata = dict(artifact.metadata)
        metadata["registered_from"] = str(src)
        return Artifact(path=str(dst), backend=self.backend, metadata=metadata)

    def load_trained_model(self, artifact_or_path: Artifact | str) -> Artifact:
        """Load a previously registered/trained model back into this adapter."""

        path = artifact_or_path.path if isinstance(artifact_or_path, Artifact) else artifact_or_path
        self.load_weights(path)
        if isinstance(artifact_or_path, Artifact):
            return artifact_or_path
        return Artifact(path=str(path), backend=self.backend, metadata={"variant": self._safe_variant()})

    def unload(self) -> None:
        """Release the resident Ultralytics model for an interactive session."""

        self._model = None
        self._loaded_from = None

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def export(self, artifact: Artifact, target: str, config: dict[str, Any] | None = None) -> ExportedArtifact:
        """Export a trained model to a deployment format.

        `target` is an Ultralytics export format string ('onnx', 'engine',
        'torchscript', 'openvino', ...). `config` carries export knobs
        (half, dynamic, simplify, opset, imgsz).
        """

        self.load_weights(artifact.path)
        cfg = dict(config or {})
        cfg["format"] = target
        exported_path = self.model.export(**cfg)
        return ExportedArtifact(
            path=str(exported_path),
            target=target,
            metadata={"source_weights": artifact.path, "export_kwargs": cfg},
        )

    def export_many(
        self, artifact: Artifact, targets: list[str], config: dict[str, Any] | None = None
    ) -> list[ExportedArtifact]:
        """Export to several formats in one call (each independently)."""

        return [self.export(artifact, fmt, config=config) for fmt in targets]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _safe_variant(self) -> str:
        try:
            return resolve_variant(self.name)
        except KeyError:
            return str(self.name)


def _box_iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0
