"""`ModelAdapter` implementation backed by `torchvision.models.detection`,
covering the full training lifecycle for Faster R-CNN and Mask R-CNN
(ResNet50-FPN, v1 and v2 heads):

    pretrained-weight loading -> train (real loop, see engine.py)
        -> validate (native torchmetrics mAP) -> predict
        -> register/save the trained model -> load it back
        -> export (TorchScript / ONNX best-effort).

torchvision.models.detection replaces mmdetection as this project's
"comprehensive supervised detection comparison" backend: `mmdetection`'s
`mmcv` dependency has no macOS arm64 wheel and doesn't build on Python 3.14
(confirmed by a real install attempt against this project's environment —
see the git history / conversation, not a hypothetical), while
torchvision is pure PyTorch, actively maintained by the PyTorch team, and
installs everywhere torch does. The tradeoff: Cascade R-CNN, DETR and DINO
(which mmdetection ships and torchvision does not) are out of scope for
this backend; Faster R-CNN and Mask R-CNN cover the "classic two-stage
detector" comparison point on their own.

Detection-only + optional instance segmentation: `predict()` fills `boxes`,
`labels`, `scores` (Faster R-CNN) and additionally `masks` (Mask R-CNN).

Requires the `torchvision` extra: `pip install -e ".[torchvision]"`.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.progress import ProgressReporter
from fabric_defect_hub.core.provenance import describe_training
from fabric_defect_hub.core.registry import register_model
from fabric_defect_hub.core.train_config import TrainConfig, resolve_train_config
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.model_statistics import parameter_counts
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter, ModelCapabilities
from fabric_defect_hub.models.torchvision.dataset import (
    SampleDetectionDataset,
    SampleSegmentationDataset,
    build_class_map,
    detection_collate_fn,
    segmentation_collate_fn,
)
from fabric_defect_hub.models.torchvision.presets import (
    build_model,
    build_transforms,
    resolve_variant,
    uses_masks,
    variant_task,
)


@register_model("torchvision")
class TorchvisionAdapter(ModelAdapter):
    """Full-lifecycle wrapper around `torchvision.models.detection.{faster,mask}_rcnn`.

    `name` is a variant ('fasterrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn',
    ...; see `presets.list_supported_variants()`) or a path to a previously
    saved checkpoint (a dict with `state_dict` + `class_map`, as written by
    `register_trained_model`).
    """

    backend = "torchvision"

    def __init__(self, name: str = "fasterrcnn_resnet50_fpn", **kwargs):
        super().__init__(name=name, **kwargs)
        self._model = None
        self._class_map: dict[str, int] | None = None
        self._device = None

    # ------------------------------------------------------------------ #
    # Model / weight loading
    # ------------------------------------------------------------------ #
    def _resolve_device(self, requested: str | None = None):
        import torch

        if requested:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_pretrained(
        self,
        class_names: list[str],
        variant: str | None = None,
        trainable_backbone_layers: int | None = None,
        device: str | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
        offline: bool = False,
    ):
        """Load COCO-pretrained detection weights with the classifier head
        swapped for `class_names` (transfer learning starting point).
        """

        variant = variant or self.name
        from fabric_defect_hub.models.torchvision.presets import variant_task
        is_seg = (variant_task(variant) == "segmentation")
        self._class_map = build_class_map([], class_names=class_names)
        num_classes = 1 if is_seg else (len(class_names) + 1)
        self._device = self._resolve_device(device)
        self._model = build_model(
            variant, num_classes=num_classes, pretrained=True,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size, max_size=max_size,
            offline=offline,
        ).to(self._device)
        return self._model

    def load_scratch(
        self,
        class_names: list[str],
        variant: str | None = None,
        trainable_backbone_layers: int | None = None,
        device: str | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
        offline: bool = False,
    ):
        """Load an ImageNet-pretrained-backbone-only model with a
        random-init head (train from scratch).
        """

        variant = variant or self.name
        from fabric_defect_hub.models.torchvision.presets import variant_task
        is_seg = (variant_task(variant) == "segmentation")
        self._class_map = build_class_map([], class_names=class_names)
        num_classes = 1 if is_seg else (len(class_names) + 1)
        self._device = self._resolve_device(device)
        self._model = build_model(
            variant, num_classes=num_classes, pretrained=False,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size, max_size=max_size,
            backbone_weights=not offline,
        ).to(self._device)
        return self._model

    def load_weights(
        self, path: str, device: str | None = None, allow_unsafe_pickle: bool = False
    ):
        """Load a checkpoint previously produced by `register_trained_model`
        (contains `state_dict`, `class_map`, `variant`).

        Builds the bare architecture with `backbone_weights=False` — every
        parameter is about to be overwritten by `checkpoint['state_dict']`
        anyway, so there is no reason to download ImageNet backbone weights
        first (and every reason not to: it makes reloading a trained model
        needlessly dependent on network access).
        """

        import torch

        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Torchvision checkpoint does not exist: {checkpoint_path}")
        self._device = self._resolve_device(device)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self._device, weights_only=not allow_unsafe_pickle
            )
        except Exception as exc:
            mode = "unsafe pickle mode" if allow_unsafe_pickle else "safe weights-only mode"
            raise ValueError(
                f"Could not load Torchvision checkpoint {checkpoint_path} in {mode}. "
                "Only set allow_unsafe_pickle=True for a checkpoint from a trusted source."
            ) from exc
        class_map, variant, state_dict = _validate_checkpoint(checkpoint, checkpoint_path, self.name)
        self._class_map = class_map
        from fabric_defect_hub.models.torchvision.presets import variant_task
        is_seg = (variant_task(variant) == "segmentation")
        num_classes = 1 if is_seg else (len(self._class_map) + 1)
        self._model = build_model(
            variant, num_classes=num_classes, pretrained=False, backbone_weights=False,
        ).to(self._device)
        try:
            self._model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise ValueError(
                f"Checkpoint {checkpoint_path} is incompatible with Torchvision variant {variant!r} "
                f"and class map {self._class_map!r}."
            ) from exc
        self.name = variant
        return self._model

    @property
    def model(self):
        if self._model is None:
            self.load_pretrained(class_names=["defect"])
        return self._model

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    # Canonical `TrainConfig` field -> this backend's real key. No
    # `image_size` entry: torchvision detection resizes internally via
    # `min_size`/`max_size`, which is not one number (pass them through
    # `backend_specific`).
    TRAIN_CONFIG_KEYS = {
        "epochs": "epochs",
        "lr": "lr",
        "batch_size": "batch_size",
        "device": "device",
        "seed": "seed",
        "num_workers": "num_workers",
        "work_dir": "run_dir",
    }

    def capabilities(self) -> ModelCapabilities:
        """Variant-dependent: the same adapter is a detector as
        `fasterrcnn_resnet50_fpn`, an instance segmenter as
        `maskrcnn_resnet50_fpn`, and a semantic segmenter as
        `unetplusplus_resnet34`.
        """

        task = variant_task(self.name)
        if task == "detect":
            return ModelCapabilities(
                tasks=("detection",),
                prediction_fields=("boxes", "labels", "scores"),
                required_annotations=("boxes", "labels"),
                export_targets=("exported_program", "torchscript", "onnx"),
                # `engine.run_training` wraps the forward in `torch.autocast`
                # with a `GradScaler`; resolved to False off CUDA.
                supports_amp=True,
                # torchvision's detection models take List[Tensor], not a
                # batched NCHW tensor -- the one place in this project where
                # that is true, and the reason the field exists.
                export_input_style="list",
            )
        if task == "instance_segmentation":
            return ModelCapabilities(
                tasks=("instance_segmentation",),
                prediction_fields=("boxes", "labels", "scores", "masks"),
                required_annotations=("masks", "labels"),
                export_targets=("exported_program", "torchscript", "onnx"),
                # `engine.run_training` wraps the forward in `torch.autocast`
                # with a `GradScaler`; resolved to False off CUDA.
                supports_amp=True,
                # Mask R-CNN shares the detection models' List[Tensor] input.
                export_input_style="list",
            )
        return ModelCapabilities(
            tasks=("segmentation",),
            prediction_fields=("masks", "labels", "scores"),
            required_annotations=("masks",),
            export_targets=("exported_program", "torchscript", "onnx"),
            supports_amp=True,
        )

    def train(self, config: dict[str, Any] | TrainConfig) -> Artifact:
        """Run a full fine-tuning job and return an `Artifact` pointing at
        the best (by validation mAP) checkpoint.

        Required:
          * `config['train_samples']` — `Sample` list for training.
          * `config['val_samples']`   — `Sample` list for validation (used
            for mAP-based checkpointing and early stopping).

        Optional (else fall back to fabric presets — see `presets.py`):
          * `class_names`, `weights`, `pretrained`, `trainable_backbone_layers`,
            `epochs`, `batch_size`, `optimizer`, `lr`, `momentum`,
            `weight_decay`, `lr_scheduler`, `step_size`, `gamma`,
            `warmup_epochs`, `grad_clip_norm`, `patience`, `num_workers`,
            `device`, `seed`, `amp`, `resume`, `run_dir`, `save_every_epoch`.
        """

        config = resolve_train_config(config, self.TRAIN_CONFIG_KEYS)

        from torch.utils.data import DataLoader

        from fabric_defect_hub.models.torchvision.engine import run_training
        from fabric_defect_hub.models.torchvision.presets import default_train_kwargs

        cfg = {**default_train_kwargs(self.name), **config}
        train_samples: list[Sample] = cfg["train_samples"]
        val_samples: list[Sample] = cfg["val_samples"]
        class_names = cfg.get("class_names") or ["defect"]
        with_masks = uses_masks(self.name)

        seed = cfg.get("seed")
        if seed is not None:
            _seed_everything(seed)

        weights = cfg.get("weights")
        pretrained = cfg.get("pretrained", True)
        offline = cfg.get("offline", False)
        device_str = cfg.get("device")
        tbl = cfg.get("trainable_backbone_layers")
        min_size = cfg.get("min_size")
        max_size = cfg.get("max_size")

        run_dir = Path(cfg.get("run_dir", "runs/fabric_defect_hub_tv")) / cfg.get("name", "torchvision_exp")
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / "best.pt"
        last_path = run_dir / "last.pt"
        history_path = run_dir / "history.csv"

        resume_state = None
        if cfg.get("resume") and last_path.is_file():
            resume_state = self._load_resume_checkpoint(
                last_path, device=device_str, trainable_backbone_layers=tbl, offline=offline,
            )
        elif weights:
            self.load_weights(weights, device=device_str)
        elif pretrained:
            self.load_pretrained(
                class_names, trainable_backbone_layers=tbl, device=device_str,
                min_size=min_size, max_size=max_size, offline=offline,
            )
        else:
            self.load_scratch(
                class_names, trainable_backbone_layers=tbl, device=device_str,
                min_size=min_size, max_size=max_size, offline=offline,
            )

        from fabric_defect_hub.models.torchvision.presets import variant_task
        task = variant_task(self.name)
        is_seg = (task == "segmentation")

        class_map = self._class_map
        if is_seg:
            train_ds = SampleSegmentationDataset(
                train_samples,
                transforms=build_transforms(
                    train=True,
                    hflip_prob=cfg.get("hflip_prob", 0.5),
                    vflip_prob=cfg.get("vflip_prob", 0.5),
                    color_jitter=cfg.get("color_jitter"),
                ),
            )
            val_ds = SampleSegmentationDataset(
                val_samples,
                transforms=build_transforms(train=False),
            )
            collate = segmentation_collate_fn
        else:
            train_ds = SampleDetectionDataset(
                train_samples, class_map=class_map, with_masks=with_masks,
                transforms=build_transforms(
                    train=True,
                    hflip_prob=cfg.get("hflip_prob", 0.5),
                    vflip_prob=cfg.get("vflip_prob", 0.5),
                    color_jitter=cfg.get("color_jitter"),
                ),
            )
            val_ds = SampleDetectionDataset(
                val_samples, class_map=class_map, with_masks=with_masks,
                transforms=build_transforms(train=False),
            )
            collate = detection_collate_fn

        num_workers = cfg.get("num_workers", 0)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.get("batch_size", 4), shuffle=True,
            num_workers=num_workers, collate_fn=collate,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.get("batch_size", 4), shuffle=False,
            num_workers=num_workers, collate_fn=collate,
        )

        # `run_training` builds the optimizer/scheduler internally; the epoch
        # callback is the only place they surface.
        seen_facilities: dict[str, Any] = {}

        def on_epoch_end(log, optimizer, scheduler, best_map, improved):
            seen_facilities["optimizer"] = optimizer
            seen_facilities["scheduler"] = scheduler
            self._save_checkpoint(
                last_path, class_names, optimizer=optimizer, scheduler=scheduler,
                epoch=log.epoch, best_map=best_map,
            )
            if improved:
                self._save_checkpoint(
                    best_path, class_names, optimizer=optimizer, scheduler=scheduler,
                    epoch=log.epoch, best_map=best_map,
                )
            if cfg.get("save_every_epoch"):
                self._save_checkpoint(run_dir / f"epoch_{log.epoch:03d}.pt", class_names)
            _append_history_row(history_path, log)

        logs, best_map = run_training(
            self.model, train_loader, val_loader, self._device,
            epochs=cfg.get("epochs", 30),
            optimizer_name=cfg.get("optimizer", "sgd"),
            lr=cfg.get("lr", 0.002),
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0.0005),
            lr_scheduler_name=cfg.get("lr_scheduler", "cosine"),
            step_size=cfg.get("step_size", 10),
            gamma=cfg.get("gamma", 0.1),
            warmup_epochs=cfg.get("warmup_epochs", 1),
            grad_clip_norm=cfg.get("grad_clip_norm", 5.0),
            patience=cfg.get("patience", 8),
            with_masks=with_masks,
            amp=cfg.get("amp", False),
            resume_state=resume_state,
            on_epoch_end=on_epoch_end,
            task=task,
        )

        final_path = best_path if best_path.exists() else last_path
        self.load_weights(str(final_path), device=device_str)

        final_metrics = logs[-1].val_metrics if logs else {}
        return Artifact(
            path=str(final_path),
            backend=self.backend,
            metadata={
                "variant": resolve_variant(self.name),
                "run_dir": str(run_dir),
                "last_checkpoint": str(last_path),
                "history_csv": str(history_path),
                "class_names": class_names,
                "epochs_run": len(logs),
                "resumed_from_epoch": resume_state["epoch"] if resume_state else None,
                "best_map": best_map,
                "final_val_metrics": final_metrics,
                # `engine.run_training` resolves `amp` to False off CUDA;
                # mirror that resolution so the record states what ran.
                "training": describe_training(
                    seen_facilities.get("optimizer", cfg.get("optimizer", "sgd")),
                    seen_facilities.get("scheduler", cfg.get("lr_scheduler", "cosine")),
                    precision="amp-mixed"
                    if cfg.get("amp", False) and self._device.type == "cuda"
                    else "fp32",
                ),
                "batch_spec": train_ds.batch_spec().as_run_metadata(),
                **parameter_counts(self.model),
            },
        )

    def _save_checkpoint(
        self,
        path: Path,
        class_names: list[str],
        optimizer=None,
        scheduler=None,
        epoch: int | None = None,
        best_map: float | None = None,
    ) -> None:
        """Write a checkpoint. `optimizer`/`scheduler`/`epoch`/`best_map`,
        when given, make it resumable (see `_load_resume_checkpoint`) — the
        extra keys are additive and never required by `_validate_checkpoint`,
        so plain inference-only checkpoints (no optimizer passed) stay
        loadable exactly as before.
        """

        import torch

        checkpoint = {
            "state_dict": self._model.state_dict(),
            "class_map": self._class_map,
            "class_names": class_names,
            "variant": resolve_variant(self.name),
        }
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()
            checkpoint["scheduler_state"] = scheduler.state_dict() if scheduler is not None else None
            checkpoint["epoch"] = epoch
            checkpoint["best_map"] = best_map
        torch.save(checkpoint, path)

    def _load_resume_checkpoint(
        self,
        path: Path,
        device: str | None = None,
        trainable_backbone_layers: int | None = None,
        offline: bool = False,
    ) -> dict[str, Any]:
        """Load a checkpoint written mid-training by `_save_checkpoint`
        (model + optimizer + scheduler + epoch + best_map) so `train()` can
        continue exactly where a previous run left off, instead of
        restarting fine-tuning from `weights`/`pretrained`/scratch.

        Rebuilds with `backbone_weights=True` (unlike `load_weights`, which
        correctly uses `backbone_weights=False` for plain inference-only
        loading): torchvision's own `_validate_trainable_layers` forces
        *every* backbone stage trainable whenever no pretrained/backbone
        weights are requested, regardless of `trainable_backbone_layers`
        (see `presets.build_model`'s "no effect without backbone weights"
        warning). The original training run built its model via
        `load_pretrained` (`backbone_weights=True`), so resuming with
        `backbone_weights=False` silently produces a different (fully
        unfrozen) set of trainable parameters -- a differently-shaped flat
        optimizer parameter group that `optimizer.load_state_dict` then
        rejects with "doesn't match the size of optimizer's group". The
        immediately-following `load_state_dict(state_dict)` overwrites
        every weight anyway, so the extra backbone download this costs
        (typically a cache hit, since training already fetched it once)
        only buys back the correct trainable-parameter structure.

        Sets `self._model`/`self._class_map`/`self.name` as a side effect
        (like `load_weights`); returns the resume state `engine.run_training`
        expects (`optimizer_state`, `scheduler_state`, `epoch`, `best_map`).
        """

        import torch

        self._device = self._resolve_device(device)
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        class_map, variant, state_dict = _validate_checkpoint(checkpoint, path, self.name)
        if "optimizer_state" not in checkpoint or "epoch" not in checkpoint:
            raise ValueError(
                f"Checkpoint {path} has no optimizer/epoch state to resume from "
                "(it looks like a plain weights checkpoint, not one saved by train())."
            )
        self._class_map = class_map
        num_classes = len(class_map) + 1
        self._model = build_model(
            variant, num_classes=num_classes, pretrained=False, backbone_weights=True,
            trainable_backbone_layers=trainable_backbone_layers, offline=offline,
        ).to(self._device)
        self._model.load_state_dict(state_dict)
        self.name = variant
        return {
            "optimizer_state": checkpoint["optimizer_state"],
            "scheduler_state": checkpoint.get("scheduler_state"),
            "epoch": checkpoint["epoch"],
            "best_map": checkpoint.get("best_map", -1.0),
        }

    # ------------------------------------------------------------------ #
    # Validation / metrics
    # ------------------------------------------------------------------ #
    def validate(
        self, samples: list[Sample], artifact: Artifact | None = None, config: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """Run native torchmetrics mAP evaluation and return a flat metrics
        dict (`map`, `map_50`, `map_75`, `mar_100`, ...; with `segm_*`
        variants too when the loaded model is Mask R-CNN).
        """

        from torch.utils.data import DataLoader

        from fabric_defect_hub.models.torchvision.engine import evaluate as run_evaluate

        if artifact is not None:
            self.load_weights(artifact.path)

        cfg = config or {}
        with_masks = uses_masks(self.name)
        from fabric_defect_hub.models.torchvision.presets import variant_task
        task = variant_task(self.name)
        is_seg = (task == "segmentation")

        if is_seg:
            dataset = SampleSegmentationDataset(
                samples,
                transforms=build_transforms(train=False),
            )
            collate = segmentation_collate_fn
        else:
            dataset = SampleDetectionDataset(
                samples, class_map=self._class_map, with_masks=with_masks,
                transforms=build_transforms(train=False),
            )
            collate = detection_collate_fn

        loader = DataLoader(
            dataset, batch_size=cfg.get("batch_size", 4), shuffle=False,
            num_workers=cfg.get("num_workers", 0), collate_fn=collate,
        )
        return run_evaluate(self.model, loader, self._device, with_masks=with_masks, task=task)

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
        """Run inference over `samples`. `output_dir` is part of the uniform
        `ModelAdapter.predict` signature and unused here: box/mask output goes
        into the returned `Prediction`s, not to disk. `config` overrides `score_threshold`
        (default 0.5), `max_detections` (default 100), `nms_iou_threshold`
        (None = rely solely on the model's own built-in per-class NMS at its
        default IoU; a value applies one more class-aware NMS pass on top,
        via `torchvision.ops.batched_nms`), and `device` (None = keep
        whatever device the loaded weights are already on; the model is
        moved back afterwards, so this does not affect a later `train()`
        call on the same adapter instance). If `artifact` is given its
        weights are loaded first.
        """

        import torch
        from torchvision.ops import batched_nms

        if artifact is not None:
            self.load_weights(artifact.path)

        cfg = config or {}
        score_threshold = cfg.get("score_threshold", 0.5)
        max_detections = cfg.get("max_detections", 100)
        nms_iou_threshold = cfg.get("nms_iou_threshold")
        with_masks = uses_masks(self.name)
        id_to_label = {v: k for k, v in (self._class_map or {}).items()}

        predict_device = self._device
        requested_device = cfg.get("device")
        if requested_device is not None:
            predict_device = self._resolve_device(requested_device)
            self._model.to(predict_device)

        from fabric_defect_hub.models.torchvision.presets import variant_task
        task = variant_task(self.name)
        is_seg = (task == "segmentation")

        if is_seg:
            dataset = SampleSegmentationDataset(
                samples,
                transforms=build_transforms(train=False),
            )
        else:
            dataset = SampleDetectionDataset(
                samples, class_map=self._class_map, with_masks=with_masks,
                transforms=build_transforms(train=False),
            )

        predictions: list[Prediction] = []
        self.model.eval()
        progress = ProgressReporter(
            f"torchvision/{self.name} predict", total=len(samples), unit="img"
        )
        with torch.no_grad(), progress:
            for sample, (image, _target) in zip(samples, dataset):
                progress.update()
                if is_seg:
                    logits = self.model(image.unsqueeze(0).to(predict_device))
                    probs = torch.sigmoid(logits)[0]
                    binary_mask = (probs > score_threshold).squeeze(0).cpu().numpy().tolist()
                    predictions.append(
                        Prediction(sample_id=sample.id, masks=[binary_mask])
                    )
                else:
                    output = self.model([image.to(predict_device)])[0]
                    keep = output["scores"] >= score_threshold
                    boxes = output["boxes"][keep]
                    scores = output["scores"][keep]
                    labels_id = output["labels"][keep]
                    masks_full = output["masks"][keep] if with_masks and "masks" in output else None

                    if nms_iou_threshold is not None and boxes.shape[0] > 0:
                        extra_keep = batched_nms(boxes, scores, labels_id, nms_iou_threshold)
                        boxes, scores, labels_id = boxes[extra_keep], scores[extra_keep], labels_id[extra_keep]
                        if masks_full is not None:
                            masks_full = masks_full[extra_keep]

                    boxes_list = boxes[:max_detections].detach().cpu().tolist()
                    scores_list = scores[:max_detections].detach().cpu().tolist()
                    labels = [
                        id_to_label.get(int(c), str(int(c)))
                        for c in labels_id[:max_detections].detach().cpu().tolist()
                    ]
                    masks = None
                    if masks_full is not None:
                        masks = (
                            (masks_full[:max_detections] > 0.5)
                            .squeeze(1)
                            .detach()
                            .cpu()
                            .numpy()
                            .tolist()
                        )
                    predictions.append(
                        Prediction(sample_id=sample.id, boxes=boxes_list, labels=labels, scores=scores_list, masks=masks)
                    )

        if requested_device is not None:
            self._model.to(self._device)
        return predictions

    # ------------------------------------------------------------------ #
    # Model registry: persist / reload trained models
    # ------------------------------------------------------------------ #
    def register_trained_model(
        self, artifact: Artifact, registry_dir: str, model_name: str | None = None
    ) -> Artifact:
        """Copy a trained checkpoint out of its transient run directory into
        a stable, named location so it can be reloaded later.
        """

        src = Path(artifact.path)
        if not src.exists():
            raise FileNotFoundError(f"cannot register missing weights: {src}")

        registry = Path(registry_dir)
        registry.mkdir(parents=True, exist_ok=True)
        variant = artifact.metadata.get("variant") or resolve_variant(self.name)
        filename = model_name or f"{variant}_{Path(artifact.metadata.get('run_dir', 'run')).name}.pt"
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
        return Artifact(path=str(path), backend=self.backend, metadata={"variant": resolve_variant(self.name)})

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def export(self, artifact: Artifact, target: str, config: dict[str, Any] | None = None) -> ExportedArtifact:
        """`target`: 'exported_program' (`torch.export`, preferred for
        Python 3.14+), 'torchscript' (legacy compatibility), or 'onnx'.
        Export failures raise immediately so callers never receive a
        fake-success path.
        """

        import torch

        self.load_weights(artifact.path)
        self.model.eval()

        run_dir = Path(artifact.metadata.get("run_dir", "runs/fabric_defect_hub_tv"))
        run_dir.mkdir(parents=True, exist_ok=True)

        from fabric_defect_hub.models.torchvision.presets import variant_task
        is_seg = (variant_task(self.name) == "segmentation")

        if target == "exported_program":
            cfg = config or {}
            height, width = cfg.get("input_size", (800, 800))
            if is_seg:
                dummy = torch.rand(1, 3, height, width, device=self._device)
                args = (dummy,)
            else:
                dummy = [torch.rand(3, height, width, device=self._device)]
                args = (dummy,)
            out_path = run_dir / "model.pt2"
            try:
                exported = torch.export.export(self.model, args, strict=False)
                torch.export.save(exported, out_path)
            except Exception as exc:
                out_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Torchvision torch.export failed. The selected model or installed PyTorch "
                    "version may contain operators not yet supported by torch.export."
                ) from exc
            return ExportedArtifact(
                path=str(out_path), target=target,
                metadata={"source_weights": artifact.path, "input_size": [height, width]},
            )

        if target == "torchscript":
            scripted = torch.jit.script(self.model)
            out_path = run_dir / "model.torchscript.pt"
            scripted.save(str(out_path))
            return ExportedArtifact(path=str(out_path), target=target, metadata={"source_weights": artifact.path})

        if target == "onnx":
            cfg = config or {}
            opset = cfg.get("opset", 17)
            out_path = run_dir / "model.onnx"
            if is_seg:
                dummy = torch.rand(1, 3, 512, 512, device=self._device)
                args = (dummy,)
                output_names = ["logits"]
            else:
                dummy = [torch.rand(3, 800, 800, device=self._device)]
                args = (dummy,)
                output_names = ["boxes", "labels", "scores"]
            try:
                torch.onnx.export(
                    self.model, args, str(out_path),
                    input_names=["images"], output_names=output_names,
                    opset_version=opset, dynamic_axes={"images": {0: "batch"}},
                )
            except Exception as exc:
                out_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Torchvision ONNX export failed. The selected model internal layers "
                    "support varies across torch versions; use target='torchscript' or a "
                    "compatible torch/onnx stack."
                ) from exc
            if not out_path.is_file() or out_path.stat().st_size == 0:
                out_path.unlink(missing_ok=True)
                raise RuntimeError("Torchvision ONNX export completed without producing a valid file.")
            return ExportedArtifact(
                path=str(out_path), target=target,
                metadata={"source_weights": artifact.path, "opset": opset},
            )

        raise ValueError(
            f"unsupported export target {target!r}; expected 'exported_program', 'torchscript', or 'onnx'."
        )


def _seed_everything(seed: int) -> None:
    """Seed every RNG this backend's training loop actually draws from:
    `torch`'s global generator (model init, `DataLoader(shuffle=True)`'s
    `torch.randperm`, and `torchvision.transforms.v2`'s random augmentations
    all consume it) and the stdlib `random` module (unused directly here
    today, but seeded for any `extra`-passed training code that might use
    it). NumPy is not seeded: nothing on this training path draws from it.
    """

    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)


def _append_history_row(path: Path, log) -> None:
    is_new = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["epoch", "train_loss", "lr", *sorted(log.val_metrics)])
        writer.writerow(
            [log.epoch, f"{log.train_loss:.6f}", f"{log.lr:.8f}", *(f"{log.val_metrics[k]:.6f}" for k in sorted(log.val_metrics))]
        )


def _validate_checkpoint(checkpoint: object, path: Path, fallback_variant: str) -> tuple[dict[str, int], str, dict]:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} must be a dictionary produced by TorchvisionAdapter.")
    missing = {"state_dict", "class_map"} - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint {path} is missing required keys: {sorted(missing)}.")
    class_map = checkpoint["class_map"]
    if not isinstance(class_map, dict) or not class_map:
        raise ValueError(f"Checkpoint {path} has an invalid non-empty 'class_map'.")
    if not all(isinstance(label, str) and isinstance(index, int) and index > 0 for label, index in class_map.items()):
        raise ValueError(f"Checkpoint {path} class_map must map non-empty labels to positive integer ids.")
    if len(set(class_map.values())) != len(class_map):
        raise ValueError(f"Checkpoint {path} class_map contains duplicate class ids.")
    variant = checkpoint.get("variant", fallback_variant)
    if not isinstance(variant, str):
        raise ValueError(f"Checkpoint {path} has an invalid 'variant'.")
    try:
        variant = resolve_variant(variant)
    except KeyError as exc:
        raise ValueError(f"Checkpoint {path} requests unsupported Torchvision variant {variant!r}.") from exc
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Checkpoint {path} has an invalid empty 'state_dict'.")
    return class_map, variant, state_dict
