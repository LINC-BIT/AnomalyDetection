"""`ModelAdapter` implementation backed by the `anomalib` package.

Covers the anomalib models the README commits to — PatchCore, PaDiM, RD4AD,
EfficientAD, SuperSimpleNet, WinCLIP — via `presets.py`, which resolves
README/paper names to anomalib's actual class names and supplies
fabric-tailored default constructor kwargs (see `presets.py` for why each
default was picked). WinCLIP is CLIP-based and needs no gradient training
(zero-shot by default, few-shot when `k_shot > 0`); it flows through the
same one-pass `engine.fit` path as PatchCore/PaDiM.

Anomaly-only: `predict()` always fills `anomaly_score` (image-level) and
can optionally persist pixel-level `anomaly_map`s (see its docstring) for
`evaluation.anomaly.AnomalyEvaluator`'s pixel AUROC/AUPRO.

Requires the `anomalib` extra: `pip install -e ".[anomalib]"`.
"""

from __future__ import annotations

import inspect
import shutil
import tempfile
from types import MethodType
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.provenance import describe_training
from fabric_defect_hub.core.registry import register_model
from fabric_defect_hub.core.train_config import TrainConfig, resolve_train_config
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.datasets.anomalib_folder import anomalib_folder_staging_dir
from fabric_defect_hub.model_statistics import parameter_counts
from fabric_defect_hub.models.anomalib.presets import (
    IMAGE_LEVEL_ONLY,
    default_model_kwargs,
    resolve_model_class,
    resolve_model_class_name,
    prompt_class_for_samples,
)
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter, ModelCapabilities


@register_model("anomalib")
class AnomalibAdapter(ModelAdapter):
    """Wraps an `anomalib.models` class.

    `name` may be a README/paper alias ('PatchCore', 'RD4AD', 'EfficientAD',
    'SuperSimpleNet', 'PaDiM', 'WinCLIP' — case-insensitive) or the literal
    anomalib class name ('Patchcore', 'ReverseDistillation', 'WinClip', ...).
    See `presets.list_supported_variants()` for the full set.
    """

    backend = "anomalib"

    def __init__(self, name: str = "PatchCore", **kwargs):
        super().__init__(name=name, **kwargs)
        # Fail fast on an unknown model name rather than at train() time.
        self.resolved_class_name = resolve_model_class_name(name)
        self._model = None
        self._loaded_path: str | None = None

    def _model_cls(self):
        return resolve_model_class(self.name)

    # Canonical `TrainConfig` field -> this backend's real key. Anomalib's
    # run length and device live inside Lightning's `engine_kwargs`, not as
    # flat keys, so only what is genuinely flat here is mapped.
    TRAIN_CONFIG_KEYS = {
        "num_workers": "num_workers",
    }

    def capabilities(self) -> ModelCapabilities:
        # Per-model, not per-backend: most anomalib models return a pixel-level
        # `anomaly_map` alongside the image score, but some (GANomaly) return
        # only the score -- see `presets.IMAGE_LEVEL_ONLY`. Declaring
        # `anomaly_map` for those would tell an evaluator that pixel
        # AUROC/AUPRO is computable when it is not, which is exactly the
        # implicit-convention problem `ModelCapabilities` exists to remove.
        fields: tuple[str, ...] = ("anomaly_score", "labels")
        if self.resolved_class_name not in IMAGE_LEVEL_ONLY:
            fields = ("anomaly_score", "anomaly_map", "labels")

        return ModelCapabilities(
            tasks=("anomaly",),
            prediction_fields=fields,
            # One-class training needs normal images only; no labels required.
            required_annotations=(),
            export_targets=("onnx", "openvino", "torch"),
            # Lightning's `precision` is configurable via `trainer` kwargs, but
            # this project has not verified a mixed-precision run, so it is not
            # advertised. Verify before flipping this.
            supports_amp=False,
        )

    def train(self, config: dict[str, Any] | TrainConfig) -> Artifact:
        """Two ways to point this at data:

        - `config['datamodule_kwargs']`: passed straight through to
          `anomalib.data.Folder(**datamodule_kwargs)` — use this if you
          already have an MVTec-style dataset on disk.
        - `config['train_samples']` + `config['test_samples']`: raw
          `Sample` lists straight out of `DatasetAdapter.load_samples()`
          (`train_samples` all-normal, `test_samples` mixed — e.g.
          `ZJULeaperDataset(..., use_defect=False)` for the former and
          `ZJULeaperDataset(..., use_defect=True, defect_ratio=...)` for
          the latter). These are symlinked into a temporary MVTec-style
          folder for the duration of this call only (see
          `datasets.anomalib_folder`); nothing is left on disk afterwards.

        Other keys: `model_kwargs` (merged over the fabric-tailored preset
        for this model — caller keys win), `engine_kwargs` (passed to
        `Engine`).
        """

        config = resolve_train_config(config, self.TRAIN_CONFIG_KEYS)

        model_kwargs = {**default_model_kwargs(self.name), **config.get("model_kwargs", {})}
        if self.resolved_class_name == "WinClip" and "prompt_class" not in model_kwargs:
            model_kwargs["class_name"] = prompt_class_for_samples(config.get("train_samples") or config.get("test_samples"))
        self._validate_model_kwargs(model_kwargs)
        if self._is_zero_shot_winclip(model_kwargs):
            return self._zero_shot_winclip_artifact(model_kwargs, config)

        from anomalib.data import Folder
        from anomalib.engine import Engine

        model = self._model_cls()(**model_kwargs)
        engine = Engine(**config.get("engine_kwargs", {}))

        train_samples = config.get("train_samples")
        test_samples = config.get("test_samples")
        if train_samples is not None and test_samples is not None:
            # num_workers=0: the staged directory is symlinks into a
            # tempfile.mkdtemp() dir that lives only for this `with` block;
            # worker subprocesses opening it introduce a shutdown race with
            # no benefit at the sample counts this path is meant for
            # (few-shot / low-shot). Override via config['num_workers'] if
            # you really want parallel loading for a large staged set.
            datamodule_kwargs = {"num_workers": config.get("num_workers", 0)}
            with anomalib_folder_staging_dir(train_samples, test_samples) as layout:
                datamodule = Folder(
                    name=self.resolved_class_name.lower(), **layout.as_kwargs(), **datamodule_kwargs
                )
                engine.fit(model=model, datamodule=datamodule)
        else:
            datamodule = Folder(**config["datamodule_kwargs"])
            engine.fit(model=model, datamodule=datamodule)

        ckpt_path = engine.trainer.checkpoint_callback.best_model_path

        # Lightning owns optimizer construction; an empty list is a real
        # state (PatchCore fits a memory bank, no gradient step), not a bug.
        optimizers = getattr(engine.trainer, "optimizers", None) or []
        schedulers = [
            scheduler_config.scheduler
            for scheduler_config in getattr(engine.trainer, "lr_scheduler_configs", None) or []
        ]
        training = describe_training(
            optimizers[0] if optimizers else "none (non-gradient fit)",
            schedulers[0] if schedulers else None,
            precision=str(getattr(engine.trainer, "precision", "32-true")),
        )

        return Artifact(
            path=str(ckpt_path),
            backend=self.backend,
            metadata={
                "model_class": self.resolved_class_name,
                "model_kwargs": model_kwargs,
                "trusted": True,
                "training": training,
                **parameter_counts(model),
            },
        )

    def _is_zero_shot_winclip(self, model_kwargs: dict[str, Any]) -> bool:
        return self.resolved_class_name == "WinClip" and int(model_kwargs.get("k_shot", 0)) == 0

    def _zero_shot_winclip_artifact(
        self, model_kwargs: dict[str, Any], config: dict[str, Any]
    ) -> Artifact:
        """Persist a reconstructable handle for parameter-free WinCLIP."""

        engine_kwargs = config.get("engine_kwargs", {})
        root = Path(engine_kwargs.get("default_root_dir") or tempfile.mkdtemp(prefix="fdh_winclip_"))
        root.mkdir(parents=True, exist_ok=True)
        path = root / "winclip_zero_shot.ckpt"
        path.write_text("WinCLIP zero-shot artifact; rebuild from metadata.\n")
        return Artifact(
            path=str(path),
            backend=self.backend,
            metadata={
                "model_class": self.resolved_class_name,
                "model_kwargs": dict(model_kwargs),
                "trusted": True,
                "zero_shot": True,
            },
        )

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]) -> None:
        """Catch fabric-specific misconfigurations before they surface as an
        opaque failure deep inside a Lightning training loop.
        """

        if self.resolved_class_name == "EfficientAd":
            imagenet_dir = model_kwargs.get("imagenet_dir")
            if not imagenet_dir or not Path(imagenet_dir).exists():
                raise ValueError(
                    "EfficientAD requires model_kwargs['imagenet_dir'] to point at an "
                    "existing natural-image dataset (used for its regularization loss); "
                    f"got {imagenet_dir!r}. There is no fabric-appropriate default — "
                    "pass a real path, e.g. an Imagenette download."
                )

        if self.resolved_class_name == "Draem":
            # Unlike EfficientAD, anomalib's Draem does not fail on a missing
            # texture source: it calls `download_and_extract(dtd_dir,
            # DTD_DOWNLOAD_INFO)` and fetches ~600MB mid-training. That is a
            # surprise on a metered or offline training box, so require the
            # directory to exist and make the download an explicit opt-in
            # rather than a side effect of starting a run.
            dtd_dir = model_kwargs.get("dtd_dir")
            allow_download = bool(model_kwargs.get("allow_dtd_download", False))
            if not allow_download and (not dtd_dir or not Path(dtd_dir).is_dir()):
                raise ValueError(
                    f"DRAEM needs a DTD texture directory to synthesize anomalies; "
                    f"model_kwargs['dtd_dir']={dtd_dir!r} is not an existing directory. "
                    "Stage DTD at datasets/general/DTD, "
                    "or set train.model_kwargs.dtd_dir to a writable path and pass "
                    "train.model_kwargs.allow_dtd_download: true to let anomalib "
                    "download ~600MB there itself."
                )

        # `allow_dtd_download` is this adapter's own opt-in flag, not an
        # anomalib constructor argument — strip it before the kwargs reach the
        # model, or anomalib rejects it as unknown.
        model_kwargs.pop("allow_dtd_download", None)

    def predict(
        self,
        samples: list[Sample],
        artifact: Artifact | None = None,
        output_dir: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Always fills `anomaly_score`. Pass `output_dir` to also persist
        each sample's pixel-level `anomaly_map` as a `.npy` file there and
        fill `Prediction.anomaly_map` with its path — needed for
        `evaluation.anomaly.AnomalyEvaluator`'s pixel-level metrics
        (pixel AUROC/AUPRO). Omit it to skip that disk write when you only
        need image-level scores.
        """

        if not artifact.metadata.get("trusted", False):
            raise ValueError(
                "Refusing to load an untrusted Anomalib checkpoint. Use load_trained_model(..., "
                "allow_unsafe_checkpoint=True) only for a checkpoint from a trusted source."
            )

        import numpy as np
        from anomalib.data import ImageBatch, PredictDataset
        from anomalib.engine import Engine
        from lightning.pytorch import Trainer
        from torch.utils.data import DataLoader

        model = self._load_artifact(artifact)

        maps_dir = None
        if output_dir is not None:
            maps_dir = Path(output_dir)
            maps_dir.mkdir(parents=True, exist_ok=True)
        # Keep Lightning's predict logs beside the requested runtime output,
        # rather than allowing its default ``results/`` directory to appear
        # at the repository root during an interactive UI prediction.
        engine_root = maps_dir.parent if maps_dir is not None else Path(artifact.path).parent
        engine_kwargs = _prediction_engine_kwargs(config)
        engine = Engine(default_root_dir=str(engine_root), **engine_kwargs)

        predictions = []
        for sample in samples:
            dataset = PredictDataset(path=sample.image_path)
            if artifact.metadata.get("zero_shot", False):
                # Anomalib's Engine routes WinCLIP through validation, but
                # WinCLIP intentionally has no val_dataloader. Supplying an
                # explicit prediction loader keeps Lightning on predict_step.
                trainer = Trainer(
                    default_root_dir=str(engine_root),
                    logger=False,
                    enable_checkpointing=False,
                    **engine_kwargs,
                )
                batches = trainer.predict(
                    model=model,
                    dataloaders=DataLoader(dataset, batch_size=1, collate_fn=ImageBatch.collate),
                ) or []
            else:
                batches = engine.predict(model=model, dataset=dataset) or []
            if not batches:
                raise RuntimeError(
                    f"Anomalib produced no prediction output for sample {sample.id!r} "
                    f"({sample.image_path})."
                )
            score = None
            predicted_label = None
            anomaly_map_path = None
            if batches:
                batch = batches[0]
                if batch.pred_score is not None:
                    score = float(batch.pred_score[0])
                raw_label = getattr(batch, "pred_label", None)
                if raw_label is not None:
                    value = raw_label[0]
                    value = value.item() if hasattr(value, "item") else value
                    predicted_label = "anomaly" if int(value) else "normal"
                raw_map = getattr(batch, "anomaly_map", None)
                if maps_dir is not None and raw_map is not None:
                    arr = raw_map[0]
                    arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
                    map_path = maps_dir / f"{sample.id}.npy"
                    # `sample.id` is an opaque identifier, not guaranteed to be a
                    # single path segment — datasets like MVTecADDataset use
                    # "category/defect_type/stem" ids, which need their own
                    # subdirectories created before `np.save` can write there.
                    map_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(map_path, np.squeeze(arr))
                    anomaly_map_path = str(map_path)
            if score is None:
                raise RuntimeError(
                    f"Anomalib prediction for sample {sample.id!r} has no anomaly score."
                )
            predictions.append(
                Prediction(
                    sample_id=sample.id,
                    labels=[predicted_label] if predicted_label is not None else None,
                    anomaly_score=score,
                    anomaly_map=anomaly_map_path,
                )
            )
        return predictions
    def export(
        self, artifact: Artifact, target: str, config: dict[str, Any] | None = None
    ) -> ExportedArtifact:
        """`target` is an `anomalib.deploy.ExportType` value, e.g. 'onnx', 'openvino'."""

        if not artifact.metadata.get("trusted", False):
            raise ValueError("Refusing to export an untrusted Anomalib checkpoint.")

        from anomalib.engine import Engine

        model = self._load_artifact(artifact)
        engine = Engine()
        exported_path = engine.export(model=model, export_type=target)
        return ExportedArtifact(path=str(exported_path), target=target)

    # ------------------------------------------------------------------ #
    # Model registry: persist / reload trained models
    # ------------------------------------------------------------------ #
    def register_trained_model(
        self, artifact: Artifact, registry_dir: str, model_name: str | None = None
    ) -> Artifact:
        """Copy a trained checkpoint out of `Engine`'s versioned working
        directory (`<default_root_dir>/<ModelClass>/<name>/v{N}/weights/
        lightning/model.ckpt`) into a stable, named location so it can be
        reloaded later independent of that version path.

        Unlike `UltralyticsAdapter.register_trained_model`, the destination
        filename doesn't need to embed a run-directory name to disambiguate
        runs: `artifact.metadata['model_class']` already uniquely identifies
        which of the five algorithms produced the checkpoint, so the default
        filename is just `<model_class>.ckpt`. Pass `model_name` explicitly
        if you're registering more than one run of the *same* model and want
        to keep both.
        """

        src = Path(artifact.path)
        if not src.exists():
            raise FileNotFoundError(f"cannot register missing checkpoint: {src}")

        registry = Path(registry_dir)
        registry.mkdir(parents=True, exist_ok=True)
        model_class = artifact.metadata.get("model_class", self.resolved_class_name)
        filename = model_name or f"{model_class}.ckpt"
        dst = registry / filename
        shutil.copy2(src, dst)

        metadata = dict(artifact.metadata)
        metadata["registered_from"] = str(src)
        return Artifact(path=str(dst), backend=self.backend, metadata=metadata)

    def load_trained_model(
        self, artifact_or_path: Artifact | str, allow_unsafe_checkpoint: bool = False
    ) -> Artifact:
        """Load a previously registered/trained checkpoint back into this
        adapter. Unlike `predict()`/`export()`, which resolve the model
        class from `artifact.metadata['model_class']` internally, this just
        validates the checkpoint exists — model-class resolution still
        happens lazily, at the point `predict()`/`export()` actually needs it.
        """

        path = artifact_or_path.path if isinstance(artifact_or_path, Artifact) else artifact_or_path
        if not Path(path).exists():
            raise FileNotFoundError(f"cannot load missing checkpoint: {path}")
        if isinstance(artifact_or_path, Artifact):
            if not artifact_or_path.metadata.get("trusted", False):
                raise ValueError("Anomalib artifact is not marked as trusted.")
            _restore_zero_shot_metadata(artifact_or_path, self.resolved_class_name, path)
            self._load_artifact(artifact_or_path)
            return artifact_or_path
        if not allow_unsafe_checkpoint:
            raise ValueError(
                "Loading a raw Anomalib checkpoint requires allow_unsafe_checkpoint=True because "
                "Lightning checkpoints can deserialize arbitrary Python objects."
            )
        artifact = Artifact(
            path=str(path), backend=self.backend,
            metadata={"model_class": self.resolved_class_name, "trusted": True},
        )
        _restore_zero_shot_metadata(artifact, self.resolved_class_name, path)
        self._load_artifact(artifact)
        return artifact

    def unload(self) -> None:
        """Release the resident Lightning module for an interactive session."""

        self._model = None
        self._loaded_path = None

    def _load_artifact(self, artifact: Artifact):
        if not artifact.metadata.get("trusted", False):
            raise ValueError("Refusing to load an untrusted Anomalib checkpoint.")
        if self._model is None or self._loaded_path != artifact.path:
            model_cls = resolve_model_class(artifact.metadata.get("model_class", self.name))
            if artifact.metadata.get("zero_shot", False):
                self._model = model_cls(**artifact.metadata["model_kwargs"])
            else:
                self._model = _load_checkpoint(model_cls, artifact.path)
            _patch_winclip_open_clip_layout(self._model)
            self._loaded_path = artifact.path
        return self._model


def _prediction_engine_kwargs(config: dict[str, Any] | None) -> dict[str, Any]:
    """Translate the uniform adapter device config into Lightning options.

    An unspecified device means "whichever accelerator this host has", not
    "every accelerator it has". Returning `{}` left `devices` at Lightning's
    own default of `"auto"` — all visible GPUs — so predicting a single image
    became a multi-process DDP run. Lightning's DDP launcher starts the extra
    ranks by re-executing *this process's entry script*
    (`lightning.fabric.strategies.launchers.subprocess_script.
    _basic_subprocess_cmd`), which inside an embedding host like the Gradio UI
    means every rank launched another `adh-ui`, collided with the port the
    serving process already held, and exited — and Lightning's child observer
    then SIGKILLs the main process when a rank dies. `devices=1` keeps
    prediction in-process everywhere while still letting `accelerator="auto"`
    pick CUDA/MPS/CPU; callers that care *which* GPU pin it themselves, as
    `run_experiment` does with the benchmark's assigned `cuda:N`.
    """

    device = str((config or {}).get("device") or "").lower()
    if device.startswith("cuda:"):
        return {"accelerator": "gpu", "devices": [int(device.partition(":")[2])]}
    if device == "cuda":
        return {"accelerator": "gpu", "devices": 1}
    if device == "mps":
        return {"accelerator": "mps", "devices": 1}
    if device == "cpu":
        return {"accelerator": "cpu", "devices": 1}
    return {"devices": 1}


def _patch_winclip_open_clip_layout(model: Any) -> None:
    """Adapt anomalib 2.5 WinCLIP to OpenCLIP's batch-first transformer.

    Anomalib's window path unconditionally converts NLD to LND before calling
    the transformer. OpenCLIP 3 uses a batch-first transformer and therefore
    interprets that converted tensor incorrectly, producing identical window
    embeddings and a spatially constant anomaly map. Keep anomalib's method
    unchanged for older OpenCLIP releases and replace only the incompatible
    batch-first path.
    """

    torch_model = getattr(model, "model", None)
    clip = getattr(torch_model, "clip", None)
    visual = getattr(clip, "visual", None)
    transformer = getattr(visual, "transformer", None)
    if torch_model is None or not getattr(transformer, "batch_first", False):
        return

    def _get_window_embeddings_batch_first(self, feature_map, masks):
        import torch

        batch_size = feature_map.shape[0]
        n_masks = masks.shape[1]
        class_index = torch.zeros(1, n_masks, dtype=torch.long, device=feature_map.device)
        indices = torch.cat((class_index, masks + 1)).T
        masked = torch.cat([torch.index_select(feature_map, 1, index) for index in indices])
        masked = self.clip.visual.patch_dropout(masked)
        masked = self.clip.visual.ln_pre(masked)
        masked = self.clip.visual.transformer(masked)
        masked = self.clip.visual.ln_post(masked)
        pooled, _ = self.clip.visual._global_pool(masked)
        if self.clip.visual.proj is not None:
            pooled = pooled @ self.clip.visual.proj
        return pooled.reshape((n_masks, batch_size, -1)).permute(1, 0, 2)

    torch_model._get_window_embeddings = MethodType(_get_window_embeddings_batch_first, torch_model)


def _load_checkpoint(model_cls, path: str):
    """`weights_only=False`: PyTorch >=2.6 defaults `torch.load` to
    `weights_only=True`, which rejects anomalib's own checkpoint globals
    (e.g. `anomalib.PrecisionType`) unless explicitly allowlisted. Safe
    here because callers are required to pass an `Artifact` marked trusted
    by `train()`/`register_trained_model()`, or to explicitly opt into an
    unsafe raw-checkpoint load in `load_trained_model()`.
    """

    load_kwargs: dict[str, Any] = {"weights_only": False}
    if "pre_trained" in inspect.signature(model_cls).parameters:
        # A trained Lightning checkpoint already contains the feature extractor.
        # Avoid an unnecessary network request before those weights are restored.
        load_kwargs["pre_trained"] = False
    return model_cls.load_from_checkpoint(path, **load_kwargs)


def _restore_zero_shot_metadata(artifact: Artifact, model_class: str, path: str) -> None:
    """Rebuild metadata-only WinCLIP artifacts created by zero-shot training.

    Zero-shot WinCLIP has no learned checkpoint. The published slot contains
    a small text marker so the registry can still hold a stable artifact path;
    loading that marker as a Lightning checkpoint would produce an opaque
    ``UnpicklingError`` beginning with the letter ``W``.
    """

    if model_class != "WinClip" or artifact.metadata.get("zero_shot"):
        return
    try:
        marker = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return
    if not marker.startswith("WinCLIP zero-shot artifact;"):
        return
    artifact.metadata.update({
        "model_class": "WinClip",
        "model_kwargs": default_model_kwargs("WinClip"),
        "zero_shot": True,
    })
