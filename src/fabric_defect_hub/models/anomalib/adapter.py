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

import contextlib
import inspect
import shutil
import tempfile
from types import MethodType
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.progress import ProgressReporter
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

        `config["raw_anomaly"]` returns the model's own score and map instead
        of the checkpoint's stored normalization — an evaluation wants that,
        a display does not. See `_raw_anomaly_predictions`.
        """

        if not artifact.metadata.get("trusted", False):
            raise ValueError(
                "Refusing to load an untrusted Anomalib checkpoint. Use load_trained_model(..., "
                "allow_unsafe_checkpoint=True) only for a checkpoint from a trusted source."
            )

        import numpy as np
        from anomalib.engine import Engine
        from lightning.pytorch import Trainer

        model = self._load_artifact(artifact)
        raw_predictions = (
            _raw_anomaly_predictions(model)
            if (config or {}).get("raw_anomaly")
            else contextlib.nullcontext()
        )

        maps_dir = None
        if output_dir is not None:
            maps_dir = Path(output_dir)
            maps_dir.mkdir(parents=True, exist_ok=True)
        # Keep Lightning's predict logs beside the requested runtime output,
        # rather than allowing its default ``results/`` directory to appear
        # at the repository root during an interactive UI prediction.
        engine_root = maps_dir.parent if maps_dir is not None else Path(artifact.path).parent
        engine_kwargs = _prediction_engine_kwargs(config)
        # Lightning's RichProgressBar is not concurrency-safe: every Trainer in a
        # benchmark worker thread starts and tears down a live display on the
        # console shared by the whole process, and when two of them overlap one
        # thread pops the console's already-empty live stack --
        # `rich/console.py::clear_live` -> `IndexError: pop from empty list` --
        # during `teardown`, which Lightning reports as this model's evaluation
        # failing while its siblings are fine. Prediction is called once per
        # sample here and this repo already prints its own per-image progress
        # (`ProgressReporter`), so the bar carries nothing extra; keep it
        # overridable for a caller that wants it and is running alone.
        engine_kwargs["enable_progress_bar"] = bool((config or {}).get("enable_progress_bar", False))
        engine = Engine(default_root_dir=str(engine_root), **engine_kwargs)

        # Predict in chunks, not one sample per `engine.predict` call.
        #
        # The per-sample loop this replaces built a Lightning Trainer for every
        # image, reprinting its banner, its callback notice and anomalib's
        # "ckpt_path is not provided" warning each time — 350 Trainer setups for
        # a 350-image benchmark, which pinned throughput at ~1.4 images/s on an
        # A100 that needs tens of milliseconds per image. Chunking amortizes
        # that setup (11 calls for 350 images) while keeping progress lines
        # meaningful: one reporter update per batch as the results come back.
        zero_shot = bool(artifact.metadata.get("zero_shot", False))
        chunk_size = max(1, int((config or {}).get("predict_chunk_size", _PREDICT_CHUNK_SIZE)))
        trainer = None
        if zero_shot:
            # Anomalib's Engine routes WinCLIP through validation, but WinCLIP
            # intentionally has no val_dataloader. Supplying explicit prediction
            # loaders keeps Lightning on predict_step.
            trainer = Trainer(
                default_root_dir=str(engine_root),
                logger=False,
                enable_checkpointing=False,
                **engine_kwargs,
            )

        predictions = []
        progress = ProgressReporter(f"anomalib/{self.name} predict", total=len(samples), unit="img")
        with raw_predictions, progress:
            for start in range(0, len(samples), chunk_size):
                chunk = samples[start : start + chunk_size]
                loader = _predict_loader(chunk)
                batches = (
                    trainer.predict(model=model, dataloaders=loader)
                    if trainer is not None
                    else engine.predict(model=model, dataloaders=loader)
                ) or []
                if len(batches) != len(chunk):
                    raise RuntimeError(
                        f"Anomalib returned {len(batches)} prediction batch(es) for {len(chunk)} "
                        f"sample(s) starting at {chunk[0].id!r}."
                    )
                for sample, batch in zip(chunk, batches):
                    score = None
                    predicted_label = None
                    anomaly_map_path = None
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
                    progress.update()
        return predictions
    def export(
        self, artifact: Artifact, target: str, config: dict[str, Any] | None = None
    ) -> ExportedArtifact:
        """`target` is an `anomalib.deploy.ExportType` value, e.g. 'onnx', 'openvino'."""

        if not artifact.metadata.get("trusted", False):
            raise ValueError("Refusing to export an untrusted Anomalib checkpoint.")

        from anomalib.engine import Engine

        model = self._load_artifact(artifact)
        # Anomalib writes its export to `<default_root_dir>/weights/<format>/model.<ext>`,
        # and a benchmark runs one worker process per GPU in parallel. The
        # default root is the process CWD, so every anomalib model in a batch
        # used to export to the *same* `results/weights/onnx/model.onnx`:
        #
        #   * two exports racing hand one reader a half-written protobuf —
        #     `InvalidProtobuf: Load model from results/weights/onnx/model.onnx
        #     failed: Protobuf parsing failed` (observed 2026-09-18, Reverse
        #     Distillation; the file parses fine once the run stops, which is
        #     how a race looks from the outside);
        #   * a benign ordering is worse: the second export overwrites the first
        #     before its profiler reads it, so one model is measured as
        #     another's graph and nothing reports an error at all.
        #
        # One directory per export. `tempfile.mkdtemp` rather than a
        # `TemporaryDirectory`: the artifact has to outlive this call (the
        # profiler and the resolution sweep read it afterwards), so the system
        # temp directory owns its lifetime instead.
        export_root = Path(tempfile.mkdtemp(prefix="fdh-anomalib-export-"))
        engine = Engine(default_root_dir=str(export_root))
        exported_path = Path(engine.export(model=model, export_type=target))
        if not exported_path.is_file():
            raise FileNotFoundError(
                f"anomalib reported an export at {exported_path}, which does not exist"
            )
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


# How many images one `engine.predict` call covers. Trainer construction, not
# inference, dominated this backend's per-image cost (see `predict`), and every
# image in a chunk is one progress update.
_PREDICT_CHUNK_SIZE = 32


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


class _SamplePredictDataset:
    """Anomalib's `PredictDataset`, over our `Sample` list.

    `anomalib.data.PredictDataset` takes a single path or a directory; benchmark
    samples come from a dataset adapter and can span directories, so this yields
    the same `ImageItem` for an explicit list — built from the same `read_image`
    call the per-sample path used, so the pixels reaching the model are
    unchanged. Deliberately not a `torch.utils.data.Dataset` subclass: torch's
    loader only needs `__len__`/`__getitem__`, and inheriting would drag torch
    into this module's import time (it is imported by framework-free tests).
    """

    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Any:
        from anomalib.data import ImageItem
        from anomalib.data.utils import read_image

        path = self._samples[index].image_path
        return ImageItem(image=read_image(path, as_tensor=True), image_path=str(path))


def _predict_loader(samples: list[Sample]):
    """A batch-size-1 loader `anomalib.engine.Engine.predict` accepts."""

    from anomalib.data import ImageBatch
    from torch.utils.data import DataLoader

    return DataLoader(_SamplePredictDataset(samples), batch_size=1, collate_fn=ImageBatch.collate)


@contextlib.contextmanager
def _raw_anomaly_predictions(model: Any):
    """Suspend anomalib's stored normalization for one prediction pass.

    Every anomalib checkpoint carries a post-processor with the min/max (and
    threshold) its *training* validation pass measured, and `predict_step`
    applies it to every output:

        y = clamp(((x - threshold) / (max - min)) + 0.5, 0, 1)

    That clamp is harmless while the inference-time scores stay inside the
    stored range — a global affine does not change AUROC/AUPRO/IAP, which is
    why the healthy models report the same numbers either way. It is *not*
    harmless once the raw error sits above the ceiling: every pixel of every
    image then lands on exactly 1.0, every `pred_score` becomes exactly 1.0,
    and the evaluator faithfully reports AUROC 0.5, AUPRO 0.0 and an IAP equal
    to the dataset's positive-pixel ratio for a model that may be perfectly
    discriminative (observed 2026-09-17: Reverse Distillation and
    SuperSimpleNet, whose stored ceilings are raw >= 1.96 and >= 0.70 and whose
    saved maps were byte-identical all-ones tensors).

    An evaluation wants the model's own ranking, not a training-time rescale
    of it, and the thresholded metrics come from the benchmark's own
    train-split calibration (`evaluation.anomaly.calibrate_thresholds`), so
    the stored normalization has nothing left to contribute. Display paths
    keep it: it is what makes a heat map renderable.
    """

    post_processor = getattr(model, "post_processor", None)
    if post_processor is None or not hasattr(post_processor, "enable_normalization"):
        yield
        return

    previous = post_processor.enable_normalization
    post_processor.enable_normalization = False
    try:
        yield
    finally:
        post_processor.enable_normalization = previous


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

    # `map_location="cpu"`: these checkpoints are GPU-saved, so the default
    # would materialize the whole state dict on whichever CUDA device the
    # training run happened to use — before this worker has even pinned its
    # own device, and on a host that may expose a different GPU set than the
    # one that saved it. Restoring on the CPU and letting Lightning move the
    # finished model is deterministic and costs one copy.
    load_kwargs: dict[str, Any] = {"weights_only": False, "map_location": "cpu"}
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
