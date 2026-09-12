"""`ModelAdapter` implementation for MambaAD -- a **clean-room
reimplementation**, not a vendored `components/mambaad` checkout.

A vendored submodule was tried once already (see `components/README.md`'s
git history: "remove infeasible MambaAD"). The reasons hold for the
official repo (`lewandofskee/MambaAD`) too, not just the fork that was
tried:

* Its `model/`/`trainer/` files are not a runnable project on their own --
  `model/mambaad.py` imports `from model import get_model, MODEL` and
  `trainer/mambaad_trainer.py` imports `util.*`/`data.get_loader`/
  `optim.*`/`loss.*`/`trainer._base_trainer.BaseTrainer`, none of which
  exist in that repo. Its own README says as much: "Clone this repo to
  [ADer]" -- it is a plugin dropped into an ADer checkout, not a
  self-contained project the way Dinomaly and MoECLIP each are. Vendoring
  it would mean vendoring ADer too, a second, general-purpose framework,
  which no other entry under `components/` does.
* Its selective-scan core calls `mamba_ssm.ops.selective_scan_interface
  .selective_scan_fn`, a compiled CUDA kernel. `pip install mamba_ssm`
  needs a matching CUDA/nvcc toolchain to build at all -- it does not
  install on this project's dev machine (Apple Silicon, no CUDA), so
  depending on the real package would gate this backend to a CUDA host
  exactly as hard as vendoring it would have.

This module reimplements the architecture instead: `ssm.py`/`scan.py` port
the selective-scan recurrence and its five multi-directional scan orders
in portable PyTorch/NumPy; `decoder.py`/`network.py` port the LSS/HSS
decoder and the teacher-fusion-decoder wiring. What's reused directly is
the published recipe's *values* (`presets.py`, sourced from upstream's own
`configs/mambaad/mambaad_mvtec.py`), not its code. Note this does *not*
mean `mamba_ssm` goes unused when it is available: `ssm.py` dispatches to
its fused kernel at run time on CUDA, so a training host that has it gets
upstream's exact kernel, while every other machine still runs.

**Fidelity, verified against upstream source rather than assumed:**

* Training loss is `L2Loss` = MSE, not the cosine objective the rest of
  this family uses (ADer `loss/base_loss.py`; see `_reconstruction_loss`).
* The anomaly map is per-level `1 - cosine`, upsampled, summed, divided by
  `len(levels) * sum(weights)`, then Gaussian-blurred at sigma 4 with
  scipy's kernel width *and* boundary convention (ADer `util/metric.py::
  cal_anomaly_map`; see `_anomaly_map`/`_gaussian_blur`).
* Image score is the max over the anomaly map (`pr_sp_max` in ADer's
  evaluator).
* The LR schedule is upstream's warmup + step decay (see `train`).
* MambaAD is **multi-class unified** -- one model over all categories at
  once (see `presets.DEFAULT_TRAIN_KWARGS`).

**Known remaining gaps** (honest scope, not silently glossed):

* No end-to-end accuracy validation against the published numbers. Doing
  that means a real multi-class MVTec-AD run on a CUDA host at upstream's
  full budget; nothing here has been checked to land at 98.6 mAUROC.
* `total_iters` defaults to this project's convention, not upstream's
  1000-epoch budget (see `presets.py`).
* Upstream's optional mixup/AMP/EMA and distributed training are not
  ported; they are ADer trainer infrastructure rather than MambaAD itself,
  and its published config leaves mixup's probability at 0.

Architecturally this is the same "reverse distillation" family as
`models/dinomaly/adapter.py`'s `ViTill`: frozen pretrained encoder,
trainable decoder, multi-scale feature reconstruction, anomaly score from
encoder/decoder discrepancy. Key differences from Dinomaly: MambaAD's
encoder is a plain CNN (`timm` `features_only`), not a ViT; its decoder is
Mamba-based (`MambaUPNet`) rather than a linear-attention transformer; and
it trains on MSE rather than cosine distance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch.nn.functional as F

from fabric_defect_hub.core.progress import ProgressReporter
from fabric_defect_hub.core.provenance import describe_training
from fabric_defect_hub.core.registry import register_model
from fabric_defect_hub.core.train_config import TrainConfig, resolve_train_config
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.model_statistics import parameter_counts
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter, ModelCapabilities
from fabric_defect_hub.models.mambaad import presets
from fabric_defect_hub.models.mambaad.data import ImageOnlyDataset


@register_model("mambaad")
class MambaADAdapter(ModelAdapter):
    """Wraps the reimplemented MambaAD network (frozen `timm` teacher +
    `MultiScaleFusion` + `MambaUPNet`).

    `name` selects the teacher preset (see `presets.ENCODER_PRESETS`),
    e.g. 'resnet34' (default, upstream's flagship MVTec-AD recipe) or
    'wide_resnet50_2'.
    """

    backend = "mambaad"

    def __init__(self, name: str = presets.DEFAULT_ENCODER_NAME, **kwargs):
        super().__init__(name=name, **kwargs)
        self.encoder_name = presets.resolve_encoder_name(name)
        self._model = None
        self._loaded_path: str | None = None

    # ------------------------------------------------------------------ #
    # Model construction, shared by train() and _load_artifact()
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

    def _build_model(self, encoder_name: str, arch: dict[str, Any], device, pretrained_teacher: bool = True):
        import timm

        from fabric_defect_hub.models.mambaad.network import MambaADNet

        preset = presets.encoder_preset(encoder_name)
        image_size = arch["image_size"]
        teacher_downsample = 32  # deepest of out_indices [1,2,3] (stride 16) + fusion's own extra /2
        if image_size % teacher_downsample:
            raise ValueError(
                f"image_size={image_size} must be a multiple of {teacher_downsample} "
                f"(the teacher's stride-16 deepest feature level, halved once more by fusion)."
            )
        base_size = image_size // teacher_downsample

        teacher = timm.create_model(
            preset["timm_name"], pretrained=pretrained_teacher,
            features_only=True, out_indices=preset["out_indices"],
        )
        model = MambaADNet(
            teacher=teacher,
            teacher_channels=preset["channels"],
            dims_decoder=arch["dims_decoder"],
            depths_decoder=arch["depths_decoder"],
            d_state=arch["d_state"],
            drop_path_rate=arch["drop_path_rate"],
            base_size=base_size,
            scan_type=arch["scan_type"],
            num_direction=arch["num_direction"],
        )
        for param in model.teacher.parameters():
            param.requires_grad = False
        return model.to(device)

    def _resolved_arch(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        arch = {
            "image_size": presets.DEFAULT_TRAIN_KWARGS["image_size"],
            "dims_decoder": list(presets.DIMS_DECODER),
            "depths_decoder": list(presets.DEPTHS_DECODER),
            "d_state": presets.D_STATE,
            "drop_path_rate": presets.DROP_PATH_RATE,
            "scan_type": presets.DEFAULT_SCAN_TYPE,
            "num_direction": presets.DEFAULT_NUM_DIRECTION,
        }
        if config:
            arch.update({k: v for k, v in config.items() if k in arch})
        return arch

    # ------------------------------------------------------------------ #
    # Loss / anomaly map -- shared math between train() and predict()
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reconstruction_loss(teacher_features, student_features, lam: float = 1.0):
        """Upstream's training objective, `L2Loss`: plain MSE between each
        teacher level and the student's reconstruction of it, summed over
        levels, each scaled by `lam`.

        Verified against ADer's `loss/base_loss.py::L2Loss`
        (`nn.MSELoss(reduction='mean')` accumulated over the feature lists
        and multiplied by `lam`), which `configs/mambaad/mambaad_mvtec.py`
        selects as `dict(type='L2Loss', name='pixel', lam=5.0)` and
        `trainer/mambaad_trainer.py::optimize_parameters` calls as
        `self.loss_terms['pixel'](self.feats_t, self.feats_s)`.

        Note this is *not* the cosine-distance objective the rest of the
        reverse-distillation family (RD4AD, Dinomaly) uses -- MambaAD
        trains on MSE and only uses cosine at *inference* to build the
        anomaly map (see `_anomaly_map`). Getting this wrong changes what
        the decoder optimizes, so it is matched exactly rather than
        approximated by the family's more common choice.
        """

        loss = teacher_features[0].new_zeros(())
        for teacher_level, student_level in zip(teacher_features, student_features):
            loss = loss + F.mse_loss(student_level, teacher_level, reduction="mean") * lam
        return loss

    @staticmethod
    def _anomaly_map(teacher_features, student_features, image_size: int, gaussian_sigma: float = 4.0):
        """Upstream's inference-time anomaly map.

        Verified against ADer's `util/metric.py::cal_anomaly_map` as called
        by `trainer/mambaad_trainer.py::test` with
        `uni_am=False, amap_mode='add', gaussian_sigma=4` (and `use_cos=True`
        by default): per level take `1 - cosine_similarity` across channels,
        bilinearly upsample to the input resolution, sum the levels, divide
        by `len(levels) * sum(weights)` (i.e. 3 * 3 = 9 with the default
        unit weights), then Gaussian-blur each map with sigma 4.

        The `/9` is a constant rescale and so cannot change any ranking
        metric, but it is kept anyway so a score printed here is directly
        comparable to one from upstream rather than 9x larger.
        """

        import torch

        levels = len(teacher_features)
        total = None
        for teacher_level, student_level in zip(teacher_features, student_features):
            level_map = (1 - F.cosine_similarity(teacher_level, student_level, dim=1)).unsqueeze(1)
            level_map = F.interpolate(level_map, size=image_size, mode="bilinear", align_corners=True)
            total = level_map if total is None else total + level_map
        total = total / (levels * levels)  # upstream: len(ft_list) * sum(weights), weights all 1
        if gaussian_sigma > 0:
            total = MambaADAdapter._gaussian_blur(total, gaussian_sigma)
        return total

    @staticmethod
    def _symmetric_pad(x, radius: int, dim: int):
        """scipy's default `mode='reflect'` boundary: `(d c b a | a b c d |
        d c b a)`, i.e. the edge sample is *repeated* across the boundary.

        `torch.nn.functional.pad(mode='reflect')` implements the *other*
        convention (`(d c b | a b c d | c b a)`, scipy's `mode='mirror'`),
        so it cannot be used directly here -- hence index arithmetic.
        """

        import torch

        length = x.shape[dim]
        positions = torch.arange(-radius, length + radius, device=x.device)
        period = 2 * length
        wrapped = positions % period
        indices = torch.where(wrapped < length, wrapped, period - 1 - wrapped)
        return x.index_select(dim, indices)

    @staticmethod
    def _gaussian_blur(x, sigma: float):
        """Separable Gaussian blur matching `scipy.ndimage.gaussian_filter`
        -- which is what upstream applies to the anomaly map -- in all three
        respects that change the result: kernel width, boundary handling,
        and normalization.

        Kernel radius is scipy's `int(truncate * sigma + 0.5)` with the
        default `truncate=4.0`, i.e. 33 taps at sigma 4. The width matters
        as much as sigma: a 5x5 window at sigma 4 is nearly a box filter and
        blurs far less, which would shift pixel-level AUPRO away from the
        published numbers for reasons that have nothing to do with the model.
        """

        import torch

        radius = int(4.0 * sigma + 0.5)
        axis = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
        kernel = torch.exp(-(axis**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        channels = x.shape[1]

        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        x = MambaADAdapter._symmetric_pad(x, radius, dim=3)
        x = F.conv2d(x, horizontal, groups=channels)
        x = MambaADAdapter._symmetric_pad(x, radius, dim=2)
        return F.conv2d(x, vertical, groups=channels)

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    # Canonical `TrainConfig` field -> this backend's real key (upstream's
    # schedule is iteration-based, so `max_iters` maps to `total_iters`).
    TRAIN_CONFIG_KEYS = {
        "max_iters": "total_iters",
        "lr": "lr",
        "batch_size": "batch_size",
        "image_size": "image_size",
        "device": "device",
        "num_workers": "num_workers",
        "work_dir": "work_dir",
    }

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tasks=("anomaly",),
            prediction_fields=("anomaly_score", "anomaly_map"),
            required_annotations=(),
            # See `export()`: the multi-directional selective scan has not been
            # verified to trace.
            export_targets=(),
            supports_amp=False,
        )

    def train(self, config: dict[str, Any] | TrainConfig) -> Artifact:
        """One-class training loop: minimize `_reconstruction_loss`
        (upstream's MSE-based `L2Loss`) between the frozen teacher's
        features and the trainable fusion+decoder's reconstruction of
        them, over normal-only fabric images. `config['train_samples']` must be all-normal (checked
        below) -- the same one-class contract as PatchCore/Dinomaly, not
        MoECLIP's labelled-defect training (see
        `training._ONE_CLASS_BACKENDS`).

        Optional keys, defaulting from `presets.DEFAULT_TRAIN_KWARGS` /
        the decoder architecture defaults in `presets.py`: `total_iters`,
        `batch_size`, `image_size`, `lr`, `weight_decay`, `warmup_iters`,
        `loss_lambda`, plus `dims_decoder`/`depths_decoder`/`d_state`/
        `drop_path_rate`/`scan_type`/`num_direction`/`device`/`num_workers`/
        `work_dir` (checkpoint destination; defaults to a fresh temp dir).

        The learning-rate schedule mirrors upstream's
        (`trainer.scheduler_kwargs` in `configs/mambaad/mambaad_mvtec.py`):
        a linear warmup from `lr/1000`, then a step decay by `decay_rate`
        at `decay_at` of the run, floored at `lr/100` -- restated in
        iterations rather than epochs, since `total_iters` replaces
        upstream's `epoch_full` here (see `presets.DEFAULT_TRAIN_KWARGS`).

        Note this trains ONE model across whatever the train split
        contains: MambaAD is a *multi-class unified* detector, so pointing
        it at the `fabric-train` composite (all fabric sources) or at
        ZJU-Leaper's full 19 patterns is using it as intended, not a
        shortcut -- see `presets.DEFAULT_TRAIN_KWARGS`' comment.
        """

        config = resolve_train_config(config, self.TRAIN_CONFIG_KEYS)

        import torch
        from torch.utils.data import DataLoader

        train_samples = config.get("train_samples")
        if not train_samples:
            raise ValueError("MambaAD training requires config['train_samples'] (a list of Sample).")
        bad = [s for s in train_samples if s.annotations.is_anomalous]
        if bad:
            raise ValueError(
                f"train_samples must be all-normal (one-class training); got {len(bad)} "
                f"defective samples, e.g. {bad[0].id!r}. Load them with use_defect=False."
            )

        kwargs = {**presets.default_train_kwargs(),
                  **{k: v for k, v in config.items() if k in presets.DEFAULT_TRAIN_KWARGS}}
        arch = self._resolved_arch(config)
        device = self._resolve_device(config.get("device"))
        num_workers = int(config.get("num_workers", 0))

        dataset = ImageOnlyDataset(train_samples, arch["image_size"])
        effective_batch_size = min(int(kwargs["batch_size"]), len(dataset))
        loader = DataLoader(
            dataset, batch_size=effective_batch_size, shuffle=True, num_workers=num_workers,
            drop_last=len(dataset) > effective_batch_size,
        )

        model = self._build_model(self.encoder_name, arch, device, pretrained_teacher=True)
        model.train()

        trainable_params = list(model.fusion.parameters()) + list(model.decoder.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params, lr=float(kwargs["lr"]), betas=(0.9, 0.999), eps=1e-8,
            weight_decay=float(kwargs["weight_decay"]),
        )
        base_lr = float(kwargs["lr"])
        warmup_iters = int(kwargs["warmup_iters"])
        total_iters = int(kwargs["total_iters"])
        decay_start = int(total_iters * float(kwargs["decay_at"]))
        decay_rate = float(kwargs["decay_rate"])
        min_lr = base_lr / 100.0  # upstream's lr_min

        def lr_at(iteration: int) -> float:
            if warmup_iters > 0 and iteration < warmup_iters:
                # Upstream warms up from lr/1000 to lr.
                start = base_lr / 1000.0
                return start + (base_lr - start) * (iteration + 1) / warmup_iters
            if iteration >= decay_start:
                return max(base_lr * decay_rate, min_lr)
            return base_lr

        it = 0
        with ProgressReporter(f"mambaad/{self.encoder_name} train", total=total_iters) as progress:
            while it < total_iters:
                for images in loader:
                    images = images.to(device)
                    for group in optimizer.param_groups:
                        group["lr"] = lr_at(it)

                    teacher_features, student_features = model(images)
                    loss = self._reconstruction_loss(
                        teacher_features, student_features, lam=float(kwargs["loss_lambda"])
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    it += 1
                    progress.update(loss=loss.detach().item(), lr=optimizer.param_groups[0]["lr"])
                    if it >= total_iters:
                        break

        work_dir = Path(config.get("work_dir") or tempfile.mkdtemp(prefix="fdh_mambaad_"))
        work_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = work_dir / f"mambaad_{self.encoder_name}.pth"
        torch.save(model.state_dict(), ckpt_path)

        return Artifact(
            path=str(ckpt_path),
            backend=self.backend,
            metadata={
                "model_class": "MambaADNet",
                "encoder_name": self.encoder_name,
                "image_size": arch["image_size"],
                "dims_decoder": arch["dims_decoder"],
                "depths_decoder": arch["depths_decoder"],
                "d_state": arch["d_state"],
                "drop_path_rate": arch["drop_path_rate"],
                "scan_type": arch["scan_type"],
                "num_direction": arch["num_direction"],
                "gaussian_sigma": presets.ANOMALY_MAP_GAUSSIAN_SIGMA,
                "trusted": True,
                # No scheduler object exists; the schedule is the inline
                # `lr_at` above (upstream's warmup + step decay).
                "training": describe_training(
                    optimizer, "linear-warmup+step-decay (upstream schedule)"
                ),
                "batch_spec": dataset.batch_spec().as_run_metadata(),
                **parameter_counts(model),
            },
        )

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict(
        self,
        samples: list[Sample],
        artifact: Artifact | None = None,
        output_dir: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Loads the checkpoint and scores each sample with the same
        per-level cosine-distance map training minimizes -- higher
        teacher/student disagreement at a pixel means the frozen teacher
        saw something the (normal-data-only-trained) decoder can't
        reconstruct, i.e. an anomaly. Pass `output_dir` to also persist
        each sample's pixel-level anomaly map as a `.npy` file (needed for
        `evaluation.anomaly.AnomalyEvaluator`'s pixel metrics).
        """

        import numpy as np
        import torch

        from fabric_defect_hub.models.mambaad.data import build_transform

        device = self._resolve_device(None)
        model = self._load_artifact(artifact, device)
        image_size = int(artifact.metadata.get("image_size", presets.DEFAULT_TRAIN_KWARGS["image_size"]))
        transform = build_transform(image_size)
        gaussian_sigma = float(artifact.metadata.get("gaussian_sigma", presets.ANOMALY_MAP_GAUSSIAN_SIGMA))

        maps_dir = None
        if output_dir is not None:
            maps_dir = Path(output_dir)
            maps_dir.mkdir(parents=True, exist_ok=True)

        predictions = []
        progress = ProgressReporter(
            f"mambaad/{self.encoder_name} predict", total=len(samples), unit="img"
        )
        with torch.no_grad(), progress:
            for sample in samples:
                from PIL import Image

                image = transform(Image.open(sample.image_path).convert("RGB")).unsqueeze(0).to(device)
                teacher_features, student_features = model(image)
                anomaly_map = self._anomaly_map(
                    teacher_features, student_features, image_size, gaussian_sigma
                )
                score = float(anomaly_map.flatten(1).max(dim=1)[0].item())

                anomaly_map_path = None
                if maps_dir is not None:
                    array = anomaly_map[0, 0].detach().cpu().numpy()
                    map_path = maps_dir / f"{sample.id}.npy"
                    map_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(map_path, array)
                    anomaly_map_path = str(map_path)

                predictions.append(
                    Prediction(sample_id=sample.id, anomaly_score=score, anomaly_map=anomaly_map_path)
                )
                progress.update(score=score)
        return predictions

    def export(
        self, artifact: Artifact, target: str, config: dict[str, Any] | None = None
    ) -> ExportedArtifact:
        raise NotImplementedError(
            "MambaAD export is not implemented: the multi-directional selective scan "
            "(SS2D.forward_core) loops over scan directions and uses index-tensor "
            "scatter/gather (HSCANS.encode/decode) that hasn't been verified to trace "
            "cleanly through torch.onnx.export. Add and verify this explicitly before "
            "relying on it."
        )

    # ------------------------------------------------------------------ #
    # Model registry: persist / reload trained models
    # ------------------------------------------------------------------ #
    def register_trained_model(
        self, artifact: Artifact, registry_dir: str, model_name: str | None = None
    ) -> Artifact:
        import shutil

        src = Path(artifact.path)
        if not src.exists():
            raise FileNotFoundError(f"cannot register missing checkpoint: {src}")

        registry = Path(registry_dir)
        registry.mkdir(parents=True, exist_ok=True)
        filename = model_name or f"mambaad_{artifact.metadata.get('encoder_name', self.encoder_name)}.pth"
        dst = registry / filename
        shutil.copy2(src, dst)

        metadata = dict(artifact.metadata)
        metadata["registered_from"] = str(src)
        return Artifact(path=str(dst), backend=self.backend, metadata=metadata)

    def load_trained_model(self, artifact_or_path: Artifact | str) -> Artifact:
        """Load a previously registered/trained checkpoint back into this
        adapter. Given a bare path, the architecture defaults to this
        adapter's `encoder_name` + `presets.py`'s decoder defaults -- pass
        an `Artifact` (with the metadata `train()` produced) if the
        checkpoint was trained with different ones.
        """

        if isinstance(artifact_or_path, Artifact):
            artifact = artifact_or_path
        else:
            path = artifact_or_path
            if not Path(path).exists():
                raise FileNotFoundError(f"cannot load missing checkpoint: {path}")
            arch = self._resolved_arch()
            artifact = Artifact(
                path=str(path), backend=self.backend,
                metadata={"model_class": "MambaADNet", "encoder_name": self.encoder_name,
                          "trusted": True, **arch},
            )
        if not Path(artifact.path).exists():
            raise FileNotFoundError(f"cannot load missing checkpoint: {artifact.path}")
        self._load_artifact(artifact, self._resolve_device(None))
        return artifact

    def unload(self) -> None:
        self._model = None
        self._loaded_path = None

    def _load_artifact(self, artifact: Artifact, device):
        if self._model is None or self._loaded_path != artifact.path:
            import torch

            encoder_name = artifact.metadata.get("encoder_name", self.encoder_name)
            arch = self._resolved_arch(artifact.metadata)
            # Teacher weights are loaded from the checkpoint's full
            # state_dict below, not re-downloaded from timm -- pretrained
            # ImageNet weights are only needed to *seed* a fresh training
            # run.
            model = self._build_model(encoder_name, arch, device, pretrained_teacher=False)
            state_dict = torch.load(artifact.path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            self._model = model
            self._loaded_path = artifact.path
        return self._model
