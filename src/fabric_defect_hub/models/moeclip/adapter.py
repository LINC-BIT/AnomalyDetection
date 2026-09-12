"""`ModelAdapter` implementation wrapping the vendored `components/moeclip`
checkout (see `components/README.md` and `vendor.py`).

Like Dinomaly, MoECLIP ships as research scripts rather than a library:
`train.py`/`test.py` are argparse mains that read `.jsonl` metadata files
for datasets they hardcode, write their own logs, and print a pandas
leaderboard. This adapter re-implements those two loops -- upstream's
`train.py::train_adapter` and `test.py::get_predictions` -- against this
project's `Sample`/`Prediction`/`Artifact` contracts, importing only the
vendored `nn.Module`s and the scoring/loss helpers in `forward_utils`, so
the math stays upstream's.

Two things differ from Dinomaly's adapter, both forced by what MoECLIP is:

* **Training is not one-class, and not on fabric.** MoECLIP is a
  *zero-shot* anomaly detector: it learns prompt-aligned patch
  representations from labelled anomalies (image label + pixel mask) on an
  auxiliary corpus and is then applied to categories it has never seen. So
  its train split needs defective samples *with masks*, not normal-only
  data (the opposite of PatchCore/Dinomaly -- see `_select_train_samples`),
  and that corpus is a cross-domain object benchmark (VisA / MVTec AD /
  LOCO), not this project's fabric sets. Training it on fabric would make
  its fabric numbers in-domain and void the transfer claim; `fdh train`
  rejects it (`training.ZERO_SHOT_TRAINABLE_DATASETS`), and the evaluation
  target is a separate config key (`data.test_dataset`).
* **Only the adapters are trained.** The CLIP backbone stays frozen;
  checkpoints hold just `text_adapter` + `image_adapter` (which contains
  the MoE LoRA experts, the segmentation projections and the detection
  projection), matching upstream's `save_checkpoint`. They are plain
  `state_dict`s saved with `torch.save`, so -- as with Dinomaly, unlike
  anomalib's Lightning checkpoints -- there's no untrusted-pickle gate.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

from fabric_defect_hub.core.progress import ProgressReporter
from fabric_defect_hub.core.provenance import describe_training
from fabric_defect_hub.core.registry import register_model
from fabric_defect_hub.core.train_config import TrainConfig, resolve_train_config
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.model_statistics import parameter_counts
from fabric_defect_hub.models.base import Artifact, ExportedArtifact, ModelAdapter, ModelCapabilities
from fabric_defect_hub.models.moeclip import presets
from fabric_defect_hub.models.moeclip.data import SampleDataset
from fabric_defect_hub.models.moeclip.vendor import cuda_free_module_init, import_vendor, vendor_root


@register_model("moeclip")
class MoECLIPAdapter(ModelAdapter):
    """Wraps upstream's `MoECLIP` (frozen CLIP ViT-L/14-336 + per-patch
    LoRA mixture-of-experts + patch-average aggregation).

    `name` selects the backbone preset; upstream supports exactly one
    (`ViT-L-14-336`, the default) -- see `presets.MODEL_PRESETS`.
    `**kwargs` may override any of `presets.DEFAULT_ARCH_KWARGS`
    (`moe_num_experts`, `moe_layers`, `img_size`, ...); whatever is used is
    recorded in the artifact's metadata so `predict` rebuilds the same
    architecture.

    Two further kwargs configure the *prompts*, which is the only part of
    this backend with no counterpart in the other adapters (see
    `presets.py`'s module docstring):

    * `prompt_class`: pin every sample to one prompt class instead of
      reading it from the sample's dataset metadata. This is the usual
      setting for fabric inference — one texture being inspected, one
      prompt — e.g. `prompt_class="fabric"`.
    * `prompts`: `{class name: description}` overriding the noun phrase
      templated into that class's prompts, e.g.
      `{"fabric": "woven cotton fabric"}`.

    Both are inference-time-tunable on purpose and deliberately *not*
    pinned from the training artifact: MoECLIP is trained on one domain
    (VisA's objects) and applied to another (fabric), so the prompts that
    trained it are not the prompts it should be run with.
    """

    backend = "moeclip"

    def __init__(self, name: str = presets.DEFAULT_MODEL_NAME, **kwargs):
        super().__init__(name=name, **kwargs)
        self.model_name = presets.resolve_model_name(name)
        self.arch_kwargs = {**presets.default_arch_kwargs(),
                            **{k: v for k, v in kwargs.items() if k in presets.DEFAULT_ARCH_KWARGS}}
        self.prompt_class: str | None = kwargs.get("prompt_class")
        self.prompts: dict[str, str] = dict(kwargs.get("prompts") or {})
        self._model = None
        self._loaded_path: str | None = None

    def class_name_for(self, sample: Sample) -> str:
        """This adapter's prompt-class policy for one sample (see the class
        docstring). Passed to `SampleDataset` so a batch's `class_name`
        entries decide which text embeddings it is scored against.
        """

        return presets.class_name_for(sample, forced=self.prompt_class)

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

    def _check_backbone_weights(self, model_name: str) -> Path:
        """MoECLIP loads its OpenCLIP backbone from a file the upstream
        repo expects you to download by hand (`model/ViT-L-14-336px.pt`);
        nothing fetches it, and `create_model` would fail deep inside
        `load_openai_model` with `Model <PosixPath> not found`. Fail here
        instead, with the location and the download link.
        """

        preset = presets.model_preset(model_name)
        path = vendor_root() / "model" / preset["checkpoint_file"]
        if not path.is_file():
            raise FileNotFoundError(
                f"MoECLIP backbone weights not found at {path}. Download the OpenCLIP "
                f"{model_name}px checkpoint from {presets.CHECKPOINT_DOWNLOAD_URL} and "
                "place it there (see components/moeclip/README.md)."
            )
        return path

    def _build_model(self, model_name: str, arch: dict[str, Any], device):
        """Build `MoECLIP` exactly as upstream's `train.py`/`test.py` main
        does: a frozen OpenCLIP backbone from the local checkpoint, wrapped
        with MoE adapters at `moe_layers`.
        """

        vendored = import_vendor()
        create_model = vendored["model.clip"].create_model
        MoECLIP = vendored["model.moe_adapter"].MoECLIP

        self._check_backbone_weights(model_name)

        clip_model = create_model(
            model_name=model_name,
            img_size=arch["img_size"],
            device=device,
            pretrained="openai",
            require_pretrained=True,
        )
        clip_model.eval()

        with cuda_free_module_init(device):
            model = MoECLIP(
                clip_model=clip_model,
                use_paa=arch["use_paa"],
                seg_proj_sharing_strategy=arch["seg_proj_sharing_strategy"],
                image_adapt_weight=arch["image_adapt_weight"],
                moe_r=arch["moe_r"],
                moe_lora_alpha=arch["moe_lora_alpha"],
                moe_num_experts=arch["moe_num_experts"],
                moe_top_k=arch["moe_top_k"],
                moe_layers=list(arch["moe_layers"]),
                use_fofs=arch["use_fofs"],
                relu=arch["relu"],
            )
        return model.to(device)

    def _resolved_arch(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        arch = dict(self.arch_kwargs)
        if config:
            arch.update({k: v for k, v in config.items() if k in presets.DEFAULT_ARCH_KWARGS})
        self._validate_arch(arch)
        return arch

    def _validate_arch(self, arch: dict[str, Any]) -> None:
        preset = presets.model_preset(self.model_name)
        patch = preset["patch_size"]
        img_size = int(arch["img_size"])
        if img_size % patch:
            raise ValueError(
                f"img_size={img_size} must be a multiple of the backbone's patch size "
                f"({patch}); MoECLIP reshapes patch tokens into a square grid."
            )
        depth = preset["num_layers"]
        bad = [layer for layer in arch["moe_layers"] if not 0 <= int(layer) < depth]
        if bad:
            raise ValueError(
                f"moe_layers {bad} out of range for a {depth}-block backbone (expected 0..{depth - 1})."
            )

    # ------------------------------------------------------------------ #
    # Prompt / text-embedding helpers
    # ------------------------------------------------------------------ #
    def _text_embeddings(self, model, class_names, device) -> dict[str, Any]:
        """One `(768, 2)` normal/abnormal text embedding per class name,
        built by upstream's own prompt ensemble (`forward_utils`), against
        the prompt classes registered by `presets.register_class_prompts`.
        """

        get_embedding = import_vendor()["forward_utils"].get_adapted_single_class_text_embedding
        dataset_key = presets.register_class_prompts(class_names, self.prompts)
        return {
            class_name: get_embedding(model, dataset_key, class_name, device)
            for class_name in class_names
        }

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    def _select_train_samples(self, samples: list[Sample]) -> tuple[list[Sample], dict[str, int]]:
        """Drop defective training samples that carry no pixel mask.

        MoECLIP's loss is dominated by a per-pixel segmentation term
        (focal + dice against the mask). A defective image with no mask
        would enter that term as an all-zero mask, i.e. as *supervision
        that the defect is not there* — actively worse than not training on
        it. Some in-domain sources are partially masked on purpose (Fabric
        Defects ships masks for hole/vertical/horizontal only, TILDA-400
        has none at all), so this filters rather than raises, and reports
        the counts on the returned artifact.
        """

        kept: list[Sample] = []
        dropped = 0
        for sample in samples:
            if sample.annotations.is_anomalous and not sample.annotations.anomaly_mask:
                dropped += 1
                continue
            kept.append(sample)
        stats = {
            "train_samples": len(kept),
            "train_defective": sum(1 for s in kept if s.annotations.is_anomalous),
            "dropped_unmasked_defects": dropped,
        }
        if not kept:
            raise ValueError(
                "no usable MoECLIP training samples: every sample was either absent or a "
                "defective image without a pixel mask. Select a mask-bearing split "
                "(e.g. raw-fabric / zju-leaper / fabric-defects with task: segmentation)."
            )
        if stats["train_defective"] == 0:
            raise ValueError(
                "MoECLIP training needs defective samples with pixel masks — it learns "
                "prompt-aligned anomaly features from labelled anomalies, unlike the "
                "one-class backends (PatchCore/Dinomaly). Set use_defect: true and a "
                "task that attaches masks in the train_selection."
            )
        return kept, stats

    # Canonical `TrainConfig` field -> this backend's real key.
    TRAIN_CONFIG_KEYS = {
        "epochs": "epochs",
        "lr": "lr",
        "batch_size": "batch_size",
        "device": "device",
        "seed": "seed",
        "num_workers": "num_workers",
        "work_dir": "work_dir",
    }

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tasks=("anomaly",),
            prediction_fields=("anomaly_score", "anomaly_map"),
            # Unlike the one-class backends, MoECLIP trains on *labelled*
            # defects with pixel masks (see `_select_train_samples`).
            required_annotations=("anomaly_mask", "is_anomalous"),
            # See `export()`: the MoE router's data-dependent dispatch does not
            # trace to a static graph.
            export_targets=(),
            supports_amp=False,
        )

    def train(self, config: dict[str, Any] | TrainConfig) -> Artifact:
        """Re-implements `train.py::train_adapter` against this project's
        data. Data comes in as `config['train_samples']` — a `Sample` list
        that must contain defective samples *with* `annotations.anomaly_mask`
        (see `_select_train_samples`); unlike the one-class backends there
        is no on-disk folder mode, because upstream reads `.jsonl` metadata
        rather than a folder layout (see `data.py`).

        Optional keys, defaulting from `presets.DEFAULT_TRAIN_KWARGS` /
        `DEFAULT_ARCH_KWARGS`: `epochs`, `batch_size`, `lr`,
        `balance_loss_lambda`, `etf_loss_lambda`, `lr_milestones`,
        `lr_gamma`, `seed`, `num_workers`, `device`, `work_dir`
        (checkpoint destination; defaults to a fresh temp dir — pass this,
        or call `register_trained_model()` afterwards, to keep the
        checkpoint past process exit), plus any architecture knob.
        """

        config = resolve_train_config(config, self.TRAIN_CONFIG_KEYS)

        import torch
        import torch.nn.functional as F
        from torch.optim.lr_scheduler import MultiStepLR
        from torch.utils.data import DataLoader

        vendored = import_vendor()
        calculate_similarity_map = vendored["forward_utils"].calculate_similarity_map
        calculate_seg_loss = vendored["forward_utils"].calculate_seg_loss

        kwargs = {**presets.default_train_kwargs(),
                  **{k: v for k, v in config.items() if k in presets.DEFAULT_TRAIN_KWARGS}}
        arch = self._resolved_arch(config)
        device = self._resolve_device(config.get("device"))
        num_workers = int(config.get("num_workers", 0))

        train_samples = config.get("train_samples")
        if not train_samples:
            raise ValueError("MoECLIP training requires config['train_samples'] (a list of Sample).")
        samples, stats = self._select_train_samples(list(train_samples))

        torch.manual_seed(int(kwargs["seed"]))

        img_size = int(arch["img_size"])
        dataset = SampleDataset(samples, img_size, self.class_name_for, train=True)
        loader = DataLoader(
            dataset,
            batch_size=min(int(kwargs["batch_size"]), len(samples)),
            shuffle=True,
            num_workers=num_workers,
        )

        model = self._build_model(self.model_name, arch, device)
        # Upstream keeps the whole model in eval() during training: only
        # the adapters learn, and the frozen CLIP backbone's dropout/norm
        # statistics must not drift. The MoE blocks read `self.training`
        # to pick their dense (train) vs top-k routed (eval) path, so this
        # also matters for which routing branch runs — kept as upstream
        # has it rather than "fixed".
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        params_to_train: list[dict[str, Any]] = []
        for param in model.text_adapter.parameters():
            param.requires_grad = True
        params_to_train.append({"params": list(model.text_adapter.parameters())})

        image_params = []
        for name, param in model.image_adapter.named_parameters():
            # FOFS freezes each expert's LoRA `A` at its fixed orthogonal
            # slice of the feature space -- that separation is the point of
            # the method, so `A` must not be learned when it's enabled.
            trainable = not (arch["use_fofs"] and "lora_A" in name)
            param.requires_grad = trainable
            if trainable:
                image_params.append(param)
        params_to_train.append({"params": image_params})

        optimizer = torch.optim.Adam(params_to_train, lr=float(kwargs["lr"]), betas=(0.5, 0.999))
        scheduler = MultiStepLR(
            optimizer, milestones=list(kwargs["lr_milestones"]), gamma=float(kwargs["lr_gamma"])
        )

        class_names = sorted({self.class_name_for(sample) for sample in samples})
        epochs = int(kwargs["epochs"])
        work_dir = Path(config.get("work_dir") or tempfile.mkdtemp(prefix="fdh_moeclip_"))
        work_dir.mkdir(parents=True, exist_ok=True)
        history_path = work_dir / "history.csv"
        step = 0
        # `len(loader)` is the batch count per epoch, so this is the exact
        # step budget -- MoECLIP's loop is epoch-driven, unlike the
        # iteration-driven Dinomaly/MambaAD ones.
        progress = ProgressReporter(
            f"moeclip/{self.model_name} train", total=epochs * len(loader)
        )
        with history_path.open("w", newline="") as history_file, progress:
            writer = csv.writer(history_file)
            writer.writerow(["epoch", "iteration", "train_loss", "lr"])
            for epoch in range(epochs):
                for batch in loader:
                    image = batch["image"].to(device)
                    mask = batch["mask"].to(device)
                    label = batch["label"].to(device)
                    batch_classes = batch["class_name"]

                    # Rebuilt every step on purpose: the text adapter is being
                    # trained, so its embeddings change between steps and the
                    # loss has to backprop through them (upstream does the same).
                    embeddings = self._text_embeddings(model, sorted(set(batch_classes)), device)
                    text_feature = torch.stack([embeddings[name] for name in batch_classes], dim=0)

                    patch_features, det_feature, aux_loss, special_loss = model(image)

                    cls_preds = torch.matmul(det_feature.unsqueeze(1), text_feature)[:, 0]
                    loss = F.cross_entropy(cls_preds, label)
                    for feature in patch_features:
                        loss = loss + calculate_seg_loss(
                            calculate_similarity_map(feature, text_feature, img_size), mask
                        )
                    loss = loss + aux_loss * float(kwargs["balance_loss_lambda"])
                    loss = loss + special_loss * float(kwargs["etf_loss_lambda"])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    loss_value = loss.detach().item()
                    current_lr = optimizer.param_groups[0]["lr"]
                    writer.writerow([epoch + 1, step, f"{loss_value:.8f}", f"{current_lr:.10f}"])
                    history_file.flush()
                    progress.update(epoch=f"{epoch + 1}/{epochs}", loss=loss_value, lr=current_lr)

        ckpt_path = work_dir / f"moeclip_{self.model_name}.pth"
        # Same payload upstream's `save_checkpoint` writes (minus the
        # optimizer state, which only exists there to resume a run):
        # the two adapter state_dicts, which is everything that trained.
        torch.save(
            {
                "epoch": epochs,
                "text_adapter": model.text_adapter.state_dict(),
                "image_adapter": model.image_adapter.state_dict(),
            },
            ckpt_path,
        )

        return Artifact(
            path=str(ckpt_path),
            backend=self.backend,
            metadata={
                "model_class": "MoECLIP",
                "model_name": self.model_name,
                # The prompt classes this run trained against -- recorded so
                # a checkpoint is traceable to the corpus it saw, not to
                # configure inference (see the class docstring).
                "class_names": class_names,
                "history_csv": str(history_path),
                "trusted": True,
                "training": describe_training(optimizer, scheduler),
                "batch_spec": dataset.batch_spec().as_run_metadata(),
                **arch,
                **stats,
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
        """Loads the adapter checkpoint and scores each sample with
        upstream's own test-time path (`test.py::get_predictions`).

        * `anomaly_map` (written as `.npy` when `output_dir` is given, and
          needed for `evaluation.anomaly.AnomalyEvaluator`'s pixel metrics)
          is upstream's map exactly: per level, `100 * cos(patch, text)`
          turned into `(abnormal + 1 - normal) / 2`, Gaussian-blurred with
          the Industrial kernel and bilinearly upsampled, then summed over
          the four aggregation levels.
        * `anomaly_score` is upstream's `0.5 * image + 0.5 * pixel-max`
          blend for the Industrial domain (`forward_utils.metrics_eval`),
          with one deliberate change: upstream min-max normalizes both
          terms *across the whole test set* before blending, which makes a
          single image's score depend on what else was in the batch (and
          undefined for one image on its own — the Single Image tab's case).
          Here the pixel term is instead a softmax over the two prompt
          classes, which is per-sample, already in [0, 1], and a monotone
          function of the same `abnormal - normal` margin — so image-level
          rankings, and therefore AUROC, are unchanged for a fixed test set
          while a lone image still scores meaningfully.
        """

        import numpy as np
        import torch
        import torch.nn.functional as F

        vendored = import_vendor()
        calculate_similarity_map = vendored["forward_utils"].calculate_similarity_map

        device = self._resolve_device(None)
        model = self._load_artifact(artifact, device)
        img_size = int(artifact.metadata.get("img_size", self.arch_kwargs["img_size"]))

        class_names = sorted({self.class_name_for(sample) for sample in samples})
        dataset = SampleDataset(samples, img_size, self.class_name_for, train=False)

        maps_dir = None
        if output_dir is not None:
            maps_dir = Path(output_dir)
            maps_dir.mkdir(parents=True, exist_ok=True)

        predictions: list[Prediction] = []
        progress = ProgressReporter(
            f"moeclip/{self.model_name} predict", total=len(samples), unit="img"
        )
        with torch.no_grad(), progress:
            embeddings = self._text_embeddings(model, class_names, device)
            for index, sample in enumerate(samples):
                item = dataset[index]
                image = item["image"].unsqueeze(0).to(device)
                text_feature = embeddings[item["class_name"]]

                patch_features, det_feature, _aux, _special = model(image)

                # Image branch: cosine to the abnormal prompt, mapped to [0, 1].
                det_score = float(((det_feature @ text_feature)[:, 1] + 1).div(2).item())

                anomaly_map = None
                pixel_probs = []
                for feature in patch_features:
                    blurred = calculate_similarity_map(
                        feature, text_feature, img_size, test=True, domain=presets.PROMPT_DOMAIN
                    )
                    anomaly_map = blurred if anomaly_map is None else anomaly_map + blurred

                    logits = 100.0 * torch.matmul(feature, text_feature)  # (1, L, 2)
                    side = int(round(logits.shape[1] ** 0.5))
                    logits = logits.permute(0, 2, 1).reshape(1, 2, side, side)
                    prob = torch.softmax(logits, dim=1)[:, 1:2]
                    pixel_probs.append(
                        F.interpolate(prob, size=img_size, mode="bilinear", align_corners=True)
                    )

                pixel_score = float(torch.stack(pixel_probs).mean(dim=0).max().item())
                score = 0.5 * det_score + 0.5 * pixel_score

                anomaly_map_path = None
                if maps_dir is not None:
                    array = anomaly_map[0, 0].detach().cpu().numpy()
                    map_path = maps_dir / f"{sample.id}.npy"
                    map_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(map_path, array)
                    anomaly_map_path = str(map_path)

                predictions.append(
                    Prediction(
                        sample_id=sample.id,
                        anomaly_score=score,
                        anomaly_map=anomaly_map_path,
                    )
                )
                progress.update(score=score)
        return predictions

    def export(
        self, artifact: Artifact, target: str, config: dict[str, Any] | None = None
    ) -> ExportedArtifact:
        raise NotImplementedError(
            "MoECLIP export is not implemented: the mixture-of-experts router "
            "dispatches patches to experts with data-dependent control flow "
            "(`BaseIndependentMoE._vit_forward` loops over `torch.where` hits and "
            "skips empty experts), which does not trace to a static ONNX graph as "
            "written. Add and verify this explicitly before relying on it."
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
        filename = model_name or f"moeclip_{artifact.metadata.get('model_name', self.model_name)}.pth"
        dst = registry / filename
        shutil.copy2(src, dst)

        metadata = dict(artifact.metadata)
        history = Path(metadata["history_csv"]) if metadata.get("history_csv") else None
        if history is not None and history.is_file():
            history_dst = registry / f"{dst.stem}.history.csv"
            shutil.copy2(history, history_dst)
            metadata["history_csv"] = str(history_dst)
        metadata["registered_from"] = str(src)
        return Artifact(path=str(dst), backend=self.backend, metadata=metadata)

    def load_trained_model(self, artifact_or_path: Artifact | str) -> Artifact:
        """Load a previously registered/trained checkpoint back into this
        adapter. Given a bare path, the architecture defaults to this
        adapter's own `model_name`/arch kwargs -- pass an `Artifact`
        carrying the metadata `train()` produced if the checkpoint was
        trained with different ones (a different expert count or
        `img_size` will otherwise fail the `load_state_dict` shape check).
        """

        if isinstance(artifact_or_path, Artifact):
            artifact = artifact_or_path
        else:
            path = artifact_or_path
            if not Path(path).exists():
                raise FileNotFoundError(f"cannot load missing checkpoint: {path}")
            artifact = Artifact(
                path=str(path),
                backend=self.backend,
                metadata={
                    "model_class": "MoECLIP",
                    "model_name": self.model_name,
                    "trusted": True,
                    **self.arch_kwargs,
                },
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

            model_name = artifact.metadata.get("model_name", self.model_name)
            arch = {**self.arch_kwargs,
                    **{k: v for k, v in artifact.metadata.items() if k in presets.DEFAULT_ARCH_KWARGS}}
            model = self._build_model(model_name, arch, device)

            checkpoint = torch.load(artifact.path, map_location=device)
            if "text_adapter" not in checkpoint or "image_adapter" not in checkpoint:
                raise ValueError(
                    f"{artifact.path} is not a MoECLIP checkpoint: expected 'text_adapter' and "
                    f"'image_adapter' state_dicts, got keys {sorted(checkpoint)}."
                )
            model.text_adapter.load_state_dict(checkpoint["text_adapter"])
            model.image_adapter.load_state_dict(checkpoint["image_adapter"])

            self._model = model.eval()
            self._loaded_path = artifact.path
        return self._model
