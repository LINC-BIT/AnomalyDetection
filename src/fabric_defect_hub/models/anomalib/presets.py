"""Per-algorithm configuration for the anomalib models this project exposes.

We deliberately do not reimplement these algorithms (see the project's
"don't reimplement models we can depend on" principle in the top-level
README) — `anomalib` already ships correct, maintained implementations.
What *is* our responsibility, and what lives here, is:

1. Alias resolution: the README/paper names ("RD4AD", "EfficientAD",
   "SuperSimpleNet") don't match anomalib's class names
   (`ReverseDistillation`, `EfficientAd`, `Supersimplenet`). Callers should
   be able to use either.
2. Fabric-tailored defaults: anomalib's own defaults are tuned for MVTec-AD
   style single-object images at 256x256. `MODEL_PRESETS` below adjusts the
   handful of knobs that matter for fabric texture images (uniform
   background, defects are small/local, and — per `ZJULeaperDataset` — the
   caller can dial in a low-shot or zero-shot sample count), while still
   letting `train_config['model_kwargs']` override anything.

`MODEL_PRESETS` doubles as the registration list: a class is reachable
through this backend only if it has an entry here (see
`resolve_model_class_name` / `list_supported_variants`). An entry whose
values are simply upstream's own defaults is therefore not redundant — it
is how the model is admitted, and the comment above each entry says
explicitly whether anything was tuned for fabric or not. Per this
project's citation rule, a preset only claims a fabric adjustment where
one was actually made and can be justified.

Families covered (so a reader can see the benchmark's breadth at a glance):

  memory bank / statistics  PatchCore, PaDiM
  teacher-student           RD4AD, STFPM, UniNet
  reconstruction            DRAEM, DSR
  adversarial (GAN)         GANomaly
  normalizing flow          FastFlow
  synthetic-anomaly         SuperSimpleNet, GLASS
  zero-shot VLM             WinCLIP, AnomalyDINO
  regularized distillation  EfficientAD

Verified against anomalib 2.5.0 (`pip install -e ".[anomalib]"`); every
class name and every constructor kwarg below was introspected directly
from that installed version with `inspect.signature`, not guessed, and
`tests/test_anomalib_config.py::test_every_preset_key_is_a_real_constructor_kwarg`
re-checks that against the installed anomalib rather than trusting this
note to stay true across upgrades.
"""

from __future__ import annotations

from typing import Any


def prompt_class_for_samples(samples: list[Any] | None, forced: str | None = None) -> str:
    """Resolve the noun used by CLIP prompts from dataset sample metadata.

    General anomaly datasets expose ``category`` (for example ``bottle`` or
    ``pcb1``); textile adapters expose ``fabric_type`` or no class metadata.
    A mixed-category batch uses the neutral ``object`` noun rather than
    silently selecting one category.
    """
    if forced and str(forced).strip():
        return str(forced).strip()
    categories: set[str] = set()
    for sample in samples or []:
        metadata = getattr(sample, "metadata", None) or {}
        value = metadata.get("category")
        if value:
            categories.add(str(value).strip())
    if len(categories) == 1:
        return next(iter(categories))
    if len(categories) > 1:
        return "object"
    return "fabric"

# README/paper name -> anomalib.models class name.
MODEL_ALIASES: dict[str, str] = {
    "patchcore": "Patchcore",
    "padim": "Padim",
    "rd4ad": "ReverseDistillation",
    "reversedistillation": "ReverseDistillation",
    "efficientad": "EfficientAd",
    "supersimplenet": "Supersimplenet",
    "winclip": "WinClip",
    # -- teacher-student --
    "stfpm": "Stfpm",
    "uninet": "UniNet",
    # -- reconstruction --
    "draem": "Draem",
    "dsr": "Dsr",
    # -- adversarial (GAN) --
    "ganomaly": "Ganomaly",
    # -- normalizing flow --
    "fastflow": "Fastflow",
    # -- synthetic-anomaly + discriminator --
    "glass": "Glass",
    # -- zero-shot VLM --
    "anomalydino": "AnomalyDINO",
    "dinov2": "AnomalyDINO",
}

# Models whose inference output carries no pixel-level anomaly map, only an
# image-level score. Checked against each class's `torch_model.forward`
# return in anomalib 2.5.0: every other model here returns an
# `InferenceBatch` with `anomaly_map` set, while GANomaly's returns
# `InferenceBatch(pred_score=scores)` and nothing else — it compares two
# latent vectors, so there is no spatial map to produce.
#
# `AnomalibAdapter.capabilities()` reads this to drop `anomaly_map` from its
# declared `prediction_fields`, so an evaluator can tell *before running
# anything* that pixel AUROC/AUPRO is not computable for this model, rather
# than computing a meaningless number over absent maps.
IMAGE_LEVEL_ONLY: frozenset[str] = frozenset({"Ganomaly"})

# Fabric-tailored default `model_kwargs` per algorithm, keyed by the
# anomalib class name (post-alias-resolution). Every key here is a real
# constructor parameter on the corresponding anomalib class.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    # Memory-bank of nominal patch features + kNN distance. Works well with
    # few normal reference images (our low-shot regime) since it needs no
    # gradient training, only a forward pass + coreset subsampling.
    "Patchcore": {
        "backbone": "wide_resnet50_2",
        "layers": ("layer2", "layer3"),
        "pre_trained": True,
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
    },
    # Per-patch multivariate Gaussian over pretrained features. Cheap and
    # stable with a moderate number of normal images; good zero-shot-style
    # baseline (no defect images needed at all, per `use_defect=False`).
    "Padim": {
        "backbone": "resnet18",
        "layers": ["layer1", "layer2", "layer3"],
        "pre_trained": True,
        "n_features": None,  # None = anomalib's per-backbone default
    },
    # Teacher/student reverse distillation. Needs an actual training loop
    # (unlike PatchCore/PaDiM), so benefits from more normal samples.
    "ReverseDistillation": {
        "backbone": "wide_resnet50_2",
        "layers": ("layer1", "layer2", "layer3"),
        "pre_trained": True,
        "anomaly_map_mode": "add",
    },
    # Distillation against a pretrained teacher, regularized on a generic
    # natural-image set (`imagenet_dir`). Fabric texture has no object
    # semantics, so `imagenet_dir` MUST be supplied by the caller — there is
    # no fabric-appropriate default, and `AnomalibAdapter.train()` raises
    # early if it is missing rather than failing deep inside Lightning.
    "EfficientAd": {
        "model_size": "small",
        "teacher_out_channels": 384,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "imagenet_dir": None,
    },
    # The one supervised-capable model in this set. ZJU-Leaper ships real
    # defect images + masks (see `ZJULeaperDataset`), so default to
    # `supervised=True` to actually use them instead of synthetic Perlin
    # anomalies; set `use_defect=False` upstream to fall back to the
    # unsupervised synthetic-anomaly mode anomalib was designed for.
    "Supersimplenet": {
        "backbone": "wide_resnet50_2.tv_in1k",
        "layers": ["layer2", "layer3"],
        "supervised": True,
        "perlin_threshold": 0.2,
    },
    # CLIP-based window model. Unlike the five above, WinCLIP does no
    # gradient training and no feature-memory fit -- it scores each window
    # against text prompts derived from `class_name` (zero-shot), optionally
    # augmented with `k_shot` normal reference images (few-shot). Defaulting
    # to `k_shot=0` keeps it a genuine zero-shot baseline that needs no
    # fabric training data at all -- its distinguishing capability in this
    # benchmark, where every other anomaly model consumes normal images.
    # Set `k_shot` > 0 (via train.model_kwargs) to switch to few-shot; the
    # normal reference images are then collected from the staged train split
    # in the model's `setup()`, exactly as the training pipeline already
    # provides them. `class_name` is the object noun WinCLIP's prompt
    # ensemble is built around; "fabric" fits this domain (anomalib would
    # otherwise fall back to the datamodule name, which is the backend id
    # here, not a meaningful prompt).
    "WinClip": {
        "class_name": "fabric",
        "k_shot": 0,
        "scales": (2, 3),
    },
    # ---------------------------------------------------------------- #
    # Teacher-student
    # ---------------------------------------------------------------- #
    # Feature-pyramid matching between a frozen pretrained teacher and a
    # student trained from scratch; the anomaly map is the per-level
    # feature discrepancy. Values are upstream's own defaults (ResNet-18,
    # layer1-3) restated -- deliberately *not* tuned to match the
    # wide_resnet50_2 the memory-bank models here use. STFPM trains its
    # student from scratch, so a heavier teacher raises cost without a
    # published fabric result to justify it; keeping layer1 matters more,
    # since that is the level where small local fabric defects show up.
    "Stfpm": {
        "backbone": "resnet18",
        "layers": ("layer1", "layer2", "layer3"),
    },
    # Contrastive teacher-student with a *pretrained* student (unlike
    # STFPM's from-scratch one) and a temperature-scaled similarity loss.
    # Upstream defaults; `wide_resnet50_2` for both sides matches the
    # backbone RD4AD/PatchCore already use here, so the three
    # teacher-student rows stay comparable on feature capacity.
    "UniNet": {
        "student_backbone": "wide_resnet50_2",
        "teacher_backbone": "wide_resnet50_2",
        "temperature": 0.1,
    },
    # ---------------------------------------------------------------- #
    # Reconstruction
    # ---------------------------------------------------------------- #
    # Reconstructive + discriminative sub-network trained on synthetic
    # anomalies blended from an external texture source (DTD).
    #
    # `dtd_dir` is pointed at this project's own `data/<Dataset>` staging
    # convention instead of upstream's `./datasets/dtd`, for one specific
    # reason: anomalib's Draem does *not* fail when that directory is
    # missing -- it calls `download_and_extract(dtd_dir, DTD_DOWNLOAD_INFO)`
    # and pulls ~600MB over the network mid-training. On a metered or
    # offline training box that is a surprise, not a convenience, so
    # `AnomalibAdapter._validate_model_kwargs` raises up front and tells the
    # caller to either stage DTD or opt into the download explicitly.
    #
    # NOTE on `beta`: upstream's default (0.1, 1.0) is the blend-factor
    # *range* the synthetic anomaly is composited with. Left as upstream --
    # no fabric-specific value has been measured.
    "Draem": {
        "dtd_dir": "datasets/general/DTD",
        "enable_sspcab": False,
        "beta": (0.1, 1.0),
    },
    # Dual-subspace re-projection: a discretely-encoded (VQ) latent plus a
    # reconstruction branch, so unlike DRAEM it needs no external texture
    # source. Upstream defaults; nothing fabric-specific to tune here --
    # both knobs govern its synthetic latent-anomaly schedule, and no
    # fabric-measured value exists to replace them with.
    "Dsr": {
        "latent_anomaly_strength": 0.2,
        "upsampling_train_ratio": 0.7,
    },
    # ---------------------------------------------------------------- #
    # Adversarial (GAN)
    # ---------------------------------------------------------------- #
    # Encoder-decoder-encoder generator + discriminator; the anomaly score
    # is the L2 distance between the two latent vectors. The benchmark's
    # only adversarial entry.
    #
    # Two things about this model are unlike every other entry here, and
    # both are load-bearing:
    #
    # 1. `batch_size` is a *constructor* argument, not an engine one, so the
    #    canonical `TrainConfig.batch_size` does NOT reach it (see
    #    `AnomalibAdapter.TRAIN_CONFIG_KEYS`, which maps only genuinely flat
    #    keys). Checked what it actually does in anomalib 2.5.0 rather than
    #    assuming: it sizes two buffers, `real_label`/`fake_label`, and
    #    nothing in `Ganomaly.training_step` reads them -- both loss modules
    #    build their BCE targets from the discriminator output's own shape.
    #    So a mismatch with the loader's batch size is inert *today*; it is
    #    documented here, and left un-validated, precisely because adding a
    #    guard would assert a coupling that this version does not have.
    # 2. It produces no pixel-level anomaly map -- see `IMAGE_LEVEL_ONLY`.
    #    Image AUROC is computable for it; pixel AUROC/AUPRO is not.
    #
    # Values are upstream's defaults. `wadv`/`wcon`/`wenc` are the
    # adversarial/contextual/encoder loss weights from the GANomaly paper;
    # `lr`/`beta1`/`beta2` are its published Adam settings.
    "Ganomaly": {
        "batch_size": 32,
        "n_features": 64,
        "latent_vec_size": 100,
        "wadv": 1,
        "wcon": 50,
        "wenc": 1,
        "lr": 2e-4,
        "beta1": 0.5,
        "beta2": 0.999,
    },
    # ---------------------------------------------------------------- #
    # Normalizing flow
    # ---------------------------------------------------------------- #
    # 2D normalizing flow over pretrained features; the anomaly map is the
    # per-position negative log-likelihood. Upstream defaults throughout.
    #
    # `conv3x3_only` is left at anomalib's `False` on purpose. The FastFlow
    # paper is reported to use 3x3-only convolutions for its CNN backbones
    # (and alternating kernels for ViT/CaiT), which would argue for `True`
    # here given the pinned ResNet-18 -- but anomalib 2.5.0 ships no config
    # tying the flag to the backbone, and that reading has not been checked
    # against the paper itself. Flipping it on an unverified recollection is
    # exactly the kind of invented anchor this project's citation rule
    # forbids, so it stays at the installed default until someone reads the
    # paper and records what it says.
    "Fastflow": {
        "backbone": "resnet18",
        "pre_trained": True,
        "flow_steps": 8,
        "conv3x3_only": False,
        "hidden_ratio": 1.0,
    },
    # ---------------------------------------------------------------- #
    # Synthetic-anomaly + discriminator
    # ---------------------------------------------------------------- #
    # Global-and-local synthetic anomaly synthesis in feature space, with a
    # discriminator trained by gradient ascent toward the normal manifold's
    # boundary. Upstream defaults except `input_shape`, which is set to the
    # 256x256 the rest of this backend's models run at (upstream's 288x288
    # is its MVTec setting) so a GLASS row is comparable with the others.
    #
    # `anomaly_source_path` is genuinely optional here, unlike DRAEM's
    # `dtd_dir`: anomalib's Glass only builds a DTD-backed generator when it
    # is given (`if anomaly_source_path is not None`), and otherwise relies
    # on its feature-space synthesis alone. Left as None -- no download, no
    # validation needed.
    "Glass": {
        "input_shape": (256, 256),
        "anomaly_source_path": None,
        "backbone": "wide_resnet50_2",
        "pre_trained": True,
        "patchsize": 3,
        "mining": True,
    },
    # ---------------------------------------------------------------- #
    # Zero-shot VLM
    # ---------------------------------------------------------------- #
    # DINOv2 patch features + nearest-neighbour scoring. Like WinCLIP it
    # needs no gradient training, but unlike WinCLIP it is *not* text
    # prompted -- so it is the control that isolates how much of WinCLIP's
    # zero-shot score comes from language priors versus from the visual
    # backbone alone. That contrast is why both are in the benchmark.
    #
    # `masking=False` is a deliberate fabric adjustment, not an upstream
    # default restated: upstream's masking step segments a foreground
    # object from its background, which is meaningful for MVTec's
    # single-object images and actively wrong for full-frame fabric
    # texture, where there is no background to remove.
    "AnomalyDINO": {
        "encoder_name": "dinov2_vit_small_14",
        "num_neighbours": 1,
        "masking": False,
        "coreset_subsampling": True,
        "sampling_ratio": 0.1,
    },
}


def resolve_model_class_name(name: str) -> str:
    """Map a README alias or literal anomalib class name to the anomalib class name."""

    canonical = MODEL_ALIASES.get(name.strip().lower())
    if canonical is not None:
        return canonical
    if name in MODEL_PRESETS:
        return name
    known = sorted(set(MODEL_ALIASES) | set(MODEL_PRESETS))
    raise KeyError(f"unknown anomalib model {name!r}. Known names: {known}")


def resolve_model_class(name: str):
    """Return the actual `anomalib.models.<Class>` for a README alias or class name."""

    import anomalib.models as anomalib_models

    class_name = resolve_model_class_name(name)
    return getattr(anomalib_models, class_name)


def default_model_kwargs(name: str) -> dict[str, Any]:
    """Fabric-tailored default constructor kwargs for `name` (copy, safe to mutate)."""

    class_name = resolve_model_class_name(name)
    return dict(MODEL_PRESETS.get(class_name, {}))


def list_supported_variants() -> list[str]:
    """Every model name this backend accepts, under the name all six
    backends' preset modules share (`api.list_models` calls it on each of
    them).
    """

    return sorted(MODEL_PRESETS)
