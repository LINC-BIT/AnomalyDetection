"""Backbone presets, upstream training defaults, and the text-prompt
registry for MoECLIP, mirrored from upstream's `train.py`/`test.py`
argparse defaults rather than reinvented.

MoECLIP is one architecture (OpenCLIP ViT-L/14-336 + per-patch LoRA
mixture-of-experts + PAA aggregation), not a model zoo, so -- as with
`models/dinomaly/presets.py` -- there is no alias table, just the backbone
it supports and the knobs that make up its training recipe. The backbone
is genuinely fixed upstream, not merely defaulted: `MoECLIP.__init__`
hardcodes `d_model = 1024` and its `forward` loops over exactly 24
transformer blocks, both ViT-L/14 specifics.

`REAL_NAME_OVERRIDES`/`class_name_for`/`register_class_prompts` cover the
other half of what upstream hardcodes, and are the *only* seam between
this project's data contracts and MoECLIP's prompt machinery: nothing in
`Sample`/`Prediction` grows a prompt field, because a prompt is not
per-image data — it is a property of what is being inspected, resolved
from the sample's dataset metadata and overridable per run via
`model.prompt_class` / `model.prompts` in the config.

MoECLIP is prompt-driven: text embeddings come
from `"a photo of a damaged {real_name}."`-style templates, and upstream
resolves `real_name` through per-dataset `CLASS_NAMES`/`REAL_NAMES` tables
in `dataset/constants.py` keyed by dataset names it ships with (MVTec,
VisA, BTAD, ...). This project feeds it arbitrary `Sample` lists instead,
so the adapter registers its own synthetic dataset key into those same
(mutable, shared-by-reference with `forward_utils`) dicts at run time --
no fork edit, and upstream's prompt machinery is used exactly as written.
"""

from __future__ import annotations

from typing import Any

# The only backbone upstream supports (see module docstring). The value is
# the model-config stem under `components/moeclip/model/model_configs/`
# and the key into its `_MODEL_CKPT_PATHS`.
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "ViT-L-14-336": {
        "d_model": 1024,
        "num_layers": 24,
        "patch_size": 14,
        # Which transformer blocks' tokens feed the segmentation head
        # (`MoECLIP.levels`, 1-indexed); fixed upstream.
        "levels": [6, 12, 18, 24],
        "checkpoint_file": "ViT-L-14-336px.pt",
    },
}

DEFAULT_MODEL_NAME = "ViT-L-14-336"

# Where the OpenCLIP backbone weights have to live -- relative to the
# vendored checkout, because `model/clip.py::_MODEL_CKPT_PATHS` resolves
# them relative to its own file and nothing in upstream downloads them.
CHECKPOINT_DOWNLOAD_URL = (
    "https://drive.google.com/file/d/1d5iKW1ojGpMkeobbxNd9h_QG27xLWuKZ/view"
)

# Architecture knobs that must match between train and predict; stored in
# `Artifact.metadata` so a checkpoint always rebuilds the model it was
# trained as. Values are upstream's `train.py` argparse defaults, with the
# two inverted `--no_use_*` store_false flags spelled out positively.
DEFAULT_ARCH_KWARGS: dict[str, Any] = {
    "img_size": 518,  # 37x37 patches at patch_size 14
    "moe_r": 8,
    "moe_lora_alpha": 16,
    "moe_num_experts": 4,
    "moe_top_k": 2,
    "moe_layers": [5, 11, 17, 23],
    "use_fofs": True,  # --no_use_fofs
    "use_paa": True,  # --no_use_paa
    "seg_proj_sharing_strategy": "shared",
    "image_adapt_weight": 0.1,
    "relu": False,  # --relu, store_true
}

# Training-loop-only defaults (not needed to rebuild the model).
DEFAULT_TRAIN_KWARGS: dict[str, Any] = {
    "epochs": 20,
    "batch_size": 2,  # upstream's --image_batch_size
    "lr": 5e-5,
    "balance_loss_lambda": 0.01,
    "etf_loss_lambda": 0.01,
    "lr_milestones": [16000, 32000],
    "lr_gamma": 0.5,
    "seed": 111,
}

# The synthetic dataset key the adapter registers into the vendored
# `dataset/constants.py` tables (see module docstring). Never collides
# with one of upstream's own keys.
PROMPT_DATASET_KEY = "FabricDefectHub"

# Domain selects the anomaly-map smoothing upstream applies at test time
# (`forward_utils.calculate_similarity_map`: sigma/kernel 1/7 for
# Industrial, 1.5/9 for Medical). Fabric inspection is industrial.
PROMPT_DOMAIN = "Industrial"

# Class name -> the noun phrase that gets templated into the prompts. Any
# class not listed here falls back to its own name with separators turned
# into spaces (see `real_name_for`), which is what makes cross-domain
# evaluation on MVTec AD/VisA/LOCO categories ("metal_nut" -> "metal nut")
# work without a per-dataset table.
REAL_NAME_OVERRIDES: dict[str, str] = {
    "fabric": "fabric texture",
    "good": "fabric texture",
}

# `Sample.metadata` keys that name what the image *is*, tried in order.
# "category" is set by every object benchmark (MVTec AD/LOCO, VisA) --
# MoECLIP's training corpora. "fabric_type" is ZJU-Leaper's own pattern
# name (see datasets/zju_leaper.py), which describes the textile being
# inspected and so makes a better inference-time prompt than the generic
# fallback. Anything else falls back to `DEFAULT_CLASS_NAME`.
#
# Deliberately NOT in this list: "defect_type". It names the *anomaly*
# (hole / oil spot / ...), i.e. the label being predicted -- prompting with
# it at test time would leak ground truth into the score. The prompt class
# must only ever describe the normal object.
CLASS_METADATA_KEYS = ("category", "fabric_type")
DEFAULT_CLASS_NAME = "fabric"


def resolve_model_name(name: str) -> str:
    if name not in MODEL_PRESETS:
        raise KeyError(
            f"unknown MoECLIP backbone {name!r}. Known backbones: {sorted(MODEL_PRESETS)}"
        )
    return name


def model_preset(name: str) -> dict[str, Any]:
    return dict(MODEL_PRESETS[resolve_model_name(name)])


def default_arch_kwargs() -> dict[str, Any]:
    return {key: (list(value) if isinstance(value, list) else value)
            for key, value in DEFAULT_ARCH_KWARGS.items()}


def default_train_kwargs() -> dict[str, Any]:
    return {key: (list(value) if isinstance(value, list) else value)
            for key, value in DEFAULT_TRAIN_KWARGS.items()}


def real_name_for(class_name: str, overrides: dict[str, str] | None = None) -> str:
    """The noun phrase templated into this class's prompts (see module
    docstring). Resolution order: the caller's `overrides` (a run's own
    `model.prompts` config), then this module's table, then the class name
    with separators turned into spaces.
    """

    key = class_name.strip().lower()
    for table in (overrides or {}, REAL_NAME_OVERRIDES):
        lowered = {name.strip().lower(): text for name, text in table.items()}
        if key in lowered:
            return lowered[key]
    return class_name.replace("_", " ").replace("-", " ").strip() or DEFAULT_CLASS_NAME


def class_name_for(sample, forced: str | None = None) -> str:
    """Which prompt class a `Sample` belongs to.

    `forced` (a run's `model.prompt_class`) pins every sample to one class,
    which is the normal case for fabric inference — one texture, one
    prompt. Otherwise it comes from the sample's own dataset metadata (see
    `CLASS_METADATA_KEYS`), which is what gives the object corpora their
    per-category prompts during training, and the fabric default otherwise.
    """

    if forced:
        return forced
    metadata = sample.metadata or {}
    for key in CLASS_METADATA_KEYS:
        value = metadata.get(key)
        if value:
            return str(value)
    return DEFAULT_CLASS_NAME


def _upstream_real_name(constants, class_name: str) -> str | None:
    """Upstream's own hand-written description for a class, if it has one.

    MoECLIP's training corpora are the very datasets `dataset/constants.py`
    ships descriptions for -- "metal nut which has four notched edges",
    "infrared sensor pcb module", ... -- and those descriptions are part of
    the published recipe, not incidental. When a `Sample`'s category is one
    of them (a VisA/MVTec/LOCO run, i.e. every training run), use
    upstream's wording rather than our generic prettifier, so the prompts
    the experts learn against are the paper's.
    """

    for dataset_key, table in constants.REAL_NAMES.items():
        if dataset_key == PROMPT_DATASET_KEY:
            continue
        if class_name in table:
            return table[class_name]
    return None


def register_class_prompts(class_names, overrides: dict[str, str] | None = None) -> str:
    """Make `class_names` resolvable by upstream's prompt builders and
    return the dataset key to pass them.

    Mutates the vendored `CLASS_NAMES`/`REAL_NAMES`/`DOMAINS` dicts *in
    place* -- `forward_utils` did `from dataset.constants import ...`, so
    it holds references to these exact objects; rebinding the names in
    `dataset.constants` instead would be invisible to it.
    """

    from fabric_defect_hub.models.moeclip.vendor import import_vendor

    constants = import_vendor()["dataset.constants"]
    known = constants.CLASS_NAMES.setdefault(PROMPT_DATASET_KEY, [])
    real = constants.REAL_NAMES.setdefault(PROMPT_DATASET_KEY, {})
    constants.DOMAINS[PROMPT_DATASET_KEY] = PROMPT_DOMAIN
    for class_name in class_names:
        if class_name not in known:
            known.append(class_name)
        # An explicit `model.prompts` entry outranks upstream's own
        # wording; otherwise upstream's is preferred (it is part of the
        # published recipe) and the generic prettifier is the last resort.
        lowered = {name.strip().lower(): text for name, text in (overrides or {}).items()}
        resolved = (
            lowered.get(class_name.strip().lower())
            or _upstream_real_name(constants, class_name)
            or real_name_for(class_name, overrides)
        )
        if real.get(class_name) != resolved:
            real[class_name] = resolved
    return PROMPT_DATASET_KEY


def list_supported_variants() -> list[str]:
    """Every backbone this backend accepts, under the name all six backends'
    preset modules share (`api.list_models` calls it on each of them). One
    entry, and that is upstream's limit rather than an omission here -- see
    this module's `MODEL_PRESETS` comment.
    """

    return sorted(MODEL_PRESETS)
