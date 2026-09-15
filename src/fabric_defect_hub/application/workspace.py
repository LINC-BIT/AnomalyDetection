"""State and inference helpers for the single-image Gradio workspace."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from fabric_defect_hub.catalog import CANONICAL_MODELS, metadata_for, published_path, read_published_metadata
from fabric_defect_hub.application import load_dataset, load_model
from fabric_defect_hub.core.availability import backend_is_importable
from fabric_defect_hub.core.checkpoint import inspect_checkpoint
from fabric_defect_hub.core.serialization import sample_from_dict, sample_to_dict
from fabric_defect_hub.core.types import Prediction, Sample
from fabric_defect_hub.models.base import Artifact
from fabric_defect_hub.i18n import DEFAULT_LANGUAGE, tr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ANOMALY_MAP_ROOT = PROJECT_ROOT / "artifacts" / "runtime" / "anomaly_maps"
_SSD_VOLUME_PARENT = Path("anomaly-detection-challenges") / "datasets"

# Generated from `catalog.CANONICAL_MODELS`
def _model_catalog_row(model):
    published_metadata = read_published_metadata(model)
    return {
        "backend": model.backend,
        "name": model.variant,
        "checkpoint": published_path(model),
        "task": model.task,
        "metadata": metadata_for(model),
        "category": model.category,
        "subtype": model.subtype,
        "integration": model.integration,
        "training_mode": model.training_mode,
        "method": model.method,
        "trained_on": published_metadata.get("trained_on", list(model.trained_on)),
        "training_split": published_metadata.get("training_split", model.training_split),
        "evaluated_on": published_metadata.get("evaluated_on", list(model.evaluated_on)),
        "domain": model.domain,
    }


MODEL_CATALOG = {
    model.label: _model_catalog_row(model)
    for model in CANONICAL_MODELS
}

# Frozen on the first `available_model_labels()` call. See that function for
# why availability is decided once per process rather than per widget.
_AVAILABLE_MODEL_LABELS: list[str] | None = None


def model_is_available(model_label: str) -> bool:
    """Whether this catalog entry's published weight is staged on this
    machine.

    A published slot is a path (README 2.2.7); the file at it is what
    `adh train --publish` writes or `tools/download_weights.py` downloads
    (README 2.2.6). No file means the model was never trained or fetched
    here, so it cannot be run.
    """

    return Path(MODEL_CATALOG[model_label]["checkpoint"]).is_file()


def available_model_labels() -> list[str]:
    """The catalog labels a model selector may offer, in manifest order.

    Both selectors read this — the Single Image "Model" dropdown and the
    Benchmark "Models to benchmark" list — so an untrained slot is never
    offered and then refused. The general-domain training slots
    (`PatchCore · General`, ...) are the everyday example: the manifest
    registers them so `adh train` has a destination, but until a run
    publishes into that slot there is nothing to load.

    Two deliberate exceptions:

    * If *nothing* is staged the full catalog is returned. An empty dropdown
      would hide the "Checkpoint missing — expected `<path>`" panel README
      5.2 documents, which is the one place that says what to download.
    * The answer is frozen on first call, so every selector on the page
      agrees for the life of the process. README 4.1.1 already makes a
      restart the way to pick a manifest change up, and a newly published
      weight is the same kind of change.
    """

    global _AVAILABLE_MODEL_LABELS
    if _AVAILABLE_MODEL_LABELS is None:
        staged = [label for label in MODEL_CATALOG if model_is_available(label)]
        _AVAILABLE_MODEL_LABELS = staged or list(MODEL_CATALOG)
    return list(_AVAILABLE_MODEL_LABELS)


# Display label -> the *presentation* facts about a registered dataset, and
# nothing else.
#
# What each entry may hold:
#   `name`         the registered `DatasetAdapter` name — the join key into
#                  `core.dataset_capabilities`
#   `env`          an environment variable a user may set to point elsewhere
#   `slice_kwarg`  which DatasetAdapter constructor kwarg the "Texture /
#                  pattern" dropdown feeds ("pattern" for ZJU-Leaper's
#                  textures, "category" for MVTec AD's object classes; None
#                  when the dataset has no such subdivision)
#   `task`         the default `Sample.task` for the Single Image gallery,
#                  which needs no ground truth and so needs only one default
#
# What each entry deliberately does NOT hold: which tasks the dataset can
# supply ground truth for, and where it lives on disk. Both are declared once
# in `core.dataset_capabilities` (`tasks`, `default_root`) and read back here
# via `dataset_tasks()` / `_dataset_dir()`. They used to be restated in this
# table as `tasks` and `dir`; the two copies happened to agree, which is the
# most dangerous state for a duplicated fact to be in.
#
# Adding a dataset means a `register_capabilities(...)` declaration plus an
# entry here, and no change to the page interaction contract, as long as
# `texture_choices`/`slice_value` below know how to enumerate its slice_kwarg.
DATASET_CATALOG = {
    "ZJU-Leaper": {
        "name": "zju-leaper",
        "env": "ZJU_LEAPER_ROOT",
        "slice_kwarg": "pattern",
        "task": "detection",
        # Every ZJU-Leaper sample carries a normal/defect flag unconditionally
        # (`ZJULeaperDataset._build_sample`'s `Annotations(is_anomalous=...)`,
        # set regardless of `task`), so it's valid anomaly-evaluation ground
        # truth too, not just detection/segmentation — this is what makes all
        # 14 canonical models (not just the 9 detection/segmentation ones)
        # benchmarkable against it. The anomaly models are "Normal Lab
        # trained" (see catalog.py's `source`), so scoring them here is a
        # cross-domain generalization check, not an in-domain one.
    },
    "RAW-FABRID": {
        "name": "raw-fabric",
        "env": "RAW_FABRIC_ROOT",
        "slice_kwarg": None,
        "task": "anomaly",
    },
    "MVTec AD": {
        "name": "mvtec-ad",
        "env": "MVTEC_AD_ROOT",
        "slice_kwarg": "category",
        "task": "anomaly",
    },
    # General object benchmark (logical + structural anomalies); per-image
    # ground-truth mask dirs (datasets/mvtec_loco.py).
    "MVTec LOCO": {
        "name": "mvtec-loco",
        "env": "MVTEC_LOCO_ROOT",
        "slice_kwarg": "category",
        "task": "anomaly",
    },
    # General object benchmark; pixel masks for the Anomaly split
    # (datasets/visa.py).
    "VisA": {
        "name": "visa",
        "env": "VISA_ROOT",
        "slice_kwarg": "category",
        "task": "anomaly",
    },
    # In-domain fabric, image-level only (no pixel masks) — good for
    # image-level AUROC, and a training-eligible source (datasets/tilda.py).
    "TILDA-400": {
        "name": "tilda-400",
        "env": "TILDA_400_ROOT",
        "slice_kwarg": None,
        "task": "anomaly",
    },
    # In-domain fabric, training-eligible (datasets/fabric_defects.py).
    # Partial pixel masks: hole/Vertical/horizontal ship binary masks
    # (the dataset's "_processed" files, parsed as masks rather than
    # separate photos); lines/stain have none. "segmentation" is still
    # offered since some samples do carry ground truth.
    "Fabric Defects": {
        "name": "fabric-defects",
        "env": "FABRIC_DEFECTS_ROOT",
        "slice_kwarg": None,
        "task": "anomaly",
    },
    # In-domain fabric, native bbox annotations (datasets/tianchi.py) --
    # both a detection training source and, via its normal_Images pool, an
    # anomaly-eligible / `fabric-train` member source. No pixel masks.
    "Tianchi": {
        "name": "tianchi",
        "env": "TIANCHI_ROOT",
        "slice_kwarg": None,
        "task": "detection",
    },
}
ALL_TEXTURES = "All textures"
ALL_IMAGES = "All images"
DEFECT_ONLY = "Defect only"
NORMAL_ONLY = "Normal only"
SHOT_FULL = "Full-shot"
SHOT_FEW = "Few-shot"
FEW_SHOT_SAMPLE_COUNT = 350
FEW_SHOT_DEFECT_RATIO = 0.3


# Gradio `(display_label, value)` tuples for every choice-based control whose
# *value* is also compared elsewhere in this module (`shot_mode == SHOT_FULL`,
# `image_scope == DEFECT_ONLY`, ...). Only the display half is localized —
# see `i18n.py`'s module docstring for why the value must stay stable.
def task_choices(lang: str = DEFAULT_LANGUAGE) -> list[tuple[str, str]]:
    """The "Task type" dropdown's `(display_label, value)` pairs. README
    4.1.2 names the two entries (`Anomaly detection` / `Defect detection`)
    and `task_text` reports the same pair in the model panel, so the two
    never drift into three names for two things."""

    return [(tr(lang, "choice_task_defect"), "defect"), (tr(lang, "choice_task_anomaly"), "anomaly")]


def split_choices(lang: str = DEFAULT_LANGUAGE) -> list[tuple[str, str]]:
    return [(tr(lang, "split_test"), "test"), (tr(lang, "split_train"), "train")]


def image_scope_choices(lang: str = DEFAULT_LANGUAGE) -> list[tuple[str, str]]:
    return [
        (tr(lang, "choice_all_images"), ALL_IMAGES),
        (tr(lang, "choice_defect_only"), DEFECT_ONLY),
        (tr(lang, "choice_normal_only"), NORMAL_ONLY),
    ]


def shot_mode_choices(lang: str = DEFAULT_LANGUAGE) -> list[tuple[str, str]]:
    return [(tr(lang, "choice_full_shot"), SHOT_FULL), (tr(lang, "choice_few_shot"), SHOT_FEW)]


def task_text(lang: str, task: str) -> str:
    """The task-type name a user sees, using the same two labels as the
    "Task type" dropdown (README 4.1.2: `Anomaly detection` / `Defect
    detection`) rather than the internal `anomaly`/`segmentation` slug.

    Every supervised task the manifest declares — detection, segmentation and
    instance segmentation — is one "Defect detection" entry in README 3.1's
    Task Type column, so the internal slugs must not leak into the panel as a
    third name for the same thing.
    """

    return tr(lang, "choice_task_anomaly" if task == "anomaly" else "choice_task_defect")


def _scope_text(lang: str, image_scope: str) -> str:
    mapping = {ALL_IMAGES: "choice_all_images", DEFECT_ONLY: "choice_defect_only", NORMAL_ONLY: "choice_normal_only"}
    return tr(lang, mapping.get(image_scope, "choice_all_images"))


def shot_text(lang: str, shot_mode: str) -> str:
    mapping = {SHOT_FULL: "choice_full_shot", SHOT_FEW: "choice_few_shot"}
    return tr(lang, mapping.get(shot_mode, "choice_full_shot"))


def dataset_tasks(dataset_name: str) -> tuple[str, ...]:
    """Every task the registered dataset `dataset_name` can supply real
    ground truth for.

    Keyed by the *registered adapter name*, not by the UI's display label:
    label -> name is presentation (`DATASET_CATALOG`), name -> tasks is the
    contract (`core.dataset_capabilities`). Keeping the two lookups separate
    is what lets the Benchmark tab swap in its own catalog without also
    having to restate what its datasets can be scored on.

    The Benchmark tab uses this to decide which models are evaluable against
    a dataset, and to pick the `task` each model's load needs — only
    `task == "segmentation"` makes a `DatasetAdapter` attach
    `Sample.annotations.masks` (see `datasets/zju_leaper.py::_build_sample`).
    """

    from fabric_defect_hub.core.dataset_capabilities import all_capabilities

    return all_capabilities()[dataset_name].tasks


def _dataset_dir(dataset_label: str) -> str:
    """The directory name this dataset is staged under, from its declared
    `default_root` (`"data/ZJU-Leaper"` -> `"ZJU-Leaper"`).
    """

    from fabric_defect_hub.core.dataset_capabilities import all_capabilities

    return Path(all_capabilities()[DATASET_CATALOG[dataset_label]["name"]].default_root).name


def _dataset_roots(dataset_label: str) -> list[Path]:
    """Return likely local roots for `dataset_label`, including the
    repository's `data/<dir>` symlink convention and the SSD layout it
    typically points at. Roots are returned as-given (not yet resolved) so
    callers can decide whether to follow symlinks to the underlying storage.
    """

    spec = DATASET_CATALOG[dataset_label]
    directory = _dataset_dir(dataset_label)
    roots: list[Path] = []
    configured = os.getenv(spec["env"])
    if configured:
        roots.append(Path(configured).expanduser())
    # Canonical layout: datasets are grouped by domain at repository level.
    # Keep the old data/ and Data/ probes for existing checkouts and callers.
    from fabric_defect_hub.core.dataset_capabilities import all_capabilities

    dataset_name = spec["name"]
    declared_root = all_capabilities().get(dataset_name)
    if declared_root and declared_root.default_root:
        declared = Path(declared_root.default_root)
        if declared.is_absolute():
            roots.append(declared)
        else:
            roots.extend((PROJECT_ROOT / declared, PROJECT_ROOT.parent / declared))
    roots.extend(PROJECT_ROOT / parent / directory for parent in ("datasets/textile", "datasets/general", "data", "Data"))
    volumes = Path("/Volumes")
    if volumes.is_dir():
        roots.extend(volume / _SSD_VOLUME_PARENT / directory for volume in volumes.iterdir())
    return roots


def default_dataset_root(dataset_label: str = "ZJU-Leaper") -> str:
    """Resolve the selected dataset without exposing an editable path field.

    `data/<dir>` is expected to be a symlink onto external storage (an SSD,
    a mounted share, ...) rather than the data itself, so every candidate is
    resolved with `Path.resolve()` before being handed to the dataset
    adapter or to Gradio's `allowed_paths` — both need the real on-disk
    location, not the symlink path, to serve/read files reliably.
    """

    if dataset_label not in DATASET_CATALOG:
        raise KeyError(f"Unknown dataset selection {dataset_label!r}.")
    for candidate in _dataset_roots(dataset_label):
        if candidate.is_dir():
            return str(candidate.resolve())
    return ""


def _mvtec_ad_categories(root: str) -> list[str]:
    if not root:
        return []
    root_path = Path(root)
    return sorted(
        path.name
        for path in root_path.iterdir()
        if path.is_dir() and (path / "train" / "good").is_dir()
    )


def _visa_categories(root: str) -> list[str]:
    """VisA marks a category by `<cat>/Data/Images/Normal` (not `train/good`),
    so it needs its own probe — this also filters out non-category entries at
    the VisA root like `split_csv/` and `LICENSE-DATASET`.
    """

    if not root:
        return []
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    return sorted(
        path.name
        for path in root_path.iterdir()
        if path.is_dir() and (path / "Data" / "Images" / "Normal").is_dir()
    )


TRAIN_PATTERNS_PRESET = "Pattern 1-4 (Train patterns)"


def texture_choices(dataset_label: str) -> list[str]:
    """Discover available texture/category slices from the registered
    dataset root. Datasets without a subdivision (`slice_kwarg` is None)
    only ever offer the "All textures" choice.
    """

    choices = [ALL_TEXTURES]
    spec = DATASET_CATALOG.get(dataset_label)
    if spec is None or spec["slice_kwarg"] is None:
        return choices
    root = default_dataset_root(dataset_label)

    if spec["name"] == "zju-leaper":
        patterns = Path(root) / "ImageSets" / "Patterns" if root else None
        if patterns is None or not patterns.is_dir():
            return choices + [TRAIN_PATTERNS_PRESET]
        pattern_ids = sorted(
            (path.stem.removeprefix("pattern") for path in patterns.glob("pattern*.json")),
            key=lambda value: int(value) if value.isdigit() else value,
        )
        return choices + [TRAIN_PATTERNS_PRESET] + [f"Pattern {pattern_id}" for pattern_id in pattern_ids]

    if spec["name"] in ("mvtec-ad", "mvtec-loco"):
        return choices + _mvtec_ad_categories(root)

    if spec["name"] == "visa":
        return choices + _visa_categories(root)

    return choices


def slice_value(dataset_label: str, texture_label: str | list[str] | tuple[str, ...] | None) -> str | list[str] | None:
    """Resolve the "Texture / pattern" dropdown's selection into the value
    passed to the selected dataset's `slice_kwarg` (e.g. "pattern7" for
    ZJU-Leaper, "bottle" for MVTec AD). Accepts single strings or lists/tuples
    for multi-pattern filtering."""

    if not texture_label or texture_label == ALL_TEXTURES:
        return None

    if isinstance(texture_label, (list, tuple)):
        if ALL_TEXTURES in texture_label:
            return None
        resolved = []
        for item in texture_label:
            val = slice_value(dataset_label, item)
            if val is None:
                return None
            if isinstance(val, list):
                resolved.extend(val)
            else:
                resolved.append(val)
        dedup = list(dict.fromkeys(resolved))
        return dedup if len(dedup) > 1 else (dedup[0] if dedup else None)

    spec = DATASET_CATALOG[dataset_label]
    if spec["name"] == "zju-leaper":
        if texture_label == TRAIN_PATTERNS_PRESET:
            return ["pattern1", "pattern2", "pattern3", "pattern4"]
        if texture_label.lower().startswith("pattern "):
            return f"pattern{texture_label.split()[-1]}"
        raise ValueError(f"Unknown texture selection {texture_label!r}.")
    if spec["name"] in ("mvtec-ad", "mvtec-loco", "visa"):
        return texture_label
    raise ValueError(f"{dataset_label!r} does not support texture/category selection.")


def shot_regime_kwargs(shot_mode: str) -> tuple[int | None, float]:
    """Map the UI's full-shot/few-shot toggle to `(num_samples, defect_ratio)`
    kwargs shared by every `DatasetAdapter` in this project."""

    if shot_mode == SHOT_FULL:
        return None, 0.5
    if shot_mode == SHOT_FEW:
        return FEW_SHOT_SAMPLE_COUNT, FEW_SHOT_DEFECT_RATIO
    raise ValueError(f"Unknown sample regime {shot_mode!r}.")


def empty_gallery_state() -> dict[str, Any]:
    return {"samples": [], "index": 0, "dataset": None}


def model_status(model_label: str, lang: str = DEFAULT_LANGUAGE) -> str:
    """Whether this catalog entry can be run right now, and why not if it can't.

    "Is this backend installed" is answered by `core.availability`, which
    imports the backend's own adapter module. This panel used to keep its own
    table of proxy pip packages to probe instead (`timm` for Dinomaly,
    `kornia` for MoECLIP, ...), which was both a second copy of per-backend
    knowledge and a weaker question: a present proxy package does not mean
    the backend imports.
    """

    spec = MODEL_CATALOG[model_label]
    backend = spec["backend"]
    if not backend_is_importable(backend):
        return tr(lang, "model_status_unavailable", package=backend)
    path = Path(spec["checkpoint"])
    if not path.is_file():
        return tr(lang, "model_status_missing", path=path)
    # `training_split` is a manifest value (README 6.2: `training_split:
    # normal_only`), so it is shown verbatim in both languages — the same
    # rule that leaves dataset and model identifiers untranslated. The panel
    # used to spell it "normal only", a third form that matched neither the
    # manifest nor README 3.1's "Normal-only (one-class)".
    training_split = spec["training_split"] or tr(lang, "value_none")
    return tr(
        lang, "model_status_ready",
        task=task_text(lang, spec["task"]),
        method=spec["method"],
        domain=spec["domain"],
        training_corpus=", ".join(spec["trained_on"]) or tr(lang, "value_none"),
        training_split=training_split,
        filename=path.name,
    )


def checkpoint_diagnostic(model_label: str, lang: str = DEFAULT_LANGUAGE) -> str:
    """On-demand, non-executing provenance for the selected model's weights.

    Offered for every model, not just the anomalib ones. The backend check
    that used to sit here answered "is this an anomalib checkpoint?" and told
    everyone else "native Ultralytics artifact" — untrue for torchvision,
    Dinomaly, MoECLIP and MambaAD, and beside the point either way: a `.pt`
    is as much an executable pickle as a `.ckpt` is, and
    `core.checkpoint.inspect_checkpoint` reads both without loading either.
    """

    spec = MODEL_CATALOG[model_label]
    diagnostic = inspect_checkpoint(spec["checkpoint"])
    if not diagnostic.exists:
        return tr(lang, "checkpoint_diag_missing", path=diagnostic.path)
    globals_summary = ", ".join(diagnostic.unsafe_globals) or tr(lang, "value_none")
    return "  \n".join((
        tr(lang, "checkpoint_diag_trusted_header"),
        tr(lang, "checkpoint_diag_sha", sha=diagnostic.sha256),
        tr(lang, "checkpoint_diag_size", size=diagnostic.size_bytes / (1024 * 1024)),
        tr(lang, "checkpoint_diag_globals", globals=globals_summary),
    ))


def dataset_declared_root(dataset_label: str) -> str:
    """The **declared root** README 2.2.7 says a dataset is staged under
    (`datasets/textile/ZJU-Leaper`, `datasets/general/MVTec AD`).

    The page must name this path rather than the legacy `data/<dir>` probe:
    the declared root is the one README documents and the one `adh doctor`
    reports, so a user told to "connect the storage at X" can compare X with
    the path the CLI names.
    """

    from fabric_defect_hub.core.dataset_capabilities import all_capabilities

    return all_capabilities()[DATASET_CATALOG[dataset_label]["name"]].default_root


def dataset_status(dataset_label: str, lang: str = DEFAULT_LANGUAGE) -> str:
    root = default_dataset_root(dataset_label)
    if root:
        return tr(lang, "dataset_ready", label=dataset_label)
    spec = DATASET_CATALOG[dataset_label]
    return tr(
        lang, "dataset_unavailable",
        label=dataset_label, root=dataset_declared_root(dataset_label), env=spec["env"],
    )


def build_gallery_state(samples: list[Sample], count: int, seed: int, dataset_label: str) -> dict[str, Any]:
    available = [sample for sample in samples if Path(sample.image_path).is_file()]
    if not available:
        raise ValueError("No readable image files were found in the selected dataset/split.")
    selected_count = min(max(int(count), 1), len(available))
    selected = random.Random(int(seed)).sample(available, selected_count)
    return {"samples": [sample_to_dict(sample) for sample in selected], "index": 0, "dataset": dataset_label}


def load_random_samples(
    dataset_label: str,
    split: str,
    sample_count: int,
    seed: int | None = None,
    texture_label: str = ALL_TEXTURES,
    image_scope: str = ALL_IMAGES,
    shot_mode: str = SHOT_FULL,
    lang: str = DEFAULT_LANGUAGE,
) -> tuple[dict[str, Any], str | None, str, str]:
    root = default_dataset_root(dataset_label)
    if not root:
        env = DATASET_CATALOG[dataset_label]["env"]
        raise FileNotFoundError(
            f"no staged copy of `{dataset_label}` was found at its declared root "
            f"`{dataset_declared_root(dataset_label)}` (or under `${env}`)"
        )
    spec = DATASET_CATALOG[dataset_label]
    actual_seed = random.SystemRandom().randrange(2**32) if seed is None else int(seed)
    num_samples, defect_ratio = shot_regime_kwargs(shot_mode)
    dataset_kwargs: dict[str, Any] = dict(
        root=root,
        split=split,
        task=spec["task"],
        use_defect=image_scope != NORMAL_ONLY,
        num_samples=num_samples,
        defect_ratio=defect_ratio,
    )
    if spec["slice_kwarg"] is not None:
        dataset_kwargs[spec["slice_kwarg"]] = slice_value(dataset_label, texture_label)
    dataset = load_dataset(spec["name"], **dataset_kwargs)
    samples = dataset.load_samples()
    if image_scope == DEFECT_ONLY:
        samples = [sample for sample in samples if sample.annotations.is_anomalous]
    elif image_scope not in (ALL_IMAGES, NORMAL_ONLY):
        raise ValueError(f"Unknown image selection {image_scope!r}.")
    state = build_gallery_state(samples, sample_count, actual_seed, dataset_label)
    path, position = current_image(state, lang)
    texture = ALL_TEXTURES if texture_label == ALL_TEXTURES else texture_label
    status = tr(
        lang, "dataset_load_success",
        count=len(state["samples"]), scope=_scope_text(lang, image_scope), shot=shot_text(lang, shot_mode),
        name=dataset_label, texture=texture, split=split,
    )
    return state, path, position, status


def current_image(state: dict[str, Any], lang: str = DEFAULT_LANGUAGE) -> tuple[str | None, str]:
    samples = state.get("samples", [])
    if not samples:
        return None, tr(lang, "caption_no_image")
    index = int(state.get("index", 0)) % len(samples)
    state["index"] = index
    sample = sample_from_dict(samples[index])
    return sample.image_path, _sample_caption(sample, index, len(samples), lang)


def move_image(state: dict[str, Any], direction: int, lang: str = DEFAULT_LANGUAGE) -> tuple[dict[str, Any], str | None, str]:
    if not state.get("samples"):
        return state, None, tr(lang, "move_need_dataset")
    state = dict(state)
    state["index"] = (int(state.get("index", 0)) + direction) % len(state["samples"])
    path, caption = current_image(state, lang)
    return state, path, caption


def detect_current(state: dict[str, Any], model_label: str, lang: str = DEFAULT_LANGUAGE) -> tuple[Any, dict[str, Any], str]:
    if not state.get("samples"):
        return None, {}, tr(lang, "inference_need_dataset")
    spec = MODEL_CATALOG[model_label]
    status = model_status(model_label, lang)
    if status.startswith("🔴") or status.startswith("🟠"):
        return None, {}, status

    sample = sample_from_dict(state["samples"][state["index"]])
    try:
        model = load_model(spec["backend"], spec["name"])
        prediction = _predict_with_model(model, spec, model_label, sample)[0]
        image = render_prediction(sample.image_path, prediction)
    except Exception as exc:
        return None, {}, tr(lang, "inference_failed", error_type=type(exc).__name__, error=exc)
    return image, prediction_summary(prediction, model.capabilities()), tr(lang, "inference_complete")


def load_selected_model(session_manager: Any, model_label: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a catalog entry through the UI-independent inference service.

    Checks the checkpoint file exists *before* handing off to the adapter.
    This isn't just a nicer error message: Ultralytics' `YOLO(path)` loader
    treats a missing path whose filename matches a known official release
    asset (e.g. "yolo11n.pt") as a request to auto-download that asset —
    confirmed live, it silently downloaded generic COCO-pretrained weights
    into this catalog's published slot for an unfinished model, which the
    UI would then present as "Ready — Fabric trained". Failing fast here
    keeps every catalog entry either genuinely fabric-trained or clearly
    marked unavailable, never a substitute pretending to be the real thing.
    """

    spec = MODEL_CATALOG[model_label]
    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"no trained checkpoint at {checkpoint} — train {model_label!r} "
            f"first (see `adh train-all`), nothing was loaded"
        )
    samples = [sample_from_dict(item) for item in (state or {}).get("samples", [])]
    return session_manager.load(model_label, spec, artifact_for_model(spec), samples=samples)


def unload_selected_model(session_manager: Any) -> dict[str, Any]:
    """Unload the active model through the UI-independent inference service."""

    return session_manager.unload()


def detect_loaded_model(
    session_manager: Any, state: dict[str, Any], model_label: str, lang: str = DEFAULT_LANGUAGE
) -> tuple[Any, dict[str, Any], str]:
    """Predict through a preloaded backend session without creating an adapter."""

    if not state.get("samples"):
        return None, {}, tr(lang, "inference_need_dataset")
    sample = sample_from_dict(state["samples"][state["index"]])
    capabilities = session_manager.capabilities(model_label)
    try:
        prediction = session_manager.predict(
            model_label, [sample], **_anomaly_map_kwargs(capabilities, model_label)
        )[0]
        image = render_prediction(sample.image_path, prediction)
    except Exception as exc:
        return None, {}, tr(lang, "inference_failed", error_type=type(exc).__name__, error=exc)
    return image, prediction_summary(prediction, capabilities), tr(lang, "inference_complete")


def _anomaly_map_kwargs(capabilities: Any, model_label: str) -> dict[str, str]:
    """`{"output_dir": ...}` for a model that produces a pixel-level anomaly
    map, `{}` for one that doesn't.

    The model's own `ModelCapabilities` decides, not a list of backend names
    kept here. Two reasons that matters: this panel is not the place that
    knows which backends have spatial output, and the backend-name version
    was wrong within a backend — GANomaly is an anomalib model with no map
    (`anomalib.presets.IMAGE_LEVEL_ONLY`), so it was being handed a directory
    to write nothing into.
    """

    if capabilities is None or not capabilities.fills("anomaly_map"):
        return {}
    return {"output_dir": str(RUNTIME_ANOMALY_MAP_ROOT / _model_slug(model_label))}


def _predict_with_model(model: Any, spec: dict[str, Any], model_label: str, sample: Sample) -> list[Prediction]:
    return model.predict(
        [sample],
        artifact_for_model(spec),
        **_anomaly_map_kwargs(model.capabilities(), model_label),
    )


def artifact_for_model(spec: dict[str, Any]) -> Artifact:
    return Artifact(path=str(spec["checkpoint"]), backend=spec["backend"], metadata=dict(spec["metadata"]))


def _model_slug(model_label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in model_label).strip("-")


def prediction_summary(prediction: Prediction, capabilities: Any = None) -> dict[str, Any]:
    """Flatten a `Prediction` for the result panel.

    `capabilities` (the model's `ModelCapabilities`, when the caller has it)
    separates two cases the panel would otherwise show identically: a model
    that *can* produce a pixel-level anomaly map but didn't this run, and one
    that structurally cannot — GANomaly scores the distance between two
    latent vectors, so there is no map to render, ever. `None` keeps the
    older, non-committal wording for callers that don't know.
    """

    return {
        "sample_id": prediction.sample_id,
        "task": "anomaly" if prediction.anomaly_score is not None else "detection",
        "detections": len(prediction.boxes or []),
        "labels": prediction.labels or [],
        "scores": [round(score, 4) for score in prediction.scores or []],
        "anomaly_score": prediction.anomaly_score,
        "has_masks": bool(prediction.masks),
        "has_anomaly_map": prediction.anomaly_map is not None,
        "pixel_map_supported": None if capabilities is None else capabilities.fills("anomaly_map"),
    }


def render_prediction_tags(summary: dict[str, Any], lang: str = DEFAULT_LANGUAGE) -> str:
    """Render the unified prediction contract as colored HTML chips instead
    of prose: one tag per detected defect naming its actual predicted class
    (`summary["labels"]`, e.g. "defect" or a checkpoint's own class name),
    paired with a confidence tag whose background opacity scales with the
    score — darker means more confident, lighter means less. See the app's
    `CSS` for the `.fdh-tag*` classes this emits into the `Detection result`
    card's `gr.HTML` panel."""

    import html as _html

    if not summary:
        return f'<div class="fdh-tagpanel-empty">{_html.escape(tr(lang, "prediction_none"))}</div>'

    if summary["task"] == "anomaly":
        score = summary["anomaly_score"]
        is_anomalous = bool(score is not None and score >= 0.5)
        verdict_key = "tag_anomalous" if is_anomalous else "tag_normal"
        verdict_class = "fdh-tag-anomalous" if is_anomalous else "fdh-tag-normal"
        chips = [f'<span class="fdh-tag {verdict_class}">{_html.escape(tr(lang, verdict_key))}</span>']
        if score is not None:
            chips.append(_confidence_chip(tr(lang, "tag_anomaly_score"), float(score)))
        if summary["has_anomaly_map"]:
            heatmap_key = "tag_heatmap_available"
        elif summary.get("pixel_map_supported") is False:
            # The model has no spatial output at all (see prediction_summary):
            # say so, rather than implying a map that merely failed to appear.
            heatmap_key = "tag_heatmap_unsupported"
        else:
            heatmap_key = "tag_heatmap_unavailable"
        chips.append(f'<span class="fdh-tag fdh-tag-neutral">{_html.escape(tr(lang, heatmap_key))}</span>')
        return f'<div class="fdh-tags">{"".join(chips)}</div>'

    detections = int(summary["detections"])
    if detections == 0:
        return (
            '<div class="fdh-tags">'
            f'<span class="fdh-tag fdh-tag-normal">{_html.escape(tr(lang, "prediction_no_defect"))}</span>'
            "</div>"
        )
    header = f'<div class="fdh-tagpanel-header">{_html.escape(tr(lang, "prediction_regions", count=detections))}</div>'
    rows = []
    for label, score in zip(summary["labels"], summary["scores"]):
        label_chip = f'<span class="fdh-tag fdh-tag-label">{_html.escape(str(label))}</span>'
        conf_chip = _confidence_chip(tr(lang, "tag_confidence"), float(score))
        rows.append(f'<div class="fdh-tag-row">{label_chip}{conf_chip}</div>')
    return header + f'<div class="fdh-tags fdh-tags-column">{"".join(rows)}</div>'


def _confidence_chip(caption: str, score: float) -> str:
    import html as _html

    clamped = max(0.0, min(1.0, score))
    alpha = 0.45 + 0.5 * clamped  # darker = higher confidence, lighter = lower
    return (
        f'<span class="fdh-tag fdh-tag-conf" style="background: rgba(234,110,24,{alpha:.2f})">'
        f"{_html.escape(caption)} {clamped * 100:.1f}%</span>"
    )


def render_prediction(image_path: str, prediction: Prediction):
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGBA")
    image = _overlay_masks(image, prediction.masks)
    image = _overlay_anomaly_map(image, prediction.anomaly_map)
    draw = ImageDraw.Draw(image)
    boxes = prediction.boxes or []
    labels = prediction.labels or ["defect"] * len(boxes)
    scores = prediction.scores or [None] * len(boxes)
    for box, label, score in zip(boxes, labels, scores):
        text = label if score is None else f"{label} · {score:.3f}"
        draw.rectangle(box, outline="#f97316", width=4)
        draw.text((box[0], max(0, box[1] - 18)), text, fill="#f97316", stroke_width=1, stroke_fill="white")
    return image.convert("RGB")


def _overlay_masks(image, masks):
    if not masks:
        return image
    import numpy as np
    from PIL import Image

    values = np.asarray(masks, dtype=bool)
    if values.ndim == 3:
        values = values.any(axis=0)
    alpha = Image.fromarray((values * 96).astype("uint8")).resize(image.size)
    overlay = Image.new("RGBA", image.size, "#f97316")
    overlay.putalpha(alpha)
    return Image.alpha_composite(image, overlay)


def _overlay_anomaly_map(image, map_path: str | None):
    if map_path is None:
        return image
    import numpy as np
    from PIL import Image

    values = np.asarray(np.load(map_path), dtype=float)
    values -= values.min()
    if values.max() > 0:
        values /= values.max()
    red = Image.fromarray((values * 255).astype("uint8")).resize(image.size)
    alpha = Image.fromarray((values * 128).astype("uint8")).resize(image.size)
    overlay = Image.merge("RGBA", (red, Image.new("L", image.size), Image.new("L", image.size), alpha))
    return Image.alpha_composite(image, overlay)


def _sample_caption(sample: Sample, index: int, total: int, lang: str = DEFAULT_LANGUAGE) -> str:
    state = tr(lang, "state_defect" if sample.annotations.is_anomalous else "state_normal")
    return f"**{index + 1} / {total}** · `{sample.id}` · {state}"
