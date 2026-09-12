"""Training resolution, dataset override, and execution entry points."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

from fabric_defect_hub.core.dataset_capabilities import default_dataset_roots, names_with_role

ShotMode = Literal["full", "medium", "few", "test"]

# "test" shot: a tiny end-to-end smoke run (per model/dataset combination)
# meant to verify the pipeline wiring, not to produce a usable checkpoint.
TEST_SHOT_NUM_SAMPLES = 8

# "medium" shot caps each selected ZJU-Leaper pattern rather than taking its
# full pool. An unrestricted config still selects all 19 patterns; the textile
# YOLO recipes intentionally pin patterns 1-4, so their medium budget is
# 4 * these per-pattern limits.
ZJU_LEAPER_PATTERN_COUNT = 19
MEDIUM_SHOT_TRAIN_PER_PATTERN = 150
MEDIUM_SHOT_VAL_PER_PATTERN = 50

# Where `resolve_model_config` looks for configs when given a bare name
# instead of a path (e.g. "ultralytics_example" or "yolov8n").
DEFAULT_MODEL_CONFIG_DIR = Path("configs/models")
DEFAULT_TRAINING_PROFILE = Path("configs/training_profile.yaml")

# Every registered dataset is expected to be reachable through a symlink
# under the project's own `data/` directory (e.g. `data/ZJU-Leaper ->
# /wherever/that machine/keeps it`), set up once per machine — this is the
# same convention already used to stage datasets on both the dev laptop and
# the training server. `apply_default_dataset_root` falls back to these
# relative paths so a config's `data.dataset_root` doesn't have to be a
# machine-specific absolute path or an `${ENV_VAR}` the caller has to
# remember to export before every run.
#
# Derived from `core.dataset_capabilities` (single declaration per dataset)
# rather than hand-maintained here — see that module's docstring for why.
DEFAULT_DATASET_ROOTS: dict[str, str] = default_dataset_roots()

# Anomaly (one-class) training is deliberately restricted to *in-domain
# fabric* sources: the individual fabric datasets and the `fabric-train`
# union that combines them (see datasets/fabric_train.py). Cross-domain
# object benchmarks (MVTec AD/LOCO, VisA) are eval-only — training a fabric
# model on them would defeat the benchmark — and detection-only sets
# (SDUST-FDD) belong to the ultralytics/torchvision backends, not the
# one-class anomaly ones. `_enforce_trainable_dataset` rejects anything
# outside this set for the one-class backends.
ANOMALY_TRAINABLE_DATASETS: set[str] = names_with_role("anomaly_train")

# ...and the exact mirror image, for zero-shot (ZSAD) backends. MoECLIP
# learns prompt-aligned anomaly features from labelled defects on an
# *auxiliary* corpus and is then applied to categories it has never seen —
# that transfer is the claim being benchmarked. Training it on fabric
# would make its fabric numbers in-domain and quietly void that claim, so
# the fabric sources above are rejected here and the cross-domain object
# benchmarks (eval-only for every other model) are the allowed training
# corpora instead. Which fabric set it is then *evaluated* on is a
# separate config key (`data.test_dataset`), unrestricted.
ZERO_SHOT_TRAINABLE_DATASETS: set[str] = names_with_role("zero_shot_train")

# Per backend: (train-split selection key, val/test-split selection key) in
# that backend's `data` config section.
_BACKEND_DATA_SELECTIONS: dict[str, tuple[str, str]] = {
    "anomalib": ("train_selection", "test_selection"),
    "dinomaly": ("train_selection", "test_selection"),
    "moeclip": ("train_selection", "test_selection"),
    "mambaad": ("train_selection", "test_selection"),
    "torchvision": ("train_selection", "val_selection"),
    "ultralytics": ("train_selection", "val_selection"),
}

_BACKEND_PIPELINE_MODULES = {
    "ultralytics": "fabric_defect_hub.models.ultralytics.pipeline",
    "torchvision": "fabric_defect_hub.models.torchvision.pipeline",
    "anomalib": "fabric_defect_hub.models.anomalib.pipeline",
    "dinomaly": "fabric_defect_hub.models.dinomaly.pipeline",
    "moeclip": "fabric_defect_hub.models.moeclip.pipeline",
    "mambaad": "fabric_defect_hub.models.mambaad.pipeline",
}

_BACKEND_CONFIG_CLASSES = {
    "ultralytics": ("fabric_defect_hub.models.ultralytics.config", "UltralyticsConfig"),
    "torchvision": ("fabric_defect_hub.models.torchvision.config", "TorchvisionConfig"),
    "anomalib": ("fabric_defect_hub.models.anomalib.config", "AnomalibConfig"),
    "dinomaly": ("fabric_defect_hub.models.dinomaly.config", "DinomalyConfig"),
    "moeclip": ("fabric_defect_hub.models.moeclip.config", "MoECLIPConfig"),
    "mambaad": ("fabric_defect_hub.models.mambaad.config", "MambaADConfig"),
}

# Per backend: which key under `model:` selects which model gets trained.
# Ultralytics/torchvision key their model families by `variant`
# (yolov8n/yolov8s/..., fasterrcnn_resnet50_fpn/...); the anomaly backends'
# models and Dinomaly's encoder presets are keyed by `name` instead
# (PatchCore/PaDiM/... or dinov2reg_vit_base_14/...).
#
# Public because `inference.runner.run_predict` reads this same table to find
# the variant it should score. It used to answer that question itself, from an
# unrelated list of backend names that happened to give the same answer.
BACKEND_MODEL_KEY: dict[str, str] = {
    "anomalib": "name",
    "dinomaly": "name",
    "moeclip": "name",
    "mambaad": "name",
    "torchvision": "variant",
    "ultralytics": "variant",
}

# Backends whose adapter trains one-class (normal-only) -- their train
# split's `use_defect` is always forced to False regardless of CLI/UI
# overrides (see `apply_dataset_overrides`). MoECLIP is deliberately not
# here: it is a zero-shot detector that learns from *labelled* anomalies
# (image label + pixel mask), so forcing its train split normal-only would
# leave its segmentation loss with nothing to learn from.
_ONE_CLASS_BACKENDS = {"anomalib", "dinomaly", "mambaad"}


# Per anomaly backend: which datasets it may *train* on, and a short label
# for the error message explaining why the other kind is refused.
_ZERO_SHOT_BACKENDS = {"moeclip"}

_BACKEND_TRAINABLE_DATASETS: dict[str, tuple[set[str], str]] = {
    "anomalib": (ANOMALY_TRAINABLE_DATASETS, "one-class"),
    "dinomaly": (ANOMALY_TRAINABLE_DATASETS, "one-class"),
    "mambaad": (ANOMALY_TRAINABLE_DATASETS, "one-class"),
    "moeclip": (ZERO_SHOT_TRAINABLE_DATASETS, "zero-shot"),
}


@dataclass
class DatasetOverrides:
    """CLI-level overrides layered onto a model config's `data` section.

    Every field defaults to `None`, meaning "leave the model config's value
    alone". `mode`/`num_samples` set the train split's (and, unless
    `val_num_samples` is given, the val/test split's) `num_samples`; the
    rest map straight onto the per-split selection dict's matching key
    (`use_defect`, `defect_ratio`, `pattern`, `category`, `seed`).

    `test_dataset`/`test_dataset_root` only apply to the zero-shot
    backends, whose evaluation target is a different dataset from their
    training corpus (see `MoECLIPConfig.DataSpec`); passing them for any
    other backend is an error rather than a silent no-op, since it would
    otherwise read as "evaluate on X" and quietly not.
    """

    dataset: str | None = None
    dataset_root: str | None = None
    test_dataset: str | None = None
    test_dataset_root: str | None = None
    mode: ShotMode | None = None
    num_samples: int | None = None
    val_num_samples: int | None = None
    use_defect: bool | None = None
    defect_ratio: float | None = None
    pattern: str | int | None = None
    category: str | None = None
    seed: int | None = None

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))


# Where each backend's run length lives in its own config, and what that
# length actually counts. The six backends genuinely disagree on both --
# anomalib's is a Lightning engine kwarg, the two distillation backends
# count optimizer *iterations* rather than passes over the data -- so any
# caller wanting to say "train for N" has to know this table.
#
# It exists so that knowledge stays in this layer, next to the config
# handling it belongs to, instead of being re-derived by `api.load_config`
# (whose whole contract is to hold no logic of its own) or by every user
# script that wants to shorten a run.
#
# `tests/test_train_config.py::test_run_length_keys_match_test_speed_overrides`
# pins this against what `_apply_test_speed_overrides` actually writes, so
# the two cannot drift apart.
RUN_LENGTH_KEYS: dict[str, tuple[str, str]] = {
    "anomalib": ("train.engine_kwargs.max_epochs", "epochs"),
    "ultralytics": ("train.epochs", "epochs"),
    "torchvision": ("train.epochs", "epochs"),
    "moeclip": ("train.epochs", "epochs"),
    "dinomaly": ("train.total_iters", "iterations"),
    "mambaad": ("train.total_iters", "iterations"),
}


# The same idea for "which device does this run use", which the six
# backends also spell three different ways: anomalib's is a Lightning
# accelerator *name* inside `engine_kwargs`, Ultralytics wants a CUDA
# *index*, and the other four take a torch device string. Only anomalib
# needs an entry for "auto" -- every other backend's `train.device: null`
# already means "cuda > mps > cpu", so writing anything there could only
# make it wrong.
#
# This matters more than it looks: the four anomalib configs pin
# `accelerator: cpu` (the honest default on the Apple-Silicon dev machine),
# so on a CUDA host they train on CPU, slowly and silently, unless
# something overrides them.
_DEVICE_KEYS: dict[str, tuple[str, dict[str, str]]] = {
    "anomalib": ("train.engine_kwargs.accelerator", {"gpu": "gpu", "cpu": "cpu", "auto": "auto"}),
    "ultralytics": ("train.device", {"gpu": "0", "cpu": "cpu"}),
    "torchvision": ("train.device", {"gpu": "cuda", "cpu": "cpu"}),
    "dinomaly": ("train.device", {"gpu": "cuda", "cpu": "cpu"}),
    "moeclip": ("train.device", {"gpu": "cuda", "cpu": "cpu"}),
    "mambaad": ("train.device", {"gpu": "cuda", "cpu": "cpu"}),
}


def device_override(backend: str, accelerator: str) -> dict[str, Any]:
    """`{dotted key: value}` that puts `backend` on `accelerator`.

    `accelerator` is 'gpu', 'cpu' or 'auto'. Returns `{}` when the backend's
    own default already means that ('auto' everywhere except anomalib) --
    layer the result into `set_overrides` / pass it as `--set`.
    """

    if backend not in _DEVICE_KEYS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(_DEVICE_KEYS)}")
    key, values = _DEVICE_KEYS[backend]
    if accelerator not in values:
        if accelerator == "auto":
            return {}
        raise ValueError(f"unknown accelerator {accelerator!r}; expected 'gpu', 'cpu' or 'auto'")
    return {key: values[accelerator]}


def infer_backend(raw: dict[str, Any]) -> str:
    """Keyword-based backend detection over a parsed model-config mapping.

    Checked in order:
    1. An explicit top-level `backend:` key (always wins).
    2. `model.name` present -> resolved against anomalib's model aliases
       first (e.g. 'PatchCore'), then Dinomaly's encoder presets (e.g.
       'dinov2reg_vit_base_14'), then MoECLIP's backbones (e.g.
       'ViT-L-14-336'), then MambaAD's encoder presets (e.g. 'resnet34')
       -- all four are keyed by `name` rather than `variant`, so a config
       using this key should set an explicit top-level `backend:` if any
       two ever collide on a name.
    3. `model.variant` keyword: `yolo*` -> `ultralytics`,
       `fasterrcnn*`/`maskrcnn*` -> `torchvision`.
    """

    if isinstance(raw, dict) and raw.get("backend"):
        backend = str(raw["backend"]).lower()
        if backend not in _BACKEND_PIPELINE_MODULES:
            raise ValueError(
                f"unknown backend '{backend}'; expected one of {sorted(_BACKEND_PIPELINE_MODULES)}"
            )
        return backend
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError(
            "cannot infer backend: config requires a 'model' mapping (or an explicit 'backend' key)"
        )
    model = raw["model"]
    if "name" in model:
        name = str(model["name"])
        from fabric_defect_hub.models.anomalib.presets import resolve_model_class_name
        from fabric_defect_hub.models.dinomaly.presets import ENCODER_PRESETS
        from fabric_defect_hub.models.mambaad.presets import ENCODER_PRESETS as MAMBAAD_ENCODER_PRESETS
        from fabric_defect_hub.models.moeclip.presets import MODEL_PRESETS

        try:
            resolve_model_class_name(name)
            return "anomalib"
        except KeyError:
            pass
        if name in ENCODER_PRESETS:
            return "dinomaly"
        if name in MODEL_PRESETS:
            return "moeclip"
        if name in MAMBAAD_ENCODER_PRESETS:
            return "mambaad"
        raise ValueError(
            f"model.name={name!r} matches no anomalib model alias, Dinomaly encoder preset, "
            "MoECLIP backbone, or MambaAD encoder preset; pass backend explicitly or add a "
            "'backend' key to the config"
        )
    variant = str(model.get("variant", "")).lower()
    # The registry, not a name pattern, decides what is an ultralytics
    # variant — "fabricmamba" carries no yolo prefix. The prefix check
    # stays as a fallback for unregistered size sweeps (e.g. "yolov8x").
    from fabric_defect_hub.models.ultralytics.presets import resolve_variant

    try:
        resolve_variant(variant)
        return "ultralytics"
    except KeyError:
        pass
    if variant.startswith("yolo"):
        return "ultralytics"
    if variant.startswith(("fasterrcnn", "maskrcnn", "cascadercnn", "detr", "unet", "deeplab")):
        return "torchvision"
    raise ValueError("cannot infer backend; pass backend explicitly or add a 'backend' key to the config")


def find_model_configs(config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR) -> list[Path]:
    """Every `*.yaml` model config under `config_dir`, sorted by filename."""

    directory = Path(config_dir)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.yaml"))


def resolve_model_config(model: str, config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR) -> Path:
    """Resolve a `train` CLI argument to a model-config YAML path.

    See `resolve_model_config_and_variant` for the resolution rules; this
    wrapper is kept for callers that only need the path.
    """

    return resolve_model_config_and_variant(model, config_dir=config_dir)[0]


def resolve_model_config_and_variant(
    model: str, config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR
) -> tuple[Path, str | None]:
    """Resolve a `train`/`predict` CLI argument to a config path plus the
    variant/name it implies (or `None` when it doesn't imply one).

    The rule this implements, in one sentence: **a bare model name resolves
    to that backend's general-purpose config; a purpose-built config (an
    MVTec reproduction, a textile-tuned variant) is addressed by its
    filename.** Everything below is that sentence made precise.

    `model` may be, in order of precedence:

    1. An existing file path (used as-is, anywhere on disk) — the original
       `fdh train configs/models/ultralytics_example.yaml` form still works.
       Implies no variant.
    2. A filename stem under `config_dir` (e.g. "ultralytics_example",
       with or without the ".yaml" suffix) — `fdh train ultralytics_example`.
       Implies no variant. This is how a purpose-built config is reached:
       `fdh train patchcore_mvtec_repro`.
    3. **Declared** (`_resolve_by_declaration`): the keyword is a config's
       own `model.variant`/`model.name`, or one of its `variants.<name>`
       profile keys — `fdh train yolov8n`, `fdh train stfpm`. Used only when
       exactly one config declares it; several declaring it (three configs
       declare PatchCore) falls through to step 4 rather than erroring, so
       the general-purpose one can win.
    4. **Catalogued** (`_resolve_by_catalog`): the keyword is a published
       model's catalog key or variant, and `CanonicalModel.config` already
       records which config trains it. Not a heuristic — that field is
       maintained precisely because the answer is not derivable. It is what
       makes `fdh train padim` / `winclip` / `unetplusplus_resnet34` work at
       all (none is declared by any config), and it is why this step must
       come before step 5: `unetplusplus_resnet34` is a `segmentation`
       variant, and *both* torchvision configs declare something else
       (`detect` and `instance_segmentation`), so any rule that guessed from
       the task would pick the wrong file. The catalog just says.
    5. **Backend-supported** (`_resolve_by_backend_support`): the keyword is
       a variant some backend's presets accept (`FastFlow`, `GLASS`,
       `dinov2reg_vit_large_14` — runnable but not published, so absent from
       the catalog) → that backend's `<backend>_example.yaml`. Guarded
       against a name two backends both claim, which none currently do.

    Steps 3-5 imply `variant=<the resolved name>` (the caller's explicit
    `--variant` still wins), so the model actually trained is the one asked
    for rather than whatever the config's own default happens to be. Step 4
    supplies the catalog's canonical spelling (`"PaDiM"`, not `"padim"`).

    Steps 4 and 5 pick a config the caller did not name. That is deliberate
    — without them half of `CANONICAL_MODELS` is unreachable by name — but
    it means the choice has to be *visible*: `fdh train` reports
    `resolved_config`/`resolved_variant` in its output for exactly this
    reason.
    """

    direct = Path(model)
    if direct.is_file():
        return direct, None

    directory = Path(config_dir)
    filename = model if model.endswith(".yaml") else f"{model}.yaml"
    by_filename = directory / filename
    if by_filename.is_file():
        return by_filename, None

    # A relative `config_dir` that doesn't exist almost always means the
    # caller is not in the project root, not that the model name is wrong --
    # and every keyword resolver below reads this directory, so without this
    # check all 22 model names fail with "could not resolve <name>", pointing
    # at the name instead of at the cwd.
    if not directory.is_dir():
        raise FileNotFoundError(
            f"model-config directory '{directory}' does not exist"
            + (
                f" relative to the current directory ({Path.cwd()}). Run `fdh` from the "
                "project root, or pass --config-dir with an absolute path."
                if not directory.is_absolute()
                else "; pass --config-dir with a directory that does."
            )
        )

    needle = model.strip().lower()
    for resolver in (_resolve_by_declaration, _resolve_by_catalog, _resolve_by_backend_support):
        resolved = resolver(needle, directory)
        if resolved is not None:
            return resolved

    # Nothing resolved. If several configs *declared* the name, say so --
    # step 3 deferred to the catalog rather than erroring, and with the
    # catalog also silent that ambiguity is the actual reason this failed.
    declaring = [path for path in find_model_configs(directory) if needle in _config_keywords(path)]
    if len(declaring) > 1:
        raise ValueError(
            f"'{model}' is declared by multiple configs under {directory} "
            f"({', '.join(path.name for path in declaring)}) and is not a published model the "
            f"catalog can disambiguate; pass one of those filenames."
        )

    raise FileNotFoundError(
        f"could not resolve model config '{model}' (checked as a path, as a filename under "
        f"'{directory}', as a model declared by a config, as a published model in the "
        f"catalog, and as a variant any backend's presets accept). Configs available under "
        f"'{directory}': {', '.join(path.stem for path in find_model_configs(directory)) or '<none found>'}. "
        f"Model names: `fdh models`."
    )


def _resolve_by_declaration(needle: str, directory: Path) -> tuple[Path, str] | None:
    """Step 3: a config that declares this exact model name.

    Returns None when several configs declare it, deferring to the catalog
    rather than erroring — three configs declare PatchCore
    (`anomalib_example`, `patchcore_textile`, `patchcore_mvtec_repro`), and
    the right answer for a bare `patchcore` is the general-purpose one,
    which only the catalog knows.
    """

    matches = [path for path in find_model_configs(directory) if needle in _config_keywords(path)]
    return (matches[0], needle) if len(matches) == 1 else None


def _resolve_by_catalog(needle: str, directory: Path) -> tuple[Path, str] | None:
    """Step 4: a published model, whose config the catalog already records."""

    from fabric_defect_hub.catalog import CANONICAL_MODELS

    for entry in CANONICAL_MODELS:
        if needle not in {entry.key.strip().lower(), entry.variant.strip().lower()}:
            continue
        path = directory / entry.config
        if not path.is_file():
            # The catalog names a config that isn't in this directory --
            # possible when `config_dir` is overridden (tests, a caller's own
            # config tree). Fall through instead of failing: the later steps
            # may still resolve it against the directory actually in use.
            return None
        return path, entry.variant
    return None


def _resolve_by_backend_support(needle: str, directory: Path) -> tuple[Path, str] | None:
    """Step 5: a variant some backend's presets accept but no config
    declares and the catalog doesn't publish -- `fdh train fastflow`.
    """

    from fabric_defect_hub.api import list_models

    owners = [
        backend
        for backend, variants in list_models().items()
        if needle in {variant.strip().lower() for variant in variants}
    ]
    if len(owners) > 1:
        raise ValueError(
            f"'{needle}' is a variant of more than one backend ({', '.join(sorted(owners))}); "
            f"pass a config filename, or --backend, to disambiguate."
        )
    if not owners:
        return None
    path = directory / f"{owners[0]}_example.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"'{needle}' is a {owners[0]} variant, but that backend's general-purpose config "
            f"{path} does not exist; pass a config path explicitly."
        )
    return path, needle


def _config_keywords(path: Path) -> set[str]:
    """Case-insensitive keywords a config can be looked up by: its filename
    stem, its declared `model.variant`/`model.name`, and (for configs using
    the `variants.<name>` profile mechanism, see
    `models.ultralytics.config.resolve_variant_profile`) every profile key.
    """

    keywords = {path.stem.lower()}
    try:
        raw = load_raw_config(path)
    except Exception:
        return keywords
    model_section = raw.get("model") if isinstance(raw, dict) else None
    if isinstance(model_section, dict):
        for key in ("variant", "name"):
            value = model_section.get(key)
            if value:
                keywords.add(str(value).strip().lower())
    variants = raw.get("variants") if isinstance(raw, dict) else None
    if isinstance(variants, dict):
        keywords.update(str(key).strip().lower() for key in variants)
    return keywords


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """Parse a model-config YAML into a plain mapping, with `${VAR}`
    environment-variable expansion (matching each backend's own
    `Config.from_yaml`), but without building the backend dataclass yet —
    that happens after overrides are layered on in `run_train`.
    """

    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"model config {path} must parse to a mapping")
    return _expand_environment_variables(raw)


def load_training_profile(path: str | Path = DEFAULT_TRAINING_PROFILE) -> DatasetOverrides:
    """Load the shared ZJU data-selection policy used unless CLI overrides it."""
    import yaml

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"training profile not found: {profile_path}")
    raw = yaml.safe_load(profile_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"training profile {profile_path} must be a mapping")
    return DatasetOverrides(
        mode=raw.get("mode"), pattern=raw.get("pattern"), defect_ratio=raw.get("defect_ratio")
    )


def merge_dataset_overrides(base: DatasetOverrides, override: DatasetOverrides) -> DatasetOverrides:
    """CLI values win over the global profile; omitted CLI values inherit it."""
    return DatasetOverrides(**{field.name: getattr(override, field.name) if getattr(override, field.name) is not None else getattr(base, field.name) for field in fields(DatasetOverrides)})


def apply_model_overrides(raw: dict[str, Any], backend: str, variant: str | None) -> dict[str, Any]:
    """Return a copy of `raw` with `variant` layered onto its `model` section.

    Each example config under `configs/models/` previously hardcoded a
    single model (`model.variant: yolov8n`, `model.name: PatchCore`, ...),
    so trying a different model in that family meant hand-copying the whole
    file. This lets `--variant` pick any model that backend supports
    (`yolov8s`, `fasterrcnn_resnet50_fpn`, `PaDiM`, ...) from the same
    config, the same way `--dataset`/`--mode`/etc. already override the
    `data` section.

    `variant` is written under `model.variant` for ultralytics/torchvision
    and `model.name` for anomalib (see `BACKEND_MODEL_KEY`) — whichever
    field that backend's `ModelSpec` actually reads. `None` leaves the
    config's own value alone.

    Also renamespaces `checkpoint.name` (all three backends share that key)
    by prefixing it with `variant`, so overriding the model also gives it
    its own run directory / registry filename instead of colliding with
    (and Ultralytics/torchvision auto-incrementing past) whatever run the
    config's original model variant already produced there.
    """

    if variant is None:
        return raw

    raw = dict(raw)
    model = dict(raw.get("model") or {})
    model[BACKEND_MODEL_KEY[backend]] = variant
    raw["model"] = model

    checkpoint = dict(raw.get("checkpoint") or {})
    current_name = checkpoint.get("name")
    slug = str(variant).strip().lower()
    if current_name and not str(current_name).lower().startswith(slug):
        checkpoint["name"] = f"{slug}_{current_name}"
        raw["checkpoint"] = checkpoint
    return raw


def apply_dataset_overrides(
    raw: dict[str, Any], backend: str, overrides: DatasetOverrides
) -> dict[str, Any]:
    """Return a copy of `raw` with `overrides` layered onto its `data` section.

    Per-model shot logic: Anomalib's five models and Dinomaly all train
    one-class, so their train split always has `use_defect` forced to
    `False` (and `defect_ratio` dropped) regardless of `overrides.use_defect`
    — that flag only ever reaches their *test* split. Ultralytics/torchvision
    have no such constraint; both overrides apply to their train split as given.
    """

    if overrides.is_empty():
        return raw

    data = dict(raw.get("data") or {})
    if overrides.dataset is not None:
        data["dataset"] = overrides.dataset
        data.pop("data_yaml", None)
        data.pop("datamodule_kwargs", None)
        if overrides.dataset_root is None:
            # Switching datasets without also passing --dataset-root: drop
            # whatever root the config declared for its *original* dataset
            # instead of silently reusing it for the new one.
            # `apply_default_dataset_root` (run after this) then resolves
            # the new dataset's own default root.
            data.pop("dataset_root", None)
    if overrides.dataset_root is not None:
        data["dataset_root"] = overrides.dataset_root

    if overrides.test_dataset is not None or overrides.test_dataset_root is not None:
        if backend not in _ZERO_SHOT_BACKENDS:
            raise ValueError(
                f"--test-dataset/--test-dataset-root only apply to the zero-shot backends "
                f"({', '.join(sorted(_ZERO_SHOT_BACKENDS))}); the '{backend}' backend evaluates "
                "on the same dataset it trains on, so use --dataset."
            )
        if overrides.test_dataset is not None:
            data["test_dataset"] = overrides.test_dataset
            if overrides.test_dataset_root is None:
                data.pop("test_dataset_root", None)
        if overrides.test_dataset_root is not None:
            data["test_dataset_root"] = overrides.test_dataset_root

    if not data.get("dataset"):
        raise ValueError(
            "dataset-selection overrides (--mode/--num-samples/--use-defect/...) require "
            "a DatasetAdapter-based 'data.dataset' config; pass --dataset to set one."
        )

    train_key, other_key = _BACKEND_DATA_SELECTIONS[backend]
    train_selection = dict(data.get(train_key) or {})
    other_selection = dict(data.get(other_key) or {})

    dataset_name = data["dataset"]
    _apply_selection_overrides(train_selection, overrides, is_train_split=True, dataset=dataset_name)
    _apply_selection_overrides(other_selection, overrides, is_train_split=False, dataset=dataset_name)

    if backend in _ONE_CLASS_BACKENDS:
        # One-class training: the train split is always normal-only, no
        # matter what --use-defect says (that only ever affects the test
        # split's mix, applied above via `other_selection`).
        train_selection["use_defect"] = False
        train_selection.pop("defect_ratio", None)

    data[train_key] = train_selection
    data[other_key] = other_selection
    raw = dict(raw)
    raw["data"] = data
    return raw


def apply_default_dataset_root(raw: dict[str, Any]) -> dict[str, Any]:
    """Fall back `data.dataset_root` (and `data.test_dataset_root`, where a
    backend has one) to this project's `data/<Dataset>` symlink (see
    `DEFAULT_DATASET_ROOTS`) when the config doesn't already have a usable
    one for the corresponding dataset.

    Runs unconditionally (unlike `apply_dataset_overrides`, which is a
    no-op with no CLI overrides) — a bare `fdh train some_config.yaml`,
    with no flags at all, still needs to resolve to a real path. "Usable"
    means a non-empty string that isn't a leftover `${ENV_VAR}` placeholder
    (see `_expand_environment_variables`: `os.path.expandvars` leaves those
    untouched, rather than erroring or blanking them, when the variable
    isn't set) — an explicit path or a successfully-expanded env var always
    wins over this fallback.
    """

    data = raw.get("data")
    if not isinstance(data, dict):
        return raw

    resolved = dict(data)
    changed = False
    # ("dataset", "dataset_root") is every backend's training corpus;
    # ("test_dataset", "test_dataset_root") only exists for the zero-shot
    # backends, whose evaluation target is a *different* dataset from the
    # one they train on (see `MoECLIPConfig.DataSpec`) — a no-op elsewhere.
    for dataset_key, root_key in (("dataset", "dataset_root"), ("test_dataset", "test_dataset_root")):
        dataset = data.get(dataset_key)
        if not dataset or dataset not in DEFAULT_DATASET_ROOTS:
            continue
        current_root = data.get(root_key)
        has_usable_root = (
            isinstance(current_root, str) and current_root.strip() and "${" not in current_root
        )
        if has_usable_root:
            continue
        resolved[root_key] = DEFAULT_DATASET_ROOTS[dataset]
        changed = True

    if not changed:
        return raw
    raw = dict(raw)
    raw["data"] = resolved
    return raw


def apply_available_dataset(raw: dict[str, Any], backend: str) -> dict[str, Any]:
    """After `apply_default_dataset_root` has resolved paths and
    `_enforce_trainable_dataset` has checked role-legality, substitute an
    actually-*staged* alternative dataset when the configured one isn't
    present on this machine (see `core.decision.decide_dataset`), instead of
    failing deep inside the backend's training loop the moment a dataset
    this particular machine doesn't have gets requested.

    No-op for detection backends (not in `_BACKEND_TRAINABLE_DATASETS`) and
    for the `data_root`/`datamodule_kwargs` on-disk modes (no `data.dataset`
    key at all -- mirrors `_enforce_trainable_dataset`'s own no-op there).
    Raises `FileNotFoundError` -- not deep inside a dataloader, but here,
    with an actionable message -- when nothing in the backend's allowed set
    is staged at all.
    """

    if backend not in _BACKEND_TRAINABLE_DATASETS:
        return raw
    data = raw.get("data")
    if not isinstance(data, dict):
        return raw
    requested = data.get("dataset")
    if not requested:
        return raw

    from fabric_defect_hub.core.decision import decide_dataset

    allowed_set, _ = _BACKEND_TRAINABLE_DATASETS[backend]
    decision = decide_dataset(requested, allowed_set, root_map=DEFAULT_DATASET_ROOTS)
    if not decision.runnable:
        raise FileNotFoundError(f"'{backend}' has no stageable training dataset available. {decision.reason}")
    if not decision.substituted:
        return raw

    import warnings

    warnings.warn(f"[{backend}] {decision.reason}", stacklevel=2)
    resolved = dict(data)
    resolved["dataset"] = decision.dataset
    resolved["dataset_root"] = DEFAULT_DATASET_ROOTS[decision.dataset]
    raw = dict(raw)
    raw["data"] = resolved
    return raw


def apply_raw_overrides(raw: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """The hyperparameter/config "tuning window": layer arbitrary
    dotted-path overrides (e.g. `{"train.model_kwargs.coreset_sampling_ratio":
    0.05}`) onto a parsed config, highest priority of any layer -- above the
    model config file, above a recipe/config-profile's defaults, above every
    other override in this module. Lets one run tweak a single knob (`fdh
    train ... --set train.model_kwargs.lr=0.0005`) without hand-editing or
    forking the YAML.

    Each key is a `.`-separated path; intermediate mappings are created (as
    plain dicts) if they don't already exist, so a profile-only knob can be
    set even when the config file never mentioned that section. A `None`/
    empty `overrides` is a no-op, matching every other `apply_*` function's
    "no CLI input -> leave the config alone" contract.
    """

    if not overrides:
        return raw
    raw = dict(raw)
    for dotted_path, value in overrides.items():
        *parents, leaf = dotted_path.split(".")
        cursor = raw
        for key in parents:
            existing = cursor.get(key)
            if not isinstance(existing, dict):
                existing = {}
                cursor[key] = existing
            else:
                # Don't mutate a dict shared with an earlier layer in place.
                existing = dict(existing)
                cursor[key] = existing
            cursor = existing
        cursor[leaf] = value
    return raw


def _zju_selected_pattern_count(selection: dict[str, Any], overrides: DatasetOverrides) -> int:
    """Return the number of ZJU patterns represented by a selection.

    A missing/``None`` pattern is ZJU-Leaper's documented whole-benchmark
    selector. Lists are the normal explicit multi-pattern form used by the
    fabric recipes. A scalar selector denotes one pattern; group selectors
    are intentionally treated as one configured selection because their
    membership is dataset metadata rather than a stable CLI contract.
    """

    pattern = overrides.pattern if overrides.pattern is not None else selection.get("pattern")
    if pattern is None or pattern == "total":
        return ZJU_LEAPER_PATTERN_COUNT
    if isinstance(pattern, (list, tuple)):
        return max(1, len(pattern))
    return 1


def _resolve_num_samples(
    overrides: DatasetOverrides,
    current: int | None,
    *,
    is_train_split: bool,
    selected_pattern_count: int = ZJU_LEAPER_PATTERN_COUNT,
) -> tuple[bool, int | None]:
    """Return (should_set, value) for one split's `num_samples`."""

    if not is_train_split and overrides.val_num_samples is not None:
        return True, overrides.val_num_samples
    if overrides.num_samples is not None:
        return True, overrides.num_samples
    if overrides.mode == "test":
        return True, TEST_SHOT_NUM_SAMPLES
    if overrides.mode == "full":
        return True, None
    if overrides.mode == "medium":
        per_pattern = MEDIUM_SHOT_TRAIN_PER_PATTERN if is_train_split else MEDIUM_SHOT_VAL_PER_PATTERN
        return True, per_pattern * selected_pattern_count
    if overrides.mode == "few":
        return False, current  # leave the config's own few-shot count as-is
    return False, current


def _apply_selection_overrides(
    selection: dict[str, Any], overrides: DatasetOverrides, *, is_train_split: bool, dataset: str
) -> None:
    selected_pattern_count = (
        _zju_selected_pattern_count(selection, overrides)
        if dataset == "zju-leaper"
        else ZJU_LEAPER_PATTERN_COUNT
    )
    should_set, num_samples = _resolve_num_samples(
        overrides,
        selection.get("num_samples"),
        is_train_split=is_train_split,
        selected_pattern_count=selected_pattern_count,
    )
    if should_set:
        selection["num_samples"] = num_samples
    if overrides.use_defect is not None:
        selection["use_defect"] = overrides.use_defect
    if overrides.defect_ratio is not None:
        selection["defect_ratio"] = overrides.defect_ratio
    if overrides.pattern is not None:
        selection["pattern"] = overrides.pattern
    elif (
        dataset == "zju-leaper"
        and overrides.mode in ("full", "medium")
        and selection.get("pattern") is None
    ):
        # An unrestricted config retains the historic whole-benchmark
        # behavior. Explicit recipe patterns (such as [pattern1, ..., pattern4])
        # are never widened by a shot-mode choice.
        selection["pattern"] = None if overrides.mode == "full" else list(range(1, ZJU_LEAPER_PATTERN_COUNT + 1))
    if overrides.category is not None:
        selection["category"] = overrides.category
    if overrides.seed is not None:
        selection["seed"] = overrides.seed


def _apply_test_speed_overrides(raw: dict[str, Any], backend: str) -> dict[str, Any]:
    """`mode == "test"` also caps epochs so the 8-image smoke run finishes
    fast — it exists to prove the pipeline wiring works end to end, not to
    produce a usable checkpoint.

    Forces the cap (rather than `setdefault`ing it) because every example
    config already declares its own `epochs`/`patience` — a `setdefault`
    would be a no-op against those and let `--mode test` silently run the
    config's full epoch count (confirmed: `ultralytics_example.yaml`
    declares `patience: 30`, so an 8-image `--mode test` run trained for 35
    epochs before early-stopping kicked in instead of the intended 1).
    """

    raw = dict(raw)
    train = dict(raw.get("train") or {})
    if backend == "anomalib":
        engine_kwargs = dict(train.get("engine_kwargs") or {})
        engine_kwargs["max_epochs"] = 1
        train["engine_kwargs"] = engine_kwargs
    elif backend == "dinomaly":
        # Also shrink batch/image size, not just total_iters: at the
        # config's own defaults (batch_size 16, 448/392 images) even 8
        # iterations of a DINOv2-base forward+backward on CPU is minutes,
        # not seconds -- defeats the point of a "prove the wiring" smoke run.
        train["total_iters"] = TEST_SHOT_NUM_SAMPLES
        train["batch_size"] = min(int(train.get("batch_size", TEST_SHOT_NUM_SAMPLES)), TEST_SHOT_NUM_SAMPLES)
        train["image_size"] = 98
        train["crop_size"] = 84
    elif backend == "moeclip":
        # Same reasoning as Dinomaly's branch: one epoch over the 8 staged
        # images, at the smallest image size the backbone's 14px patch grid
        # allows, so the smoke run proves the wiring in seconds rather than
        # forward+backwarding a ViT-L at 518px on CPU.
        train["epochs"] = 1
        train["batch_size"] = min(int(train.get("batch_size", 2)), 2)
        raw = _apply_test_speed_model_overrides(raw, {"img_size": 112})
    elif backend == "mambaad":
        # A handful of iterations over the 8 staged images at the smallest
        # image size its teacher's 32x total downsampling allows (a
        # power-of-two multiple of 32, since the default scan_type
        # 'hilbert' also requires each decoder stage's grid to be a power
        # of two -- see scan.py) -- proves the wiring, not a usable model.
        train["total_iters"] = TEST_SHOT_NUM_SAMPLES
        train["batch_size"] = min(int(train.get("batch_size", TEST_SHOT_NUM_SAMPLES)), TEST_SHOT_NUM_SAMPLES)
        train["image_size"] = 64
    else:
        train["epochs"] = 1
        train["patience"] = 1
    raw["train"] = train
    return raw


def _apply_test_speed_model_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """`mode == "test"` overrides that live under `model:` rather than
    `train:` (MoECLIP's `img_size` is an architecture knob, not a training
    one -- it has to match at inference time, so it is declared where the
    rest of the architecture is).
    """

    raw = dict(raw)
    model = dict(raw.get("model") or {})
    model.update(overrides)
    raw["model"] = model
    return raw


@dataclass
class TrainRunResult:
    """What `run_train` produced, tagged with the backend that ran it."""

    backend: str
    result: Any  # UltralyticsRunResult | TorchvisionRunResult | AnomalibRunResult
    published_path: str | None = None  # set when (backend, variant) is one of catalog.CANONICAL_MODELS
    weight_manifest_path: str | None = None
    # What the `model` argument actually resolved to. Reported because steps
    # 4 and 5 of `resolve_model_config_and_variant` can pick a config the
    # caller never named (`fdh train padim` -> anomalib_example.yaml), and a
    # choice made on someone's behalf has to be inspectable afterwards.
    config_path: str | None = None
    variant: str | None = None


def _enforce_trainable_dataset(raw: dict[str, Any], backend: str) -> None:
    """Reject training an anomaly backend on a dataset that isn't a valid
    training source *for that backend* — which cuts in opposite directions
    for the one-class and zero-shot backends (see
    `ANOMALY_TRAINABLE_DATASETS` / `ZERO_SHOT_TRAINABLE_DATASETS`). No-op
    for detection backends (which legitimately train on detection sets)
    and for the `data_root`/`datamodule_kwargs` on-disk modes, where no
    registered dataset name is involved and the caller has taken explicit
    responsibility for the folder.

    Only the *training* corpus (`data.dataset`) is checked; a zero-shot
    backend's evaluation target (`data.test_dataset`) is deliberately
    unrestricted — evaluating on fabric is the whole point.
    """

    if backend not in _BACKEND_TRAINABLE_DATASETS:
        return
    data = raw.get("data")
    dataset = data.get("dataset") if isinstance(data, dict) else None
    allowed_set, kind = _BACKEND_TRAINABLE_DATASETS[backend]
    if not dataset or dataset in allowed_set:
        return
    allowed = ", ".join(sorted(allowed_set))
    if kind == "zero-shot":
        raise ValueError(
            f"dataset {dataset!r} is not a training source for the zero-shot "
            f"'{backend}' backend. It is trained on an auxiliary cross-domain "
            f"corpus and applied to unseen categories, so training is restricted "
            f"to: {allowed}. Training it on fabric would make its fabric scores "
            "in-domain and void the zero-shot claim — point data.test_dataset at "
            "the fabric set you want it evaluated on instead."
        )
    raise ValueError(
        f"dataset {dataset!r} is not a training source for the one-class "
        f"'{backend}' backend. Anomaly training is restricted to in-domain "
        f"fabric sources: {allowed}. Cross-domain benchmarks (mvtec-ad, "
        "mvtec-loco, visa) are eval-only for these models — use them for "
        "inference/benchmark, not training. To train on the combined fabric "
        "corpus use 'fabric-train'."
    )


def run_train(
    model: str | Path,
    backend: str | None = None,
    overrides: DatasetOverrides | None = None,
    config_dir: str | Path = DEFAULT_MODEL_CONFIG_DIR,
    variant: str | None = None,
    publish: bool = True,
    set_overrides: dict[str, Any] | None = None,
    profile: str | Path | None = DEFAULT_TRAINING_PROFILE,
) -> TrainRunResult:
    """The unified training entry point.

    1. Resolve `model` to a config path (see `resolve_model_config` — a
       full path, a filename stem under `config_dir`, or a model keyword
       like "yolov8n"/"patchcore" all work), parse it, and resolve its
       backend (`backend`, if given, skips keyword detection).
    2. Layer `variant` onto its `model` section (which concrete model in
       that backend's family to train — see `apply_model_overrides`).
    3. Layer `overrides` onto its `data` section (dataset choice, shot mode,
       sample count, defect mix, ...), then fall back any still-unresolved
       `dataset_root` to this project's own `data/<Dataset>` symlink
       convention (see `apply_default_dataset_root`) — so this works the
       same on every machine that has its datasets staged there, with no
       environment variable or machine-specific path required.
    4. If the resolved dataset isn't actually staged *on this machine*,
       substitute an available alternative from the same allowed set rather
       than failing deep inside the backend (see `apply_available_dataset` /
       `core.decision.decide_dataset`) — not every machine (a cloud training
       box especially) has every dataset this project knows about staged.
    5. Layer `set_overrides` on top of everything else — the tuning window:
       arbitrary dotted-path config overrides (e.g.
       `{"train.model_kwargs.lr": 0.0005}`), highest priority of any layer
       (see `apply_raw_overrides`).
    6. Build that backend's own config dataclass and run its full
       train/val/export lifecycle via `models.<backend>.pipeline.run_from_config`.
    7. If `publish` (default) and the resolved (backend, variant) is one of
       `catalog.CANONICAL_MODELS`, copy the registered artifact to its fixed
       published path — the location the frontend's `MODEL_CATALOG` (see
       `web/single_image.py`) actually reads from. A no-op for any run that
       isn't one of those models (e.g. a one-off backbone sweep); the run
       itself is unaffected either way, only `TrainRunResult.published_path`.
    """

    model_config, implied_variant = resolve_model_config_and_variant(str(model), config_dir=config_dir)
    variant = variant if variant is not None else implied_variant
    raw = load_raw_config(model_config)
    resolved_backend = backend or infer_backend(raw)
    if resolved_backend not in _BACKEND_PIPELINE_MODULES:
        raise ValueError(
            f"unknown backend '{resolved_backend}'; expected one of {sorted(_BACKEND_PIPELINE_MODULES)}"
        )

    raw = apply_model_overrides(raw, resolved_backend, variant)
    if resolved_backend == "ultralytics":
        from fabric_defect_hub.models.ultralytics.config import resolve_variant_profile

        raw = resolve_variant_profile(raw)

    overrides = overrides or DatasetOverrides()
    if profile is not None and raw.get("data", {}).get("dataset") == "zju-leaper":
        overrides = merge_dataset_overrides(load_training_profile(profile), overrides)
    raw = apply_dataset_overrides(raw, resolved_backend, overrides)
    _enforce_trainable_dataset(raw, resolved_backend)
    raw = apply_default_dataset_root(raw)
    raw = apply_available_dataset(raw, resolved_backend)
    if overrides.mode == "test":
        raw = _apply_test_speed_overrides(raw, resolved_backend)
    raw = apply_raw_overrides(raw, set_overrides)

    config_module_name, config_cls_name = _BACKEND_CONFIG_CLASSES[resolved_backend]
    config_cls = getattr(importlib.import_module(config_module_name), config_cls_name)
    config = config_cls.from_dict(raw)

    pipeline_module = importlib.import_module(_BACKEND_PIPELINE_MODULES[resolved_backend])
    result = pipeline_module.run_from_config(config)

    published: str | None = None
    if publish and result.registered_artifact is not None:
        from fabric_defect_hub.catalog import publish_artifact

        model_key = BACKEND_MODEL_KEY[resolved_backend]
        resolved_variant = raw.get("model", {}).get(model_key)
        if resolved_variant:
            destination = publish_artifact(resolved_backend, resolved_variant, result.registered_artifact.path)
            published = str(destination) if destination is not None else None

    manifest_path: str | None = None
    if result.registered_artifact is not None:
        from fabric_defect_hub.weight_registry import record_weight

        model_key = BACKEND_MODEL_KEY[resolved_backend]
        manifest_path = str(
            record_weight(
                project_root=Path(__file__).resolve().parents[2],
                backend=resolved_backend,
                variant=raw.get("model", {}).get(model_key),
                config_path=model_config,
                resolved_config=raw,
                registered_artifact=result.registered_artifact,
                published_path=published,
                metrics=result.metrics,
                run_id=os.environ.get("FDH_BATCH_RUN_ID"),
                model_key=os.environ.get("FDH_BATCH_MODEL_KEY"),
            )
        )

    return TrainRunResult(
        backend=resolved_backend,
        result=result,
        published_path=published,
        weight_manifest_path=manifest_path,
        config_path=str(model_config),
        variant=variant,
    )


def _expand_environment_variables(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value
