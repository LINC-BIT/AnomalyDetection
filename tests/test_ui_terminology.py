"""One concept, one name: the front end speaks README's vocabulary.

The page used to name the same thing several ways — the model dropdown was
"Local trained model" in one place and "Model" in the workflow it belongs to,
the pixel-level output chip said "Heatmap" while README writes "heat map", the
single-image tab mixed "detection" and "inference", and the run-history table
headed its columns with the JSONL's own keys (`image_auroc`) beside benchmark
tables that call the same metric "Image AUROC".

README.md is the project's terminology. Section 4.1 names the controls a reader
is told to click, section 1 defines what a benchmark is, and section 3.3 names
the metric tables. The tests below make that mechanical:

* every control README names is labelled verbatim in the UI table;
* a banned second name cannot come back;
* the two languages stay in lockstep, key for key and placeholder for
  placeholder, so a concept cannot exist in one language only;
* no key is orphaned, so a renamed string leaves no stale name behind.
"""

from __future__ import annotations

import string
from pathlib import Path

import pytest

from fabric_defect_hub.i18n import _STRINGS

# A control README 4.1 names, and the key whose value must be that exact name.
# Quoting README verbatim is the point: a paraphrase here is the bug.
README_NAMES: dict[str, str] = {
    "tab_single_image": "Single Image Detection",
    "tab_benchmark": "Benchmark",
    "tab_run_history": "Run History",
    "task_type_label": "Task type",
    "choice_task_anomaly": "Anomaly detection",
    "choice_task_defect": "Defect detection",
    "application_domain_label": "Application domain",
    "model_dropdown_label": "Model",
    "btn_load_model": "Load model",
    "btn_run_detection": "Run detection",
    "dataset_dropdown_label": "Dataset",
    "benchmark_dataset_label": "Dataset",
    "texture_dropdown_label": "Texture / pattern",
    "benchmark_texture_label": "Texture / pattern",
    "split_label": "Split",
    "sample_regime_label": "Sample regime",
    "benchmark_shot_label": "Sample regime (test split)",
    "choice_full_shot": "Full-shot",
    "choice_few_shot": "Few-shot",
    "choice_all_images": "All images",
    "btn_load_random_images": "Load random images",
    "btn_run_benchmark": "Run benchmark",
    "history_metric_label": "Metric to Chart",
    "btn_history_refresh": "Refresh",
}

# README names these with an explanatory parenthetical, so the control keeps a
# short tail; the README phrase itself must still lead the label.
README_PHRASES: dict[str, str] = {
    "benchmark_profiling_label": "Include profiling",
    "benchmark_resolution_sweep_label": "Include resolution sweep",
    "benchmark_cross_domain_label": "Cross-domain degradation target dataset",
}

# Second names this change retired. Each is a real string that used to appear
# on the page for a concept README already names.
BANNED_SECOND_NAMES: tuple[str, ...] = (
    "Heatmap",                      # README writes "heat map"
    "Local trained model",          # README 4.1.2: "Model"
    "Inference result",             # the tab's action is detection
    "Include performance profiling",  # README 4.1.3: "Include profiling"
    "resolution-sensitivity",       # README 4.1.3: "Include resolution sweep"
    "Metric to chart",              # README 4.1.4: "Metric to Chart"
    "Trained on",                   # README 4.1.2: "training corpus"
    "Model session",                # README 4.1.2: "Model Setup"
    "Dataset sampler",              # README 4.1.2: "Dataset & Sample Selection"
    "Workspace · Datasets · Models · Results",
)

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "fabric_defect_hub"


def _strings(lang: str) -> dict[str, str]:
    return _STRINGS[lang]


def test_every_readme_control_name_is_used_verbatim():
    en = _strings("en")
    wrong = {
        key: (name, en.get(key))
        for key, name in README_NAMES.items()
        if en.get(key) != name
    }
    assert not wrong, (
        f"README 4.1 names these controls; the UI must use the same words, not a "
        f"second name for them. Offending key -> (README, UI): {wrong}"
    )


def test_readme_phrases_lead_their_labels():
    en = _strings("en")
    wrong = {
        key: (phrase, en.get(key))
        for key, phrase in README_PHRASES.items()
        if not (en.get(key, "").startswith(phrase))
    }
    assert not wrong, f"a README-named option must start with its README name: {wrong}"


@pytest.mark.parametrize("lang", sorted(_STRINGS))
def test_no_retired_second_name_survives(lang: str):
    table = _strings(lang)
    offenders = {
        key: (banned, text)
        for key, text in table.items()
        for banned in BANNED_SECOND_NAMES
        if banned in text
    }
    assert not offenders, (
        f"{lang} still carries a second name for a concept README already names: {offenders}"
    )


def test_languages_cover_the_same_concepts():
    en, zh = set(_strings("en")), set(_strings("zh"))
    assert en == zh, (
        f"en-only keys: {sorted(en - zh)}; zh-only keys: {sorted(zh - en)}. A concept "
        f"must exist in both languages or the toggle leaves a key visible as its own name."
    )


def test_translations_keep_the_same_placeholders():
    en, zh = _strings("en"), _strings("zh")
    mismatched = {
        key: (sorted(_placeholders(en[key])), sorted(_placeholders(zh[key])))
        for key in en
        if _placeholders(en[key]) != _placeholders(zh[key])
    }
    assert not mismatched, (
        f"a translation with different placeholders renders a literal `{{name}}` or "
        f"raises on format: {mismatched}"
    )


def test_no_orphaned_string_keys():
    """Every key is still referenced somewhere, so a renamed control leaves no
    stale name behind. Scans the package's source for the quoted key rather
    than only `tr(lang, "key")` calls, because several keys are selected
    through a variable (`verdict_key`, `mapping.get(...)`)."""

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in SOURCE_ROOT.rglob("*.py")
        if path.name != "i18n.py"
    )
    orphaned = sorted(
        key for key in _strings("en")
        if f'"{key}"' not in source and f"'{key}'" not in source
    )
    assert not orphaned, (
        f"these keys are no longer used anywhere under src/fabric_defect_hub/: {orphaned}"
    )


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}
