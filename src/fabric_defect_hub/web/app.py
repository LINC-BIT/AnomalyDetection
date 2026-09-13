"""Gradio application entry point for AnomalyDetection."""

from __future__ import annotations

import os

from fabric_defect_hub.i18n import DEFAULT_LANGUAGE, LANGUAGES, tr
from fabric_defect_hub.inference.session import InferenceSessionManager, format_session_status
from fabric_defect_hub.reporting import flatten_run_log_rows, latest_run_per_model, read_run_log
from fabric_defect_hub.metrics_taxonomy import UNIMPLEMENTED
from fabric_defect_hub.web.tables import empty_sections, render_sections, status_line
from fabric_defect_hub.application.benchmark import (
    DEFAULT_RUN_LOG_PATH,
    compatible_models,
    run_benchmark,
    score_preset_choices,
)
from fabric_defect_hub.application.workspace import (
    DATASET_CATALOG,
    MODEL_CATALOG,
    checkpoint_diagnostic,
    current_image,
    dataset_status as dataset_availability_status,
    default_dataset_root,
    detect_loaded_model,
    empty_gallery_state,
    image_scope_choices,
    load_random_samples,
    load_selected_model,
    model_status,
    move_image,
    render_prediction_tags,
    shot_mode_choices,
    split_choices,
    texture_choices,
    unload_selected_model,
)


CSS = """
html, body, .gradio-container { min-height: 100%; }
body.dark, body.dark .gradio-container { background: #07111f !important; color: #e5edf8 !important; }
body.dark .gradio-container { --body-background-fill: #07111f; --block-background-fill: #111d2e; --block-border-color: #25354c; --block-label-text-color: #a9bad0; --input-background-fill: #0b1626; --input-border-color: #30445f; --input-placeholder-color: #71849d; --body-text-color: #e5edf8; --color-accent: #fb923c; }
body:not(.dark), body:not(.dark) .gradio-container { background: #f4f7fb !important; color: #172033 !important; }
body:not(.dark) .gradio-container { --body-background-fill: #f4f7fb; --block-background-fill: #ffffff; --block-border-color: #d7e0ec; --block-label-text-color: #475569; --input-background-fill: #ffffff; --input-border-color: #cbd5e1; --input-placeholder-color: #94a3b8; --body-text-color: #172033; --color-accent: #ea6e18; }
.fdh-shell { max-width: 1440px; margin: 0 auto; padding: 0 8px 28px; }
.fdh-nav-row { display: flex; align-items: center; padding: 12px 2px 26px; gap: 16px; }
.fdh-nav { display: flex; align-items: center; justify-content: space-between; flex: 1; }
.fdh-brand { display: flex; gap: 12px; align-items: center; font-weight: 800; font-size: 22px; }
.fdh-logo { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 10px; background: #f97316; color: #fff; }
.fdh-links { font-size: 14px; }
.fdh-lang { max-width: 120px; }
.fdh-lang .wrap-inner { min-height: 0 !important; }
.fdh-card, .fdh-control-card, .fdh-dataset-card { border-radius: 14px; padding: 10px; }
.fdh-control-card { min-height: 0; }
.fdh-card label, .fdh-card .wrap, .fdh-card .prose, .fdh-control-card label, .fdh-control-card .wrap, .fdh-dataset-card label { font-weight: 600; }
.fdh-status { padding: 10px 14px; border-radius: 10px; }
.fdh-caption { text-align: center; min-height: 26px; }
.fdh-nav-button button { min-width: 110px; border-radius: 10px !important; }
.fdh-primary button { background: #ea6e18 !important; border-color: #fb923c !important; color: #fff !important; }
.fdh-action-run button { height: 52px !important; min-height: 52px !important; max-height: 52px !important; font-size: 15px; font-weight: 700; width: 100%; margin-top: 6px; }
.fdh-dataset-actions { align-items: end; }
.fdh-placeholder { padding: 80px 30px; text-align: center; }
.fdh-tagpanel-header { font-weight: 700; margin-bottom: 8px; font-size: 15px; }
.fdh-tagpanel-empty { padding: 10px 2px; }
.fdh-tags { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.fdh-tags-column { flex-direction: column; align-items: flex-start; gap: 6px; }
.fdh-tag-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.fdh-tag { display: inline-flex; align-items: center; padding: 5px 12px; border-radius: 999px; font-size: 13px; font-weight: 700; color: #fff; white-space: nowrap; line-height: 1.6; }
.fdh-tag-label { background: var(--color-accent); }
.fdh-tag-normal { background: #16a34a; }
.fdh-tag-anomalous { background: #dc2626; }
.fdh-tag-neutral { background: #64748b; }
body.dark .fdh-brand { color: #f8fafc; }
body.dark .fdh-links { color: #9fb0c8; }
body.dark .fdh-tagpanel-empty { color: #bdcbe0; }
body.dark .fdh-card, body.dark .fdh-control-card, body.dark .fdh-dataset-card { border: 1px solid #293b54; background: #101c2d; box-shadow: 0 8px 24px rgba(0,0,0,.18); }
body.dark .fdh-card label, body.dark .fdh-card .wrap, body.dark .fdh-card .prose, body.dark .fdh-control-card label, body.dark .fdh-control-card .wrap, body.dark .fdh-dataset-card label { color: #dce7f5 !important; }
body.dark .fdh-card input:not([type="checkbox"]):not([type="radio"]), body.dark .fdh-card textarea, body.dark .fdh-card button.secondary, body.dark .fdh-control-card input:not([type="checkbox"]):not([type="radio"]), body.dark .fdh-control-card button.secondary, body.dark .fdh-dataset-card input:not([type="checkbox"]):not([type="radio"]), body.dark .fdh-dataset-card button.secondary { background: #0a1524 !important; color: #e5edf8 !important; border-color: #30445f !important; }
body.dark .fdh-card label.selected, body.dark .fdh-control-card label.selected, body.dark .fdh-dataset-card label.selected { background: #ea6e18 !important; border-color: #fb923c !important; color: #fff !important; }
body.dark .fdh-card button.secondary:hover, body.dark .fdh-control-card button.secondary:hover, body.dark .fdh-dataset-card button.secondary:hover { background: #16283f !important; border-color: #3d5776 !important; }
body.dark .fdh-card button.secondary:active, body.dark .fdh-control-card button.secondary:active, body.dark .fdh-dataset-card button.secondary:active { background: #1d3552 !important; border-color: #fb923c !important; }
body.dark .fdh-status { background: #182a3d; border: 1px solid #36516d; color: #dbeafe; }
body.dark .fdh-caption, body.dark .fdh-placeholder { color: #bdcbe0; }
body:not(.dark) .fdh-brand { color: #172033; }
body:not(.dark) .fdh-links { color: #64748b; }
body:not(.dark) .fdh-tagpanel-empty { color: #52627a; }
body:not(.dark) .fdh-card, body:not(.dark) .fdh-control-card, body:not(.dark) .fdh-dataset-card { border: 1px solid #d7e0ec; background: #ffffff; box-shadow: 0 6px 18px rgba(15,23,42,.07); }
body:not(.dark) .fdh-card label, body:not(.dark) .fdh-card .wrap, body:not(.dark) .fdh-card .prose, body:not(.dark) .fdh-control-card label, body:not(.dark) .fdh-control-card .wrap, body:not(.dark) .fdh-dataset-card label { color: #172033 !important; }
body:not(.dark) .fdh-card input:not([type="checkbox"]):not([type="radio"]), body:not(.dark) .fdh-card textarea, body:not(.dark) .fdh-card button.secondary, body:not(.dark) .fdh-control-card input:not([type="checkbox"]):not([type="radio"]), body:not(.dark) .fdh-control-card button.secondary, body:not(.dark) .fdh-dataset-card input:not([type="checkbox"]):not([type="radio"]), body:not(.dark) .fdh-dataset-card button.secondary { background: #ffffff !important; color: #172033 !important; border-color: #cbd5e1 !important; }
body:not(.dark) .fdh-card label.selected, body:not(.dark) .fdh-control-card label.selected, body:not(.dark) .fdh-dataset-card label.selected { background: #ea6e18 !important; border-color: #ea6e18 !important; color: #fff !important; }
body:not(.dark) .fdh-card button.secondary:hover, body:not(.dark) .fdh-control-card button.secondary:hover, body:not(.dark) .fdh-dataset-card button.secondary:hover { background: #f4f7fb !important; border-color: #94a3b8 !important; }
body:not(.dark) .fdh-card button.secondary:active, body:not(.dark) .fdh-control-card button.secondary:active, body:not(.dark) .fdh-dataset-card button.secondary:active { background: #e9edf3 !important; border-color: #ea6e18 !important; }
body:not(.dark) .fdh-status { background: #f8fafc; border: 1px solid #cbd5e1; color: #24344d; }
body:not(.dark) .fdh-caption, body:not(.dark) .fdh-placeholder { color: #52627a; }

/* -- Checkboxes and radios ------------------------------------------------
   These were invisible when ticked: the `.fdh-card input` rules above set a
   flat background with `!important`, which also matched `input[type=checkbox]`
   and painted over the checked-state fill the browser draws — so clicking
   "Include profiling" changed nothing you could see. The rules above now
   exclude checkbox/radio; these give the control a size and a checked colour
   in both themes so the state is unmistakable rather than merely present. */
.fdh-card input[type="checkbox"], .fdh-card input[type="radio"],
.fdh-control-card input[type="checkbox"], .fdh-control-card input[type="radio"],
.fdh-dataset-card input[type="checkbox"], .fdh-dataset-card input[type="radio"] {
  appearance: auto; -webkit-appearance: auto;
  width: 18px; height: 18px; min-width: 18px; flex: 0 0 18px;
  accent-color: var(--color-accent); cursor: pointer; margin-right: 8px;
}
.fdh-control-card label:has(input[type="checkbox"]) { display: inline-flex; align-items: center; cursor: pointer; }
/* A ring on the checked control, so the state survives any theme in which
   `accent-color` alone is too low-contrast to notice at a glance. */
.fdh-card input[type="checkbox"]:checked, .fdh-control-card input[type="checkbox"]:checked,
.fdh-dataset-card input[type="checkbox"]:checked {
  outline: 2px solid var(--color-accent); outline-offset: 1px; border-radius: 3px;
}
.fdh-card input[type="checkbox"]:focus-visible, .fdh-control-card input[type="checkbox"]:focus-visible {
  outline: 3px solid var(--color-accent); outline-offset: 2px;
}

"""


def _nav_html(lang: str) -> str:
    return (
        "<div class='fdh-nav'><a class='fdh-brand' href='https://github.com/LINC-BIT/AnomalyDetection' "
        "target='_blank' rel='noopener noreferrer'><span class='fdh-logo'>AD</span>"
        f"AnomalyDetection</a><div class='fdh-links'>{tr(lang, 'nav_links')}</div></div>"
    )


def _lang_choices() -> list[tuple[str, str]]:
    return [(display, code) for code, display in LANGUAGES.items()]


def _tables_from_leaderboard(
    scored: list[dict], columns: list[str], profiling_requested: bool = False
) -> dict[str, dict]:
    """Regroup benchmark rows into the taxonomy's tables.

    The leaderboard produces one wide row per model; the page shows several
    narrow tables. The reshaping goes through `api.BenchmarkResult` — the
    same object `adh benchmark` exposes to scripts — precisely so the UI
    cannot become a second opinion about which table a metric belongs in.
    A metric with no `MetricSpec` is dropped here and reported by
    `tables.status_line`, rather than being quietly rendered somewhere.
    """

    from fabric_defect_hub.api import BenchmarkResult

    rows = [
        {"model": row.get("model", ""), "status": "ok", "metrics": {
            key: value for key, value in row.items() if key != "model"
        }}
        for row in (scored or [])
    ]
    result = BenchmarkResult(rows=rows)
    rendered = {
        table["name"]: table
        for section in render_sections(result)
        for table in section["tables"]
    }
    tables = {
        name: {"headers": spec["headers"], "rows": spec["rows"], "note": spec["note"]}
        for name, spec in rendered.items()
    }
    # The default hint for compute/memory says "tick Include profiling" —
    # actively wrong once the box *was* ticked and the pass failed. In that
    # case the real explanation is in the status line's "profiling skipped
    # (...)" entry, so point there instead of at the checkbox.
    if profiling_requested:
        for name in ("compute", "memory"):
            if not tables[name]["rows"]:
                tables[name]["note"] = (
                    "Profiling was requested but produced no metrics for these models — "
                    "see the status message above for the reason it was skipped."
                )
    return tables


def create_app():
    try:
        gr = _import_gradio_for_local_ui()
    except ImportError as exc:
        raise RuntimeError("Install the UI dependencies first: pip install -r requirements.txt") from exc

    sessions = InferenceSessionManager()
    default_model = next(iter(MODEL_CATALOG))
    default_spec = MODEL_CATALOG[default_model]
    default_task = "anomaly" if default_spec.get("task") == "anomaly" else "defect"
    default_domain = default_spec.get("domain", "general")
    domain_values = sorted({spec.get("domain", "general") for spec in MODEL_CATALOG.values()})
    default_dataset = next(iter(DATASET_CATALOG))
    lang0 = DEFAULT_LANGUAGE

    with gr.Blocks(title="AnomalyDetection") as app:
        lang_state = gr.State(lang0)
        with gr.Column(elem_classes="fdh-shell"):
            with gr.Row(elem_classes="fdh-nav-row"):
                nav_html = gr.HTML(_nav_html(lang0))
                lang_choice = gr.Dropdown(
                    choices=_lang_choices(), value=lang0, show_label=False,
                    container=False, scale=0, min_width=100, elem_classes="fdh-lang",
                )
            with gr.Tabs():
                with gr.Tab(tr(lang0, "tab_single_image"), id="single-image") as tab_single:
                    state = gr.State(empty_gallery_state())

                    # Column 1: model session + Run detection. Column 2: the
                    # sampled dataset image. Column 3: the annotated result.
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            model_header = gr.Markdown(tr(lang0, "model_session_header"))
                            task_choice = gr.Dropdown(
                                choices=[("Defect detection", "defect"), ("Anomaly detection", "anomaly")],
                                value=default_task, label="Task type",
                            )
                            domain_choice = gr.Dropdown(
                                choices=domain_values, value=default_domain, label="Application domain",
                            )
                            model_choice = gr.Dropdown(
                                list(MODEL_CATALOG), value=default_model, label=tr(lang0, "model_dropdown_label")
                            )
                            model_state = gr.Markdown(model_status(default_model, lang0), elem_classes="fdh-status")
                            with gr.Row():
                                load_model_button = gr.Button(tr(lang0, "btn_load_model"), variant="secondary")
                                unload_model_button = gr.Button(tr(lang0, "btn_unload_model"), variant="secondary")
                            verify_model_button = gr.Button(tr(lang0, "btn_inspect_checkpoint"), variant="secondary")
                            detect_button = gr.Button(
                                tr(lang0, "btn_run_detection"), variant="primary",
                                elem_classes="fdh-primary fdh-action-run",
                            )
                        with gr.Column(scale=4, elem_classes="fdh-card"):
                            source_image = gr.Image(
                                label=tr(lang0, "image_selected_label"), height=360, interactive=False
                            )
                            position = gr.Markdown(tr(lang0, "caption_no_image"), elem_classes="fdh-caption")
                            with gr.Row():
                                previous = gr.Button(tr(lang0, "btn_previous"), elem_classes="fdh-nav-button")
                                next_image = gr.Button(tr(lang0, "btn_next"), elem_classes="fdh-nav-button")
                        with gr.Column(scale=5, elem_classes="fdh-card"):
                            result_image = gr.Image(
                                label=tr(lang0, "image_result_label"), height=360, interactive=False
                            )

                    # Runtime memory sits under the sampled image; the new
                    # colored-tag prediction result sits under the annotated
                    # result image right above it.
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=4, elem_classes="fdh-card"):
                            runtime_header = gr.Markdown(tr(lang0, "runtime_memory_header"))
                            runtime_state = gr.Markdown(
                                format_session_status(sessions.status(), lang0), elem_classes="fdh-status"
                            )
                        with gr.Column(scale=5, elem_classes="fdh-card"):
                            result_summary = gr.HTML(render_prediction_tags({}, lang0), elem_classes="fdh-status")
                            inference_status = gr.Markdown(tr(lang0, "inference_hint_start"), elem_classes="fdh-status")

                    with gr.Column(elem_classes="fdh-dataset-card"):
                        dataset_header = gr.Markdown(tr(lang0, "dataset_sampler_header"))
                        with gr.Row():
                            with gr.Column(scale=3):
                                dataset_choice = gr.Dropdown(
                                    list(DATASET_CATALOG), value=default_dataset, label=tr(lang0, "dataset_dropdown_label")
                                )
                            with gr.Column(scale=3):
                                texture_choice = gr.Dropdown(
                                    texture_choices(default_dataset), value="All textures",
                                    label=tr(lang0, "texture_dropdown_label"),
                                )
                            with gr.Column(scale=2):
                                split = gr.Radio(choices=split_choices(lang0), value="test", label=tr(lang0, "split_label"))
                            with gr.Column(scale=2):
                                sample_count = gr.Slider(4, 12, value=8, step=1, label=tr(lang0, "slider_random_images_label"))
                            with gr.Column(scale=2):
                                image_scope = gr.Dropdown(
                                    choices=image_scope_choices(lang0), value="All images",
                                    label=tr(lang0, "image_selection_label"),
                                )
                            with gr.Column(scale=2):
                                shot_mode = gr.Radio(
                                    choices=shot_mode_choices(lang0), value="Full-shot", label=tr(lang0, "sample_regime_label")
                                )
                            with gr.Column(scale=2, elem_classes="fdh-dataset-actions"):
                                load_button = gr.Button(tr(lang0, "btn_load_random_images"))
                        dataset_status = gr.Markdown(
                            dataset_availability_status(default_dataset, lang0), elem_classes="fdh-status"
                        )

                    def load_handler(dataset, texture, selected_split, count, selected_scope, selected_shot_mode, lang):
                        try:
                            new_state, image, caption, status = load_random_samples(
                                dataset,
                                selected_split,
                                count,
                                texture_label=texture,
                                image_scope=selected_scope,
                                shot_mode=selected_shot_mode,
                                lang=lang,
                            )
                            return (
                                new_state, image, caption, status, None,
                                render_prediction_tags({}, lang), tr(lang, "inference_hint_ready"),
                            )
                        except Exception as exc:
                            return (
                                empty_gallery_state(), None, tr(lang, "caption_no_image"),
                                tr(lang, "dataset_load_error", error=exc), None, render_prediction_tags({}, lang), "",
                            )

                    def move_handler(current_state, direction, lang):
                        new_state, image, caption = move_image(current_state, direction, lang)
                        return new_state, image, caption, None, render_prediction_tags({}, lang), tr(lang, "inference_hint_changed")

                    def detect_handler(current_state, model_label, lang):
                        image, summary, status = detect_loaded_model(sessions, current_state, model_label, lang)
                        return image, render_prediction_tags(summary, lang), status, format_session_status(sessions.status(), lang)

                    def load_model_handler(model_label, current_state, lang):
                        try:
                            return format_session_status(load_selected_model(sessions, model_label, current_state), lang)
                        except Exception as exc:
                            return tr(lang, "model_load_failed", error_type=type(exc).__name__, error=exc)

                    def unload_model_handler(lang):
                        return format_session_status(unload_selected_model(sessions), lang)

                    load_button.click(
                        load_handler,
                        inputs=[dataset_choice, texture_choice, split, sample_count, image_scope, shot_mode, lang_state],
                        outputs=[state, source_image, position, dataset_status, result_image, result_summary, inference_status],
                    )
                    previous.click(
                        lambda current_state, lang: move_handler(current_state, -1, lang),
                        inputs=[state, lang_state],
                        outputs=[state, source_image, position, result_image, result_summary, inference_status],
                    )
                    next_image.click(
                        lambda current_state, lang: move_handler(current_state, 1, lang),
                        inputs=[state, lang_state],
                        outputs=[state, source_image, position, result_image, result_summary, inference_status],
                    )
                    model_choice.change(
                        model_status,
                        inputs=[model_choice, state, lang_state],
                        outputs=model_state,
                    )
                    def model_choices(task_type, domain):
                        return [
                            label for label, spec in MODEL_CATALOG.items()
                            if (spec.get("task") == "anomaly") == (task_type == "anomaly")
                            and spec.get("domain", "general") == domain
                        ]

                    def update_models(task_type, domain):
                        choices = model_choices(task_type, domain)
                        value = choices[0] if choices else None
                        return gr.Dropdown(choices=choices, value=value), model_status(value, lang0) if value else "No registered model matches this selection."

                    task_choice.change(
                        update_models, inputs=[task_choice, domain_choice], outputs=[model_choice, model_state]
                    )
                    domain_choice.change(
                        update_models, inputs=[task_choice, domain_choice], outputs=[model_choice, model_state]
                    )
                    verify_model_button.click(
                        checkpoint_diagnostic,
                        inputs=[model_choice, lang_state],
                        outputs=model_state,
                    )
                    load_model_button.click(
                        load_model_handler,
                        inputs=[model_choice, lang_state],
                        outputs=runtime_state,
                    )
                    unload_model_button.click(
                        unload_model_handler,
                        inputs=lang_state,
                        outputs=runtime_state,
                    )
                    dataset_choice.change(
                        lambda dataset: gr.Dropdown(choices=texture_choices(dataset), value="All textures"),
                        inputs=dataset_choice,
                        outputs=texture_choice,
                    )
                    for selection in (texture_choice, split, image_scope, shot_mode):
                        selection.change(
                            load_handler,
                            inputs=[dataset_choice, texture_choice, split, sample_count, image_scope, shot_mode, lang_state],
                            outputs=[state, source_image, position, dataset_status, result_image, result_summary, inference_status],
                        )
                    detect_button.click(
                        detect_handler,
                        inputs=[state, model_choice, lang_state],
                        outputs=[result_image, result_summary, inference_status, runtime_state],
                    )

                with gr.Tab(tr(lang0, "tab_benchmark"), id="benchmark") as tab_bench:
                    bench_header = gr.Markdown(tr(lang0, "benchmark_header"))
                    with gr.Row():
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            bench_dataset = gr.Dropdown(
                                list(DATASET_CATALOG), value=default_dataset, label=tr(lang0, "benchmark_dataset_label")
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            bench_texture = gr.Dropdown(
                                texture_choices(default_dataset),
                                value=["Pattern 1-4 (Train patterns)"] if default_dataset == "ZJU-Leaper" else ["All textures"],
                                multiselect=True,
                                label=tr(lang0, "benchmark_texture_label"),
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            bench_shot_mode = gr.Radio(
                                choices=shot_mode_choices(lang0), value="Full-shot",
                                label=tr(lang0, "benchmark_shot_label"),
                            )
                    with gr.Row():
                        with gr.Column(scale=7, elem_classes="fdh-control-card"):
                            bench_models = gr.CheckboxGroup(
                                compatible_models(default_dataset), label=tr(lang0, "benchmark_models_label")
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card fdh-action-run"):
                            bench_run_button = gr.Button(
                                tr(lang0, "btn_run_benchmark"), variant="primary", elem_classes="fdh-primary"
                            )
                    with gr.Row():
                        with gr.Column(scale=4, elem_classes="fdh-control-card"):
                            bench_profiling = gr.Checkbox(
                                value=False, label=tr(lang0, "benchmark_profiling_label")
                            )
                        with gr.Column(scale=4, elem_classes="fdh-control-card"):
                            bench_resolution_sweep = gr.Checkbox(
                                value=False, label=tr(lang0, "benchmark_resolution_sweep_label")
                            )
                        with gr.Column(scale=4, elem_classes="fdh-control-card"):
                            bench_cross_domain = gr.Dropdown(
                                choices=[tr(lang0, "benchmark_cross_domain_none"), *DATASET_CATALOG],
                                value=tr(lang0, "benchmark_cross_domain_none"),
                                label=tr(lang0, "benchmark_cross_domain_label"),
                            )
                    with gr.Row():
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            bench_score_preset = gr.Dropdown(
                                choices=score_preset_choices(lang0), value="balanced",
                                label=tr(lang0, "benchmark_score_preset_label"),
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            bench_custom_weight = gr.Slider(
                                0, 1, value=0.5, step=0.05,
                                label=tr(lang0, "benchmark_custom_weight_label"), visible=False,
                            )
                    bench_status = gr.Markdown(tr(lang0, "benchmark_placeholder"), elem_classes="fdh-status")

                    bench_rows_state = gr.State([])
                    # Tables only — no bar chart, no radar. The taxonomy
                    # (`metrics_taxonomy`) decides which tables exist and
                    # which metric goes in which one; this loop only lays
                    # them out, so a metric can never appear here under a
                    # heading the report disagrees with.
                    bench_tables: dict[str, Any] = {}
                    bench_notes: dict[str, Any] = {}
                    for section in empty_sections():
                        gr.Markdown(f"### {section['title']}")
                        for spec in section["tables"]:
                            with gr.Column(elem_classes="fdh-card"):
                                gr.Markdown(f"**{spec['title']}**")
                                if spec["name"] in UNIMPLEMENTED:
                                    gr.Markdown(
                                        f"_{UNIMPLEMENTED[spec['name']]}_",
                                        elem_classes="fdh-status",
                                    )
                                    continue
                                # An empty table with no explanation reads as
                                # a broken page: "YOLO has no pixel metrics"
                                # and "you did not tick profiling" both look
                                # like nothing happened.
                                bench_notes[spec["name"]] = gr.Markdown(
                                    f"_{spec['note']}_" if spec.get("note") else "",
                                    elem_classes="fdh-status",
                                    visible=bool(spec.get("note")),
                                )
                                # Always visible. Toggling `visible` in the
                                # same `gr.update` that changes `headers` and
                                # `value` desynced the streamed re-render —
                                # the Memory table stayed one row behind and
                                # dropped a model. Emptiness is conveyed by
                                # the note above instead, which is the only
                                # component whose visibility changes.
                                bench_tables[spec["name"]] = gr.Dataframe(
                                    headers=spec["headers"], value=[],
                                    interactive=False, wrap=True,
                                )
                    bench_results = gr.Dataframe(
                        label=tr(lang0, "leaderboard_label"), interactive=False, wrap=True, visible=False,
                    )

                    def bench_dataset_change_handler(dataset_label, lang):
                        choices = texture_choices(dataset_label)
                        default_val = ["Pattern 1-4 (Train patterns)"] if dataset_label == "ZJU-Leaper" else ["All textures"]
                        return (
                            gr.Dropdown(choices=choices, value=default_val, multiselect=True),
                            gr.CheckboxGroup(choices=compatible_models(dataset_label), value=[]),
                        )

                    def bench_run_handler(
                        dataset_label, texture_label, shot_mode_value, model_labels,
                        include_profiling, include_resolution_sweep, cross_domain_choice,
                        score_preset, custom_weight, lang,
                    ):
                        # `run_benchmark` always yields at least once (it
                        # yields an error row when no model is selected), but
                        # seeding this keeps the final build below total.
                        last: tuple = ([], "", [])
                        cross_domain_label = (
                            None if cross_domain_choice in (None, tr(lang, "benchmark_cross_domain_none"))
                            else cross_domain_choice
                        )
                        for columns, rows, status, scored in run_benchmark(
                            dataset_label, texture_label, shot_mode_value, model_labels, lang,
                            include_profiling=include_profiling,
                            include_resolution_sweep=include_resolution_sweep,
                            cross_domain_dataset_label=cross_domain_label,
                            score_preset=score_preset,
                            custom_technical_weight=custom_weight,
                        ):
                            # Progress streams; the tables do not. Re-sending
                            # a Dataframe's headers *and* value on every
                            # model dropped rows in Gradio -- the Memory
                            # table stayed a model behind while Compute,
                            # updated identically, did not. Rather than
                            # depend on that behaviour, the tables are built
                            # once from the completed run below; the status
                            # line is what tells the user it is still going.
                            last = (rows, status, scored)
                            yield (
                                gr.update(value=rows or []), status, scored,
                                *[gr.skip() for _ in bench_tables],
                                *[gr.skip() for _ in bench_notes],
                            )

                        rows, status, scored = last
                        # The leaderboard rows are regrouped through the
                        # taxonomy rather than rendered as one wide table:
                        # `BenchmarkResult` is the same object a script gets
                        # from the application measurement service, so the page and a JSON
                        # report are two views of one result, not two
                        # implementations.
                        grouped = _tables_from_leaderboard(
                            scored or rows, columns, profiling_requested=include_profiling
                        )
                        yield (
                            gr.update(value=rows or []),
                            status,
                            scored,
                            *[
                                gr.update(
                                    headers=grouped[name]["headers"],
                                    value=grouped[name]["rows"],
                                )
                                for name in bench_tables
                            ],
                            *[
                                gr.update(
                                    value=f"_{grouped[name]['note']}_" if grouped[name]["note"] else "",
                                    visible=bool(grouped[name]["note"]),
                                )
                                for name in bench_notes
                            ],
                        )

                    bench_dataset.change(
                        bench_dataset_change_handler,
                        inputs=[bench_dataset, lang_state],
                        outputs=[bench_texture, bench_models],
                    )
                    bench_score_preset.change(
                        lambda preset: gr.Slider(visible=preset == "custom"),
                        inputs=bench_score_preset,
                        outputs=bench_custom_weight,
                    )
                    bench_run_button.click(
                        bench_run_handler,
                        inputs=[
                            bench_dataset, bench_texture, bench_shot_mode, bench_models,
                            bench_profiling, bench_resolution_sweep, bench_cross_domain,
                            bench_score_preset, bench_custom_weight, lang_state,
                        ],
                        outputs=[
                            bench_results, bench_status, bench_rows_state,
                            *bench_tables.values(), *bench_notes.values(),
                        ],
                    )

                with gr.Tab(tr(lang0, "tab_run_history"), id="run-history") as tab_history:
                    history_header = gr.Markdown(tr(lang0, "history_header"))
                    with gr.Row():
                        with gr.Column(scale=5, elem_classes="fdh-control-card"):
                            history_path = gr.Textbox(
                                value=DEFAULT_RUN_LOG_PATH, label=tr(lang0, "history_path_label")
                            )
                        with gr.Column(scale=3, elem_classes="fdh-control-card"):
                            history_metric = gr.Dropdown(choices=[], label=tr(lang0, "history_metric_label"))
                        with gr.Column(scale=2, elem_classes="fdh-control-card fdh-action-run"):
                            history_refresh_button = gr.Button(
                                tr(lang0, "btn_history_refresh"), variant="secondary"
                            )
                    history_status = gr.Markdown(tr(lang0, "history_no_runs"), elem_classes="fdh-status")
                    history_table = gr.Dataframe(
                        label=tr(lang0, "history_table_label"), interactive=False, wrap=True
                    )

                    def history_refresh_handler(path, metric, lang):
                        try:
                            rows = read_run_log(path)
                        except (OSError, ValueError) as exc:
                            return (
                                gr.Dataframe(value=[]), gr.Dropdown(choices=[]),
                                tr(lang, "history_load_error", error=exc),
                            )
                        if not rows:
                            return gr.Dataframe(value=[]), gr.Dropdown(choices=[]), tr(lang, "history_no_runs")

                        columns, table = flatten_run_log_rows(rows)
                        excluded = {"timestamp_utc", "model", "backend", "task", "dataset", "device"}
                        metric_choices = [column for column in columns if column not in excluded]
                        selected_metric = metric if metric in metric_choices else (
                            metric_choices[0] if metric_choices else None
                        )
                        return (
                            gr.Dataframe(headers=columns, value=table),
                            gr.Dropdown(choices=metric_choices, value=selected_metric),
                            tr(lang, "history_table_label"),
                        )

                    history_refresh_button.click(
                        history_refresh_handler,
                        inputs=[history_path, history_metric, lang_state],
                        outputs=[history_table, history_metric, history_status],
                    )

        # -- Language toggle: rebuilds every static label/header/button/
        # placeholder in the new language, and recomputes the handful of
        # dynamic panels we have live data for (model/dataset/runtime
        # status, the currently-shown image's caption) instead of just
        # resetting them. Transient result panels (the last inference's
        # tags, the benchmark leaderboard's status line) reset to their
        # placeholder text -- there's no stored "was a result already
        # shown" state to re-render them from, and resetting on a language
        # switch is an acceptable, expected trade-off.
        def apply_language(lang, model_label, dataset_label, gallery_state):
            _, caption = current_image(gallery_state, lang)
            return (
                lang,
                _nav_html(lang),
                gr.Tab(label=tr(lang, "tab_single_image")),
                gr.Tab(label=tr(lang, "tab_benchmark")),
                gr.Tab(label=tr(lang, "tab_run_history")),
                tr(lang, "model_session_header"),
                gr.Dropdown(label=tr(lang, "model_dropdown_label")),
                model_status(model_label, lang),
                tr(lang, "btn_load_model"),
                tr(lang, "btn_unload_model"),
                tr(lang, "btn_inspect_checkpoint"),
                tr(lang, "btn_run_detection"),
                gr.Image(label=tr(lang, "image_selected_label")),
                caption,
                tr(lang, "btn_previous"),
                tr(lang, "btn_next"),
                gr.Image(label=tr(lang, "image_result_label")),
                tr(lang, "runtime_memory_header"),
                format_session_status(sessions.status(), lang),
                render_prediction_tags({}, lang),
                tr(lang, "inference_hint_ready") if gallery_state.get("samples") else tr(lang, "inference_hint_start"),
                tr(lang, "dataset_sampler_header"),
                gr.Dropdown(label=tr(lang, "dataset_dropdown_label")),
                gr.Dropdown(label=tr(lang, "texture_dropdown_label")),
                gr.Radio(choices=split_choices(lang), label=tr(lang, "split_label")),
                gr.Slider(label=tr(lang, "slider_random_images_label")),
                gr.Dropdown(choices=image_scope_choices(lang), label=tr(lang, "image_selection_label")),
                gr.Radio(choices=shot_mode_choices(lang), label=tr(lang, "sample_regime_label")),
                tr(lang, "btn_load_random_images"),
                dataset_availability_status(dataset_label, lang),
                tr(lang, "benchmark_header"),
                gr.Dropdown(label=tr(lang, "benchmark_dataset_label")),
                gr.Dropdown(label=tr(lang, "benchmark_texture_label")),
                gr.Radio(choices=shot_mode_choices(lang), label=tr(lang, "benchmark_shot_label")),
                gr.CheckboxGroup(label=tr(lang, "benchmark_models_label")),
                tr(lang, "btn_run_benchmark"),
                gr.Checkbox(label=tr(lang, "benchmark_profiling_label")),
                gr.Dropdown(choices=score_preset_choices(lang), label=tr(lang, "benchmark_score_preset_label")),
                gr.Slider(label=tr(lang, "benchmark_custom_weight_label")),
                tr(lang, "benchmark_placeholder"),
                gr.Dataframe(label=tr(lang, "leaderboard_label")),
                tr(lang, "history_header"),
                gr.Textbox(label=tr(lang, "history_path_label")),
                gr.Dropdown(label=tr(lang, "history_metric_label")),
                tr(lang, "btn_history_refresh"),
                tr(lang, "history_no_runs"),
                gr.Dataframe(label=tr(lang, "history_table_label")),
            )

        lang_choice.change(
            apply_language,
            inputs=[lang_choice, model_choice, dataset_choice, state],
            outputs=[
                lang_state, nav_html, tab_single, tab_bench, tab_history,
                model_header, model_choice, model_state,
                load_model_button, unload_model_button, verify_model_button, detect_button,
                source_image, position, previous, next_image, result_image,
                runtime_header, runtime_state, result_summary, inference_status,
                dataset_header, dataset_choice, texture_choice, split, sample_count, image_scope, shot_mode,
                load_button, dataset_status,
                bench_header, bench_dataset, bench_texture, bench_shot_mode, bench_models,
                bench_run_button, bench_profiling, bench_score_preset, bench_custom_weight,
                bench_status, bench_results,
                history_header, history_path, history_metric, history_refresh_button,
                history_status, history_table,
            ],
        )
    return app


def _import_gradio_for_local_ui():
    """Import Gradio without letting an unrelated SOCKS proxy break local UI.

    Gradio constructs an httpx client during import. If a shell exports a
    ``socks5://`` proxy but the optional ``socksio`` package is absent, that
    import fails before the local server can start. Localhost serving does not
    need the proxy, so remove proxy variables for this process only.
    """
    import importlib.util

    proxy_keys = ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    has_socks_proxy = any("socks" in os.environ.get(key, "").lower() for key in proxy_keys)
    if has_socks_proxy and importlib.util.find_spec("socksio") is None:
        for key in proxy_keys:
            os.environ.pop(key, None)
    import gradio as gr
    return gr


def _port_holder(port: int) -> str:
    """`"PID 1234 (python adh-ui)"` for whatever is listening on `port`,
    or `""` if that cannot be determined. Diagnostic only — never raises."""

    import subprocess

    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return ""
    pid = command = ""
    for line in out.splitlines():
        if line.startswith("p"):
            pid = line[1:]
        elif line.startswith("c"):
            command = line[1:]
    return f"PID {pid} ({command})" if pid else ""


def launch(**kwargs):
    import os

    kwargs.setdefault("css", CSS)
    # The cloud host this project is deployed to only has port 6008 open;
    # standardize on it everywhere so `adh-ui` works unmodified there.
    # `GRADIO_SERVER_PORT` still wins for local use — hard-coding the port
    # unconditionally also disabled the documented env-var escape hatch.
    kwargs.setdefault("server_name", "0.0.0.0")
    kwargs.setdefault("server_port", int(os.environ.get("GRADIO_SERVER_PORT", "6008")))
    # Gradio 6 only caches files below the workspace or system temp directory
    # unless an external location is explicitly trusted. Every catalog
    # dataset typically lives on external storage reached through a
    # `data/<dir>` symlink, so register each one's *resolved* root (the real
    # on-disk location, not the symlink) for image display.
    configured_paths = list(kwargs.pop("allowed_paths", []) or [])
    for dataset_label in DATASET_CATALOG:
        dataset_root = default_dataset_root(dataset_label)
        if dataset_root and dataset_root not in configured_paths:
            configured_paths.append(dataset_root)
    kwargs["allowed_paths"] = configured_paths
    try:
        gr = _import_gradio_for_local_ui()

        kwargs.setdefault("theme", gr.themes.Base(primary_hue="orange", neutral_hue="slate"))
    except ImportError:
        pass
    try:
        return create_app().launch(**kwargs)
    except OSError as exc:
        # Without this, a leftover `adh-ui` (a backgrounded run, or one
        # orphaned by a closed terminal) holding the port makes every new
        # launch die instantly with a bare "Cannot find empty port" — which
        # reads as "the app is broken", not "the app is already running".
        port = kwargs["server_port"]
        holder = _port_holder(port)
        raise SystemExit(
            f"adh-ui: port {port} is already in use"
            + (f" by {holder}" if holder else "")
            + " — an earlier adh-ui is probably still running.\n"
            f"  Reuse it:   open http://127.0.0.1:{port} in the browser\n"
            + (f"  Or stop it: kill {holder.split()[1]}\n" if holder else "")
            + f"  Or move:    GRADIO_SERVER_PORT=7860 adh-ui"
        ) from exc
