"""UI text tables for the Gradio workspace's language toggle.

Every user-facing string that isn't a stable data value is looked up here
by key via `tr(lang, key, **kwargs)`. Dataset/model names (e.g. "ZJU-Leaper",
"YOLOv8n · ZJU-Leaper") are deliberately left untranslated in both
languages — that's standard practice for ML tooling and, more importantly,
they're also used as literal dict keys throughout `single_image.py`/
`benchmark.py` (`MODEL_CATALOG[label]`, `DATASET_CATALOG[label]`), so
translating the display text would require a second lookup layer with no
real benefit.

Gradio `Dropdown`/`Radio`/`CheckboxGroup` choices that double as internal
lookup keys (`SHOT_FULL`, `ALL_IMAGES`, ...) are localized via Gradio's
`(display_label, value)` tuple form instead of translating the constant
itself — see `single_image.py`'s `*_choices()` helpers — so the underlying
value never changes and every `if shot_mode == SHOT_FULL` comparison
elsewhere keeps working regardless of the selected UI language.

--------------------------------------------------------------------------
Glossary: one concept, one name
--------------------------------------------------------------------------
README.md is the vocabulary of this project, and the front end speaks it.
Section 4.1 names the controls a reader is told to click, section 1 defines
what a *benchmark* is, and section 3.3 names the metric tables. The strings
below must not invent a second name for anything README already names:

* ``benchmark``  — scoring one or more models, on one dataset, under one
  configuration (README section 1). The tab, the run button ("Run
  benchmark") and the progress lines ("Scored 2/5") all use this one verb,
  so "test"/"evaluate" never appear for the same action here.
* ``technical`` / ``overhead`` — the two metric parts (README section 3.3),
  used by the section headings, the score presets and the weight slider.
* ``heat map`` (two words) — the pixel-level output a model renders;
  README writes it that way in all 13 occurrences.
* ``anomaly score`` / ``image-level score`` — the image-level output.
* ``Task type`` with values ``Anomaly detection`` / ``Defect detection``
  (README section 4.1.2) — the model-status line reports the same pair.
* ``Application domain`` (README section 4.1.2) — never just "domain".
* ``training corpus`` / ``training split`` (README section 4.1.2) — the two
  facts the model panel shows about a weight.
* ``detection`` — the single-image tab's action ("Single Image Detection",
  "Run detection", "Detection result"), so the tab never mixes in
  "inference" as a label; README uses "inference" only for the *process*.

When a string below names a control that README also names, quote README
verbatim rather than paraphrasing it.
"""

from __future__ import annotations

LANGUAGES: dict[str, str] = {"en": "EN", "zh": "中文"}
DEFAULT_LANGUAGE = "en"

_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # -- Shell ---------------------------------------------------------
        "nav_links": "Models · Datasets · Metrics · Workflows",
        "tab_single_image": "Single Image Detection",
        "tab_benchmark": "Benchmark",
        "tab_run_history": "Run History",

        # -- Model setup (README 4.1.2, Phase 1) ---------------------------
        "model_session_header": "### Model setup",
        "task_type_label": "Task type",
        "choice_task_anomaly": "Anomaly detection",
        "choice_task_defect": "Defect detection",
        "application_domain_label": "Application domain",
        "model_dropdown_label": "Model",
        "model_none_matching": "No staged model matches this selection.",
        "btn_load_model": "Load model",
        "btn_unload_model": "Unload model",
        "btn_inspect_checkpoint": "Inspect checkpoint",
        "btn_run_detection": "Run detection",
        "model_status_unavailable": "🔴 **Unavailable** — install the `{package}` extra before loading this backend.",
        "model_status_missing": "🟠 **Checkpoint missing** — expected `{path}`.",
        "model_status_ready": (
            "🟢 **Ready** — {task}\n\n"
            "**Method:** {method}  \n**Application domain:** {domain}  \n"
            "**Training corpus:** {training_corpus}  \n**Training split:** {training_split}  \n"
            "**Weight:** `{filename}`"
        ),
        "checkpoint_diag_missing": "🟠 **Checkpoint missing** — `{path}` was not found.",
        "checkpoint_diag_trusted_header": "🟢 **Trusted checkpoint diagnostic**",
        "checkpoint_diag_sha": "SHA-256: `{sha}`",
        "checkpoint_diag_size": "Size: `{size:.1f} MiB`",
        "checkpoint_diag_globals": "Declared checkpoint globals: `{globals}`",
        "model_load_failed": "🔴 **Model load failed** — {error_type}: {error}",
        "runtime_memory_header": "### Runtime memory",
        "runtime_active_model": "**Active model:** `{model}`",
        "runtime_device": "**Runtime device:** `{device}`",
        "runtime_params": "**Model parameters and buffers:** `{value}`",
        "runtime_rss": "**Process RSS:** `{value}`",
        "runtime_cuda": "**CUDA allocated / reserved:** `{alloc} / {reserved}`",
        "runtime_mps": "**MPS allocated:** `{value}`",
        "runtime_load_time": "**Load time:** `{ms:.1f} ms`",
        "value_none": "none",
        "value_unavailable": "unavailable",

        # -- Single-image result (README 4.1.2, Phase 3) -------------------
        "image_selected_label": "Selected dataset image",
        "image_result_label": "Detection result",
        "caption_no_image": "No image loaded yet.",
        "btn_previous": "← Previous",
        "btn_next": "Next →",
        "prediction_none": "No detection result yet.",
        "prediction_no_defect": "No defect detected",
        "prediction_regions": "Detected {count} defect region(s)",
        # A segmentation model reports a pixel map, not boxes: this is the
        # panel's verdict for it, with `tag_defect_area` carrying the only
        # number such a model can give.
        "prediction_defect_found": "Detected defect region(s)",
        "tag_defect_area": "defect area {pct}%",
        "tag_confidence": "confidence",
        "tag_anomaly_score": "anomaly score",
        "tag_heatmap_available": "Heat map available",
        "tag_heatmap_unavailable": "Heat map not available",
        # Distinct from "not available": this model produces an image-level
        # score only and has no pixel-level output to render at all.
        "tag_heatmap_unsupported": "Image-level score only",
        "tag_normal": "Normal",
        "tag_anomalous": "Anomalous",
        "inference_hint_start": "Select an image and a ready model to begin.",
        "inference_hint_ready": "Image ready. Choose a model and run detection.",
        "inference_hint_changed": "Image changed. Run detection again for this image.",
        "inference_need_dataset": "🟠 Load a dataset and select an image first.",
        "inference_failed": "🔴 **Detection failed** — {error_type}: {error}",
        "inference_complete": "🟢 **Detection complete**",

        # -- Dataset & sample selection (README 4.1.2, Phase 2) ------------
        "dataset_sampler_header": "### Dataset & sample selection",
        "dataset_dropdown_label": "Dataset",
        "texture_dropdown_label": "Texture / pattern",
        "split_label": "Split",
        "split_test": "test",
        "split_train": "train",
        "slider_random_images_label": "Random images",
        "image_selection_label": "Image selection",
        "choice_all_images": "All images",
        "choice_defect_only": "Defect only",
        "choice_normal_only": "Normal only",
        "sample_regime_label": "Sample regime",
        "choice_full_shot": "Full-shot",
        "choice_few_shot": "Few-shot",
        "btn_load_random_images": "Load random images",
        "dataset_ready": "🟢 **Ready** — using the registered `{label}` dataset.",
        "dataset_unavailable": "🟠 **Dataset unavailable** — connect the storage containing `{label}` (declared root `{root}`, or set `${env}`), then restart the app.",
        "dataset_load_error": "🔴 **Dataset unavailable** — {error}",
        "dataset_load_success": "🟢 Loaded **{count}** images — {scope}, {shot} — from `{name}` / `{texture}` / `{split}`.",
        "move_need_dataset": "Load a dataset before browsing images.",
        "state_defect": "defect",
        "state_normal": "normal",

        # -- Benchmark (README 4.1.3) --------------------------------------
        "benchmark_header": (
            "### Benchmark workspace\n"
            "A benchmark scores one or more models, on one dataset, under one "
            "configuration, and reports the metrics of section 3.3 — image "
            "AUROC/F1 for anomaly models, mAP/precision/recall for detection "
            "models, mIoU/Dice/pixel-F1 for segmentation models. Every selected "
            "model is mounted, scored, and unmounted before the next one loads, "
            "and all of them see the exact same test-split sample."
        ),
        "benchmark_dataset_label": "Dataset",
        "benchmark_texture_label": "Texture / pattern",
        "benchmark_shot_label": "Sample regime (test split)",
        "benchmark_models_label": "Models to benchmark",
        "btn_run_benchmark": "Run benchmark",
        "benchmark_placeholder": "Select a dataset, a sample regime, and one or more models to begin.",
        "leaderboard_label": "Leaderboard",
        "bench_select_model": "🟠 Select at least one model.",
        "bench_dataset_unavailable": "🔴 **Dataset unavailable** — connect `{label}` first.",
        "bench_starting": "🔵 Starting benchmark — 0/{total} models scored.",
        "bench_task_mismatch": "{model}: {dataset} has no ground truth for its task ({task}).",
        "bench_progress": "🔵 Scored {index}/{total} — last: {model}.",
        "bench_done": "🟢 Scored {count} model(s) on {samples} samples ({shot}).",
        "bench_no_results": "🔴 No results produced.",
        "bench_profiling_no_metrics": (
            "Profiling was requested but produced no metrics for these models — "
            "see the status message above for the reason it was skipped."
        ),
        "benchmark_profiling_label": "Include profiling (FPS / latency / memory)",
        "benchmark_resolution_sweep_label": "Include resolution sweep (throughput decay slope)",
        "benchmark_cross_domain_label": "Cross-domain degradation target dataset (optional)",
        "benchmark_cross_domain_none": "None",
        "benchmark_score_preset_label": "Score preset",
        "benchmark_custom_weight_label": "Technical vs. overhead weight (higher = favor technical metrics)",
        "choice_score_accuracy_first": "Technical-first",
        "choice_score_balanced": "Balanced",
        "choice_score_efficiency_first": "Overhead-first",
        "choice_score_custom": "Custom",

        # -- Run history (README 4.1.4) ------------------------------------
        "history_header": (
            "### Benchmark run history\n"
            "Every completed benchmark run (from this UI or `adh benchmark`) is "
            "appended as one line to a shared JSONL log — this reads it back."
        ),
        "history_path_label": "Run log path",
        "btn_history_refresh": "Refresh",
        "history_metric_label": "Metric to Chart",
        "history_no_runs": "No runs found at this path yet.",
        "history_load_error": "🔴 Could not read run log — {error}",
        "history_table_label": "Benchmark runs",
        "history_loaded": "🟢 Loaded **{count}** benchmark run(s).",
    },
    "zh": {
        # -- Shell ---------------------------------------------------------
        "nav_links": "模型 · 数据集 · 指标 · 工作流",
        "tab_single_image": "单图检测",
        "tab_benchmark": "基准测试",
        "tab_run_history": "运行历史",

        # -- Model setup (README 4.1.2, Phase 1) ---------------------------
        "model_session_header": "### 模型设置",
        "task_type_label": "任务类型",
        "choice_task_anomaly": "异常检测",
        "choice_task_defect": "缺陷检测",
        "application_domain_label": "应用领域",
        "model_dropdown_label": "模型",
        "model_none_matching": "没有符合当前筛选条件且已就绪的模型。",
        "btn_load_model": "加载模型",
        "btn_unload_model": "卸载模型",
        "btn_inspect_checkpoint": "检查权重文件",
        "btn_run_detection": "运行检测",
        "model_status_unavailable": "🔴 **不可用** — 使用前请先安装 `{package}` 扩展。",
        "model_status_missing": "🟠 **权重文件缺失** — 期望路径 `{path}`。",
        "model_status_ready": (
            "🟢 **就绪** — {task}\n\n"
            "**方法：** {method}  \n**应用领域：** {domain}  \n"
            "**训练语料：** {training_corpus}  \n**训练划分：** {training_split}  \n**权重：** `{filename}`"
        ),
        "checkpoint_diag_missing": "🟠 **权重文件缺失** — 未找到 `{path}`。",
        "checkpoint_diag_trusted_header": "🟢 **可信权重诊断信息**",
        "checkpoint_diag_sha": "SHA-256：`{sha}`",
        "checkpoint_diag_size": "大小：`{size:.1f} MiB`",
        "checkpoint_diag_globals": "声明的权重全局对象：`{globals}`",
        "model_load_failed": "🔴 **模型加载失败** — {error_type}：{error}",
        "runtime_memory_header": "### 运行时内存",
        "runtime_active_model": "**当前模型：** `{model}`",
        "runtime_device": "**运行设备：** `{device}`",
        "runtime_params": "**模型参数与缓冲区：** `{value}`",
        "runtime_rss": "**进程常驻内存：** `{value}`",
        "runtime_cuda": "**CUDA 已分配 / 已保留：** `{alloc} / {reserved}`",
        "runtime_mps": "**MPS 已分配：** `{value}`",
        "runtime_load_time": "**加载耗时：** `{ms:.1f} 毫秒`",
        "value_none": "无",
        "value_unavailable": "不可用",

        # -- Single-image result (README 4.1.2, Phase 3) -------------------
        "image_selected_label": "已选数据集图像",
        "image_result_label": "检测结果",
        "caption_no_image": "尚未加载图像。",
        "btn_previous": "← 上一张",
        "btn_next": "下一张 →",
        "prediction_none": "暂无检测结果。",
        "prediction_no_defect": "未检测到缺陷",
        "prediction_regions": "检测到 {count} 处缺陷",
        "prediction_defect_found": "检测到缺陷区域",
        "tag_defect_area": "缺陷面积 {pct}%",
        "tag_confidence": "置信度",
        "tag_anomaly_score": "异常分数",
        "tag_heatmap_available": "热力图可用",
        "tag_heatmap_unavailable": "热力图不可用",
        "tag_heatmap_unsupported": "仅图像级分数",
        "tag_normal": "正常",
        "tag_anomalous": "异常",
        "inference_hint_start": "请先选择图像并加载可用模型。",
        "inference_hint_ready": "图像已就绪，请选择模型并运行检测。",
        "inference_hint_changed": "图像已切换，请重新运行检测。",
        "inference_need_dataset": "🟠 请先加载数据集并选择图像。",
        "inference_failed": "🔴 **检测失败** — {error_type}：{error}",
        "inference_complete": "🟢 **检测完成**",

        # -- Dataset & sample selection (README 4.1.2, Phase 2) ------------
        "dataset_sampler_header": "### 数据集与样本选择",
        "dataset_dropdown_label": "数据集",
        "texture_dropdown_label": "纹理 / 图案",
        "split_label": "数据划分",
        "split_test": "测试集",
        "split_train": "训练集",
        "slider_random_images_label": "随机图像数量",
        "image_selection_label": "图像范围",
        "choice_all_images": "全部图像",
        "choice_defect_only": "仅缺陷",
        "choice_normal_only": "仅正常",
        "sample_regime_label": "采样档位",
        "choice_full_shot": "全量",
        "choice_few_shot": "少样本",
        "btn_load_random_images": "加载随机图像",
        "dataset_ready": "🟢 **就绪** — 正在使用已注册的 `{label}` 数据集。",
        "dataset_unavailable": "🟠 **数据集不可用** — 请连接包含 `{label}` 的存储（声明的根目录为 `{root}`，或设置环境变量 `${env}`），然后重启应用。",
        "dataset_load_error": "🔴 **数据集不可用** — {error}",
        "dataset_load_success": "🟢 已加载 **{count}** 张图像 — {scope}，{shot} — 来自 `{name}` / `{texture}` / `{split}`。",
        "move_need_dataset": "请先加载数据集，再浏览图像。",
        "state_defect": "缺陷",
        "state_normal": "正常",

        # -- Benchmark (README 4.1.3) --------------------------------------
        "benchmark_header": (
            "### 基准测试工作台\n"
            "一次基准测试在同一个数据集、同一套配置下为一个或多个模型评分，"
            "并输出 3.3 节的各项指标——异常检测模型给出图像级 AUROC/F1，"
            "检测模型给出 mAP/精确率/召回率，分割模型给出 mIoU/Dice/像素级 F1。"
            "每个被选中的模型都会依次挂载、评分、卸载后再加载下一个，"
            "且全部模型使用完全相同的测试集样本。"
        ),
        "benchmark_dataset_label": "数据集",
        "benchmark_texture_label": "纹理 / 图案",
        "benchmark_shot_label": "采样档位（测试集）",
        "benchmark_models_label": "待评分模型",
        "btn_run_benchmark": "运行基准测试",
        "benchmark_placeholder": "请选择数据集、采样档位以及至少一个模型。",
        "leaderboard_label": "排行榜",
        "bench_select_model": "🟠 请至少选择一个模型。",
        "bench_dataset_unavailable": "🔴 **数据集不可用** — 请先连接 `{label}`。",
        "bench_starting": "🔵 开始基准测试 — 已评分 0/{total} 个模型。",
        "bench_task_mismatch": "{model}：{dataset} 没有该任务（{task}）所需的真实标注。",
        "bench_progress": "🔵 已评分 {index}/{total} — 最近完成：{model}。",
        "bench_done": "🟢 已在 {samples} 个样本（{shot}）上评分 {count} 个模型。",
        "bench_no_results": "🔴 未产生任何结果。",
        "bench_profiling_no_metrics": "已请求性能剖析，但这些模型没有产生任何指标 — 跳过原因见上方状态信息。",
        "benchmark_profiling_label": "启用性能剖析（FPS / 延迟 / 显存）",
        "benchmark_resolution_sweep_label": "启用分辨率扫描（吞吐量衰减斜率）",
        "benchmark_cross_domain_label": "跨域退化率目标数据集（可选）",
        "benchmark_cross_domain_none": "不启用",
        "benchmark_score_preset_label": "评分预设",
        "benchmark_custom_weight_label": "技术 vs 开销权重（越高越偏重技术指标）",
        "choice_score_accuracy_first": "技术优先",
        "choice_score_balanced": "均衡",
        "choice_score_efficiency_first": "开销优先",
        "choice_score_custom": "自定义",

        # -- Run history (README 4.1.4) ------------------------------------
        "history_header": (
            "### 基准测试运行历史\n"
            "每一次完成的基准测试（无论来自本界面还是 `adh benchmark`）都会作为一行"
            "追加到共享的 JSONL 日志中——这里将其读取回来展示。"
        ),
        "history_path_label": "运行日志路径",
        "btn_history_refresh": "刷新",
        "history_metric_label": "图表指标",
        "history_no_runs": "该路径下暂无运行记录。",
        "history_load_error": "🔴 无法读取运行日志 — {error}",
        "history_table_label": "基准测试运行记录",
        "history_loaded": "🟢 已加载 **{count}** 条基准测试运行记录。",
    },
}


def tr(lang: str, key: str, **kwargs) -> str:
    """Look up `key` in `lang`'s string table (falling back to English for
    an unknown language or a key missing from a non-English table), then
    `.format(**kwargs)` it. Returns the bare key if it's missing everywhere,
    so a translation gap fails loud (a visible `some_key` in the UI) rather
    than silently swallowing an exception."""

    table = _STRINGS.get(lang, _STRINGS[DEFAULT_LANGUAGE])
    template = table.get(key, _STRINGS[DEFAULT_LANGUAGE].get(key, key))
    return template.format(**kwargs) if kwargs else template
