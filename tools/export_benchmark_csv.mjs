import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const root = new URL("..", import.meta.url).pathname;
const inputPath = `${root}artifacts/runtime/full_sweep.jsonl`;
const outputDir = `${root}artifacts/runtime`;
const longPath = `${outputDir}/benchmark_results_long.csv`;
const summaryPath = `${outputDir}/benchmark_model_summary.csv`;

const lines = (await fs.readFile(inputPath, "utf8"))
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));
const header = lines[0];
const rows = lines.slice(1);

const baseColumns = [
  "model", "backend", "variant", "group", "status", "reason", "duration_s",
  "recorded_at", "weights",
];
const metricColumns = [
  "sample_count", "image_auroc", "image_precision", "image_recall", "image_f1",
  "image_threshold", "pixel_auroc", "pixel_aupro", "pixel_f1", "iap",
  "dice", "miou", "num_evaluated", "map", "map_50", "map_75",
  "precision_at_threshold", "recall_at_threshold", "recall_normal", "recall_small",
  "true_positives", "false_positives", "false_negatives", "classes", "mar_1",
  "mar_10", "mar_100", "flops_g", "params_m", "fps",
  "instantaneous_fps_mean", "instantaneous_fps_cv", "instantaneous_fps_std",
  "latency_ms_mean", "latency_ms_std", "latency_ms_cv", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99",
  "avg_memory_mb", "peak_memory_mb", "model_size_mb", "lmei",
  "resolution_slope_alpha", "resolution_slope_beta", "resolution_sweep_points",
  "concurrency_probe_points", "frame_budget_ms", "max_concurrent_streams",
  "single_stream_latency_ms", "model_transfer_mb", "model_transfer_bytes",
  "export_transfer_mb", "export_transfer_bytes", "metric", "mode", "acc_src",
  "mean_degradation_pct", "top_k_mean_degradation_pct", "ci_low", "ci_high",
  "k_effective", "scored_pattern_count", "selected_patterns", "skipped_patterns",
  "per_pattern_degradation_pct", "traceback_tail",
];

function csvValue(value) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(columns, data) {
  return [columns.join(","), ...data.map((row) => columns.map((c) => csvValue(row[c])).join(","))].join("\n") + "\n";
}

const longRows = rows.map((row) => ({
  ...row,
  ...Object.fromEntries(Object.entries(row.metrics ?? {}).map(([key, value]) => [key, value])),
}));
await fs.writeFile(longPath, toCsv([...baseColumns, ...metricColumns], longRows), "utf8");

const byModel = new Map();
for (const row of rows) {
  const key = row.model;
  if (!byModel.has(key)) {
    byModel.set(key, {
      model: row.model, backend: row.backend, variant: row.variant,
      weights: row.weights, groups_expected: header.groups.length,
      groups_recorded: 0, ok_groups: 0, skipped_groups: 0, failed_groups: 0,
    });
  }
  const summary = byModel.get(key);
  summary.groups_recorded += 1;
  summary[`${row.group}_status`] = row.status;
  summary[`${row.group}_duration_s`] = row.duration_s;
  if (row.status === "ok") summary.ok_groups += 1;
  if (row.status === "skipped") summary.skipped_groups += 1;
  if (row.status === "failed") summary.failed_groups += 1;
  for (const [metric, value] of Object.entries(row.metrics ?? {})) {
    summary[`${row.group}__${metric}`] = value;
  }
  if (row.reason) summary[`${row.group}_reason`] = row.reason;
}
const summaryColumns = [
  "model", "backend", "variant", "weights", "groups_expected", "groups_recorded",
  "ok_groups", "skipped_groups", "failed_groups",
  ...header.groups.flatMap((group) => [`${group}_status`, `${group}_duration_s`, `${group}_reason`]),
  ...[
    "accuracy__sample_count", "accuracy__image_auroc", "accuracy__image_precision",
    "accuracy__image_recall", "accuracy__image_f1", "accuracy__pixel_auroc",
    "accuracy__pixel_aupro", "accuracy__pixel_f1", "accuracy__iap", "accuracy__dice",
    "accuracy__miou", "accuracy__map", "accuracy__map_50", "accuracy__map_75",
    "cross_domain__metric", "cross_domain__acc_src", "cross_domain__mean_degradation_pct",
    "cross_domain__top_k_mean_degradation_pct", "cross_domain__ci_low", "cross_domain__ci_high",
    "cross_domain__k_effective", "cross_domain__scored_pattern_count",
    "cross_domain__selected_patterns", "cross_domain__skipped_patterns",
    "cross_domain__per_pattern_degradation_pct",
    "runtime__fps", "runtime__latency_ms_mean", "runtime__peak_memory_mb",
    "runtime__avg_memory_mb", "runtime__model_size_mb", "runtime__params_m", "runtime__flops_g",
    "runtime__lmei", "scaling__resolution_slope_alpha", "scaling__resolution_slope_beta",
    "scaling__resolution_sweep_points", "concurrency__frame_budget_ms",
    "concurrency__max_concurrent_streams", "concurrency__single_stream_latency_ms",
    "concurrency__concurrency_probe_points", "communication__model_transfer_mb",
    "communication__model_transfer_bytes", "communication__export_transfer_mb",
    "communication__export_transfer_bytes",
  ],
];
const summaryRows = [...byModel.values()];
await fs.writeFile(summaryPath, toCsv(summaryColumns, summaryRows), "utf8");

// Parse the generated CSV through the bundled spreadsheet engine as a
// structural check, so quoting and column alignment are validated.
const workbook = await Workbook.fromCSV(await fs.readFile(longPath, "utf8"), { sheetName: "Results" });
const used = workbook.worksheets.getItem("Results").getUsedRange(true);
console.log(JSON.stringify({
  source_records: rows.length,
  expected_records: header.model_count * header.groups.length,
  long_csv: longPath,
  summary_csv: summaryPath,
  long_rows: used.rowCount - 1,
  long_columns: used.columnCount,
}));
