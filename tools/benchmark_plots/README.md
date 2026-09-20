# Local Benchmark Plots

These tools use only `runs/benchmark_rows.jsonl`. They do not read the 4090 Excel or Markdown files.

## Finding the data

A benchmark run writes **one dated JSON per run** to `runs/benchmark_snapshots/<UTC timestamp>.json`,
holding every row it scored — technical and overhead metrics, `selection`, `provenance` and
`warnings` together. That is what the figures read. They do not read the 4090 Excel or Markdown
files, and they do not read the append-only `runs/benchmark_rows.jsonl` unless no dated snapshot
exists yet.

`data.load_latest_snapshot()` reads the dated files **newest first and merges them per model,
newest row winning**. That is what makes a partial re-run usable: re-scoring three models writes a
three-row snapshot, and taking only the newest file would drop the other sixteen models from every
figure. Instead the re-run contributes exactly the models it re-scored and the rest come from the
last complete run.

Each run prints what it found, so a caller can say which "latest" it used:

```
snapshot commit: 275578d885a5534ddfe9fe58714659e66a86e2f9
snapshot source: runs/benchmark_snapshots/20260920T095036Z.json + runs/benchmark_snapshots/20260920T101500Z.json
snapshot rows:   19 models, 350 samples, latest row 2026-09-20T10:15:00+00:00
```

Pass `--commit <sha>` to look at one commit's snapshot on purpose.

## Drawing one figure

Neither script has to be run whole. Both take the same three flags, and figure ids run 01–07 (bar
and overhead figures) then 08–10 (PR/ROC curves):

```bash
# what can be drawn, and what is already on disk
python tools/benchmark_plots/plot_local_benchmark.py --list
python tools/benchmark_plots/plot_anomaly_curves.py --list

# one figure
python tools/benchmark_plots/plot_local_benchmark.py --only 02
python tools/benchmark_plots/plot_anomaly_curves.py --only 10

# several
python tools/benchmark_plots/plot_local_benchmark.py --only 01 03

# only what is not on disk yet
python tools/benchmark_plots/plot_local_benchmark.py --missing
```

`--list` answers without loading a snapshot, so a driver can ask a script what it can draw instead
of hardcoding the list. `--only` with an unknown id fails loudly and prints the known ids.

Run from the repository root:

```bash
python tools/benchmark_plots/plot_local_benchmark.py
```

The command selects the newest complete 19-model local snapshot with the benchmark protocol fields, then writes outputs to `artifacts/local_benchmark_plots/`.

To select a snapshot explicitly:

```bash
python tools/benchmark_plots/plot_local_benchmark.py \
  --commit 275578d885a5534ddfe9fe58714659e66a86e2f9
```

Generated figures:

- `01_image_level`: image AUROC / F1 / precision / recall, anomaly scores and detection boxes
- `02_pixel_level`: pixel AUROC / AUPRO / IAP / Pixel F1 / mIoU, anomaly maps and segmentation masks
- `03_instance_level`: detection mAP and F1
- `04_compute_throughput_latency`: FPS, mean and p95 latency, wall time
- `05_parameters_vs_flops`: model complexity scatter plot
- `06_quality_vs_latency`: quality/latency plots, one panel per metric granularity
- `07_memory_by_measurement_kind`: peak memory, by instrument (whole-process RSS vs CUDA allocator)
- `08_instance_size_breakdown`: size-bucketed AP (small/medium/large) and AR@1/10/100
- `09_instance_counts`: TP/FP/FN and precision/recall/F1 at the evaluation threshold
- `10_ranking`: mean rank and blended composite score
- `11_anomaly_image_level_pr_roc`: image-level PR/ROC, anomaly models
- `12_detection_image_level_pr_roc`: image-level PR/ROC, detection models
- `13_anomaly_pixel_level_pr_roc`: pixel-level PR/ROC, anomaly-map models

No figure carries a title: the report writes its own heading. Set `FIGURE_TITLES = True` in
`plot_local_benchmark.py` to bake one in for a standalone look at a figure.

Every figure is written as PNG and PDF. `snapshot_audit.csv` records every candidate metric cell, including missing values and their reason, under a `family` column. `snapshot.json` records the selected commit and model list.

## Bar layout and styles

`01_image_level`, `02_pixel_level` and `03_instance_level` are one grouped-bar panel each: one
group per model, one bar per metric, with the same style used for the same metric in every
figure. Four separate panels made a metric look comparable across a different model set than its
neighbours; one panel keeps a model's whole profile in a single column and turns the metric into a
matter of bar style.

| Bar | Style |
| --- | --- |
| 1 | grey solid fill |
| 2 | red diagonal hatch (`/`) |
| 3 | blue diagonal hatch (`/`) |
| 4 | grey diagonal hatch (`/`) |

The hatch is a single character — the sparsest matplotlib draws; repeating the character (`//`)
is what makes a hatch look dense. `hatch.linewidth` is lowered to 0.9 so the strokes read as a
light texture.

A model that reports nothing for a metric gets no bar there rather than a zero-height bar, so a
missing measurement never reads as a bad score. `02_pixel_level` carries a footnote saying why a
bar can be absent.

Metric subsets are presentation choices on top of the taxonomy grouping:

- image level: AUROC, F1, precision, recall — all 16 models report all four
- pixel level: AUROC, AUPRO, Pixel F1, mIoU — see below
- instance level: mAP, mAP@0.5, mAP@0.75, F1@0.5

Everything measured, including the metrics left off a figure, stays in `snapshot_audit.csv`.

## Pixel level: two metric families

The pixel-level chart mixes two kinds of metric, and the mix is why the bars are not uniform:

- **Threshold-free ranking** — `pixel_auroc`, `pixel_aupro`, `iap`. These sweep a threshold over a
  *continuous* score map, so they need one to exist.
- **Thresholded overlap** — `pixel_f1`, `miou`, `dice`. These need only one binary mask.

The nine anomaly models persist a continuous map (`Prediction.anomaly_map`, written under
`artifacts/runtime/anomaly_maps/benchmark/`). The three mask models (UNet++, DeepLabV3+, Mask
R-CNN) used to have no AUROC/AUPRO/IAP bars, because the torchvision adapter binarised its
sigmoid probability map and dropped the continuous one:

```python
probs = torch.sigmoid(logits)[0]                       # continuous map existed here
binary_mask = (probs > score_threshold).squeeze(0)...  # binarised
Prediction(sample_id=sample.id, masks=[binary_mask])    # only the mask was kept
```

**That gap is fixed in code** (`src/fabric_defect_hub/`, 2026-09-20):

- `models/torchvision/adapter.py` persists the map when `output_dir` is given — the sigmoid
  probability map for semantic segmentation, the per-pixel max over instance masks for Mask
  R-CNN — and sets `anomaly_score` to the map's maximum, mirroring the anomaly backends'
  `.npy` convention.
- The `segmentation` and `instance_segmentation` capabilities now declare `anomaly_map` and
  `anomaly_score`, which is the switch `loader.run_experiment` reads before forwarding
  `output_dir`.
- `evaluation/segmentation.py` reports `pixel_auroc` / `pixel_aupro` / `iap` from that map, via the
  same `_pixel_level_metrics` routine `AnomalyEvaluator` uses, taking its ground truth from the
  segmentation convention (`annotations.masks`, unioned).

The snapshot in `runs/benchmark_rows.jsonl` predates the fix, so its three mask rows still carry
overlap metrics only. Re-running the benchmark over UNet++, DeepLabV3+ and Mask R-CNN is what adds
the bars back; do it in the environment the snapshot came from (`anomalib_env`, torch 2.14 /
torchvision 0.29 — see `provenance.python_executable` on any row), and the new fields land in the
JSONL and in `snapshot_audit.csv` automatically.

`dice` is deliberately not plotted: for a binary mask it **is** `pixel_f1`. The evaluator's
`_dice` (`2·|g∩p| / (|g|+|p|)`) and `_pixel_f1` (`2PR/(P+R)`) are the same formula, and in this
snapshot the two columns agree to every quoted digit. Showing both would put two identical bars in
every mask-model group. `iap` is left off the figure too: it covers the same nine anomaly models
as AUROC/AUPRO and several of its values are near zero.

## Families

Only two families are reported: **anomaly** and **detection**. A segmentation model predicts a
binary pixel mask that is scored with pixel-level overlap metrics, which is pixel-level defect
detection, so `data.family()` groups it with the anomaly models rather than giving it a third
colour and a third legend entry. That is why `05_parameters_vs_flops` and `06_quality_vs_latency`
show two legend entries, not three.

The raw task strings (`anomaly` / `segmentation` / `detection`) still exist in `data.task()`, which
is what decides whether a row reports box metrics; only the figure and audit grouping is
two-family.

## Where the grouping comes from

Figures and the audit table group metrics by `fabric_defect_hub.metrics_taxonomy`, the project's
single source of truth, rather than by a table this script invents. Two consequences worth
knowing:

- **Segmentation is pixel-level.** `SegmentationEvaluator` binarises the predicted mask (`> 0`,
  instance masks OR-ed together) and scores mIoU / Dice / Pixel F1 against the ground-truth mask,
  and the taxonomy files all three under `pixel_level`. So a segmentation model appears in
  `02_pixel_level`, next to the anomaly-map models, not in a `segmentation_level` figure of its
  own. On the current snapshot `02_pixel_level` holds 12 models: 9 anomaly-map models (pixel
  AUROC / AUPRO / Pixel F1) and 3 mask models (Pixel F1 / mIoU), with no bar where a model does
  not report a metric.
- **One metric lives in one group.** `pixel_f1` used to be listed under both an
  `anomaly_pixel_level` and a `segmentation_level` chart, so `snapshot_audit.csv` counted the same
  cells twice. It now appears once, under `pixel_level`.

`01_image_level` covers all 16 models that report an image-level score, including the detection
backends, whose score is derived from the highest-confidence box per image (see
`metrics_taxonomy.EMPTY_HINTS`). It is no longer limited to anomaly models.

`03_instance_level` drops detection models whose `map` is below `INSTANCE_MAP_FLOOR` (0.01): a
zero-height bar is indistinguishable from a missing model. The excluded models and their measured
value are printed when the script runs, never dropped silently. Models above the floor are always
shown, and the excluded ones still appear in `06_quality_vs_latency`.

`06_quality_vs_latency` uses the taxonomy's granularities as its three panels (image / pixel /
instance) rather than the model families, so anomaly-map and segmentation models share the
pixel-level panel through Pixel F1.

## Scatter annotations

The two scatter figures label their points through `_annotate_points`, not a fixed offset. Each
label tries `LABEL_OFFSETS` in order — right first, then left, then the diagonals and pure
vertical placements — and takes the first that overlaps neither an already placed label, nor an
unrelated marker, nor the axes frame. Near-identical models therefore end up labelled on opposite
sides instead of on top of each other. The collision test works in pixels, so it runs after the
final `tight_layout` and `_save(..., layout=False)`.

## Anomaly ROC/PR curves

`plot_anomaly_curves.py` reads the saved per-sample predictions and anomaly maps under
`artifacts/runtime/anomaly_maps/benchmark/` and writes two-sided PR/ROC figures:

```bash
python tools/benchmark_plots/plot_anomaly_curves.py
```

- `08_anomaly_image_level_pr_roc`: image-level PR and ROC for anomaly detection models
- `09_detection_image_level_pr_roc`: image-level PR and ROC for bounding-box defect detection models
- `10_anomaly_pixel_level_pr_roc`: pixel-level PR and ROC for anomaly-map models

Anomaly detection and defect detection are drawn as separate figures because mixing the two
families into one panel makes the curves unreadable.

Segmentation models cannot appear in these curves. A PR/ROC trace needs a score to sweep the
threshold over, and a segmentation model persists a binarised mask (`anomaly_map` is `null` for
all 350 rows) — a single operating point, not a curve. Their pixel quality is shown as
thresholded bars in `02_pixel_level` instead.

Layout conventions:

- The shared legend on the right lists model names only, inside a rounded frame.
- Panel labels `(a)` and `(b)` sit below their panels, as in a figure caption.
- Curves are told apart by hue and by line style: colours come from `tab10` and the dash pattern
  cycles through `LINE_STYLES`. `tab20` is deliberately avoided because it pairs a dark and a
  light variant of the same hue, which reads as one colour.

Useful options:

```bash
# render only one of the three figures
python tools/benchmark_plots/plot_anomaly_curves.py --only pixel

# zoom the main panels yourself, e.g. the high-score region
python tools/benchmark_plots/plot_anomaly_curves.py --xlim 0 0.3 --ylim 0.7 1.0
```

The traces are smoothed for display only: the empirical staircase is re-sampled on a dense grid
and convolved with a Gaussian kernel (`SMOOTH_SIGMA`), then passed through a monotonicity pass.
The reported AP and AUROC values are always computed from the unsmoothed traces.

## Known gaps

`docs/writting/benchmark_20260807_latest_19.xlsx` is a hand-maintained export that predates the
current log. Its `Method_Notes` sheet still says *"The current anomaly log contains image-level
metrics only"* and it keeps a separate `Segmentation` sheet. Both are stale: the current
`runs/benchmark_rows.jsonl` reports `pixel_auroc` / `pixel_aupro` / `iap` / `pixel_f1` for nine
anomaly models. The Excel is not regenerated by these scripts.
