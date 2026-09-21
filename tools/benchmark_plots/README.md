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

Neither script has to be run whole. Both take the same three flags, and figure ids run 01–10 (bar
and overhead figures) then 11–13 (PR/ROC curves):

```bash
# what can be drawn, and what is already on disk
python tools/benchmark_plots/plot_local_benchmark.py --list
python tools/benchmark_plots/plot_anomaly_curves.py --list

# one figure
python tools/benchmark_plots/plot_local_benchmark.py --only 02
python tools/benchmark_plots/plot_anomaly_curves.py --only 11

# several
python tools/benchmark_plots/plot_local_benchmark.py --only 01 03

# fill in any output format that is missing (png/pdf/svg)
python tools/benchmark_plots/plot_local_benchmark.py --missing
```

`--list` answers without loading a snapshot, so a driver can ask a script what it can draw instead
of hardcoding the list; it prints the formats still missing for each figure. `--only` with an
unknown id fails loudly and prints the known ids. `--missing` renders a figure when *any* of its
three formats is absent, which is how a format added later (SVG, for instance) is filled in for
existing figures without redrawing the ones that already have everything.

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

- `01_image_level`: image AUROC / AP / F1 / precision / recall, in two panels — anomaly detection
  and supervised detection, kept apart (see below)
- `02_pixel_level`: pixel AUROC / AUPRO / IAP / Pixel F1, anomaly maps and segmentation masks
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

Every figure is written as PNG, PDF and SVG. The SVG keeps its text as editable `<text>` elements (`svg.fonttype = "none"`), so a vector editor can reword a label without redrawing it. The trade-off is that the SVG no longer embeds the font: it carries a fallback chain ending in `sans-serif`, so it renders everywhere but with the local font's metrics, whereas PNG and PDF stay self-contained. `snapshot_audit.csv` records every candidate metric cell, including missing values and their reason, under a `family` column. `snapshot.json` records the selected commit and model list.

## Bar layout and styles

`01_image_level` is two grouped-bar panels — **(a)** anomaly detection, **(b)** supervised defect
detection — because the two families' image scores are not the same measurement; see
"Image level is two panels" below. `02_pixel_level` and `03_instance_level` are one grouped-bar
panel each: one group per model, one bar per metric, with the same style used for the same metric
in every figure. Four separate panels made a metric look comparable across a different model set
than its neighbours; one panel keeps a model's whole profile in a single column and turns the
metric into a matter of bar style.

| Bar | Style |
| --- | --- |
| 1 | light grey solid fill (`#d1d5db`) |
| 2 | red dots (`.`) |
| 3 | blue diagonal hatch (`/`) |
| 4 | grey diagonal hatch (`/`) |
| 5 | amber diagonal hatch (`\`) |
| 6 | teal diagonal hatch (`\`) |

In `01_image_level` both panels pass the *same* metric list in the same order, so a metric takes one
style across the whole figure rather than one style per panel.

The hatch is a single character — the sparsest matplotlib draws; repeating the character (`//`)
is what makes a hatch look dense. `hatch.linewidth` is lowered to 0.9 so the strokes read as a
light texture.

A model that reports nothing for a metric gets no bar there rather than a zero-height bar, so a
missing measurement never reads as a bad score. A model that reports *no* metric in a figure's set
is left out of that figure entirely, which is why `03_instance_level` holds only the detectors.

Metric subsets are presentation choices on top of the taxonomy grouping:

- image level, both panels: AUROC, AP, F1, precision, recall
- pixel level: AUROC, AUPRO, IAP, Pixel F1 — see below
- instance level: mAP, mAP@0.5, mAP@0.75, F1@0.5

`image_ap` is reported by both evaluators (`average_precision_score` over the image-level scores,
beside `image_auroc`) as of 2026-09-21, so it appears in the snapshot only after a benchmark re-run;
until then the AP bar is absent rather than zero. It is the area under panel (a) of the PR/ROC
curve figure, which is what aligns the two figures on the precision-recall dimension.

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

The current snapshot reflects that fix: the three mask rows carry `pixel_auroc` / `pixel_aupro` /
`iap` alongside `pixel_f1`, which is why all twelve groups in `02_pixel_level` now have the same
four bars. A snapshot taken before it would show the mask models with the overlap metrics only.

`dice` is deliberately not plotted: for a binary mask it **is** `pixel_f1`. The evaluator's
`_dice` (`2·|g∩p| / (|g|+|p|)`) and `_pixel_f1` (`2PR/(P+R)`) are the same formula, and in this
snapshot the two columns agree to every quoted digit. Showing both would put two identical bars in
every mask-model group.

`miou` is not plotted either, for two reasons. Only the three segmentation models report it, so it
was a bar on three of the twelve groups and nothing on the other nine -- a reader sees a gap and
cannot tell a missing measurement from a bad score. And for a single binary mask per image IoU is a
monotone function of F1 (`IoU = F1 / (2 - F1)`), so the bar repeats `pixel_f1` rather than adding a
measurement. It stays in `snapshot_audit.csv` and in the report's segmentation tables, which is
where the metric is actually compared.

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
  own. On the current snapshot `02_pixel_level` holds 12 models -- 9 anomaly-map models and 3 mask
  models -- and all twelve report the same four bars (pixel AUROC / AUPRO / IAP / Pixel F1), so no
  group is short a bar.
- **One metric lives in one group.** `pixel_f1` used to be listed under both an
  `anomaly_pixel_level` and a `segmentation_level` chart, so `snapshot_audit.csv` counted the same
  cells twice. It now appears once, under `pixel_level`.

### Image level is two panels

`01_image_level` splits the image-level score by **purpose**, because the two families' numbers are
not the same measurement:

- **(a) Anomaly detection** — the anomaly backends report a continuous normality score for every
  image, so their AUROC, AP, F1, precision and recall are comparable with each other.
- **(b) Defect detection** — a detector's image score is `max(box confidence)`, which
  is exactly `0` on any image where it finds nothing. `YOLOv8n` scores 157 of 245 normal images at
  exactly 0 and no defective image at 0, so its AUROC is 0.997 while its recall at its own
  threshold is 0.571. Placing that AUROC beside `PatchCore`'s on one axis invites a reviewer to
  read two different quantities as one.

Splitting the families is the whole fix, so the panels themselves carry no special-casing: they use
the same metric list, the same order and therefore the same bar styles, and models are ordered by
Image AUROC in both. `mAP@0.5` is not in this figure — mAP is an instance-level metric and belongs
to `03_instance_level`. The score caveat lives here and in `CAPTIONS.md`, not in a flagged bar or a
footnote: an earlier revision marked panel (b)'s AUROC with its own hatch and a two-line note, and
the figure read as though it were arguing with itself.

`DETR` is absent from both panels for the pre-existing `EXCLUDED_MODELS` reason (its bars are ~0),
but it is still in `snapshot_audit.csv` and in `12_detection_image_level_pr_roc`, where its AUROC
of 0.65 is a real, readable measurement.

Segmentation models (`DeepLabV3+`, `UNet++`, `Mask R-CNN`) report no image-level score at all —
pixel metrics are their image-level statement — so neither panel contains them.

`03_instance_level` drops detection models whose `map` is below `INSTANCE_MAP_FLOOR` (0.01): a
zero-height bar is indistinguishable from a missing model. The excluded models and their measured
value are printed when the script runs, never dropped silently. Models above the floor are always
shown; the rest still appear in `06_quality_vs_latency` and the PR/ROC curves — except `DETR`,
which `EXCLUDED_MODELS` removes from every bar figure.

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

- `11_anomaly_image_level_pr_roc`: image-level PR and ROC for anomaly detection models
- `12_detection_image_level_pr_roc`: image-level PR and ROC for bounding-box defect detection models
- `13_anomaly_pixel_level_pr_roc`: pixel-level PR and ROC for anomaly-map models

Anomaly detection and defect detection are drawn as separate figures because mixing the two
families into one panel makes the curves unreadable.

Segmentation models cannot appear in these curves. A PR/ROC trace needs a score to sweep the
threshold over, and a segmentation model persists a binarised mask (`anomaly_map` is `null` for
all 350 rows) — a single operating point, not a curve. Their pixel quality is shown as
thresholded bars in `02_pixel_level` instead.

### What the areas mean

Each legend entry prints the area under its own curves — `AP` for panel (a), `AUROC` for panel (b) —
because those areas are the only quantitative statement the figure makes, and they are the same
quantities the bar figures report. Two conventions are worth stating:

- **ROC AUC is the Mann-Whitney statistic**, so it matches `sklearn.roc_auc_score` and the
  evaluator's `image_auroc` exactly. Tied scores are collapsed into a single threshold before the
  trace is built. Building one point per *sample* instead is wrong under ties: a set of four
  samples all scoring 0.5 has a true AUC of 0.5, and the per-sample staircase integrates to 0.75.
  Ties are not exotic here — a detector that fires on nothing emits an exact `0.0`, and
  `Faster R-CNN` / `Cascade R-CNN` have over 200 tied scores in 350 samples, which used to bias
  their AUC by ~0.004.
- **AP is the area under the interpolated (VOC2010-style) PR curve**, i.e. precision is made
  monotone non-increasing before integration, which is the curve this figure draws.
  `sklearn.average_precision_score` uses the step-wise convention and returns a slightly lower
  number; they are different definitions, so the figure names the one it uses instead of leaving
  "AP" ambiguous.

### Reading figure 13's pixel curves

`13_anomaly_pixel_level_pr_roc` cannot draw 27.5 M pixels as a readable trace, so it subsamples:
at most `MAX_IMAGES_PER_MODEL` images and `MAX_PIXELS_PER_MODEL` pixels per model (fixed seed, so
the figure is reproducible). The subsample is **display-only** and its implied AUROC is *not* the
full-pixel `pixel_auroc` in `02_pixel_level`. The gap is large for models whose per-image maps vary
— `SuperSimpleNet` integrates to ≈0.82 on the subsample against a full-pixel 0.46. Read 13 for
curve *shape*, and 02 for the pixel-level number.

### Detector image scores

A detection model's image score is `max(box confidence)`, which is exactly `0` whenever nothing is
detected: `YOLOv8n` scores 157 of 245 normal images at 0, while no defective image scores 0. That
point mass makes image-level ranking metrics easy and is one reason `11/12` and `01_image_level`
keep the families apart — a detector's `AUROC` and an anomaly detector's `AUROC` are not the same
measurement, and `image_recall` (at the calibrated threshold) can be 0.36 while `AUROC` is 0.99.
The detector's real comparison is the box metrics in `03_instance_level`.

Layout conventions:

- The shared legend on the right lists each model with its `AP · AUROC`, inside a rounded frame.
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
