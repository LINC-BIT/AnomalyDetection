# AnomalyDetection

An extensible, configuration-driven platform for supervised defect detection, supervised defect segmentation, and industrial anomaly detection. One interface serves several application domains, and a model manifest — not the source code — determines which concrete models the platform publishes.

The platform is operated in three ways: through the web interface, through the command-line tool `adh`, and through the Python package `fabric_defect_hub`. This document specifies how to install the platform, prepare data and weights, operate each of the three entry points, and extend the platform.

## Outline

<div>
<a href="#outline">Outline</a><br>
<a href="#1-minimal-working-example">1. Minimal Working Example</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#11-install-the-platform">1.1 Install the platform</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#12-verify-the-installation-without-data">1.2 Verify the installation without data</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#13-run-inference-in-the-web-interface">1.3 Run inference in the web interface</a><br>
<a href="#2-scope">2. Scope</a><br>
<a href="#3-task-families-and-outputs">3. Task families and outputs</a><br>
<a href="#4-supported-models">4. Supported models</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#41-textile-application-domain">4.1 Textile application domain</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#42-general-application-domain">4.2 General application domain</a><br>
<a href="#5-datasets">5. Datasets</a><br>
<a href="#6-architecture">6. Architecture</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#61-directory-layout">6.1 Directory layout</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#62-layering-and-the-dependency-rule">6.2 Layering and the dependency rule</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#63-registries">6.3 Registries</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#64-contracts">6.4 Contracts</a><br>
<a href="#7-requirements">7. Requirements</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#71-software">7.1 Software</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#72-hardware-profiles">7.2 Hardware profiles</a><br>
<a href="#8-installation">8. Installation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#81-obtain-the-source-and-initialise-components">8.1 Obtain the source and initialise components</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#82-create-the-python-environment">8.2 Create the Python environment</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#83-install-dependencies">8.3 Install dependencies</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#84-verify-the-installation">8.4 Verify the installation</a><br>
<a href="#9-configuration-and-environment-variables">9. Configuration and environment variables</a><br>
<a href="#10-data-preparation">10. Data preparation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#101-registered-datasets-and-default-roots">10.1 Registered datasets and default roots</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#102-staging-a-dataset-manually">10.2 Staging a dataset manually</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#103-automated-download">10.3 Automated download</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#104-verifying-data-availability">10.4 Verifying data availability</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#105-public-dataset-sources">10.5 Public dataset sources</a><br>
<a href="#11-weight-preparation">11. Weight preparation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#111-published-slot-layout">11.1 Published-slot layout</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#112-downloading-published-weights">11.2 Downloading published weights</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#113-verifying-weight-availability">11.3 Verifying weight availability</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#114-the-weight-manifest">11.4 The weight manifest</a><br>
<a href="#12-web-interface">12. Web interface</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#121-launch">12.1 Launch</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#122-model-session">12.2 Model session</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#123-benchmark">12.3 Benchmark</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#124-run-history">12.4 Run history</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#125-recorded-demonstrations">12.5 Recorded demonstrations</a><br>
<a href="#13-command-line-workflows">13. Command-line workflows</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#131-model-configuration-resolution">13.1 Model configuration resolution</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#132-training">13.2 Training</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#133-inference">13.3 Inference</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#134-evaluation">13.4 Evaluation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#135-benchmarking">13.5 Benchmarking</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#136-batch-training">13.6 Batch training</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#137-catalogue-and-diagnostic-commands">13.7 Catalogue and diagnostic commands</a><br>
<a href="#14-outputs-and-reproducibility">14. Outputs and reproducibility</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#141-output-tree">14.1 Output tree</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#142-provenance-records">14.2 Provenance records</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#143-reproducibility-checklist">14.3 Reproducibility checklist</a><br>
<a href="#15-tools-reference">15. Tools reference</a><br>
<a href="#16-testing-and-quality-gates">16. Testing and quality gates</a><br>
<a href="#17-extending-the-platform">17. Extending the platform</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#171-add-a-model">17.1 Add a model</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#172-add-a-dataset">17.2 Add a dataset</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#173-add-an-application-domain">17.3 Add an application domain</a><br>
<a href="#18-troubleshooting">18. Troubleshooting</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#181-a-dataset-is-reported-as-unavailable">18.1 A dataset is reported as unavailable</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#182-a-component-checkout-is-missing">18.2 A component checkout is missing</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#183-importing-the-web-interface-raises-a-socks-proxy-error">18.3 Importing the web interface raises a SOCKS proxy error</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#184-a-weight-is-missing">18.4 A weight is missing</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#185-the-web-interface-displays-an-obsolete-label">18.5 The web interface displays an obsolete label</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#186-no-anomaly-heat-map-is-displayed">18.6 No anomaly heat map is displayed</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#187-training-runs-out-of-memory">18.7 Training runs out of memory</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;<a href="#188-a-general-training-run-fails-because-the-dataset-is-not-a-one-class-training-source">18.8 A general training run fails because the dataset is not a one-class training source</a><br>
<a href="#19-terminology">19. Terminology</a><br>
<a href="#20-documentation-placeholders">20. Documentation placeholders</a><br>
<a href="#21-additional-documentation">21. Additional documentation</a>
</div>

## 1. Minimal Working Example

This section is the shortest path from a clean checkout to a visible result. Each step is specified in full in the section referenced beside it.

### 1.1 Install the platform

```bash
git clone --recurse-submodules https://github.com/LINC-BIT/AnomalyDetection.git
cd AnomalyDetection
conda create -n anomalib_env python=3.12 -y
conda activate anomalib_env
git submodule update --init --recursive
python -m pip install -r requirements-full.txt
```

`[[FILL: confirm the canonical clone URL and, if the project is mirrored, the mirror URL.]]`

### 1.2 Verify the installation without data

The following commands require no dataset and no weight.

```bash
python examples/01_inspect_catalog.py
adh list
adh inventory
adh doctor
```

Expected output, as recorded on the development machine:

```text
AnomalyDetection version: 0.2.0
Registered models: 29
Registered datasets: 9
Models: yolov8n, yolov8s, yolo11n, fasterrcnn_resnet50_fpn, cascadercnn_resnet50_fpn, detr_resnet50, maskrcnn_resnet50_fpn, unetplusplus_resnet34, deeplabv3plus_resnet50, PatchCore, PaDiM, RD4AD, EfficientAD, SuperSimpleNet, STFPM, GANomaly, WinCLIP, Dinomaly, MoECLIP, MambaAD, PatchCore_general, PaDiM_general, RD4AD_general, EfficientAD_general, SuperSimpleNet_general, STFPM_general, GANomaly_general, Dinomaly_general, MambaAD_general
Datasets: fabric-defects, fabric-train, mvtec-ad, mvtec-loco, raw-fabric, tianchi, tilda-400, visa, zju-leaper
```

`adh list` prints the registered datasets, backends, evaluators, and profilers. `adh inventory` prints the model manifest. `adh doctor` prints, for each backend, whether the backend is trainable on this machine. The expected outputs of all three commands are stated in Section 8.4.

### 1.3 Run inference in the web interface

Step 1. Launch the web interface as described in Section 12.1.

```bash
adh-ui
```

Step 2. Open the printed URL. The default URL is `http://127.0.0.1:6008`.

Step 3. In **Task type**, select **Anomaly detection**.

![Minimal Working Example, step 3: select the task type](docs/images/img1.png)

Step 4. In **Application domain**, select `general`, because the model used in this example belongs to the `general` application domain.

![Minimal Working Example, step 4: select the application domain](docs/images/img2.png)

Step 5. In **Model**, select `WinCLIP · LAION-400M zero-shot`. Confirm that the panel states the application domain `general` and the published slot `general/artifacts/models/published/WinCLIP.ckpt`.

![Minimal Working Example, step 5: select the model](docs/images/img3.png)

If that published slot is empty on your machine, substitute any model identifier whose `weight_status` field is `file` or `symlink` in the output of `adh inventory` (Section 11.3).

Step 6. Select **Load model**, and wait until the status reports the model as loaded.

![Minimal Working Example, step 6: load the model](docs/images/img4.png)

Step 7. Select the dataset `ZJU-Leaper`, select a pattern or **All textures**, and select the `test` split. Select **Full-shot**. Then select **Load random images**.

![Minimal Working Example, step 7: select the dataset and the split](docs/images/img5.png)

Step 8. Select **Run detection**, and wait for the process to finish.

![Minimal Working Example, step 8: run detection](docs/images/img6.png)

Expected result: the gallery displays the loaded images and the anomaly score of each image, and, for a model that reports an anomaly map, the anomaly heat map.

`[[FILL: record the observed anomaly scores, the observed runtime, and a screenshot of the completed run, so that a reader can compare a local result with a reference result.]]`

## 2. Scope

This repository delivers a modular framework organised by application domain, a runnable demonstration for each registered application domain, and documentation of deployment, architecture, inference, evaluation, and outputs.

This repository does not deliver a weight for every published slot, nor an automated download procedure for every dataset. Section 11.3 lists the published slots that are empty, and Section 10.2 lists the datasets that must be staged manually. A weight is valid only for the training corpus declared for its model identifier in the model manifest.

## 3. Task families and outputs

The platform solves four task families through one interface. A backend fills only the prediction fields that its capability declaration reports.

<table align="center">
  <thead>
    <tr>
      <th>Task family</th>
      <th>Required annotation</th>
      <th>Prediction fields</th>
      <th>Evaluator</th>
      <th>Headline metric</th>
      <th>Metric keys</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Supervised defect detection</td>
      <td><code>boxes</code>, <code>labels</code></td>
      <td><code>boxes</code>, <code>labels</code>, <code>scores</code></td>
      <td><code>detection</code></td>
      <td><code>map</code></td>
      <td><code>map</code>, <code>map_50</code>, <code>map_75</code>, <code>map_small</code>, <code>map_medium</code>, <code>map_large</code>, <code>mar_1</code>, <code>mar_10</code>, <code>mar_100</code>, <code>precision_at_threshold</code>, <code>recall_at_threshold</code>, <code>f1_at_threshold</code>, <code>recall_small</code>, <code>recall_normal</code></td>
    </tr>
    <tr>
      <td>Supervised defect segmentation</td>
      <td><code>masks</code></td>
      <td><code>masks</code></td>
      <td><code>segmentation</code></td>
      <td><code>miou</code></td>
      <td><code>miou</code>, <code>dice</code>, <code>pixel_f1</code>, <code>num_evaluated</code>, <code>num_skipped_empty</code></td>
    </tr>
    <tr>
      <td>Industrial anomaly detection</td>
      <td><code>is_anomalous</code></td>
      <td><code>anomaly_score</code>, and <code>anomaly_map</code> when supported</td>
      <td><code>anomaly</code></td>
      <td><code>image_auroc</code></td>
      <td><code>image_auroc</code>, <code>image_f1</code>, <code>image_precision</code>, <code>image_recall</code>, <code>image_threshold</code>, <code>pixel_auroc</code>, <code>pixel_f1</code>, <code>pixel_aupro</code>, <code>iap</code></td>
    </tr>
    <tr>
      <td>Production-line anomaly detection</td>
      <td><code>is_anomalous</code>, and a unit length per sample</td>
      <td><code>anomaly_score</code></td>
      <td><code>industrial</code></td>
      <td>Not declared</td>
      <td><code>under_detection_rate</code>, <code>over_detection_rate</code>, <code>chosen_threshold</code>, <code>num_alarms</code>, <code>num_samples</code>, <code>alarms_per_unit_length</code></td>
    </tr>
  </tbody>
</table>

Three rules determine which metric keys are computable.

1. Image-level anomaly metrics require a sample label and a prediction score.
2. Pixel-level anomaly metrics require a ground-truth mask and a model-produced anomaly map. A model that does not declare `anomaly_map` cannot produce a heat map, and the web interface therefore shows no heat map for it. GANomaly is such a model.
3. The `instance_segmentation` task is scored by the `segmentation` evaluator over a unioned binary mask.

## 4. Supported models

The model manifest is authoritative. Run `adh inventory` for the machine-readable view and `adh models` for the backend-grouped view.

The manifest declares 29 model identifiers: 18 in the `textile` application domain and 11 in the `general` application domain. Two identifiers may share a backend and a model variant when they belong to different application domains, which is the case for the nine general training slots.

### 4.1 Textile application domain

<table align="center">
  <thead>
    <tr>
      <th>Model identifier</th>
      <th>Backend</th>
      <th>Model variant</th>
      <th>Task</th>
      <th>Method</th>
      <th>Training corpus and protocol</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>yolov8n</code></td><td><code>ultralytics</code></td><td><code>yolov8n</code></td><td><code>detection</code></td><td>YOLO</td><td>ZJU-Leaper</td></tr>
    <tr><td><code>yolov8s</code></td><td><code>ultralytics</code></td><td><code>yolov8s</code></td><td><code>detection</code></td><td>YOLO</td><td>ZJU-Leaper</td></tr>
    <tr><td><code>yolo11n</code></td><td><code>ultralytics</code></td><td><code>yolo11n</code></td><td><code>detection</code></td><td>YOLO</td><td>ZJU-Leaper</td></tr>
    <tr><td><code>fasterrcnn_resnet50_fpn</code></td><td><code>torchvision</code></td><td><code>fasterrcnn_resnet50_fpn</code></td><td><code>detection</code></td><td>Faster R-CNN</td><td>ZJU-Leaper</td></tr>
    <tr><td><code>cascadercnn_resnet50_fpn</code></td><td><code>torchvision</code></td><td><code>cascadercnn_resnet50_fpn</code></td><td><code>detection</code></td><td>Cascade R-CNN</td><td>ZJU-Leaper</td></tr>
    <tr><td><code>detr_resnet50</code></td><td><code>torchvision</code></td><td><code>detr_resnet50</code></td><td><code>detection</code></td><td>DETR</td><td>ZJU-Leaper</td></tr>
    <tr><td><code>maskrcnn_resnet50_fpn</code></td><td><code>torchvision</code></td><td><code>maskrcnn_resnet50_fpn</code></td><td><code>instance_segmentation</code></td><td>Mask R-CNN</td><td>ZJU-Leaper</td></tr>
    <tr><td><code>unetplusplus_resnet34</code></td><td><code>torchvision</code></td><td><code>unetplusplus_resnet34</code></td><td><code>segmentation</code></td><td>UNet++</td><td>ZJU-Leaper</td></tr>
    <tr><td><code>deeplabv3plus_resnet50</code></td><td><code>torchvision</code></td><td><code>deeplabv3plus_resnet50</code></td><td><code>segmentation</code></td><td>DeepLabV3+</td><td>ZJU-Leaper</td></tr>
    <tr><td><code>PatchCore</code></td><td><code>anomalib</code></td><td><code>PatchCore</code></td><td><code>anomaly</code></td><td>PatchCore</td><td>ZJU-Leaper, normal-only split</td></tr>
    <tr><td><code>PaDiM</code></td><td><code>anomalib</code></td><td><code>PaDiM</code></td><td><code>anomaly</code></td><td>PaDiM</td><td>ZJU-Leaper, normal-only split</td></tr>
    <tr><td><code>RD4AD</code></td><td><code>anomalib</code></td><td><code>RD4AD</code></td><td><code>anomaly</code></td><td>Reverse Distillation</td><td>ZJU-Leaper, normal-only split</td></tr>
    <tr><td><code>EfficientAD</code></td><td><code>anomalib</code></td><td><code>EfficientAD</code></td><td><code>anomaly</code></td><td>EfficientAD</td><td>ZJU-Leaper, normal-only split</td></tr>
    <tr><td><code>SuperSimpleNet</code></td><td><code>anomalib</code></td><td><code>SuperSimpleNet</code></td><td><code>anomaly</code></td><td>SuperSimpleNet</td><td>ZJU-Leaper, normal-only split</td></tr>
    <tr><td><code>STFPM</code></td><td><code>anomalib</code></td><td><code>STFPM</code></td><td><code>anomaly</code></td><td>STFPM</td><td>ZJU-Leaper, normal-only split</td></tr>
    <tr><td><code>GANomaly</code></td><td><code>anomalib</code></td><td><code>GANomaly</code></td><td><code>anomaly</code></td><td>GANomaly</td><td>ZJU-Leaper, normal-only split; image-level score only</td></tr>
    <tr><td><code>Dinomaly</code></td><td><code>dinomaly</code></td><td><code>dinov2reg_vit_base_14</code></td><td><code>anomaly</code></td><td>Dinomaly</td><td>ZJU-Leaper, normal-only split</td></tr>
    <tr><td><code>MambaAD</code></td><td><code>mambaad</code></td><td><code>resnet34</code></td><td><code>anomaly</code></td><td>MambaAD</td><td>ZJU-Leaper, normal-only split; published slot empty</td></tr>
  </tbody>
</table>

### 4.2 General application domain

<table align="center">
  <thead>
    <tr>
      <th>Model identifier</th>
      <th>Backend</th>
      <th>Model variant</th>
      <th>Task</th>
      <th>Method</th>
      <th>Training corpus and protocol</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>WinCLIP</code></td><td><code>anomalib</code></td><td><code>WinClip</code></td><td><code>anomaly</code></td><td>WinCLIP</td><td>External LAION-400M pretraining; zero-shot. ZJU-Leaper is an evaluation target only.</td></tr>
    <tr><td><code>MoECLIP</code></td><td><code>moeclip</code></td><td><code>ViT-L-14-336</code></td><td><code>anomaly</code></td><td>MoECLIP</td><td>MVTec AD auxiliary anomaly training; evaluated on ZJU-Leaper</td></tr>
    <tr><td><code>PatchCore_general</code></td><td><code>anomalib</code></td><td><code>PatchCore</code></td><td><code>anomaly</code></td><td>PatchCore</td><td>General training slot; no weight trained yet</td></tr>
    <tr><td><code>PaDiM_general</code></td><td><code>anomalib</code></td><td><code>PaDiM</code></td><td><code>anomaly</code></td><td>PaDiM</td><td>General training slot; no weight trained yet</td></tr>
    <tr><td><code>RD4AD_general</code></td><td><code>anomalib</code></td><td><code>RD4AD</code></td><td><code>anomaly</code></td><td>Reverse Distillation</td><td>General training slot; no weight trained yet</td></tr>
    <tr><td><code>EfficientAD_general</code></td><td><code>anomalib</code></td><td><code>EfficientAD</code></td><td><code>anomaly</code></td><td>EfficientAD</td><td>General training slot; no weight trained yet</td></tr>
    <tr><td><code>SuperSimpleNet_general</code></td><td><code>anomalib</code></td><td><code>SuperSimpleNet</code></td><td><code>anomaly</code></td><td>SuperSimpleNet</td><td>General training slot; no weight trained yet</td></tr>
    <tr><td><code>STFPM_general</code></td><td><code>anomalib</code></td><td><code>STFPM</code></td><td><code>anomaly</code></td><td>STFPM</td><td>General training slot; no weight trained yet</td></tr>
    <tr><td><code>GANomaly_general</code></td><td><code>anomalib</code></td><td><code>GANomaly</code></td><td><code>anomaly</code></td><td>GANomaly</td><td>General training slot; no weight trained yet</td></tr>
    <tr><td><code>Dinomaly_general</code></td><td><code>dinomaly</code></td><td><code>dinov2reg_vit_base_14</code></td><td><code>anomaly</code></td><td>Dinomaly</td><td>General training slot; no weight trained yet</td></tr>
    <tr><td><code>MambaAD_general</code></td><td><code>mambaad</code></td><td><code>resnet34</code></td><td><code>anomaly</code></td><td>MambaAD</td><td>General training slot; no weight trained yet</td></tr>
  </tbody>
</table>

A weight must not be relabelled from one application domain to another. The `domain` field of the model manifest records the application domain of a weight, and a training run writes its weight into the published slot of the application domain of the dataset that produced it.

## 5. Datasets

Nine dataset identifiers are registered. Each identifier declares a default root, the tasks for which it can supply ground truth, and the training roles it may serve.

<table align="center">
  <thead>
    <tr>
      <th>Dataset identifier</th>
      <th>Application domain</th>
      <th>Default root</th>
      <th>Tasks</th>
      <th>Training roles</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>zju-leaper</code></td><td><code>textile</code></td><td><code>datasets/textile/ZJU-Leaper</code></td><td><code>detection</code>, <code>segmentation</code>, <code>anomaly</code></td><td><code>anomaly_train</code>, <code>detection_train</code>, <code>fabric_train_member</code></td></tr>
    <tr><td><code>raw-fabric</code></td><td><code>textile</code></td><td><code>datasets/textile/RAW_FABRID</code></td><td><code>anomaly</code>, <code>segmentation</code></td><td><code>anomaly_train</code>, <code>fabric_train_member</code></td></tr>
    <tr><td><code>tilda-400</code></td><td><code>textile</code></td><td><code>datasets/textile/TILDA_400</code></td><td><code>anomaly</code></td><td><code>anomaly_train</code>, <code>fabric_train_member</code></td></tr>
    <tr><td><code>fabric-defects</code></td><td><code>textile</code></td><td><code>datasets/textile/Fabric Defects Dataset</code></td><td><code>anomaly</code>, <code>segmentation</code></td><td><code>anomaly_train</code>, <code>fabric_train_member</code></td></tr>
    <tr><td><code>tianchi</code></td><td><code>textile</code></td><td><code>datasets/textile/tianchi</code></td><td><code>detection</code>, <code>anomaly</code></td><td><code>anomaly_train</code>, <code>detection_train</code>, <code>fabric_train_member</code></td></tr>
    <tr><td><code>fabric-train</code></td><td><code>textile</code></td><td><code>datasets/textile</code></td><td><code>detection</code>, <code>segmentation</code>, <code>anomaly</code></td><td><code>anomaly_train</code></td></tr>
    <tr><td><code>mvtec-ad</code></td><td><code>general</code></td><td><code>datasets/general/MVTec AD</code></td><td><code>anomaly</code>, <code>segmentation</code></td><td><code>anomaly_train</code>, <code>zero_shot_train</code></td></tr>
    <tr><td><code>mvtec-loco</code></td><td><code>general</code></td><td><code>datasets/general/MVTec LOCO</code></td><td><code>anomaly</code>, <code>segmentation</code></td><td><code>anomaly_train</code>, <code>zero_shot_train</code></td></tr>
    <tr><td><code>visa</code></td><td><code>general</code></td><td><code>datasets/general/VisA</code></td><td><code>anomaly</code>, <code>segmentation</code></td><td><code>anomaly_train</code>, <code>zero_shot_train</code></td></tr>
  </tbody>
</table>

The training roles have the following meanings.

1. `anomaly_train` — the dataset supplies a normal-only split, so it is a valid training corpus for the one-class anomaly backends `anomalib`, `dinomaly`, and `mambaad`.
2. `zero_shot_train` — the dataset supplies labelled anomalies, so it is a valid auxiliary training corpus for the zero-shot backend `moeclip`. A dataset may hold both this role and `anomaly_train`.
3. `detection_train` — the dataset supplies bounding-box annotations, so it is a valid training corpus for the detection backends `ultralytics` and `torchvision`.
4. `fabric_train_member` — the dataset contributes samples to the `fabric-train` composite.

A dataset that declares none of these roles can be evaluated but not trained on. The `fabric-train` composite is not a member of its own union.

## 6. Architecture

### 6.1 Directory layout

```text
AnomalyDetection/
├── src/fabric_defect_hub/            # reusable SDK; the import name is historical
│   ├── application/                  # business services used by the CLI and the web interface
│   ├── core/                         # registries, contracts, provenance, configuration
│   ├── datasets/                     # DatasetAdapter implementations
│   ├── models/                       # ModelAdapter implementations, one package per backend
│   ├── evaluation/                   # backend-independent metric computation
│   ├── profiling/                    # runtime and resource measurement
│   ├── quantization/                 # post-training ONNX and TensorRT quantization
│   ├── recipes/                      # optimization recipes
│   ├── reporting/                    # run log, tables, LaTeX export, training curves
│   ├── strategies/                   # sparse subsampling, tiling, test-time augmentation
│   └── web/                          # presentation and event binding only
├── components/                       # pinned upstream research checkouts (Git submodules)
├── configs/
│   ├── registry/models.yaml          # the model manifest: the published-model source of truth
│   ├── models/                       # model configurations (textile)
│   └── training_profile.yaml         # the shared training profile
├── datasets/
│   ├── textile/                      # textile datasets
│   └── general/                      # MVTec AD, MVTec LOCO, VisA, and auxiliary data
├── general/                          # general assets and artifacts
│   ├── configs/models/               # model configurations (general)
│   └── artifacts/models/published/   # general published slots
├── textile/                          # textile assets and artifacts
│   └── artifacts/models/published/   # textile published slots
├── examples/                         # Minimal Working Example scripts
├── tools/                            # conversion, export, download, and benchmark tools
├── tests/                            # runtime contracts and architecture audits
├── schemas/                          # JSON Schemas for the unified contracts
└── docs/                             # focused guides and design records
```

### 6.2 Layering and the dependency rule

A layer may import the layers above it in the following list and must not import the layers below it.

1. `core/` — contracts, registries, and utilities. Imports no backend.
2. `datasets/`, `models/`, `evaluation/`, `profiling/`, `quantization/`, `recipes/` — implementations of the contracts.
3. `application/` — business services. The only supported boundary for the CLI and the web interface.
4. `cli.py` and `web/` — presentation.

Two mechanisms enforce this rule. `tests/test_web_layering.py` parses every module of `web/` with the `ast` module and fails if a module imports a backend package or contains a backend name as a string literal. The `architecture` pytest marker, declared in `pyproject.toml`, runs the source-boundary audits.

### 6.3 Registries

Two registries answer two different questions.

1. The backend registry answers how an implementation runs. A decorator registers each implementation: `@register_model`, `@register_dataset`, `@register_evaluator`, `@register_profiler`, and `@register_recipe`.
2. The model manifest answers which concrete model is published: its method, task, integration, application domain, training corpus, evaluation corpora, model configuration, and weight source.

The application services join the two registries, so adding a conforming entry to the model manifest updates every consumer.

### 6.4 Contracts

<table align="center">
  <thead>
    <tr>
      <th>Contract</th>
      <th>Module</th>
      <th>Responsibility</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>ModelAdapter</code></td><td><code>models/base.py</code></td><td>Defines <code>train</code>, <code>predict</code>, <code>export</code>, <code>load_trained_model</code>, and <code>capabilities</code>. The signatures are identical for every backend, including parameters that a given backend ignores.</td></tr>
    <tr><td><code>ModelCapabilities</code></td><td><code>models/base.py</code></td><td>Declares the tasks, the <code>Prediction</code> fields, the required annotations, the export targets, and the exported input style of a model. Every name is validated against a fixed vocabulary at import time.</td></tr>
    <tr><td><code>DatasetAdapter</code></td><td><code>datasets/base.py</code></td><td>Converts a source dataset into <code>Sample</code> objects.</td></tr>
    <tr><td><code>DatasetCapabilities</code></td><td><code>core/dataset_capabilities.py</code></td><td>Declares the default root, the training roles, the tasks, and the application domain of a dataset. This module is the single source of truth for what a dataset may be used for.</td></tr>
    <tr><td><code>DataAdapter</code></td><td><code>core/data_adapter.py</code></td><td>Converts <code>Sample</code> objects into backend batches, and declares the resulting batch layout through <code>BatchSpec</code>.</td></tr>
    <tr><td><code>Sample</code>, <code>Prediction</code>, <code>ExperimentResult</code></td><td><code>core/types.py</code></td><td>The unified data contracts, whose JSON representations are specified by <code>schemas/sample.schema.json</code>, <code>schemas/prediction.schema.json</code>, and <code>schemas/experiment_result.schema.json</code>.</td></tr>
    <tr><td><code>Evaluator</code></td><td><code>evaluation/base.py</code></td><td>Computes metrics from <code>Sample</code> and <code>Prediction</code> objects. An evaluator imports no backend.</td></tr>
    <tr><td><code>BackendProfiler</code></td><td><code>profiling/base.py</code></td><td>Measures the runtime and the resource consumption of an exported model.</td></tr>
  </tbody>
</table>

## 7. Requirements

### 7.1 Software

<table align="center">
  <thead>
    <tr>
      <th>Item</th>
      <th>Requirement</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Python</td><td>Version 3.10 or newer is required. Versions 3.11 to 3.13 are recommended for training, because PyTorch emits TorchScript deprecation warnings on version 3.14. The commands in this document use version 3.12. The verification recorded on the development machine used version 3.14.6.</td></tr>
    <tr><td>Environment manager</td><td>Conda is recommended. The environment name used throughout this document is <code>anomalib_env</code>; the name is historical and does not restrict which backends may be installed.</td></tr>
    <tr><td>Operating system</td><td>macOS or Linux.</td></tr>
    <tr><td>Graphics processing unit</td><td>Optional for inference. A CUDA-capable NVIDIA graphics processing unit is recommended for training.</td></tr>
    <tr><td>CUDA</td><td>Required only for NVIDIA acceleration and for the CUDA-only optional dependency groups.</td></tr>
    <tr><td>Git</td><td>Required, because three components are Git submodules.</td></tr>
  </tbody>
</table>

### 7.2 Hardware profiles

These figures are recommendations, not guaranteed minimums. On a constrained machine, use a smaller batch size and the `test` shot mode.

<table align="center">
  <thead>
    <tr>
      <th>Workflow</th>
      <th>Processor and memory</th>
      <th>Graphics processing unit</th>
      <th>Disk</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Catalogue inspection, test suite, web interface layout</td><td>4 or more cores, 16 GB RAM</td><td>Not required</td><td>10 GB, plus the weights</td></tr>
    <tr><td>Minimal Working Example inference</td><td>8 or more cores, 16 GB RAM</td><td>Optional</td><td>The weights, plus the selected dataset</td></tr>
    <tr><td>Full textile training</td><td>8 or more cores, 32 GB RAM</td><td>NVIDIA recommended; capacity depends on the model and the batch size</td><td>The dataset and the run artifacts, typically tens of GB</td></tr>
    <tr><td>Full benchmark suite</td><td>16 or more cores, 32 GB to 64 GB RAM</td><td>Recommended</td><td>All datasets, weights, anomaly maps, exports, and logs</td></tr>
  </tbody>
</table>

## 8. Installation

### 8.1 Obtain the source and initialise components

```bash
git clone --recurse-submodules https://github.com/LINC-BIT/AnomalyDetection.git
cd AnomalyDetection
```

If the repository was cloned without the `--recurse-submodules` option, initialise the components explicitly.

```bash
git submodule update --init --recursive
git submodule status
```

Expected result of `git submodule status`: three lines, one each for `components/anomalydiffusion`, `components/dinomaly`, and `components/moeclip`, each beginning with the pinned revision.

Do not clone a component repository manually. The parent repository pins the tested revision through `.gitmodules` and a Git link.

### 8.2 Create the Python environment

```bash
conda create -n anomalib_env python=3.12 -y
conda activate anomalib_env
python --version
```

Expected result: the shell prompt is prefixed with `(anomalib_env)`.

### 8.3 Install dependencies

<table align="center">
  <thead>
    <tr>
      <th>Dependency set</th>
      <th>Contents</th>
      <th>Installation command</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Complete</td><td>All six backends, the profiling and quantization dependencies, and the test dependencies. The file ends with <code>-e .</code>, so it also installs the project itself in editable mode.</td><td><code>python -m pip install -r requirements-full.txt</code></td></tr>
    <tr><td>Lean</td><td>The web interface and lightweight inference only. This file does not install the project itself, so the editable installation is a separate command.</td><td><code>python -m pip install -r requirements.txt</code><br><code>python -m pip install --no-deps --no-build-isolation -e .</code></td></tr>
  </tbody>
</table>

Optional dependency groups are declared in `pyproject.toml`. Install a group only when it is needed.

```bash
python -m pip install -e ".[mambaad-cuda]"           # CUDA-only fused selective-scan kernel
python -m pip install -e ".[profiling-onnxruntime]"  # ONNX Runtime profiler
python -m pip install -e ".[profiling-flops]"        # FLOPs counter
python -m pip install -e ".[profiling-power-nvidia]" # NVIDIA power draw through NVML
python -m pip install -e ".[profiling-tensorrt]"     # TensorRT profiler; NVIDIA hardware only
python -m pip install -e ".[quantization]"           # fp16 and INT8 ONNX quantization
```

The `Makefile` provides the same two dependency sets as `make install` and `make install-ui`.

### 8.4 Verify the installation

Four commands verify the installation. None of them requires a dataset or a weight. The first is stated in Section 1.2; the remaining three are stated here.

```bash
adh list
```

Expected output on a machine with all six frameworks installed:

```json
{
  "datasets": ["fabric-defects", "fabric-train", "mvtec-ad", "mvtec-loco", "raw-fabric", "tianchi", "tilda-400", "visa", "zju-leaper"],
  "model_backends": {
    "known": ["anomalib", "dinomaly", "mambaad", "moeclip", "torchvision", "ultralytics"],
    "available": ["anomalib", "dinomaly", "mambaad", "moeclip", "torchvision", "ultralytics"]
  },
  "evaluators": ["anomaly", "detection", "industrial", "segmentation"],
  "profilers": ["onnxruntime", "pytorch", "tensorrt"]
}
```

`model_backends.known` lists every backend the project knows how to talk to; `model_backends.available` lists the backends whose framework is importable on this machine.

```bash
adh inventory
```

Expected result: a JSON object with the keys `models` and `datasets`. Each entry of `models` states the model identifier, the label, the backend, the model variant, the task, the category, the subtype, the integration, the training mode, the method, the training corpus, the training split, the evaluation corpora, the application domain, the model configuration, the published-slot path, the published-slot status, the weight source, and the legacy weight URL.

```bash
adh doctor
```

Expected result: a JSON object keyed by backend. For each backend, `framework_installed` states whether the framework is importable, `trainable_now` states whether both the framework and a suitable dataset are available, and `reason` states which dataset would be selected and why.

```json
{
  "backends": {
    "anomalib": {
      "framework_installed": true,
      "dataset_kind": "one-class",
      "dataset": "fabric-defects",
      "trainable_now": true,
      "reason": "No dataset was requested; picked 'fabric-defects' (staged). Other staged alternatives: fabric-train, mvtec-ad, mvtec-loco, raw-fabric, tianchi, tilda-400, visa, zju-leaper."
    }
  }
}
```

The real output contains one such entry for each of the six backends. The selected dataset depends on which datasets are staged on the machine.

## 9. Configuration and environment variables

Two YAML files carry configuration. The model manifest `configs/registry/models.yaml` declares the published models and is the source of truth. A model configuration under `configs/models/` or `general/configs/models/` declares the executable parameters of one run; the model manifest refers to a model configuration by filename, and the runtime resolves it under `configs/models/` unless an explicit path is supplied.

No secret and no machine-specific absolute path may enter a committed YAML file. Use an environment variable instead.

<table align="center">
  <thead>
    <tr>
      <th>Environment variable</th>
      <th>Default value</th>
      <th>Effect</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>AD_DATASETS_ROOT</code></td><td><code>&lt;repository&gt;/datasets</code></td><td>Root of the dataset tree.</td></tr>
    <tr><td><code>AD_COMPONENTS_ROOT</code></td><td><code>&lt;repository&gt;/components</code></td><td>Root of the component checkouts.</td></tr>
    <tr><td><code>AD_TEXTILE_ROOT</code></td><td><code>&lt;repository&gt;/textile</code></td><td>Root of the textile assets and artifacts.</td></tr>
    <tr><td><code>AD_GENERAL_ROOT</code></td><td><code>&lt;repository&gt;/general</code></td><td>Root of the general assets and artifacts.</td></tr>
    <tr><td><code>AD_ARTIFACTS_ROOT</code></td><td><code>&lt;repository&gt;/textile/artifacts</code></td><td>Root under which trained artifacts and the weight manifest are written.</td></tr>
    <tr><td><code>AD_CONFIGS_ROOT</code></td><td><code>&lt;repository&gt;/configs</code></td><td>Root of the configuration tree.</td></tr>
    <tr><td><code>AD_RESULTS_ROOT</code></td><td><code>&lt;repository&gt;/results</code></td><td>Root of the results tree.</td></tr>
    <tr><td><code>GRADIO_SERVER_PORT</code></td><td><code>6008</code></td><td>Port on which the web interface listens. The web interface also listens on all interfaces.</td></tr>
    <tr><td><code>FDH_PROGRESS</code></td><td><code>1</code></td><td>Whether progress lines are printed. One of <code>0</code>, <code>false</code>, <code>no</code>, or <code>off</code> disables them.</td></tr>
    <tr><td><code>FDH_PROGRESS_INTERVAL</code></td><td><code>5.0</code></td><td>Seconds between progress lines. A non-positive or non-numeric value restores the default.</td></tr>
    <tr><td><code>FDH_MODEL_CACHE</code></td><td><code>artifacts/models</code></td><td>Directory searched by the cloud-model preflight tool for a missing weight.</td></tr>
    <tr><td><code>FDH_MAMBAAD_SCAN_BUDGET</code></td><td><code>64000000</code></td><td>Element budget of one chunk of the portable MambaAD selective scan.</td></tr>
  </tbody>
</table>

The web interface additionally resolves the location of each dataset from a dataset-specific environment variable before it falls back to the declared default root. The command-line interface does not read these variables; it uses the declared default root and the `--dataset-root` option instead.

<table align="center">
  <thead>
    <tr>
      <th>Dataset label in the web interface</th>
      <th>Dataset identifier</th>
      <th>Root override variable</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>ZJU-Leaper</td><td><code>zju-leaper</code></td><td><code>ZJU_LEAPER_ROOT</code></td></tr>
    <tr><td>RAW-FABRID</td><td><code>raw-fabric</code></td><td><code>RAW_FABRIC_ROOT</code></td></tr>
    <tr><td>MVTec AD</td><td><code>mvtec-ad</code></td><td><code>MVTEC_AD_ROOT</code></td></tr>
    <tr><td>MVTec LOCO</td><td><code>mvtec-loco</code></td><td><code>MVTEC_LOCO_ROOT</code></td></tr>
    <tr><td>VisA</td><td><code>visa</code></td><td><code>VISA_ROOT</code></td></tr>
    <tr><td>TILDA-400</td><td><code>tilda-400</code></td><td><code>TILDA_400_ROOT</code></td></tr>
    <tr><td>Fabric Defects</td><td><code>fabric-defects</code></td><td><code>FABRIC_DEFECTS_ROOT</code></td></tr>
    <tr><td>Tianchi</td><td><code>tianchi</code></td><td><code>TIANCHI_ROOT</code></td></tr>
  </tbody>
</table>

## 10. Data preparation

### 10.1 Registered datasets and default roots

A dataset is available to the command-line interface when its directory exists at the declared default root, relative to the repository root. The roots are listed in Section 5 and are declared once in `core/dataset_capabilities.py`. The `--dataset-root` option overrides the declared root for one command.

### 10.2 Staging a dataset manually

1. Create the declared default root directory.
2. Copy the dataset into it, preserving the directory structure that the dataset adapter expects.
3. Confirm the result with `adh doctor`, then with a training run in `test` shot mode.

```bash
mkdir -p "datasets/general/MVTec LOCO"
# Copy the extracted dataset into that directory.
adh doctor
```

The directory structure that each adapter expects is documented in the module docstring of that adapter.

<table align="center">
  <thead>
    <tr>
      <th>Dataset identifier</th>
      <th>Adapter module</th>
      <th>Automated download</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>zju-leaper</code></td><td><code>datasets/zju_leaper.py</code></td><td>Available</td></tr>
    <tr><td><code>raw-fabric</code></td><td><code>datasets/raw_fabric.py</code></td><td>Not available; stage manually</td></tr>
    <tr><td><code>tilda-400</code></td><td><code>datasets/tilda.py</code></td><td>Not available; stage manually</td></tr>
    <tr><td><code>fabric-defects</code></td><td><code>datasets/fabric_defects.py</code></td><td>Not available; stage manually</td></tr>
    <tr><td><code>tianchi</code></td><td><code>datasets/tianchi.py</code></td><td>Not available; stage manually</td></tr>
    <tr><td><code>fabric-train</code></td><td><code>datasets/fabric_train.py</code></td><td>Composite; no separate download</td></tr>
    <tr><td><code>mvtec-ad</code></td><td><code>datasets/mvtec_ad.py</code></td><td>Available</td></tr>
    <tr><td><code>mvtec-loco</code></td><td><code>datasets/mvtec_loco.py</code></td><td>Not available; stage manually</td></tr>
    <tr><td><code>visa</code></td><td><code>datasets/visa.py</code></td><td>Available</td></tr>
  </tbody>
</table>

`[[FILL: for each dataset that must be staged manually, record the acquisition procedure, the archive size, the license, the download date, and the exact directory structure that was staged.]]`

### 10.3 Automated download

The download tool takes the dataset identifier as its first positional argument and requires `--root`. The option `--category` restricts the download to one category or class.

```bash
python tools/download_datasets.py mvtec-ad --root "datasets/general/MVTec AD" --category bottle
python tools/download_datasets.py visa --root datasets/general/VisA --category capsules
python tools/download_datasets.py zju-leaper --root datasets/textile/ZJU-Leaper
```

Pass the declared default root exactly, including the space in `MVTec AD`. A different spelling places the dataset outside the registered root, and `adh doctor` then reports the dataset as not staged.

`zju-leaper` accepts `--repo-id` to select a different Hugging Face dataset repository. The default is `AnupamaBandara/ZLU_Leaper`.

### 10.4 Verifying data availability

```bash
adh doctor
```

Expected result: for every backend whose required kind of dataset is staged, the `dataset` field names that dataset and `trainable_now` is `true`. A backend whose `dataset` field is absent has no staged dataset of the required kind.

`[[FILL: record the expected category or pattern count for each staged dataset, so that a reader can detect a partial download.]]`

### 10.5 Public dataset sources

<table align="center">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Source</th>
      <th>Automated download</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>MVTec AD</td><td><a href="https://www.mvtec.com/company/research/datasets/mvtec-ad">official page</a></td><td><code>mvtec-ad</code></td></tr>
    <tr><td>MVTec LOCO</td><td><a href="https://www.mvtec.com/company/research/datasets/mvtec-loco">official page</a></td><td>Not available</td></tr>
    <tr><td>VisA</td><td><a href="https://github.com/amazon-science/spot-diff">official repository</a></td><td><code>visa</code></td></tr>
    <tr><td>ZJU-Leaper</td><td><a href="https://huggingface.co/datasets/AnupamaBandara/ZLU_Leaper">AnupamaBandara/ZLU_Leaper</a></td><td><code>zju-leaper</code></td></tr>
    <tr><td>RAW-FABRID</td><td><code>[[FILL: source URL]]</code></td><td>Not available</td></tr>
    <tr><td>TILDA-400</td><td><code>[[FILL: source URL]]</code></td><td>Not available</td></tr>
    <tr><td>Fabric Defects Dataset</td><td><code>[[FILL: source URL]]</code></td><td>Not available</td></tr>
    <tr><td>Tianchi Guangdong fabric defect challenge</td><td><code>[[FILL: source URL]]</code></td><td>Not available</td></tr>
  </tbody>
</table>

## 11. Weight preparation

A weight is an immutable runtime artifact and is not committed to Git. The model manifest declares where each weight resolves. A published slot without a weight is reported as `missing`; a weight without a conforming model-identifier entry is unsupported.

### 11.1 Published-slot layout

A published slot resolves under the `<application domain>/artifacts/models/published/` directory, where `<application domain>` is the application domain of the model identifier.

```text
textile/artifacts/models/published/<model identifier><extension>
general/artifacts/models/published/<model identifier><extension>
```

The extension is determined by the backend: `.pt` for `ultralytics` and `torchvision`, `.ckpt` for `anomalib`, and `.pth` for `dinomaly`, `moeclip`, and `mambaad`.

A published slot holds one of four states, reported by `adh inventory` in the `weight_status` field.

<table align="center">
  <thead>
    <tr>
      <th>State</th>
      <th>Meaning</th>
      <th>Usable</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>file</code></td><td>A regular file is present.</td><td>Yes</td></tr>
    <tr><td><code>symlink</code></td><td>A symbolic link is present and resolves.</td><td>Yes</td></tr>
    <tr><td><code>broken_link</code></td><td>A symbolic link is present and does not resolve, typically because the tree was copied without <code>artifacts/models/</code>.</td><td>No</td></tr>
    <tr><td><code>missing</code></td><td>Nothing is present.</td><td>No</td></tr>
  </tbody>
</table>

A training run with publishing enabled replaces the published slot of the matching model identifier with a relative symbolic link to the trained artifact. A tree copied from another machine therefore arrives with regular files, which remain usable without a migration step.

### 11.2 Downloading published weights

The published textile weights are distributed from a Hugging Face Hub repository. The download tool takes the repository identifier and the filename inside the repository as positional arguments and requires `--output`.

```bash
python tools/download_weights.py AuroraLeeeeee/AnomalyDetection-textile-weights \
  textile/artifacts/models/published/yolov8n.pt \
  --output textile/artifacts/models/published/yolov8n.pt
python tools/download_weights.py AuroraLeeeeee/AnomalyDetection-textile-weights \
  textile/artifacts/models/published/yolov8s.pt \
  --output textile/artifacts/models/published/yolov8s.pt
python tools/download_weights.py AuroraLeeeeee/AnomalyDetection-textile-weights \
  textile/artifacts/models/published/yolo11n.pt \
  --output textile/artifacts/models/published/yolo11n.pt
```

Use the same pattern for every other file, with the exact filename that `adh inventory` reports in the `weight` field. Do not move a weight from one application domain to another.

`[[FILL: record the exact Hugging Face Hub repository identifier, the revision, and the SHA-256 checksum of every distributed file. The repository identifier shown above must be confirmed before release.]]`

`[[FILL: record the acquisition procedure for the weights that are not distributed through the Hugging Face Hub repository, namely the anomalib weights, the Dinomaly weight, the MambaAD weight, and the MoECLIP weight. The MoECLIP weight is currently a manual import from a legacy Google Drive location.]]`

Two model identifiers require special handling.

1. `WinCLIP` is an upstream LAION-400M-pretrained zero-shot model. Its published slot is a metadata handle rather than a serialized weight, because the `k_shot = 0` configuration has no learned state to restore. The adapter reconstructs the model from the metadata. `[[FILL: state whether the WinCLIP published slot must be distributed, or whether the model is always reconstructed.]]`
2. `MoECLIP` is an adapter trained on an auxiliary corpus. Its training corpus and its evaluation corpus are different datasets, and both are recorded in the model manifest.

### 11.3 Verifying weight availability

```bash
adh inventory
```

Expected result, recorded on the development machine: 19 published slots hold a weight and 10 are empty.

<table align="center">
  <thead>
    <tr>
      <th>Published slot</th>
      <th>State on the development machine</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Every textile slot except <code>MambaAD</code></td><td><code>file</code></td></tr>
    <tr><td><code>general/artifacts/models/published/WinCLIP.ckpt</code></td><td><code>file</code>; a metadata handle</td></tr>
    <tr><td><code>general/artifacts/models/published/MoECLIP.pth</code></td><td><code>file</code></td></tr>
    <tr><td><code>textile/artifacts/models/published/MambaAD.pth</code></td><td><code>missing</code></td></tr>
    <tr><td>The nine general training slots</td><td><code>missing</code></td></tr>
  </tbody>
</table>

A missing weight is resolved either by training the corresponding model identifier or by placing the weight at the exact path reported by `adh inventory`. Do not rename the weight of a different architecture to satisfy a published slot.

### 11.4 The weight manifest

Every training run that produces a trained artifact appends one provenance record to `artifacts/models/weight_manifest.jsonl` and writes a snapshot of the resolved configuration to `artifacts/models/records/<record identifier>.config.json`. The provenance record retains both the trained artifact and the published slot, so replacing the published slot never destroys the provenance of the trained artifact.

## 12. Web interface

### 12.1 Launch

```bash
conda activate anomalib_env
adh-ui
```

Expected result: the process starts a local server and prints a URL. The default URL is `http://127.0.0.1:6008`. The web interface listens on all interfaces, so a remote host is reachable at the address of that host and the same port.

```bash
GRADIO_SERVER_PORT=7860 adh-ui
```

If the port is already in use, `adh-ui` reports the process that holds the port and prints the two remedies: reuse the running instance, or start a new instance on another port. The web interface never downloads a missing weight implicitly; a missing weight is displayed with its exact expected path.

### 12.2 Model session

Select the controls in the stated order.

1. **Task type** — `Defect detection` or `Anomaly detection`.
2. **Application domain** — `textile` or `general`.
3. **Model** — the model identifiers that declare both the selected task and the selected application domain.

The model panel states the method, the training corpus, the training split, the published-slot status, and the prediction fields that the capability declaration reports. Select **Load model** before running inference. The complete click-by-click procedure is stated in Section 1.3.

### 12.3 Benchmark

The **Benchmark** tab evaluates one or more model identifiers on one dataset. Select the dataset, the category or pattern, the shot mode, the model identifiers, and optionally the profiling or resolution sweep, then select **Run benchmark**.

The results are grouped into two categories.

<table align="center">
  <thead>
    <tr>
      <th>Category</th>
      <th>Table</th>
      <th>Contents</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>technical</code></td><td><code>image_level</code></td><td>Image AUROC, image F1, image precision, and image recall.</td></tr>
    <tr><td><code>technical</code></td><td><code>pixel_level</code></td><td>Pixel AUROC, AUPRO, and IAP, when a ground-truth mask and an anomaly map are available.</td></tr>
    <tr><td><code>technical</code></td><td><code>instance_level</code></td><td>Detection boxes and instance metrics, for the models that produce boxes.</td></tr>
    <tr><td><code>technical</code></td><td><code>cross_domain</code></td><td>Scores on a held-out dataset or pattern.</td></tr>
    <tr><td><code>overhead</code></td><td><code>compute</code></td><td>Wall-clock time, frames per second, latency, FLOPs, and the edge-deployment index when the selected profiler provides it.</td></tr>
    <tr><td><code>overhead</code></td><td><code>memory</code></td><td>Peak and average memory, when the selected profiler provides them.</td></tr>
    <tr><td><code>overhead</code></td><td><code>communication</code></td><td>Exported-model transfer-size proxy. Declared but not implemented.</td></tr>
  </tbody>
</table>

A metric that cannot be computed is reported as unavailable rather than inferred from output that the model does not produce. The web interface writes each benchmark run to the run log `runs/leaderboard_log.jsonl` and writes anomaly maps under `artifacts/runtime/anomaly_maps/benchmark/`.

### 12.4 Run history

The **Run history** tab reads a saved JSON or JSONL report. Enter the report path, select **Refresh**, and optionally select a metric to filter the table. The table states the run timestamp, the model identifier, the dataset identifier, the metric values, and the report path. The tab reads existing files only and never reruns an experiment.

### 12.5 Recorded demonstrations

<table align="center">
  <thead>
    <tr>
      <th>Demonstration</th>
      <th>Recording</th>
      <th>Contents</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Web interface demonstration</td><td><a href="docs/videos/detection.mp4">detection.mp4</a></td><td>Task selection, application-domain selection, model selection, dataset selection, image loading, and detection output.</td></tr>
    <tr><td>Benchmark demonstration</td><td><a href="docs/videos/benchmark.mp4">benchmark.mp4</a></td><td>Benchmark execution and run-history reading.</td></tr>
  </tbody>
</table>

`[[FILL: record the recording date, the platform version, and the commit revision shown in each recording, so that a reader can tell which revision the recording demonstrates.]]`

## 13. Command-line workflows

### 13.1 Model configuration resolution

The `train`, `predict`, and `evaluate` commands resolve their first positional argument in the same three ways.

1. As a path to a model configuration, for example `configs/models/ultralytics_example.yaml`.
2. As a filename stem under `--config-dir`, for example `ultralytics_example`.
3. As a model keyword matched against the `model.variant` field or the `model.name` field of every model configuration under `--config-dir`, for example `yolov8n` or `patchcore`.

The default value of `--config-dir` is `configs/models`. The following three commands list three different things and must not be confused.

<table align="center">
  <thead>
    <tr>
      <th>Command</th>
      <th>Lists</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>adh train --list</code></td><td>The 13 model configurations under <code>--config-dir</code>.</td></tr>
    <tr><td><code>adh recipes</code></td><td>The registered optimization recipes, each with its paper reference and its hyperparameters.</td></tr>
    <tr><td><code>adh models</code></td><td>The model variants that each backend supports. This is not the list of published models, which is printed by <code>adh inventory</code>.</td></tr>
  </tbody>
</table>

### 13.2 Training

A training run is configured by a model configuration and then modified by the `--dataset`, `--variant`, `--mode`, and `--set` options. The option `--set` overrides any value of the resolved model configuration by dotted path and has the highest priority of any override layer.

```bash
# Train a one-class anomaly model on a general dataset.
adh train general/configs/models/anomalib.yaml \
  --variant PatchCore \
  --dataset mvtec-ad \
  --category bottle

# Train a textile detection model over the complete configured sample budget.
adh train yolov8n --dataset zju-leaper --mode full

# Train a textile one-class model and override one nested parameter.
adh train patchcore \
  --dataset zju-leaper \
  --mode medium \
  --set train.model_kwargs.coreset_sampling_ratio=0.05
```

The `moeclip` backend uses one dataset for auxiliary training and a different dataset for evaluation.

```bash
adh train moeclip \
  --dataset mvtec-ad \
  --test-dataset zju-leaper \
  --mode test \
  --no-publish
```

The `test` shot mode is an eight-image wiring check. It does not produce a usable weight and must be combined with `--no-publish`.

Publishing is enabled by default. A successful run whose model identifier appears in the model manifest replaces that identifier's published slot, which is the location that the web interface reads.

```bash
# A smoke run that does not touch any published slot.
adh train patchcore --dataset zju-leaper --mode test --no-publish
```

Expected result: a JSON object with the keys `backend`, `resolved_config`, `resolved_variant`, `metrics`, `trained_artifact`, `registered_artifact`, `published_path`, `weight_manifest_path`, and `exports`. `metrics` holds the validation metrics that the backend reports, and `weight_manifest_path` is the absolute path of the weight manifest.

### 13.3 Inference

```bash
# A single image.
adh predict patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --image /absolute/path/to/image.jpg \
  --output-dir artifacts/runtime/anomaly_maps \
  --output results/patchcore-prediction.json

# A selection of samples from a registered dataset.
adh predict patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --dataset zju-leaper \
  --split test \
  --pattern pattern1 \
  --num-samples 8 \
  --output-dir artifacts/runtime/anomaly_maps
```

The option `--output-dir` persists one anomaly map per sample as `<output directory>/<sample identifier>.npy`. It is meaningful only for a model whose capability declaration includes `anomaly_map`; it cannot make an image-level model such as GANomaly produce a heat map. The option `--output` writes the predictions as a JSON array whose schema is `schemas/prediction.schema.json`.

Expected result for the single-image command:

```json
{
  "backend": "anomalib",
  "resolved_config": "configs/models/patchcore_textile.yaml",
  "variant": "patchcore",
  "num_predictions": 1,
  "predictions": [
    {
      "sample_id": "image",
      "boxes": null,
      "labels": null,
      "scores": null,
      "masks": null,
      "anomaly_score": 0.0,
      "anomaly_map": "artifacts/runtime/anomaly_maps/image.npy"
    }
  ]
}
```

`[[FILL: replace the anomaly score above with the value observed on the reference machine for the documented reference image.]]`

### 13.4 Evaluation

```bash
adh evaluate patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --dataset zju-leaper \
  --split test \
  --num-samples 32 \
  --output-dir artifacts/runtime/anomaly_maps
```

The option `--task` forces a specific evaluator instead of using the task of the dataset samples. The option `--output-dir` is what makes the pixel-level metrics computable, because those metrics require a persisted anomaly map; without it, an anomaly model is scored on image-level metrics alone.

Expected result: a JSON object with the keys `backend`, `resolved_config`, `variant`, `sample_count`, and `metrics`. The keys of `metrics` are those listed in Section 3.

Cross-pattern robustness is measured by scoring the same weight on held-out patterns and reducing the per-pattern accuracy drops to one number.

```bash
adh evaluate patchcore \
  --weights textile/artifacts/models/published/PatchCore.ckpt \
  --dataset zju-leaper \
  --pattern pattern1-4 \
  --cross-domain-patterns 5,6,7,8 \
  --cross-domain-k 3 \
  --cross-domain-mode worst \
  --output-dir artifacts/runtime/anomaly_maps
```

The option `--cross-domain-metric` selects the metric over which the degradation is computed and defaults to the headline metric of the task. The option `--cross-domain-mode` selects whether the reported mean degradation averages over the largest drops (`worst`) or the smallest drops (`best`). The chosen mode is echoed in the output, because the two modes report opposite results.

Expected result: the JSON object of the direct evaluation, augmented with the key `cross_domain`. A pattern that cannot be scored on this machine is reported as skipped rather than counted as a degradation of zero.

### 13.5 Benchmarking

A benchmark is declared by a benchmark configuration whose top-level keys are `runs`, `output_dir`, `leaderboard`, `report_path`, and `run_log_path`. The `runs` key is a non-empty list, and each entry declares one experiment.

```bash
adh benchmark configs/archive/benchmark_example.yaml
```

Expected result: a JSON array of experiment results, one element per run. The example configuration writes its outputs under `artifacts/benchmarks/example`.

`[[FILL: add a benchmark configuration that a reader can run without editing, using a dataset and weights that the reader already has, and record its expected leaderboard.]]`

### 13.6 Batch training

The `train-all` command trains every model identifier of the model manifest in one resumable batch, with one log file and one state record per model identifier.

```bash
adh train-all --dry-run
adh train-all --only yolov8n PatchCore --mode test --no-publish
adh train-all --run-id [[FILL: batch run identifier]] --resume
```

Expected result: a JSON object with the keys `batch_state`, `succeeded`, `total`, and `results`. The directory of `batch_state` holds the per-model state and the per-model log.

### 13.7 Catalogue and diagnostic commands

<table align="center">
  <thead>
    <tr>
      <th>Command</th>
      <th>Function</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>adh inventory</code></td><td>Print the unified machine-readable inventory of models and datasets.</td></tr>
    <tr><td><code>adh list</code></td><td>Print the registered datasets, backends, evaluators, and profilers.</td></tr>
    <tr><td><code>adh models</code></td><td>Print the model variants of each backend. Restrict the output with <code>--backend</code>.</td></tr>
    <tr><td><code>adh recipes</code></td><td>Print the registered optimization recipes.</td></tr>
    <tr><td><code>adh doctor</code></td><td>Print, for each backend, whether it is trainable on this machine and which dataset would be selected.</td></tr>
    <tr><td><code>adh train --list</code></td><td>Print the model configurations that a training command can resolve.</td></tr>
    <tr><td><code>adh export-latex</code></td><td>Convert a benchmark result file into a LaTeX table.</td></tr>
  </tbody>
</table>

## 14. Outputs and reproducibility

### 14.1 Output tree

```text
artifacts/
├── models/
│   ├── <trained artifact>                       # the run-specific weight
│   ├── weight_manifest.jsonl                    # append-only provenance records
│   └── records/
│       └── <record identifier>.config.json      # the resolved configuration of one run
├── benchmarks/
│   └── <benchmark output directory>/
│       ├── leaderboard.csv                      # when report_path is configured
│       └── <experiment identifier>/
│           ├── result.json                      # ExperimentResult
│           └── predictions.json                 # Prediction list
└── runtime/
    └── anomaly_maps/
        ├── <sample identifier>.npy              # adh predict --output-dir
        └── benchmark/<experiment identifier>/   # web interface benchmark

<application domain>/artifacts/models/published/
└── <model identifier><extension>                # the published slot
    └── <model identifier><extension>.metadata.json   # optional provenance sidecar

runs/
└── leaderboard_log.jsonl                        # append-only evaluation and benchmark log

results/
└── <user-specified output>.json                 # adh predict --output
```

### 14.2 Provenance records

A provenance record states the timestamp, the Git revision, whether the working tree was modified, the host and platform, the Python environment, the versions of the installed packages, and the revisions of the component checkouts. The same provenance block is attached both to the weight manifest and to the run log, so a training result and an evaluation result can be traced to the same revision.

### 14.3 Reproducibility checklist

Before reporting a result, confirm every item of the following list.

1. The model identifier and the model configuration are stated exactly.
2. The dataset identifier and the split are stated exactly.
3. The shot mode and the sample count are stated exactly.
4. The published-slot state was `file` or `symlink` when the run started.
5. The provenance record of the run was retained.
6. The anomaly maps were retained if any pixel-level metric is reported.
7. The metric keys are quoted exactly as the evaluator emits them, and the metric key reported as the headline result is named.
8. Any metric that could not be computed is reported as unavailable rather than as zero.

Do not commit a generated dataset, a trained artifact, an anomaly map, a cache, or a result. Distribute a durable weight through the channel documented in Section 11.2.

## 15. Tools reference

<table align="center">
  <thead>
    <tr>
      <th>Tool</th>
      <th>Function</th>
      <th>Interface</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>tools/download_datasets.py</code></td><td>Download a registered dataset.</td><td><code>mvtec-ad</code>, <code>visa</code>, or <code>zju-leaper</code>; required <code>--root</code>; optional <code>--category</code> and <code>--repo-id</code></td></tr>
    <tr><td><code>tools/download_weights.py</code></td><td>Download a weight from Hugging Face Hub.</td><td>positional <code>repo_id</code> and <code>filename</code>; required <code>--output</code>; optional <code>--revision</code></td></tr>
    <tr><td><code>tools/export_model.py</code></td><td>Export a trained artifact, optionally building a TensorRT engine or applying post-training ONNX quantization.</td><td>required <code>--backend</code>, <code>--model</code>, <code>--artifact</code>, and <code>--target</code>; optional <code>--precision</code>, <code>--calibration-dir</code>, <code>--quantize-level</code>, and <code>--input-size</code></td></tr>
    <tr><td><code>tools/visualize_predictions.py</code></td><td>Render boxes, masks, and anomaly maps onto images.</td><td>positional <code>samples</code>, <code>predictions</code>, and <code>output_dir</code></td></tr>
    <tr><td><code>tools/run_metric_sweep.py</code></td><td>Measure every implemented metric over every model that has a weight.</td><td><code>--dataset</code>, <code>--groups</code>, <code>--models</code>, and <code>--output</code>; use <code>--list</code> to print the plan without measuring</td></tr>
    <tr><td><code>tools/run_full_benchmark.sh</code></td><td>Run the full post-training measurement, including the cross-domain sweep, from environment variables.</td><td>environment variables prefixed <code>FDH_</code>; notably <code>FDH_POWER_MODE</code>, which is required</td></tr>
    <tr><td><code>tools/collect_cloud_artifacts.py</code></td><td>Collect and optionally archive the artifacts of a batch run.</td><td>required <code>--run-id</code>; optional <code>--verify-only</code> and <code>--archive</code></td></tr>
    <tr><td><code>tools/preflight_cloud_models.py</code></td><td>Report, before a cloud run, which model identifiers are blocked by a missing path.</td><td>no arguments; prints a report</td></tr>
    <tr><td><code>tools/smoke_test_all_backends.py</code></td><td>Train every model identifier for one step, as a wiring check.</td><td>no arguments; starts training immediately</td></tr>
    <tr><td><code>tools/convert_annotations.py</code></td><td>Convert COCO detection annotations into a <code>Sample</code> JSON file.</td><td>positional <code>annotations</code> and <code>output</code>; optional <code>--image-root</code></td></tr>
    <tr><td><code>tools/plot_training_curves.py</code></td><td>Render SVG training curves from the YOLO <code>results.csv</code> file and the torchvision <code>history.csv</code> file.</td><td>positional <code>paths</code>, being history CSV files or directories searched recursively; optional <code>--output-dir</code></td></tr>
    <tr><td><code>tools/prune_artifacts.py</code></td><td>Inspect, and optionally apply, the storage cleanup of <code>artifacts/models/</code>.</td><td>optional <code>--dedupe-published</code>, <code>--prune-checkpoints</code>, <code>--keep</code>, <code>--run-root</code>, <code>--project-root</code>, and <code>--apply</code>. Without <code>--apply</code>, the tool prints the plan and writes nothing.</td></tr>
    <tr><td><code>tools/export_benchmark_csv.mjs</code></td><td>Convert the JSONL output of the metric sweep into two CSV files.</td><td>no arguments; reads <code>artifacts/runtime/full_sweep.jsonl</code> and writes <code>artifacts/runtime/benchmark_results_long.csv</code> and <code>artifacts/runtime/benchmark_model_summary.csv</code>. Requires the Node.js package <code>@oai/artifact-tool</code>.</td></tr>
  </tbody>
</table>

The tools `preflight_cloud_models.py` and `smoke_test_all_backends.py` accept no arguments and act when they are invoked. Do not invoke them with `--help` intending to inspect their interface.

`[[FILL: add one worked example, with its expected output, for each tool that a reader is expected to use.]]`

## 16. Testing and quality gates

<table align="center">
  <thead>
    <tr>
      <th>Command</th>
      <th>Gate</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>pytest -q</code></td><td>The runtime and integration contracts. The architecture audits are deselected by default, as declared by <code>addopts</code> in <code>pyproject.toml</code>.</td></tr>
    <tr><td><code>pytest -q -m architecture</code></td><td>The source-boundary audits, which parse the source with <code>ast</code> and enforce the layering rule of Section 6.2.</td></tr>
    <tr><td><code>git diff --check</code></td><td>Repository hygiene: whitespace errors and conflict markers.</td></tr>
    <tr><td><code>make test</code></td><td>Equivalent to <code>pytest -q</code>.</td></tr>
    <tr><td><code>make test-architecture</code></td><td>Equivalent to <code>pytest -q -m architecture</code>.</td></tr>
    <tr><td><code>make check</code></td><td>All three gates.</td></tr>
  </tbody>
</table>

A test marked `slow` requires the real framework dependencies, the real weights, and access to the dataset, and is not part of the default run.

## 17. Extending the platform

### 17.1 Add a model

1. Reuse an existing backend, or implement the `ModelAdapter` contract.
2. Implement `train`, `predict`, `export`, and `load_trained_model`, and declare a truthful capability declaration.
3. Add a model configuration under `configs/models/`, or under `general/configs/models/` when the model belongs to the `general` application domain.
4. Add exactly one entry to the model manifest, including the method, the application domain, the training corpus, the training split, the model configuration, and the weight source.
5. Supply a weight for the published slot, or leave the published slot empty so that its state is reported as `missing`.
6. Run the runtime tests and the architecture audits of Section 16.

### 17.2 Add a dataset

1. Implement `DatasetAdapter.load_samples()` and register the adapter.
2. Declare the default root, the tasks, the training roles, and the application domain in `core/dataset_capabilities.py`.
3. Add the dataset to the presentation table in `application/workspace.py` if the web interface must offer it.
4. Add contract tests for the normal case, the anomalous case, the missing case, and the malformed case.

### 17.3 Add an application domain

1. Create `datasets/<application domain>/` and the dataset adapters of that application domain.
2. Create `<application domain>/domain.yaml`, the configuration overlays, and the examples.
3. Create the `<application domain>/artifacts/` directory.
4. Declare the application domain of every dataset of the new domain in `core/dataset_capabilities.py`.
5. Train or calibrate the weights of the new domain and add separate model-manifest entries with the correct `domain` and `trained_on` values.

Do not copy a reusable algorithm into an application-domain directory, and do not relabel a weight of one application domain as a weight of another.

## 18. Troubleshooting

### 18.1 A dataset is reported as unavailable

Run `adh doctor`. Confirm that the dataset directory exists at the declared default root and that a partial download did not leave it incomplete. Pass `--dataset-root` to use a different location for one command.

### 18.2 A component checkout is missing

```bash
git submodule update --init --recursive
git submodule status
adh doctor
```

### 18.3 Importing the web interface raises a SOCKS proxy error

Install the SOCKS dependency declared by `requirements.txt` as `httpx[socks]`.

```bash
python -m pip install -r requirements.txt
```

The web interface additionally removes unusable SOCKS proxy variables from its own process when the `socksio` package is unavailable.

### 18.4 A weight is missing

Run `adh inventory` and read the `weight` field and the `weight_status` field of the model identifier. Place the weight at the reported path. Do not rename the weight of a different architecture to satisfy a published slot.

### 18.5 The web interface displays an obsolete label

The model manifest is read when the Python process starts. Stop the running process and start it again with `adh-ui`. If the port is already in use, `adh-ui` reports the process that holds it.

### 18.6 No anomaly heat map is displayed

Read the capability declaration of the model identifier. GANomaly reports an image-level score only. For a model that declares `anomaly_map`, supply an output directory, and confirm that the selected dataset carries ground-truth masks if a pixel-level metric is required.

### 18.7 Training runs out of memory

Use the `test` shot mode, reduce the batch size with `--set`, reduce the input resolution where the backend supports it, or select a smaller model variant. A CUDA-only accelerator is optional and must match the installed PyTorch and CUDA toolchain.

### 18.8 A general training run fails because the dataset is not a one-class training source

Confirm that the dataset declares the `anomaly_train` role. The roles are declared in `core/dataset_capabilities.py` and are reported by `adh doctor`.

## 19. Terminology

This section defines every technical term used in this document. Each term carries exactly one meaning. `<application domain>` is a placeholder for either `textile` or `general`.

<table align="center">
  <thead>
    <tr>
      <th>Term</th>
      <th>Definition</th>
      <th>Declared in</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>application domain</td><td>A data family and its assets. The registered application domains are <code>textile</code> and <code>general</code>.</td><td><code>configs/registry/models.yaml</code> (<code>domain</code>)</td></tr>
    <tr><td>backend</td><td>The execution framework through which a model runs. The registered backends are <code>ultralytics</code>, <code>torchvision</code>, <code>anomalib</code>, <code>dinomaly</code>, <code>moeclip</code>, and <code>mambaad</code>.</td><td><code>core/registry.py</code>, <code>loader._MODEL_BACKEND_MODULES</code></td></tr>
    <tr><td>model variant</td><td>The architecture selected within a backend, for example <code>yolov8n</code> or <code>PatchCore</code>.</td><td><code>configs/registry/models.yaml</code> (<code>variant</code>)</td></tr>
    <tr><td>model identifier</td><td>The stable identifier of one entry of the model manifest, for example <code>PatchCore</code> or <code>PatchCore_general</code>.</td><td><code>configs/registry/models.yaml</code> (<code>id</code>)</td></tr>
    <tr><td>model manifest</td><td>The single authoritative catalogue of published models: <code>configs/registry/models.yaml</code>.</td><td><code>catalog.py</code></td></tr>
    <tr><td>model configuration</td><td>A YAML file that declares the executable parameters of one run, for example <code>configs/models/ultralytics_example.yaml</code>.</td><td><code>configs/models/</code>, <code>general/configs/models/</code></td></tr>
    <tr><td>optimization recipe</td><td>A paper-anchored hyperparameter profile registered under <code>src/fabric_defect_hub/recipes/</code>. An optimization recipe is not a model configuration.</td><td><code>fabric_defect_hub.recipes</code></td></tr>
    <tr><td>dataset adapter</td><td>A <code>DatasetAdapter</code> subclass that converts one source dataset into <code>Sample</code> objects.</td><td><code>src/fabric_defect_hub/datasets/</code></td></tr>
    <tr><td>sample</td><td>One image together with its annotations, represented by <code>core.types.Sample</code>.</td><td><code>core/types.py</code></td></tr>
    <tr><td>prediction</td><td>One model output, represented by <code>core.types.Prediction</code>.</td><td><code>core/types.py</code></td></tr>
    <tr><td>task</td><td>One of <code>detection</code>, <code>segmentation</code>, <code>instance_segmentation</code>, <code>anomaly</code>, and <code>industrial</code>.</td><td><code>models/base.py</code> (<code>TASKS</code>)</td></tr>
    <tr><td>annotation</td><td>A task-native ground-truth field of a sample: <code>boxes</code>, <code>masks</code>, <code>labels</code>, <code>is_anomalous</code>, or <code>anomaly_mask</code>.</td><td><code>core/types.py</code> (<code>Annotations</code>)</td></tr>
    <tr><td>capability declaration</td><td>A <code>ModelCapabilities</code> or <code>DatasetCapabilities</code> value that states, without executing the model or reading the dataset, which tasks and which output fields are supported.</td><td><code>models/base.py</code>, <code>core/dataset_capabilities.py</code></td></tr>
    <tr><td>shot mode</td><td>The sample budget of a training run: <code>full</code>, <code>medium</code>, <code>few</code>, or <code>test</code>.</td><td><code>training.ShotMode</code></td></tr>
    <tr><td>split</td><td>The partition drawn from a dataset: <code>train</code> or <code>test</code>.</td><td><code>cli.py</code> (<code>--split</code>)</td></tr>
    <tr><td>trained artifact</td><td>The run-specific weight file written under <code>artifacts/models/</code>. The source symbol is <code>registered_artifact</code>.</td><td><code>training.run_train</code></td></tr>
    <tr><td>published slot</td><td>The fixed path <code>&lt;application domain&gt;/artifacts/models/published/&lt;model identifier&gt;&lt;extension&gt;</code> that the web interface reads. A published slot holds either a regular file or a symbolic link.</td><td><code>catalog.published_path</code></td></tr>
    <tr><td>weight</td><td>A model parameter file.</td><td>—</td></tr>
    <tr><td>weight manifest</td><td>The append-only provenance log <code>artifacts/models/weight_manifest.jsonl</code>.</td><td><code>weight_registry.py</code></td></tr>
    <tr><td>provenance record</td><td>One line of the weight manifest.</td><td><code>core/provenance.py</code></td></tr>
    <tr><td>run log</td><td>The append-only evaluation log <code>runs/leaderboard_log.jsonl</code>.</td><td><code>application/benchmark.py</code></td></tr>
    <tr><td>anomaly map</td><td>A per-pixel anomaly score array persisted as <code>&lt;output directory&gt;/&lt;sample identifier&gt;.npy</code>.</td><td><code>models/anomalib/adapter.py</code></td></tr>
    <tr><td>Minimal Working Example</td><td>The shortest procedure that produces a visible result, stated in Section 1.</td><td>this document</td></tr>
  </tbody>
</table>

Two rules follow from the table.

1. `model configuration` never means `optimization recipe`, and `published slot` never means `weight`.
2. A command-line command is written exactly as it must be typed, including the `adh` program name.

## 20. Documentation placeholders

The following placeholders must be completed by the maintainers. Each placeholder is written in the document as `[[FILL: description]]`.

<table align="center">
  <thead>
    <tr>
      <th>Section</th>
      <th>Placeholder</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1.1</td><td>Confirm the canonical clone URL and, if the project is mirrored, the mirror URL.</td></tr>
    <tr><td>1.3</td><td>Record the observed anomaly scores, the observed runtime, and a screenshot of the completed Minimal Working Example run.</td></tr>
    <tr><td>10.2</td><td>Record the acquisition procedure, the archive size, the license, the download date, and the staged directory structure for each dataset that must be staged manually.</td></tr>
    <tr><td>10.4</td><td>Record the expected category or pattern count for each staged dataset.</td></tr>
    <tr><td>10.5</td><td>Record the source URL for RAW-FABRID, TILDA-400, the Fabric Defects Dataset, and the Tianchi challenge.</td></tr>
    <tr><td>11.2</td><td>Record the Hugging Face Hub repository identifier, the revision, and the SHA-256 checksum of every distributed file.</td></tr>
    <tr><td>11.2</td><td>Record the acquisition procedure for the weights that are not distributed through the Hugging Face Hub repository.</td></tr>
    <tr><td>11.2</td><td>State whether the WinCLIP published slot must be distributed, or whether the model is always reconstructed from its metadata.</td></tr>
    <tr><td>12.5</td><td>Record the recording date, the platform version, and the commit revision shown in each recording.</td></tr>
    <tr><td>13.3</td><td>Replace the example anomaly score with the value observed on the reference machine.</td></tr>
    <tr><td>13.5</td><td>Add a benchmark configuration that a reader can run without editing, and record its expected leaderboard.</td></tr>
    <tr><td>13.6</td><td>Record a reference batch run identifier.</td></tr>
    <tr><td>15</td><td>Add one worked example, with its expected output, for each tool that a reader is expected to use.</td></tr>
    <tr><td>21</td><td>Record the maintainer contact and the license statement.</td></tr>
  </tbody>
</table>

## 21. Additional documentation

<table align="center">
  <thead>
    <tr>
      <th>Document</th>
      <th>Contents</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><a href="docs/model-integration.md">docs/model-integration.md</a></td><td>The integration types of a model, and the publishing checklist.</td></tr>
    <tr><td><a href="docs/interface.md">docs/interface.md</a></td><td>The unified interface contract.</td></tr>
    <tr><td><a href="docs/config.md">docs/config.md</a></td><td>The configuration model.</td></tr>
    <tr><td><a href="docs/deployment.md">docs/deployment.md</a></td><td>The deployment procedure and the release checks.</td></tr>
    <tr><td><a href="docs/weights.md">docs/weights.md</a></td><td>Weight storage, distribution, and provenance.</td></tr>
    <tr><td><a href="docs/cloud_training_checklist.md">docs/cloud_training_checklist.md</a></td><td>The cloud training checklist.</td></tr>
    <tr><td><a href="docs/report.md">docs/report.md</a></td><td>The benchmark report format and the metric taxonomy.</td></tr>
    <tr><td><a href="docs/add_new.md">docs/add_new.md</a></td><td>Adding a dataset, an application domain, or a backend.</td></tr>
    <tr><td><a href="docs/adr/0001-unified-registry-and-adapters.md">docs/adr/0001-unified-registry-and-adapters.md</a></td><td>The architecture decision record for the unified registry and the adapters.</td></tr>
    <tr><td><a href="general/README.md">general/README.md</a></td><td>The <code>general</code> application domain.</td></tr>
    <tr><td><a href="textile/README.md">textile/README.md</a></td><td>The <code>textile</code> application domain.</td></tr>
    <tr><td><a href="datasets/README.md">datasets/README.md</a></td><td>Dataset storage conventions.</td></tr>
    <tr><td><a href="examples/README.md">examples/README.md</a></td><td>The Minimal Working Example scripts.</td></tr>
    <tr><td><a href="CONTRIBUTING.md">CONTRIBUTING.md</a></td><td>The contribution procedure.</td></tr>
    <tr><td><a href="LICENSE">LICENSE</a></td><td>The license.</td></tr>
  </tbody>
</table>

Maintainer contact and license statement: `[[FILL: record the maintainer contact and confirm the license statement.]]`
