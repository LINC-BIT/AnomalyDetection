# General anomaly detection

This domain contains dataset-independent training configurations for industrial anomaly detection. Reusable model code lives in `src/fabric_defect_hub/models/`; upstream research code remains in `components/`.

## Supported training datasets

| Dataset | Registry name | Default root | One-class training | MoECLIP auxiliary training |
|---|---|---|---|---|
| MVTec AD | `mvtec-ad` | `datasets/general/MVTec AD` | yes | yes |
| MVTec LOCO | `mvtec-loco` | `datasets/general/MVTec LOCO` | yes | yes |
| VisA | `visa` | `datasets/general/VisA` | yes | yes |

Other datasets require adapters that describe their published directory and annotation formats. Add each adapter through the dataset interface before using its registry name.

## Training commands

Anomalib models:

```bash
adh train general/configs/models/anomalib.yaml --variant PatchCore --dataset mvtec-ad --category bottle
adh train general/configs/models/anomalib.yaml --variant PaDiM --dataset visa --category pcb1
```

Dinomaly and MambaAD:

```bash
adh train general/configs/models/dinomaly.yaml --dataset mvtec-ad --category bottle
adh train general/configs/models/mambaad.yaml --dataset visa --category pcb1
```

MoECLIP uses one dataset for supervised auxiliary training and another for evaluation:

```bash
adh train general/configs/models/moeclip.yaml \
  --dataset visa \
  --test-dataset mvtec-ad
```

Use `--mode test --no-publish` for a smoke run. General checkpoints are stored under `general/artifacts/models/`.
