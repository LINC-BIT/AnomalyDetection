# Weight storage and distribution

Weights are immutable runtime artifacts and are not committed to Git. Textile weights use `textile/artifacts/models/published/<model-id>.<extension>`. General weights use `general/artifacts/models/published/<model-id>.<extension>`. A file without a manifest row is unsupported; a row without a file is reported as `missing`.

The current published-weight repository is
[`AuroraLeeeeee/AnomalyDetection-textile-weights`](https://huggingface.co/AuroraLeeeeee/AnomalyDetection-textile-weights).
It contains both domain directories' published files, despite the repository's
historical name. Download a file with:

```bash
python tools/download_weights.py \
  AuroraLeeeeee/AnomalyDetection-textile-weights \
  general/artifacts/models/published/WinCLIP.ckpt \
  --output general/artifacts/models/published/WinCLIP.ckpt
```

Use the same command with the exact path from the manifest for every other
checkpoint. The destination path is part of the runtime contract.

## WinCLIP provenance

`WinCLIP` is a general-domain, upstream-pretrained zero-shot model. Its
canonical entry has `trained_on: [LAION-400M]`, `training_split:
external_pretraining`, and `source: upstream_pretrained`. It is not trained on
the general MVTec/VisA datasets and it is not trained on textile data. The
ZJU-Leaper split in `configs/models/anomalib_winclip.yaml` is an evaluation
target; `k_shot: 0` means no dataset examples are fitted into the model.

Public releases should use a dedicated Hugging Face Hub model repository with immutable revisions. Each entry should include repository ID, revision, filename, byte size, and SHA-256; download tooling must verify the checksum before publication. A project cache is preferable to a generic Python library cache because it keeps deployment provenance explicit. Google Drive is legacy/manual import only, including the current MoECLIP source.

The UI never downloads implicitly. Existing local weights work offline, while missing files show their exact expected path. Before release, record training dataset/split/domain, export and hash the canonical checkpoint, upload it to a versioned revision, update the manifest, and test a clean download through inference, evaluation, and the UI Minimal Working Example.
